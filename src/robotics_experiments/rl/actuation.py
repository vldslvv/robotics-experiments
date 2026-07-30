"""Pure RL-action -> per-rotor angular-velocity actuation functions (core path).

numpy + stdlib only. State lives in frozen params dataclasses and behavior in
pure top-level functions (``f(params, inputs) -> outputs``), so the actuation
stage composes into a plain function pipeline. This is the single-env
reference/spec implementation and the unit-test oracle; it is exported through
``robotics_experiments.rl``. The Isaac Lab package exposes the batched torch
twin's external diagnostic entry point as
``robotics_experiments.workflows.isaac_lab.compute_lab_rotor_omega``; it must
match this module term-for-term (both precompute the allocation pseudoinverse in
their builder).

The rotor mix inverts the exact allocation the PX4 bridge's
``PegasusQuadraticThrustModel.commands_to_dynamics`` realizes: per-rotor thrust
``T_i = rotor_constant_i * omega_i^2`` applied at the body-FLU rotor offsets, a
body-Z yaw reaction ``sum(rolling_moment_coefficient_i * omega_i^2 * rot_dir_i)``,
so feeding the output omega back through that thrust model reproduces the
commanded collective-thrust + body-torque wrench.

Two action levels (``kind``) are provided:

- ``velocity`` -- action ``[vx, vy, vz, yaw_rate]`` (normalized): a cascaded
  geometric controller tracks a yaw-frame body velocity + yaw-rate command.
- ``ctbr`` -- action ``[thrust, wx, wy, wz]`` (normalized): collective thrust +
  body-rate setpoint closed by the inner rate loop shared with ``velocity``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robotics_experiments.alignment import ControllerSettings
from robotics_experiments.config import RlActionConfig
from robotics_experiments.px4_bridge import GRAVITY_M_S2, rotate_vector_wxyz

__all__ = [
    "ActuationControllerState",
    "ActuationParams",
    "ActuationState",
    "MixerGeometry",
    "action_to_rotor_omega",
    "build_actuation_params",
    "hover_action",
    "initial_actuation_controller_state",
    "rotor_forces_to_body_wrench",
]

_EPS = 1e-9
_SATURATION_EPS = 1e-5


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MixerGeometry:
    """Airframe control-allocation geometry in PX4 motor-index order (body-FLU).

    Mirrors the fields the bridge's ``PegasusQuadraticThrustModel`` consumes; the
    per-rotor arrays are ordered by PX4 motor index. ``positions_body`` are the
    rotor marker offsets from the body origin (from the package-root
    ``DiscoveredRotorMapping``); ``rot_dir`` is ``+1`` CCW / ``-1`` CW.
    """

    positions_body: np.ndarray
    rot_dir: np.ndarray
    rotor_constant: np.ndarray
    rolling_moment_coefficient: np.ndarray
    min_rotor_velocity: np.ndarray
    max_rotor_velocity: np.ndarray


@dataclass(frozen=True)
class ActuationState:
    """Single-env vehicle state the controllers read.

    Attributes:
        attitude_flu_enu_wxyz: Body-FLU to world-ENU rotation quaternion
            ``[w, x, y, z]`` (matches ``VehicleState.orientation_flu_enu_wxyz``).
        linear_velocity_enu: World-ENU linear velocity (m/s).
        angular_velocity_body_flu: Body-FLU angular velocity (rad/s).
    """

    attitude_flu_enu_wxyz: np.ndarray
    linear_velocity_enu: np.ndarray
    angular_velocity_body_flu: np.ndarray


@dataclass(frozen=True)
class ActuationControllerState:
    """State carried by the PX4-like velocity and body-rate cascade."""

    filtered_velocity_enu: np.ndarray
    acceleration_enu: np.ndarray
    velocity_integral_enu: np.ndarray
    previous_angular_velocity_body_flu: np.ndarray
    angular_acceleration_body_flu: np.ndarray
    rate_integral_body_flu: np.ndarray
    initialized: bool


@dataclass(frozen=True)
class ActuationParams:
    """Immutable actuation state: geometry + derived allocation + gains + limits.

    Built once by :func:`build_actuation_params`; every actuation function is a
    pure function of these params and its inputs. The velocity-cascade gains and
    limits are populated for both kinds (``ctbr`` ignores them), matching the
    torch twin.
    """

    kind: str
    action_dim: int
    mass: float
    geometry: MixerGeometry
    allocation_pinv: np.ndarray
    max_total_thrust: float
    kp_rate: np.ndarray
    ki_rate: np.ndarray
    kd_rate: np.ndarray
    kff_rate: np.ndarray
    rate_output_scale_nm: np.ndarray
    rate_derivative_filter_hz: float
    rate_integral_limit_nm: np.ndarray
    max_body_rate_radps: float
    hover_thrust_fraction: float
    kp_vel: np.ndarray
    ki_vel: np.ndarray
    kd_vel: np.ndarray
    velocity_filter_hz: float
    velocity_derivative_filter_hz: float
    velocity_integral_limit_mps2: np.ndarray
    kp_att: np.ndarray
    attitude_yaw_weight: float
    max_body_velocity_mps: float
    max_yaw_rate_radps: float
    max_tilt_rad: float
    min_collective_thrust_fraction: float
    max_collective_thrust_fraction: float
    horizontal_thrust_margin_fraction: float


def _num_rotors(geometry: MixerGeometry) -> int:
    return int(geometry.rotor_constant.shape[0])


def _allocation_matrix(geometry: MixerGeometry) -> np.ndarray:
    """Return the ``(4, N)`` map from per-rotor thrust to ``[Fz, tx, ty, tz]``.

    Rows: collective thrust (sum), roll torque ``sum(y_i * f_i)``, pitch
    torque ``-sum(x_i * f_i)``, yaw torque
    ``sum((c_i / k_i) * rot_dir_i * f_i)`` -- the linear-in-thrust form of the
    thrust model's yaw reaction (``omega_i^2 = f_i / k_i``).
    """
    x = geometry.positions_body[:, 0]
    y = geometry.positions_body[:, 1]
    yaw = (geometry.rolling_moment_coefficient / geometry.rotor_constant) * geometry.rot_dir
    return np.stack([np.ones(_num_rotors(geometry)), y, -x, yaw], axis=0)


def _max_total_thrust(geometry: MixerGeometry) -> float:
    """Summed per-rotor static max thrust at ``max_rotor_velocity`` (N)."""
    return float(np.sum(geometry.rotor_constant * np.square(geometry.max_rotor_velocity)))


def _rotation_from_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Return the ``3x3`` body-FLU -> world-ENU rotation matrix (columns = axes)."""
    x_axis = rotate_vector_wxyz(q_wxyz, np.array([1.0, 0.0, 0.0]))
    y_axis = rotate_vector_wxyz(q_wxyz, np.array([0.0, 1.0, 0.0]))
    z_axis = rotate_vector_wxyz(q_wxyz, np.array([0.0, 0.0, 1.0]))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _vee(matrix: np.ndarray) -> np.ndarray:
    """Inverse hat map: the axial vector of a skew-symmetric ``3x3`` matrix."""
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]], dtype=np.float64)


def _clamp_tilt(thrust_world: np.ndarray, max_tilt_rad: float) -> np.ndarray:
    """Unit desired body-Z from a world thrust vector, tilt-limited from up."""
    norm = float(np.linalg.norm(thrust_world))
    if norm < _EPS:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    direction = thrust_world / norm
    up = np.array([0.0, 0.0, 1.0])
    cos_tilt = float(np.clip(direction[2], -1.0, 1.0))
    tilt = np.arccos(cos_tilt)
    if tilt <= max_tilt_rad:
        return direction
    horizontal = direction - cos_tilt * up
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm < _EPS:
        return up
    horizontal /= horizontal_norm
    return np.cos(max_tilt_rad) * up + np.sin(max_tilt_rad) * horizontal


def _wrench_to_omega(
    params: ActuationParams,
    collective_thrust: float,
    body_torque: np.ndarray,
) -> np.ndarray:
    """Invert the allocation: desired wrench -> clipped per-rotor omega.

    Solves ``A f = [collective, tau_x, tau_y, tau_z]`` via the precomputed
    pseudoinverse (exact for a square quad, least-squares for a redundant hexa),
    clips ``f >= 0``, then ``omega_i = sqrt(f_i / k_i)`` clipped to
    ``[min, max]``. Feeding the result through the thrust model reproduces the
    commanded wrench (up to the clips).
    """
    geometry = params.geometry
    wrench = np.array(
        [collective_thrust, body_torque[0], body_torque[1], body_torque[2]],
        dtype=np.float64,
    )
    thrust = params.allocation_pinv @ wrench
    thrust = np.clip(thrust, 0.0, None)
    omega = np.sqrt(thrust / geometry.rotor_constant)
    return np.clip(omega, geometry.min_rotor_velocity, geometry.max_rotor_velocity)


def _low_pass_alpha(*, cutoff_hz: float, dt_s: float) -> float:
    return 1.0 - float(np.exp(-2.0 * np.pi * cutoff_hz * dt_s))


def _limited_thrust_world(
    *,
    params: ActuationParams,
    desired_acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gravity = np.array([0.0, 0.0, GRAVITY_M_S2])
    requested = params.mass * (desired_acceleration + gravity)
    maximum = params.max_total_thrust * params.max_collective_thrust_fraction
    minimum = params.max_total_thrust * params.min_collective_thrust_fraction
    vertical = float(np.clip(requested[2], minimum, maximum))
    radial_limit = np.sqrt(max(maximum * maximum - vertical * vertical, 0.0))
    margin_limit = params.max_total_thrust * (
        params.max_collective_thrust_fraction - params.horizontal_thrust_margin_fraction
    )
    horizontal_limit = min(radial_limit, margin_limit)
    horizontal = requested[:2]
    norm = float(np.linalg.norm(horizontal))
    if norm > horizontal_limit:
        horizontal = horizontal * (horizontal_limit / max(norm, _EPS))
    limited = np.array([horizontal[0], horizontal[1], vertical], dtype=np.float64)
    achieved_acceleration = limited / params.mass - gravity
    return limited, achieved_acceleration


def _rate_to_omega(
    params: ActuationParams,
    controller_state: ActuationControllerState,
    collective_thrust: float,
    body_rate_setpoint: np.ndarray,
    state: ActuationState,
    dt_s: float,
    /,
) -> tuple[ActuationControllerState, np.ndarray]:
    """Stateful body-rate PID: rate setpoint -> body torque -> rotor omega."""
    rate = np.clip(body_rate_setpoint, -params.max_body_rate_radps, params.max_body_rate_radps)
    angular_velocity = np.asarray(state.angular_velocity_body_flu, dtype=np.float64)
    if controller_state.initialized:
        raw_angular_acceleration = (
            angular_velocity - controller_state.previous_angular_velocity_body_flu
        ) / dt_s
        alpha = _low_pass_alpha(
            cutoff_hz=params.rate_derivative_filter_hz,
            dt_s=dt_s,
        )
        angular_acceleration = controller_state.angular_acceleration_body_flu + alpha * (
            raw_angular_acceleration - controller_state.angular_acceleration_body_flu
        )
    else:
        angular_acceleration = np.zeros(3, dtype=np.float64)
    rate_error: np.ndarray = rate - angular_velocity
    rate_integral = np.clip(
        controller_state.rate_integral_body_flu + params.ki_rate * rate_error * dt_s,
        -params.rate_integral_limit_nm,
        params.rate_integral_limit_nm,
    )
    body_torque = params.rate_output_scale_nm * (
        params.kp_rate * rate_error
        + rate_integral
        - params.kd_rate * angular_acceleration
        + params.kff_rate * rate
    )
    successor = ActuationControllerState(
        filtered_velocity_enu=_readonly_copy(controller_state.filtered_velocity_enu),
        acceleration_enu=_readonly_copy(controller_state.acceleration_enu),
        velocity_integral_enu=_readonly_copy(controller_state.velocity_integral_enu),
        previous_angular_velocity_body_flu=_readonly_copy(angular_velocity),
        angular_acceleration_body_flu=_readonly_copy(angular_acceleration),
        rate_integral_body_flu=_readonly_copy(rate_integral),
        initialized=True,
    )
    return successor, _wrench_to_omega(params, collective_thrust, body_torque)


def _velocity_action_to_omega(
    params: ActuationParams,
    controller_state: ActuationControllerState,
    action: np.ndarray,
    state: ActuationState,
    dt_s: float,
    /,
) -> tuple[ActuationControllerState, np.ndarray]:
    """Cascaded geometric velocity controller (action ``[vx, vy, vz, yaw_rate]``).

    Yaw-frame velocity + yaw-rate command -> world acceleration -> desired thrust
    vector (gravity-compensated) -> collective thrust + desired tilt -> geometric
    attitude error -> body-rate setpoint -> shared inner rate loop -> rotor omega.
    """
    rotation = _rotation_from_wxyz(state.attitude_flu_enu_wxyz)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    heading = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )

    velocity_setpoint_world = heading @ (action[:3] * params.max_body_velocity_mps)
    measured_velocity = np.asarray(state.linear_velocity_enu, dtype=np.float64)
    if controller_state.initialized:
        velocity_alpha = _low_pass_alpha(
            cutoff_hz=params.velocity_filter_hz,
            dt_s=dt_s,
        )
        filtered_velocity = controller_state.filtered_velocity_enu + velocity_alpha * (
            measured_velocity - controller_state.filtered_velocity_enu
        )
        raw_acceleration = (filtered_velocity - controller_state.filtered_velocity_enu) / dt_s
        derivative_alpha = _low_pass_alpha(
            cutoff_hz=params.velocity_derivative_filter_hz,
            dt_s=dt_s,
        )
        acceleration = controller_state.acceleration_enu + derivative_alpha * (
            raw_acceleration - controller_state.acceleration_enu
        )
    else:
        filtered_velocity = measured_velocity
        acceleration = np.zeros(3, dtype=np.float64)
    velocity_error = velocity_setpoint_world - filtered_velocity
    desired_acceleration = (
        params.kp_vel * velocity_error
        + controller_state.velocity_integral_enu
        - params.kd_vel * acceleration
    )
    thrust_world, achieved_acceleration = _limited_thrust_world(
        params=params,
        desired_acceleration=desired_acceleration,
    )
    saturation_error = achieved_acceleration - desired_acceleration
    may_integrate = (np.abs(saturation_error) < _SATURATION_EPS) | (
        velocity_error * saturation_error >= 0.0
    )
    velocity_integral = np.where(
        may_integrate,
        controller_state.velocity_integral_enu + params.ki_vel * velocity_error * dt_s,
        controller_state.velocity_integral_enu,
    )
    velocity_integral = np.clip(
        velocity_integral,
        -params.velocity_integral_limit_mps2,
        params.velocity_integral_limit_mps2,
    )
    body_z = rotation[:, 2]
    collective = max(float(thrust_world @ body_z), 0.0)

    desired_z = _clamp_tilt(thrust_world, params.max_tilt_rad)
    heading_x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    desired_y = np.cross(desired_z, heading_x)
    desired_y /= max(float(np.linalg.norm(desired_y)), _EPS)
    desired_x = np.cross(desired_y, desired_z)
    desired_rotation = np.stack([desired_x, desired_y, desired_z], axis=1)

    attitude_error = 0.5 * _vee(desired_rotation.T @ rotation - rotation.T @ desired_rotation)
    attitude_error[2] *= params.attitude_yaw_weight
    body_rate_setpoint = -params.kp_att * attitude_error
    body_rate_setpoint[2] += action[3] * params.max_yaw_rate_radps
    outer_successor = ActuationControllerState(
        filtered_velocity_enu=_readonly_copy(filtered_velocity),
        acceleration_enu=_readonly_copy(acceleration),
        velocity_integral_enu=_readonly_copy(velocity_integral),
        previous_angular_velocity_body_flu=_readonly_copy(
            controller_state.previous_angular_velocity_body_flu
        ),
        angular_acceleration_body_flu=_readonly_copy(
            controller_state.angular_acceleration_body_flu
        ),
        rate_integral_body_flu=_readonly_copy(controller_state.rate_integral_body_flu),
        initialized=controller_state.initialized,
    )
    return _rate_to_omega(
        params,
        outer_successor,
        collective,
        body_rate_setpoint,
        state,
        dt_s,
    )


def _ctbr_collective(params: ActuationParams, thrust_command: float) -> float:
    """Map a normalized thrust command to collective thrust (N), hover-centered."""
    fraction = np.clip(params.hover_thrust_fraction * (1.0 + thrust_command), 0.0, 1.0)
    return float(fraction * params.max_total_thrust)


def _ctbr_action_to_omega(
    params: ActuationParams,
    controller_state: ActuationControllerState,
    action: np.ndarray,
    state: ActuationState,
    dt_s: float,
    /,
) -> tuple[ActuationControllerState, np.ndarray]:
    """Collective-thrust + body-rate (CTBR) model (action ``[thrust, wx, wy, wz]``).

    The inner rate loop of the ``velocity`` cascade; the attitude/velocity outer
    loops are the policy's job. ``action[0]`` maps to collective thrust centered
    on ``hover_thrust_fraction`` (``action[0] = 0`` -> hover fraction),
    ``action[1:]`` to the body-rate setpoint.
    """
    collective = _ctbr_collective(params, float(action[0]))
    body_rate_setpoint = action[1:4] * params.max_body_rate_radps
    return _rate_to_omega(
        params,
        controller_state,
        collective,
        body_rate_setpoint,
        state,
        dt_s,
    )


def rotor_forces_to_body_wrench(
    positions_body: np.ndarray,
    forces_n: np.ndarray,
    body_torque: np.ndarray,
    body_drag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce per-rotor thrust + body effects to one equivalent body wrench.

    The single-env analogue of the batched Isaac Lab force sink: per-rotor thrust
    ``[0, 0, f_i]`` at the body-FLU offset ``positions_body[i]`` becomes a net
    force + torque at the body origin, plus the thrust model's body-Z yaw
    reaction (``body_torque``) and linear drag (``body_drag``). Physically
    equivalent to applying the forces at the markers.

    Returns:
        ``(force_body, torque_body)``, both body-FLU ``(3,)``. Roll/pitch come
        from the moment arms: ``torque_body = [sum(y_i f_i), -sum(x_i f_i), 0] +
        body_torque``.
    """
    forces_n = np.asarray(forces_n, dtype=np.float64)
    x = np.asarray(positions_body, dtype=np.float64)[:, 0]
    y = np.asarray(positions_body, dtype=np.float64)[:, 1]
    force_body = np.array([0.0, 0.0, float(np.sum(forces_n))]) + np.asarray(
        body_drag, dtype=np.float64
    )
    torque_body = np.array(
        [float(np.sum(y * forces_n)), float(-np.sum(x * forces_n)), 0.0]
    ) + np.asarray(body_torque, dtype=np.float64)
    return force_body, torque_body


def build_actuation_params(
    *,
    geometry: MixerGeometry,
    action_cfg: RlActionConfig,
    controller_settings: ControllerSettings,
    mass: float,
) -> ActuationParams:
    """Build frozen actuation params from action and aligned controller contracts.

    Precomputes the allocation pseudoinverse and the max total thrust once, so
    the per-call functions consume only frozen state (mirroring the torch
    twin's builder).

    Raises:
        ValueError: for an unknown or not-yet-implemented action level.
    """
    kind = action_cfg.kind
    if kind not in ("velocity", "ctbr"):
        raise ValueError(
            f"unsupported rl.action.kind={kind!r}; implemented levels are 'velocity' and 'ctbr'"
        )
    frozen_geometry = MixerGeometry(
        positions_body=_readonly_copy(geometry.positions_body),
        rot_dir=_readonly_copy(geometry.rot_dir),
        rotor_constant=_readonly_copy(geometry.rotor_constant),
        rolling_moment_coefficient=_readonly_copy(geometry.rolling_moment_coefficient),
        min_rotor_velocity=_readonly_copy(geometry.min_rotor_velocity),
        max_rotor_velocity=_readonly_copy(geometry.max_rotor_velocity),
    )
    allocation_pinv = np.linalg.pinv(_allocation_matrix(frozen_geometry))
    allocation_pinv.setflags(write=False)
    return ActuationParams(
        kind=kind,
        action_dim=4,
        mass=float(mass),
        geometry=frozen_geometry,
        allocation_pinv=allocation_pinv,
        max_total_thrust=_max_total_thrust(frozen_geometry),
        kp_rate=_readonly_copy(controller_settings.kp_rate),
        ki_rate=_readonly_copy(controller_settings.ki_rate),
        kd_rate=_readonly_copy(controller_settings.kd_rate),
        kff_rate=_readonly_copy(controller_settings.kff_rate),
        rate_output_scale_nm=_readonly_copy(controller_settings.rate_output_scale_nm),
        rate_derivative_filter_hz=float(controller_settings.rate_derivative_filter_hz),
        rate_integral_limit_nm=_readonly_copy(controller_settings.rate_integral_limit_nm),
        max_body_rate_radps=float(action_cfg.max_body_rate_radps),
        hover_thrust_fraction=float(action_cfg.hover_thrust_fraction),
        kp_vel=_readonly_copy(controller_settings.kp_vel),
        ki_vel=_readonly_copy(controller_settings.ki_vel),
        kd_vel=_readonly_copy(controller_settings.kd_vel),
        velocity_filter_hz=float(controller_settings.velocity_filter_hz),
        velocity_derivative_filter_hz=float(controller_settings.velocity_derivative_filter_hz),
        velocity_integral_limit_mps2=_readonly_copy(
            controller_settings.velocity_integral_limit_mps2
        ),
        kp_att=_readonly_copy(controller_settings.kp_att),
        attitude_yaw_weight=float(controller_settings.attitude_yaw_weight),
        max_body_velocity_mps=float(action_cfg.max_body_velocity_mps),
        max_yaw_rate_radps=float(action_cfg.max_yaw_rate_radps),
        max_tilt_rad=float(np.radians(controller_settings.max_tilt_cmd_deg)),
        min_collective_thrust_fraction=float(controller_settings.min_collective_thrust_fraction),
        max_collective_thrust_fraction=float(controller_settings.max_collective_thrust_fraction),
        horizontal_thrust_margin_fraction=float(
            controller_settings.horizontal_thrust_margin_fraction
        ),
    )


def hover_action(params: ActuationParams) -> np.ndarray:
    """A normalized action that holds a level hover (weight-canceling)."""
    if params.kind == "ctbr":
        hover_collective = params.mass * GRAVITY_M_S2
        hover_fraction_target = hover_collective / params.max_total_thrust
        thrust_command = hover_fraction_target / params.hover_thrust_fraction - 1.0
        return np.array([thrust_command, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.zeros(params.action_dim, dtype=np.float64)


def initial_actuation_controller_state() -> ActuationControllerState:
    """Return a fresh controller state for an environment or probe reset."""
    zero = np.zeros(3, dtype=np.float64)
    return ActuationControllerState(
        filtered_velocity_enu=_readonly_copy(zero),
        acceleration_enu=_readonly_copy(zero),
        velocity_integral_enu=_readonly_copy(zero),
        previous_angular_velocity_body_flu=_readonly_copy(zero),
        angular_acceleration_body_flu=_readonly_copy(zero),
        rate_integral_body_flu=_readonly_copy(zero),
        initialized=False,
    )


def action_to_rotor_omega(
    params: ActuationParams,
    controller_state: ActuationControllerState,
    action: np.ndarray,
    state: ActuationState,
    dt_s: float,
    /,
) -> tuple[ActuationControllerState, np.ndarray]:
    """Advance the controller and return clipped per-rotor velocity references."""
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("controller dt_s must be finite and positive")
    action = np.asarray(action, dtype=np.float64)
    if params.kind == "ctbr":
        return _ctbr_action_to_omega(
            params,
            controller_state,
            action,
            state,
            dt_s,
        )
    return _velocity_action_to_omega(
        params,
        controller_state,
        action,
        state,
        dt_s,
    )

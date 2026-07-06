"""Pure RL-action -> per-rotor angular-velocity actuation functions (core path).

numpy + stdlib only. State lives in frozen params dataclasses and behavior in
pure top-level functions (``f(params, inputs) -> outputs``), so the actuation
stage composes into a plain function pipeline. This is the single-env
reference/spec implementation and the unit-test oracle; the batched torch twin
the Isaac Lab env runs lives in
``robotics_experiments.workflows.isaac_lab.tasks._flight`` and must match it
term-for-term (both precompute the allocation pseudoinverse in their builder).

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

from dataclasses import dataclass

import numpy as np

from ...px4_bridge.frames import GRAVITY_M_S2, rotate_vector_wxyz

__all__ = [
    "ActuationParams",
    "ActuationState",
    "MixerGeometry",
    "action_to_rotor_omega",
    "build_actuation_params",
    "hover_action",
    "reduce_to_body_wrench",
]

_EPS = 1e-9


@dataclass(frozen=True)
class MixerGeometry:
    """Airframe control-allocation geometry in PX4 motor-index order (body-FLU).

    Mirrors the fields the bridge's ``PegasusQuadraticThrustModel`` consumes; the
    per-rotor arrays are ordered by PX4 motor index. ``positions_body`` are the
    rotor marker offsets from the body origin (from
    ``px4_bridge.rotor_discovery.DiscoveredRotorMapping``); ``rot_dir`` is
    ``+1`` CCW / ``-1`` CW.
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
    max_body_rate_radps: float
    hover_thrust_fraction: float
    kp_vel: np.ndarray
    kd_vel: np.ndarray
    kp_att: np.ndarray
    max_body_velocity_mps: float
    max_yaw_rate_radps: float
    max_tilt_rad: float


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


def _rate_to_omega(
    params: ActuationParams,
    collective_thrust: float,
    body_rate_setpoint: np.ndarray,
    state: ActuationState,
) -> np.ndarray:
    """Shared inner rate loop: rate setpoint -> body torque -> rotor omega."""
    rate = np.clip(body_rate_setpoint, -params.max_body_rate_radps, params.max_body_rate_radps)
    rate_error: np.ndarray = rate - np.asarray(state.angular_velocity_body_flu, dtype=np.float64)
    body_torque = params.kp_rate * rate_error
    return _wrench_to_omega(params, collective_thrust, body_torque)


def _velocity_action_to_omega(
    params: ActuationParams, action: np.ndarray, state: ActuationState
) -> np.ndarray:
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
    velocity = np.asarray(state.linear_velocity_enu, dtype=np.float64)
    velocity_error = velocity_setpoint_world - velocity
    desired_acceleration = params.kp_vel * velocity_error - params.kd_vel * velocity

    thrust_world = params.mass * (desired_acceleration + np.array([0.0, 0.0, GRAVITY_M_S2]))
    body_z = rotation[:, 2]
    collective = max(float(thrust_world @ body_z), 0.0)

    desired_z = _clamp_tilt(thrust_world, params.max_tilt_rad)
    heading_x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    desired_y = np.cross(desired_z, heading_x)
    desired_y /= max(float(np.linalg.norm(desired_y)), _EPS)
    desired_x = np.cross(desired_y, desired_z)
    desired_rotation = np.stack([desired_x, desired_y, desired_z], axis=1)

    attitude_error = 0.5 * _vee(desired_rotation.T @ rotation - rotation.T @ desired_rotation)
    body_rate_setpoint = -params.kp_att * attitude_error
    body_rate_setpoint[2] += action[3] * params.max_yaw_rate_radps
    return _rate_to_omega(params, collective, body_rate_setpoint, state)


def _ctbr_collective(params: ActuationParams, thrust_command: float) -> float:
    """Map a normalized thrust command to collective thrust (N), hover-centered."""
    fraction = np.clip(params.hover_thrust_fraction * (1.0 + thrust_command), 0.0, 1.0)
    return float(fraction * params.max_total_thrust)


def _ctbr_action_to_omega(
    params: ActuationParams, action: np.ndarray, state: ActuationState
) -> np.ndarray:
    """Collective-thrust + body-rate (CTBR) model (action ``[thrust, wx, wy, wz]``).

    The inner rate loop of the ``velocity`` cascade; the attitude/velocity outer
    loops are the policy's job. ``action[0]`` maps to collective thrust centered
    on ``hover_thrust_fraction`` (``action[0] = 0`` -> hover fraction),
    ``action[1:]`` to the body-rate setpoint.
    """
    collective = _ctbr_collective(params, float(action[0]))
    body_rate_setpoint = action[1:4] * params.max_body_rate_radps
    return _rate_to_omega(params, collective, body_rate_setpoint, state)


def reduce_to_body_wrench(
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
    geometry: MixerGeometry,
    action_cfg,
    mass: float,
) -> ActuationParams:
    """Build the frozen actuation params for ``action_cfg.kind``.

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
    return ActuationParams(
        kind=kind,
        action_dim=4,
        mass=float(mass),
        geometry=geometry,
        allocation_pinv=np.linalg.pinv(_allocation_matrix(geometry)),
        max_total_thrust=_max_total_thrust(geometry),
        kp_rate=np.asarray(action_cfg.controller_kp_rate, dtype=np.float64),
        max_body_rate_radps=float(action_cfg.max_body_rate_radps),
        hover_thrust_fraction=float(action_cfg.hover_thrust_fraction),
        kp_vel=np.asarray(action_cfg.controller_kp_vel, dtype=np.float64),
        kd_vel=np.asarray(action_cfg.controller_kd_vel, dtype=np.float64),
        kp_att=np.asarray(action_cfg.controller_kp_att, dtype=np.float64),
        max_body_velocity_mps=float(action_cfg.max_body_velocity_mps),
        max_yaw_rate_radps=float(action_cfg.max_yaw_rate_radps),
        max_tilt_rad=float(np.radians(float(action_cfg.max_tilt_cmd_deg))),
    )


def hover_action(params: ActuationParams) -> np.ndarray:
    """A normalized action that holds a level hover (weight-canceling)."""
    if params.kind == "ctbr":
        hover_collective = params.mass * GRAVITY_M_S2
        hover_fraction_target = hover_collective / params.max_total_thrust
        thrust_command = hover_fraction_target / params.hover_thrust_fraction - 1.0
        return np.array([thrust_command, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.zeros(params.action_dim, dtype=np.float64)


def action_to_rotor_omega(
    params: ActuationParams,
    action: np.ndarray,
    state: ActuationState,
) -> np.ndarray:
    """Map a normalized action to clipped per-rotor angular-velocity references."""
    action = np.asarray(action, dtype=np.float64)
    if params.kind == "ctbr":
        return _ctbr_action_to_omega(params, action, state)
    return _velocity_action_to_omega(params, action, state)

"""Pure normalized command -> per-rotor angular-velocity controller.

numpy + stdlib only. State lives in frozen params dataclasses and behavior in
pure top-level functions (``f(params, inputs) -> outputs``), so the actuation
stage composes into a plain function pipeline. This is the single-env
reference/spec implementation and the unit-test oracle; it is exported through
``robotics_experiments.controller``. The Isaac Lab package exposes the batched torch
twin's external diagnostic entry point as
``robotics_experiments.workflows.isaac_lab.compute_lab_rotor_omega``; it must
match this module term-for-term. The allocator reproduces PX4's normalized
pseudo-inverse allocation and sequential desaturation before applying the
bridge-owned affine motor mapping and quadratic rotor curve.

Two action levels (``kind``) are provided:

- ``velocity`` -- action ``[vx, vy, vz, yaw_rate]`` (normalized): a cascaded
  geometric controller tracks a yaw-frame body velocity + yaw-rate command.
- ``ctbr`` -- action ``[thrust, wx, wy, wz]`` (normalized): collective thrust +
  body-rate setpoint closed by the inner rate loop shared with ``velocity``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from robotics_experiments.config import RlActionConfig
from robotics_experiments.controller._models import ControllerSettings
from robotics_experiments.px4_bridge import (
    GRAVITY_M_S2,
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    rotate_vector_wxyz,
)

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
    "seed_actuation_controller_state",
    "update_hover_thrust_estimate",
]

_EPS = 1e-9


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MixerGeometry:
    """Airframe control-allocation geometry in PX4 motor-index order (body-FLU).

    Mirrors the fields the bridge's ``PegasusQuadraticThrustModel`` consumes;
    the per-rotor arrays are ordered by PX4 motor index. ``positions_body`` are
    the rotor marker offsets from the body origin (from the package-root
    ``DiscoveredRotorMapping``); ``rot_dir`` is ``+1`` CCW / ``-1`` CW. The
    input arrays define the bridge mapping from normalized motor control to
    rotor speed.
    """

    positions_body: np.ndarray
    rot_dir: np.ndarray
    rotor_constant: np.ndarray
    rolling_moment_coefficient: np.ndarray
    min_rotor_velocity: np.ndarray
    max_rotor_velocity: np.ndarray
    input_offset: np.ndarray
    input_scaling: np.ndarray
    zero_position_armed: np.ndarray


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
    velocity_estimator_bias_enu: np.ndarray
    acceleration_enu: np.ndarray
    acceleration_setpoint_enu: np.ndarray
    velocity_integral_enu: np.ndarray
    previous_angular_velocity_body_flu: np.ndarray
    angular_acceleration_body_flu: np.ndarray
    rate_integral_body_flu: np.ndarray
    filtered_yaw_torque: float
    hover_thrust_normalized: float
    outer_loop_elapsed_s: float
    collective_thrust_setpoint: float
    body_rate_setpoint: np.ndarray
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
    normalized_allocation_mix: np.ndarray
    kp_rate: np.ndarray
    ki_rate: np.ndarray
    kd_rate: np.ndarray
    kff_rate: np.ndarray
    angular_velocity_filter_hz: float
    rate_derivative_filter_hz: float
    yaw_torque_filter_hz: float
    rate_integral_limit: np.ndarray
    max_body_rate_radps: float
    velocity_controller_rate_hz: float
    hover_thrust_normalized: float
    kp_vel: np.ndarray
    horizontal_velocity_error_gain_scale: float
    horizontal_velocity_error_gain_start_mps: float
    horizontal_velocity_error_gain_full_mps: float
    ki_vel: np.ndarray
    kd_vel: np.ndarray
    velocity_observation_scale: np.ndarray
    velocity_observation_bias_mps: np.ndarray
    velocity_estimator_bias_decay_hz: float
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


def _normalized_allocation_mix(geometry: MixerGeometry) -> np.ndarray:
    """PX4 normalized pseudo-inverse mix in FRD control coordinates."""
    x = geometry.positions_body[:, 0]
    y = geometry.positions_body[:, 1]
    yaw = -geometry.rolling_moment_coefficient / geometry.rotor_constant * geometry.rot_dir
    effectiveness = np.stack([y, x, yaw, -np.ones(_num_rotors(geometry))], axis=0)
    mix = np.linalg.pinv(effectiveness)

    roll_count = int(np.count_nonzero(np.abs(mix[:, 0]) > 1e-3))
    pitch_count = int(np.count_nonzero(np.abs(mix[:, 1]) > 1e-3))
    roll_scale = (
        np.sqrt(float(np.sum(np.square(mix[:, 0]))) / (roll_count / 2.0)) if roll_count else 1.0
    )
    pitch_scale = (
        np.sqrt(float(np.sum(np.square(mix[:, 1]))) / (pitch_count / 2.0)) if pitch_count else 1.0
    )
    roll_pitch_scale = max(roll_scale, pitch_scale)
    if roll_pitch_scale > np.finfo(np.float32).eps:
        mix[:, :2] /= roll_pitch_scale

    yaw_scale = float(np.max(mix[:, 2]))
    if yaw_scale > np.finfo(np.float32).eps:
        mix[:, 2] /= yaw_scale

    thrust_values = np.abs(mix[:, 3])
    nonzero_thrust = thrust_values > np.finfo(np.float32).eps
    if np.any(nonzero_thrust):
        mix[:, 3] /= float(np.mean(thrust_values[nonzero_thrust]))
    mix[np.abs(mix) < 1e-3] = 0.0
    return mix


def _rotation_from_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Return the ``3x3`` body-FLU -> world-ENU rotation matrix (columns = axes)."""
    x_axis = rotate_vector_wxyz(q_wxyz, np.array([1.0, 0.0, 0.0]))
    y_axis = rotate_vector_wxyz(q_wxyz, np.array([0.0, 1.0, 0.0]))
    z_axis = rotate_vector_wxyz(q_wxyz, np.array([0.0, 0.0, 1.0]))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _canonical_quat(q_wxyz: np.ndarray, /) -> np.ndarray:
    normalized = normalize_quat_wxyz(q_wxyz)
    return -normalized if normalized[0] < 0.0 else normalized


def _quat_from_two_vectors(source: np.ndarray, target: np.ndarray, /) -> np.ndarray:
    source = source / max(float(np.linalg.norm(source)), _EPS)
    target = target / max(float(np.linalg.norm(target)), _EPS)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot < -1.0 + 1e-6:
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, basis)
        axis /= max(float(np.linalg.norm(axis)), _EPS)
        return np.array([0.0, *axis], dtype=np.float64)
    return normalize_quat_wxyz(np.array([1.0 + dot, *np.cross(source, target)]))


def _quat_from_rotation(rotation: np.ndarray, /) -> np.ndarray:
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    candidates = np.array(
        [
            [1.0 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01],
            [m21 - m12, 1.0 + m00 - m11 - m22, m10 + m01, m02 + m20],
            [m02 - m20, m10 + m01, 1.0 - m00 + m11 - m22, m12 + m21],
            [m10 - m01, m02 + m20, m12 + m21, 1.0 - m00 - m11 + m22],
        ],
        dtype=np.float64,
    )
    magnitudes = np.sqrt(np.maximum(np.diag(candidates), 0.0))
    index = int(np.argmax(magnitudes))
    return normalize_quat_wxyz(candidates[index] / max(2.0 * magnitudes[index], 0.2))


def _attitude_rate_setpoint(
    *,
    attitude_wxyz: np.ndarray,
    rotation: np.ndarray,
    desired_rotation: np.ndarray,
    proportional_gain: np.ndarray,
    yaw_weight: float,
    yaw_rate_radps: float,
) -> np.ndarray:
    """PX4 reduced-quaternion attitude law in body-FLU/world-ENU coordinates."""
    attitude = normalize_quat_wxyz(attitude_wxyz)
    desired_attitude = _quat_from_rotation(desired_rotation)
    reduced_desired = _quat_from_two_vectors(rotation[:, 2], desired_rotation[:, 2])
    if abs(float(reduced_desired[1])) > 1.0 - 1e-5 or abs(float(reduced_desired[2])) > 1.0 - 1e-5:
        reduced_desired = desired_attitude
    else:
        reduced_desired = quat_multiply_wxyz(reduced_desired, attitude)
    delta_yaw = _canonical_quat(
        quat_multiply_wxyz(quat_conjugate_wxyz(reduced_desired), desired_attitude)
    )
    weighted_yaw = np.array(
        [
            np.cos(yaw_weight * np.arccos(np.clip(delta_yaw[0], -1.0, 1.0))),
            0.0,
            0.0,
            np.sin(yaw_weight * np.arcsin(np.clip(delta_yaw[3], -1.0, 1.0))),
        ],
        dtype=np.float64,
    )
    weighted_desired = quat_multiply_wxyz(reduced_desired, weighted_yaw)
    error = _canonical_quat(quat_multiply_wxyz(quat_conjugate_wxyz(attitude), weighted_desired))
    gain = np.array(proportional_gain, copy=True)
    if yaw_weight > 1e-4:
        gain[2] /= yaw_weight
    rate_setpoint = 2.0 * error[1:] * gain
    return rate_setpoint + rotation[2, :] * yaw_rate_radps


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


def _desaturation_gain(
    actuator_sp: np.ndarray,
    desaturation_vector: np.ndarray,
    *,
    actuator_max: float = 1.0,
) -> float:
    k_min = 0.0
    k_max = 0.0
    for setpoint, vector in zip(actuator_sp, desaturation_vector, strict=True):
        if abs(float(vector)) < 0.2:
            continue
        if setpoint < 0.0:
            gain = -float(setpoint) / float(vector)
            k_min = min(k_min, gain)
            k_max = max(k_max, gain)
        if setpoint > actuator_max:
            gain = (actuator_max - float(setpoint)) / float(vector)
            k_min = min(k_min, gain)
            k_max = max(k_max, gain)
    return k_min + k_max


def _desaturate(
    actuator_sp: np.ndarray,
    desaturation_vector: np.ndarray,
    *,
    increase_only: bool = False,
    actuator_max: float = 1.0,
) -> np.ndarray:
    gain = _desaturation_gain(
        actuator_sp,
        desaturation_vector,
        actuator_max=actuator_max,
    )
    if increase_only and gain < 0.0:
        return actuator_sp
    actuator_sp = actuator_sp + gain * desaturation_vector
    gain = 0.5 * _desaturation_gain(
        actuator_sp,
        desaturation_vector,
        actuator_max=actuator_max,
    )
    return actuator_sp + gain * desaturation_vector


def _normalized_controls_to_omega(
    params: ActuationParams,
    collective: float,
    body_torque: np.ndarray,
) -> np.ndarray:
    """Apply pinned PX4 sequential desaturation and the bridge motor mapping."""
    mix = params.normalized_allocation_mix
    control = np.array(
        [body_torque[0], -body_torque[1], -body_torque[2], -collective],
        dtype=np.float64,
    )
    actuator_sp = mix[:, 0] * control[0] + mix[:, 1] * control[1] + mix[:, 3] * control[3]
    actuator_sp = _desaturate(actuator_sp, mix[:, 3], increase_only=True)
    actuator_sp = _desaturate(actuator_sp, mix[:, 0])
    actuator_sp = _desaturate(actuator_sp, mix[:, 1])
    actuator_sp = actuator_sp + mix[:, 2] * control[2]
    actuator_sp = _desaturate(actuator_sp, mix[:, 2], actuator_max=1.15)
    actuator_sp = _desaturate(actuator_sp, mix[:, 3], increase_only=True)
    actuator_sp = np.clip(actuator_sp, 0.0, 1.0)
    geometry = params.geometry
    omega = (
        actuator_sp + geometry.input_offset
    ) * geometry.input_scaling + geometry.zero_position_armed
    return np.clip(omega, geometry.min_rotor_velocity, geometry.max_rotor_velocity)


def _low_pass_alpha(*, cutoff_hz: float, dt_s: float) -> float:
    if cutoff_hz <= 0.0:
        return 1.0
    scaled = 2.0 * np.pi * cutoff_hz * dt_s
    return float(scaled / (1.0 + scaled))


def _normalized_thrust_world(
    *,
    params: ActuationParams,
    hover_thrust_normalized: float,
    desired_acceleration: np.ndarray,
    velocity_error: np.ndarray,
    horizontal_proportional_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    desired_z = _clamp_tilt(
        np.array(
            [desired_acceleration[0], desired_acceleration[1], GRAVITY_M_S2],
            dtype=np.float64,
        ),
        params.max_tilt_rad,
    )
    collective = max(
        hover_thrust_normalized
        * (1.0 + desired_acceleration[2] / GRAVITY_M_S2)
        / max(float(desired_z[2]), _EPS),
        params.min_collective_thrust_fraction,
    )
    requested = desired_z * collective
    adjusted_error = np.array(velocity_error, copy=True)
    if (requested[2] <= params.min_collective_thrust_fraction and adjusted_error[2] <= 0.0) or (
        requested[2] >= params.max_collective_thrust_fraction and adjusted_error[2] >= 0.0
    ):
        adjusted_error[2] = 0.0

    horizontal = requested[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    allocated_horizontal = min(
        horizontal_norm,
        params.horizontal_thrust_margin_fraction,
    )
    vertical_max = np.sqrt(
        max(
            params.max_collective_thrust_fraction**2 - allocated_horizontal**2,
            0.0,
        )
    )
    vertical = min(float(requested[2]), vertical_max)
    horizontal_max = np.sqrt(max(params.max_collective_thrust_fraction**2 - vertical**2, 0.0))
    if horizontal_norm > horizontal_max:
        horizontal = horizontal * (horizontal_max / max(horizontal_norm, _EPS))
    limited = np.array([horizontal[0], horizontal[1], vertical], dtype=np.float64)
    produced_acceleration = limited[:2] * (GRAVITY_M_S2 / hover_thrust_normalized)
    if float(desired_acceleration[:2] @ desired_acceleration[:2]) > float(
        produced_acceleration @ produced_acceleration
    ):
        adjusted_error[:2] -= (2.0 / horizontal_proportional_gain) * (
            desired_acceleration[:2] - produced_acceleration
        )
    return limited, adjusted_error


def _effective_velocity_proportional_gain(
    params: ActuationParams, velocity_error: np.ndarray, /
) -> np.ndarray:
    horizontal_error = float(np.linalg.norm(velocity_error[:2]))
    transition = np.clip(
        (horizontal_error - params.horizontal_velocity_error_gain_start_mps)
        / (
            params.horizontal_velocity_error_gain_full_mps
            - params.horizontal_velocity_error_gain_start_mps
        ),
        0.0,
        1.0,
    )
    blend = transition * transition * (3.0 - 2.0 * transition)
    scale = 1.0 + (params.horizontal_velocity_error_gain_scale - 1.0) * blend
    gain = np.array(params.kp_vel, copy=True)
    gain[:2] *= scale
    return gain


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
    raw_angular_velocity = np.asarray(state.angular_velocity_body_flu, dtype=np.float64)
    if controller_state.initialized:
        angular_velocity_alpha = _low_pass_alpha(
            cutoff_hz=params.angular_velocity_filter_hz,
            dt_s=dt_s,
        )
        angular_velocity = controller_state.previous_angular_velocity_body_flu + (
            angular_velocity_alpha
            * (raw_angular_velocity - controller_state.previous_angular_velocity_body_flu)
        )
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
        angular_velocity = raw_angular_velocity
        angular_acceleration = np.zeros(3, dtype=np.float64)
    rate_error: np.ndarray = rate - angular_velocity
    normalized_torque = (
        params.kp_rate * rate_error
        + controller_state.rate_integral_body_flu
        - params.kd_rate * angular_acceleration
        + params.kff_rate * rate
    )
    yaw_torque_alpha = _low_pass_alpha(
        cutoff_hz=params.yaw_torque_filter_hz,
        dt_s=dt_s,
    )
    normalized_torque[2] = controller_state.filtered_yaw_torque + yaw_torque_alpha * (
        normalized_torque[2] - controller_state.filtered_yaw_torque
    )
    integral_factor = np.maximum(
        0.0,
        1.0 - np.square(rate_error / np.radians(400.0)),
    )
    rate_integral = np.clip(
        controller_state.rate_integral_body_flu
        + integral_factor * params.ki_rate * rate_error * dt_s,
        -params.rate_integral_limit,
        params.rate_integral_limit,
    )
    successor = ActuationControllerState(
        filtered_velocity_enu=_readonly_copy(controller_state.filtered_velocity_enu),
        velocity_estimator_bias_enu=_readonly_copy(controller_state.velocity_estimator_bias_enu),
        acceleration_enu=_readonly_copy(controller_state.acceleration_enu),
        acceleration_setpoint_enu=_readonly_copy(controller_state.acceleration_setpoint_enu),
        velocity_integral_enu=_readonly_copy(controller_state.velocity_integral_enu),
        previous_angular_velocity_body_flu=_readonly_copy(angular_velocity),
        angular_acceleration_body_flu=_readonly_copy(angular_acceleration),
        rate_integral_body_flu=_readonly_copy(rate_integral),
        filtered_yaw_torque=float(normalized_torque[2]),
        hover_thrust_normalized=controller_state.hover_thrust_normalized,
        outer_loop_elapsed_s=controller_state.outer_loop_elapsed_s,
        collective_thrust_setpoint=controller_state.collective_thrust_setpoint,
        body_rate_setpoint=_readonly_copy(controller_state.body_rate_setpoint),
        initialized=True,
    )
    omega = _normalized_controls_to_omega(
        params,
        collective_thrust,
        normalized_torque,
    )
    return successor, omega


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
    outer_loop_period_s = 1.0 / params.velocity_controller_rate_hz
    outer_loop_elapsed_s = controller_state.outer_loop_elapsed_s + dt_s
    if controller_state.initialized and outer_loop_elapsed_s < outer_loop_period_s - _EPS:
        held_state = replace(controller_state, outer_loop_elapsed_s=outer_loop_elapsed_s)
        return _rate_to_omega(
            params,
            held_state,
            held_state.collective_thrust_setpoint,
            held_state.body_rate_setpoint,
            state,
            dt_s,
        )
    outer_dt_s = outer_loop_period_s
    rotation = _rotation_from_wxyz(state.attitude_flu_enu_wxyz)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    heading = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )

    velocity_setpoint_world = heading @ (action[:3] * params.max_body_velocity_mps)
    physical_velocity = (
        params.velocity_observation_scale
        * np.asarray(
            state.linear_velocity_enu,
            dtype=np.float64,
        )
        + params.velocity_observation_bias_mps
    )
    measured_velocity = physical_velocity + controller_state.velocity_estimator_bias_enu
    if controller_state.initialized:
        velocity_alpha = _low_pass_alpha(
            cutoff_hz=params.velocity_filter_hz,
            dt_s=outer_dt_s,
        )
        filtered_velocity = controller_state.filtered_velocity_enu + velocity_alpha * (
            measured_velocity - controller_state.filtered_velocity_enu
        )
        raw_acceleration = (filtered_velocity - controller_state.filtered_velocity_enu) / outer_dt_s
        derivative_alpha = _low_pass_alpha(
            cutoff_hz=params.velocity_derivative_filter_hz,
            dt_s=outer_dt_s,
        )
        acceleration = controller_state.acceleration_enu + derivative_alpha * (
            raw_acceleration - controller_state.acceleration_enu
        )
        bias_decay = math.exp(-2.0 * math.pi * params.velocity_estimator_bias_decay_hz * outer_dt_s)
        velocity_estimator_bias = controller_state.velocity_estimator_bias_enu * bias_decay
    else:
        filtered_velocity = measured_velocity
        acceleration = np.zeros(3, dtype=np.float64)
        velocity_estimator_bias = np.zeros(3, dtype=np.float64)
    velocity_error = velocity_setpoint_world - filtered_velocity
    effective_kp_vel = _effective_velocity_proportional_gain(params, velocity_error)
    desired_acceleration = (
        effective_kp_vel * velocity_error
        + controller_state.velocity_integral_enu
        - params.kd_vel * acceleration
    )
    hover_thrust_normalized = (
        controller_state.hover_thrust_normalized
        if controller_state.hover_thrust_normalized > 0.0
        else params.hover_thrust_normalized
    )
    thrust_world, integration_error = _normalized_thrust_world(
        params=params,
        hover_thrust_normalized=hover_thrust_normalized,
        desired_acceleration=desired_acceleration,
        velocity_error=velocity_error,
        horizontal_proportional_gain=float(effective_kp_vel[0]),
    )
    velocity_integral = (
        controller_state.velocity_integral_enu + params.ki_vel * integration_error * outer_dt_s
    )
    velocity_integral = np.clip(
        velocity_integral,
        -params.velocity_integral_limit_mps2,
        params.velocity_integral_limit_mps2,
    )
    # PX4's attitude-setpoint boundary preserves the desired normalized thrust
    # magnitude while the attitude loop rotates toward its direction.
    collective = float(np.linalg.norm(thrust_world))

    desired_z = _clamp_tilt(thrust_world, params.max_tilt_rad)
    heading_x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    desired_y = np.cross(desired_z, heading_x)
    desired_y /= max(float(np.linalg.norm(desired_y)), _EPS)
    desired_x = np.cross(desired_y, desired_z)
    desired_rotation = np.stack([desired_x, desired_y, desired_z], axis=1)

    body_rate_setpoint = _attitude_rate_setpoint(
        attitude_wxyz=np.asarray(state.attitude_flu_enu_wxyz, dtype=np.float64),
        rotation=rotation,
        desired_rotation=desired_rotation,
        proportional_gain=params.kp_att,
        yaw_weight=params.attitude_yaw_weight,
        yaw_rate_radps=float(action[3] * params.max_yaw_rate_radps),
    )
    outer_successor = ActuationControllerState(
        filtered_velocity_enu=_readonly_copy(filtered_velocity),
        velocity_estimator_bias_enu=_readonly_copy(velocity_estimator_bias),
        acceleration_enu=_readonly_copy(acceleration),
        acceleration_setpoint_enu=_readonly_copy(desired_acceleration),
        velocity_integral_enu=_readonly_copy(velocity_integral),
        previous_angular_velocity_body_flu=_readonly_copy(
            controller_state.previous_angular_velocity_body_flu
        ),
        angular_acceleration_body_flu=_readonly_copy(
            controller_state.angular_acceleration_body_flu
        ),
        rate_integral_body_flu=_readonly_copy(controller_state.rate_integral_body_flu),
        filtered_yaw_torque=controller_state.filtered_yaw_torque,
        hover_thrust_normalized=hover_thrust_normalized,
        outer_loop_elapsed_s=max(outer_loop_elapsed_s - outer_loop_period_s, 0.0),
        collective_thrust_setpoint=collective,
        body_rate_setpoint=_readonly_copy(body_rate_setpoint),
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
    """Map a normalized thrust command to normalized hover-centered collective."""
    return float(np.clip(params.hover_thrust_normalized * (1.0 + thrust_command), 0.0, 1.0))


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
    on aligned normalized hover thrust (``action[0] = 0`` -> hover),
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

    Precomputes the physical and normalized allocation maps and maximum thrust
    once, so the per-call functions consume only frozen state (mirroring the
    torch twin's builder).

    Raises:
        ValueError: for an unknown or not-yet-implemented action level.
    """
    kind = action_cfg.kind
    if kind not in ("velocity", "ctbr"):
        raise ValueError(
            f"unsupported rl.action.kind={kind!r}; implemented levels are 'velocity' and 'ctbr'"
        )
    if not 0.0 < controller_settings.hover_thrust_normalized <= 1.0:
        raise ValueError("controller hover_thrust_normalized must be in (0, 1]")
    frozen_geometry = MixerGeometry(
        positions_body=_readonly_copy(geometry.positions_body),
        rot_dir=_readonly_copy(geometry.rot_dir),
        rotor_constant=_readonly_copy(geometry.rotor_constant),
        rolling_moment_coefficient=_readonly_copy(geometry.rolling_moment_coefficient),
        min_rotor_velocity=_readonly_copy(geometry.min_rotor_velocity),
        max_rotor_velocity=_readonly_copy(geometry.max_rotor_velocity),
        input_offset=_readonly_copy(geometry.input_offset),
        input_scaling=_readonly_copy(geometry.input_scaling),
        zero_position_armed=_readonly_copy(geometry.zero_position_armed),
    )
    normalized_allocation_mix = _normalized_allocation_mix(frozen_geometry)
    normalized_allocation_mix.setflags(write=False)
    return ActuationParams(
        kind=kind,
        action_dim=4,
        mass=float(mass),
        geometry=frozen_geometry,
        normalized_allocation_mix=normalized_allocation_mix,
        kp_rate=_readonly_copy(controller_settings.kp_rate),
        ki_rate=_readonly_copy(controller_settings.ki_rate),
        kd_rate=_readonly_copy(controller_settings.kd_rate),
        kff_rate=_readonly_copy(controller_settings.kff_rate),
        angular_velocity_filter_hz=float(controller_settings.angular_velocity_filter_hz),
        rate_derivative_filter_hz=float(controller_settings.rate_derivative_filter_hz),
        yaw_torque_filter_hz=float(controller_settings.yaw_torque_filter_hz),
        rate_integral_limit=_readonly_copy(controller_settings.rate_integral_limit),
        max_body_rate_radps=float(action_cfg.max_body_rate_radps),
        velocity_controller_rate_hz=float(controller_settings.velocity_controller_rate_hz),
        hover_thrust_normalized=float(controller_settings.hover_thrust_normalized),
        kp_vel=_readonly_copy(controller_settings.kp_vel),
        horizontal_velocity_error_gain_scale=float(
            controller_settings.horizontal_velocity_error_gain_scale
        ),
        horizontal_velocity_error_gain_start_mps=float(
            controller_settings.horizontal_velocity_error_gain_start_mps
        ),
        horizontal_velocity_error_gain_full_mps=float(
            controller_settings.horizontal_velocity_error_gain_full_mps
        ),
        ki_vel=_readonly_copy(controller_settings.ki_vel),
        kd_vel=_readonly_copy(controller_settings.kd_vel),
        velocity_observation_scale=_readonly_copy(controller_settings.velocity_observation_scale),
        velocity_observation_bias_mps=_readonly_copy(
            controller_settings.velocity_observation_bias_mps
        ),
        velocity_estimator_bias_decay_hz=float(
            controller_settings.velocity_estimator_bias_decay_hz
        ),
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
    return np.zeros(params.action_dim, dtype=np.float64)


def initial_actuation_controller_state() -> ActuationControllerState:
    """Return a fresh controller state for an environment or probe reset."""
    zero = np.zeros(3, dtype=np.float64)
    return ActuationControllerState(
        filtered_velocity_enu=_readonly_copy(zero),
        velocity_estimator_bias_enu=_readonly_copy(zero),
        acceleration_enu=_readonly_copy(zero),
        acceleration_setpoint_enu=_readonly_copy(zero),
        velocity_integral_enu=_readonly_copy(zero),
        previous_angular_velocity_body_flu=_readonly_copy(zero),
        angular_acceleration_body_flu=_readonly_copy(zero),
        rate_integral_body_flu=_readonly_copy(zero),
        filtered_yaw_torque=0.0,
        hover_thrust_normalized=0.0,
        outer_loop_elapsed_s=0.0,
        collective_thrust_setpoint=0.0,
        body_rate_setpoint=_readonly_copy(zero),
        initialized=False,
    )


def seed_actuation_controller_state(
    *,
    params: ActuationParams,
    controller_dt_s: float,
    physical_linear_velocity_enu: np.ndarray,
    filtered_linear_velocity_enu: np.ndarray,
    angular_velocity_body_flu: np.ndarray,
    acceleration_setpoint_enu_mps2: np.ndarray,
    linear_acceleration_enu_mps2: np.ndarray,
    angular_acceleration_body_flu_radps2: np.ndarray,
    rate_integral_body_flu: np.ndarray,
    hover_thrust_normalized: float,
) -> ActuationControllerState:
    """Match a neutral-command PX4 controller boundary for one replay."""
    if params.kind != "velocity":
        raise ValueError("controller-state seeding requires velocity actuation")
    if not math.isfinite(controller_dt_s) or controller_dt_s <= 0.0:
        raise ValueError("controller_dt_s must be finite and positive")

    def vector(values: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain three finite values")
        return result

    filtered_velocity = vector(
        filtered_linear_velocity_enu,
        "filtered_linear_velocity_enu",
    )
    physical_velocity = (
        params.velocity_observation_scale
        * vector(
            physical_linear_velocity_enu,
            "physical_linear_velocity_enu",
        )
        + params.velocity_observation_bias_mps
    )
    angular_velocity = vector(angular_velocity_body_flu, "angular_velocity_body_flu")
    acceleration_setpoint = vector(
        acceleration_setpoint_enu_mps2,
        "acceleration_setpoint_enu_mps2",
    )
    acceleration = vector(
        linear_acceleration_enu_mps2,
        "linear_acceleration_enu_mps2",
    )
    angular_acceleration = vector(
        angular_acceleration_body_flu_radps2,
        "angular_acceleration_body_flu_radps2",
    )
    rate_integral = vector(rate_integral_body_flu, "rate_integral_body_flu")
    if (
        not math.isfinite(hover_thrust_normalized)
        or hover_thrust_normalized <= 0.0
        or hover_thrust_normalized > 1.0
    ):
        raise ValueError("hover_thrust_normalized must be finite and in (0, 1]")
    velocity_error = -filtered_velocity
    effective_kp_vel = _effective_velocity_proportional_gain(params, velocity_error)
    velocity_integral = np.clip(
        acceleration_setpoint - effective_kp_vel * velocity_error + params.kd_vel * acceleration,
        -params.velocity_integral_limit_mps2,
        params.velocity_integral_limit_mps2,
    )
    return ActuationControllerState(
        filtered_velocity_enu=_readonly_copy(filtered_velocity),
        velocity_estimator_bias_enu=_readonly_copy(filtered_velocity - physical_velocity),
        acceleration_enu=_readonly_copy(acceleration),
        acceleration_setpoint_enu=_readonly_copy(acceleration_setpoint),
        velocity_integral_enu=_readonly_copy(velocity_integral),
        previous_angular_velocity_body_flu=_readonly_copy(angular_velocity),
        angular_acceleration_body_flu=_readonly_copy(angular_acceleration),
        rate_integral_body_flu=_readonly_copy(
            np.clip(
                rate_integral,
                -params.rate_integral_limit,
                params.rate_integral_limit,
            )
        ),
        filtered_yaw_torque=0.0,
        hover_thrust_normalized=float(hover_thrust_normalized),
        # The captured boundary is published before PX4 processes the current
        # physics tick, so the first replay step must run the outer loop.
        outer_loop_elapsed_s=controller_dt_s,
        collective_thrust_setpoint=float(hover_thrust_normalized),
        body_rate_setpoint=_readonly_copy(np.zeros(3)),
        initialized=True,
    )


def update_hover_thrust_estimate(
    *,
    params: ActuationParams,
    controller_state: ActuationControllerState,
    hover_thrust_normalized: float,
) -> ActuationControllerState:
    """Apply PX4's thrust-preserving hover-estimate integrator correction."""
    if (
        not math.isfinite(hover_thrust_normalized)
        or hover_thrust_normalized <= 0.0
        or hover_thrust_normalized > 1.0
    ):
        raise ValueError("hover_thrust_normalized must be finite and in (0, 1]")
    previous_hover = (
        controller_state.hover_thrust_normalized
        if controller_state.hover_thrust_normalized > 0.0
        else params.hover_thrust_normalized
    )
    if hover_thrust_normalized == previous_hover:
        return controller_state
    velocity_integral = np.array(controller_state.velocity_integral_enu, copy=True)
    acceleration_setpoint_z = float(controller_state.acceleration_setpoint_enu[2])
    velocity_integral[2] += (acceleration_setpoint_z + GRAVITY_M_S2) * (
        previous_hover / hover_thrust_normalized - 1.0
    )
    velocity_integral = np.clip(
        velocity_integral,
        -params.velocity_integral_limit_mps2,
        params.velocity_integral_limit_mps2,
    )
    return replace(
        controller_state,
        velocity_integral_enu=_readonly_copy(velocity_integral),
        hover_thrust_normalized=float(hover_thrust_normalized),
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

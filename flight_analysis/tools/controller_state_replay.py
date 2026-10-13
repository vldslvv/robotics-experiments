"""Compare PX4 and Isaac velocity-controller outputs at matched replay states.

The PX4 ULog is the controller oracle. At each logged motor sample in the
selected window, this tool reads PX4's held world-NED trajectory command,
converts the logged estimator state to the Isaac controller's ENU/FLU
convention, and evaluates the host numpy actuation reference. Both outputs are
reduced to the same physical boundary:
per-rotor angular velocity and the resulting body wrench.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from robotics_experiments.alignment import load_alignment_reference
from robotics_experiments.config import load_config
from robotics_experiments.controller import (
    ActuationState,
    MixerGeometry,
    build_actuation_params,
    controller_command_to_rotor_omega,
    load_controller_artifact,
    normalized_action_to_controller_command,
    seed_actuation_controller_state,
    update_hover_thrust_estimate,
)
from robotics_experiments.drone._airframe import (
    compute_body_mass_properties,
    load_builder_config,
)
from robotics_experiments.flight_parameters import load_flight_selection, open_flight_evidence
from robotics_experiments.px4_bridge import GRAVITY_M_S2

__all__ = ["main"]

_NED_TO_ENU = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)
_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])
_PHYSICS_TICK_US = 4_000
_MAX_NORMALIZED_MOTOR_RMSE = 0.01
_MAX_COLLECTIVE_RMSE_N = 0.05
_MOTOR_SATURATION_EPS = 1e-6
_BRIDGE_DYNAMICS_PATTERN = re.compile(
    r"PX4 bridge dynamics: .* sim_time_us=(?P<sim_time_us>\d+) "
    r"last_actuator_us=(?P<actuator_time_us>\d+) "
    r"actuator_lag_us=(?P<join_error_us>\d+).* "
    r"commands=(?P<rotor_speed>\[[^\]]+\]) "
    r"forces_n=(?P<forces>\[[^\]]+\]) "
    r"body_torque_n_m=(?P<body_torque>\[[^\]]+\])"
)
_BRIDGE_STATE_PATTERN = re.compile(
    r"PX4 bridge dynamics: .* sim_time_us=(?P<sim_time_us>\d+).* "
    r"position_enu=(?P<position>\[[^\]]+\]) "
    r"quat_wxyz=(?P<attitude>\[[^\]]+\]) "
    r"velocity_enu=(?P<velocity>\[[^\]]+\]) "
    r"body_velocity_flu=\[[^\]]+\] "
    r"angular_velocity_body_flu=(?P<angular_velocity>\[[^\]]+\])"
)


def _dataset(*, ulog, name: str) -> dict:
    for item in ulog.data_list:
        if item.name == name:
            return item.data
    raise KeyError(f"ULog topic not found: {name}")


def _nearest_indices(*, source_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(source_t, query_t, side="left")
    upper = np.clip(upper, 0, len(source_t) - 1)
    lower = np.clip(upper - 1, 0, len(source_t) - 1)
    use_lower = np.abs(query_t - source_t[lower]) <= np.abs(source_t[upper] - query_t)
    return np.where(use_lower, lower, upper)


def _hold_indices(*, source_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(source_t, query_t, side="right") - 1, 0, len(source_t) - 1)


def _replay_time_grid(
    *,
    start_s: float,
    end_s: float,
    controller_dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    step_count = round((end_s - start_s) / controller_dt_s)
    relative_time_s = np.arange(step_count + 1, dtype=np.float64) * controller_dt_s
    return start_s + relative_time_s, relative_time_s


def _delay_velocity_actions(
    *,
    actions: np.ndarray,
    delay_steps: int,
) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError(f"velocity actions must have shape (samples, 4), got {actions.shape}")
    if delay_steps < 0:
        raise ValueError("velocity action delay must be non-negative")
    delayed = np.zeros_like(actions)
    if delay_steps < len(actions):
        delayed[delay_steps:] = actions[: len(actions) - delay_steps]
    return delayed


def _delay_transmitted_actions(*, actions: np.ndarray) -> np.ndarray:
    """Apply a recorded setpoint after the publishing physics tick.

    Alignment samples contain the command returned by the policy observation at
    that timestamp. PX4 consumes the newly published setpoint on the following
    4 ms controller tick. The first recorded command is already active for the
    first complete replay interval, so it is also the concrete boundary value
    before that first interval.
    """
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("transmitted actions must be a non-empty matrix")
    return np.concatenate([values[:1], values[:-1]], axis=0)


def _linear_interpolate(
    *,
    source_t: np.ndarray,
    values: np.ndarray,
    query_t: np.ndarray,
) -> np.ndarray:
    columns = values.reshape(len(values), -1)
    result = np.column_stack(
        [np.interp(query_t, source_t, columns[:, index]) for index in range(columns.shape[1])]
    )
    return result.reshape((len(query_t), *values.shape[1:]))


def _px4_vertical_velocity_state(
    *,
    velocity_setpoint_ned: np.ndarray,
    velocity_ned: np.ndarray,
    position_derivative_z_ned: np.ndarray,
    landing_speed_mps: float,
) -> np.ndarray:
    """Reproduce PX4's velocity-only vertical estimator blending seam."""
    if not math.isfinite(landing_speed_mps) or landing_speed_mps <= 0.0:
        raise ValueError("MPC_LAND_SPEED must be finite and positive")
    result = np.array(velocity_ned, dtype=np.float64, copy=True)
    commanded_z = np.asarray(velocity_setpoint_ned, dtype=np.float64)[:, 2]
    position_derivative = np.asarray(position_derivative_z_ned, dtype=np.float64)
    active = np.isfinite(commanded_z) & (np.abs(commanded_z) > np.finfo(np.float32).eps)
    weight = np.minimum(np.abs(commanded_z) / landing_speed_mps, 1.0)
    result[active, 2] = position_derivative[active] * weight[active] + result[active, 2] * (
        1.0 - weight[active]
    )
    return result


def _vertical_acceleration_setpoint_enu(
    *,
    quaternion_frd_ned_wxyz: np.ndarray,
    thrust_body_z: np.ndarray,
    hover_thrust_normalized: np.ndarray,
) -> np.ndarray:
    """Recover PX4's exact vertical acceleration setpoint from its thrust output."""
    quaternions = np.asarray(quaternion_frd_ned_wxyz, dtype=np.float64)
    thrust = np.asarray(thrust_body_z, dtype=np.float64)
    hover = np.asarray(hover_thrust_normalized, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("attitude setpoints must be an N x 4 quaternion matrix")
    if thrust.shape != (len(quaternions),) or hover.shape != (len(quaternions),):
        raise ValueError("attitude thrust and hover setpoints must match quaternion count")
    if not np.all(np.isfinite(quaternions)) or not np.all(np.isfinite(thrust)):
        raise ValueError("attitude setpoints must be finite")
    if np.any(thrust > 0.0):
        raise ValueError("attitude thrust_body_z must be non-positive")
    if not np.all(np.isfinite(hover)) or np.any(hover <= 0.0) or np.any(hover > 1.0):
        raise ValueError("hover thrust estimates must be finite and in (0, 1]")
    body_z_ned = np.stack([_quat_to_matrix(quaternion_wxyz=value)[:, 2] for value in quaternions])
    thrust_ned_z = body_z_ned[:, 2] * thrust
    acceleration_ned_z = (thrust_ned_z + hover) * (GRAVITY_M_S2 / hover)
    return -acceleration_ned_z


def _interpolate_quaternions(
    *,
    source_t: np.ndarray,
    quaternions_wxyz: np.ndarray,
    query_t: np.ndarray,
) -> np.ndarray:
    upper = np.clip(np.searchsorted(source_t, query_t, side="right"), 1, len(source_t) - 1)
    lower = upper - 1
    denominator = source_t[upper] - source_t[lower]
    fraction = np.divide(
        query_t - source_t[lower],
        denominator,
        out=np.zeros_like(query_t),
        where=denominator > 0.0,
    )
    first = quaternions_wxyz[lower]
    second = quaternions_wxyz[upper].copy()
    second[np.sum(first * second, axis=1) < 0.0] *= -1.0
    interpolated = first + fraction[:, None] * (second - first)
    return interpolated / np.linalg.norm(interpolated, axis=1)[:, None]


def _quat_to_matrix(*, quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_quat(*, rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        first = axis
        second = (axis + 1) % 3
        third = (axis + 2) % 3
        scale = (
            math.sqrt(
                1.0 + rotation[first, first] - rotation[second, second] - rotation[third, third]
            )
            * 2.0
        )
        quaternion = np.empty(4)
        quaternion[0] = (rotation[third, second] - rotation[second, third]) / scale
        quaternion[first + 1] = 0.25 * scale
        quaternion[second + 1] = (rotation[second, first] + rotation[first, second]) / scale
        quaternion[third + 1] = (rotation[third, first] + rotation[first, third]) / scale
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion / np.linalg.norm(quaternion)


def _px4_attitude_to_isaac(*, quaternion_frd_ned_wxyz: np.ndarray) -> tuple[np.ndarray, float]:
    rotation_ned_frd = _quat_to_matrix(quaternion_wxyz=quaternion_frd_ned_wxyz)
    rotation_enu_flu = _NED_TO_ENU @ rotation_ned_frd @ _FLU_TO_FRD
    yaw = math.atan2(rotation_enu_flu[1, 0], rotation_enu_flu[0, 0])
    return _matrix_to_quat(rotation=rotation_enu_flu), yaw


def _geometry_and_mass(*, airframe_name: str) -> tuple[MixerGeometry, float]:
    airframe = load_builder_config(overrides=[f"airframes={airframe_name}"]).airframe
    rotor_count = len(airframe.arms)
    positions = np.zeros((rotor_count, 3))
    directions = np.zeros(rotor_count)
    local_position = np.asarray(airframe.arm_geometry.propeller.position, dtype=np.float64)
    for arm in airframe.arms:
        angle = math.radians(arm.azimuth_deg)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        positions[arm.px4_motor_index] = rotation @ local_position
        directions[arm.px4_motor_index] = 1.0 if arm.spin_dir == "CCW" else -1.0
    body_mass, _center_of_mass, _inertia = compute_body_mass_properties(airframe)
    moving_pusher_mass = 0.0 if airframe.pusher is None else airframe.pusher.pad.mass
    return (
        MixerGeometry(
            positions_body=positions,
            rot_dir=directions,
            rotor_constant=np.empty(rotor_count),
            rolling_moment_coefficient=np.empty(rotor_count),
            min_rotor_velocity=np.empty(rotor_count),
            max_rotor_velocity=np.empty(rotor_count),
            input_offset=np.empty(rotor_count),
            input_scaling=np.empty(rotor_count),
            zero_position_armed=np.empty(rotor_count),
        ),
        body_mass + moving_pusher_mass,
    )


def _wrench(*, geometry: MixerGeometry, omega: np.ndarray) -> np.ndarray:
    thrust = geometry.rotor_constant * np.square(omega)
    positions = geometry.positions_body
    yaw = geometry.rolling_moment_coefficient * np.square(omega) * geometry.rot_dir
    return np.array(
        [
            float(np.sum(thrust)),
            float(np.sum(positions[:, 1] * thrust)),
            float(-np.sum(positions[:, 0] * thrust)),
            float(np.sum(yaw)),
        ]
    )


def _root_mean_square(*, values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _normalized_boundary_metrics(
    *,
    target_motor_controls: np.ndarray,
    px4_motor_controls: np.ndarray,
    normalized_mix: np.ndarray,
    px4_collective: np.ndarray,
    px4_torque_flu: np.ndarray,
) -> dict:
    """Recover unsaturated target thrust/torque at the PX4 allocator boundary."""
    target_controls = np.asarray(target_motor_controls, dtype=np.float64)
    px4_controls = np.asarray(px4_motor_controls, dtype=np.float64)
    if target_controls.shape != px4_controls.shape:
        raise ValueError(
            "target and PX4 motor controls must have matching shapes, got "
            f"{target_controls.shape} and {px4_controls.shape}"
        )
    mix = np.asarray(normalized_mix, dtype=np.float64)
    if mix.shape != (target_controls.shape[1], 4):
        raise ValueError(
            f"normalized mix has shape {mix.shape}, expected ({target_controls.shape[1]}, 4)"
        )
    collective = np.asarray(px4_collective, dtype=np.float64)
    torque = np.asarray(px4_torque_flu, dtype=np.float64)
    if collective.shape != (len(target_controls),):
        raise ValueError(
            f"PX4 collective has shape {collective.shape}, expected ({len(target_controls)},)"
        )
    if torque.shape != (len(target_controls), 3):
        raise ValueError(
            f"PX4 torque has shape {torque.shape}, expected ({len(target_controls)}, 3)"
        )

    saturated = np.any(
        (target_controls <= _MOTOR_SATURATION_EPS)
        | (target_controls >= 1.0 - _MOTOR_SATURATION_EPS)
        | (px4_controls <= _MOTOR_SATURATION_EPS)
        | (px4_controls >= 1.0 - _MOTOR_SATURATION_EPS),
        axis=1,
    )
    unsaturated = ~saturated
    recovered = target_controls @ np.linalg.pinv(mix).T
    target_collective = -recovered[:, 3]
    target_torque_flu = recovered[:, :3] * np.array([1.0, -1.0, -1.0])
    return {
        "normalized_motor_rmse": _root_mean_square(values=target_controls - px4_controls),
        "unsaturated_sample_count": int(np.count_nonzero(unsaturated)),
        "saturated_sample_count": int(np.count_nonzero(saturated)),
        "normalized_collective_rmse": (
            _root_mean_square(values=target_collective[unsaturated] - collective[unsaturated])
            if np.any(unsaturated)
            else None
        ),
        "normalized_torque_rmse": (
            [
                _root_mean_square(
                    values=target_torque_flu[unsaturated, axis] - torque[unsaturated, axis]
                )
                for axis in range(3)
            ]
            if np.any(unsaturated)
            else None
        ),
    }


def _controller_setpoint_metrics(
    *,
    target_body_rate_flu: np.ndarray,
    px4_body_rate_flu: np.ndarray,
    target_collective: np.ndarray,
    px4_collective: np.ndarray,
) -> dict:
    """Compare the attitude-loop output before the shared rate controller."""
    target_rate = np.asarray(target_body_rate_flu, dtype=np.float64)
    px4_rate = np.asarray(px4_body_rate_flu, dtype=np.float64)
    target_thrust = np.asarray(target_collective, dtype=np.float64)
    px4_thrust = np.asarray(px4_collective, dtype=np.float64)
    if target_rate.shape != px4_rate.shape or target_rate.ndim != 2 or target_rate.shape[1] != 3:
        raise ValueError(
            "target and PX4 body-rate setpoints must have matching (samples, 3) shapes, got "
            f"{target_rate.shape} and {px4_rate.shape}"
        )
    expected_collective_shape = (len(target_rate),)
    if target_thrust.shape != expected_collective_shape or px4_thrust.shape != (
        expected_collective_shape
    ):
        raise ValueError(
            "target and PX4 collective setpoints must match the body-rate sample count"
        )
    return {
        "body_rate_setpoint_rmse_radps": [
            _root_mean_square(values=target_rate[:, axis] - px4_rate[:, axis]) for axis in range(3)
        ],
        "collective_setpoint_rmse": _root_mean_square(values=target_thrust - px4_thrust),
    }


def _collective_boundary_metrics(
    *,
    target_desired_collective: np.ndarray,
    px4_attitude_collective: np.ndarray,
    px4_rate_collective: np.ndarray,
    px4_allocator_collective: np.ndarray,
) -> dict[str, float]:
    """Keep the velocity controller's collective handoffs independently visible."""
    boundaries = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            target_desired_collective,
            px4_attitude_collective,
            px4_rate_collective,
            px4_allocator_collective,
        )
    )
    if (
        any(values.ndim != 1 for values in boundaries)
        or len({len(values) for values in boundaries}) != 1
    ):
        raise ValueError("collective boundary samples must be equally sized vectors")
    if not all(np.all(np.isfinite(values)) for values in boundaries):
        raise ValueError("collective boundary samples must be finite")
    desired, attitude, rate, allocator = boundaries
    return {
        "desired_to_attitude_collective_rmse": _root_mean_square(values=desired - attitude),
        "attitude_to_rate_collective_rmse": _root_mean_square(values=attitude - rate),
        "rate_to_allocator_collective_rmse": _root_mean_square(values=rate - allocator),
    }


def _controller_setpoint_delay_scan(
    *,
    target_body_rate_grid_flu: np.ndarray,
    target_collective_grid: np.ndarray,
    replay_indices: np.ndarray,
    px4_body_rate_flu: np.ndarray,
    px4_collective: np.ndarray,
    maximum_delay_steps: int,
) -> list[dict]:
    """Measure deterministic target handoff delays against PX4 rate setpoints."""
    if maximum_delay_steps < 0:
        raise ValueError("maximum delay steps must be non-negative")
    indices = np.asarray(replay_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0):
        raise ValueError("replay indices must be a non-negative vector")
    target_rates = np.asarray(target_body_rate_grid_flu, dtype=np.float64)
    target_collective = np.asarray(target_collective_grid, dtype=np.float64)
    if target_rates.ndim != 2 or target_rates.shape[1] != 3:
        raise ValueError("target body-rate grid must have shape (samples, 3)")
    if target_collective.shape != (len(target_rates),):
        raise ValueError("target collective grid must match target body-rate grid")
    if indices.size and int(np.max(indices)) >= len(target_rates):
        raise ValueError("replay index exceeds the target setpoint grid")
    results = []
    for delay_steps in range(maximum_delay_steps + 1):
        delayed_indices = np.maximum(indices - delay_steps, 0)
        metrics = _controller_setpoint_metrics(
            target_body_rate_flu=target_rates[delayed_indices],
            px4_body_rate_flu=px4_body_rate_flu,
            target_collective=target_collective[delayed_indices],
            px4_collective=px4_collective,
        )
        results.append({"delay_steps": delay_steps, **metrics})
    return results


def _allocator_collective_at_motor_samples(
    *,
    motor_controls: np.ndarray,
    normalized_mix: np.ndarray,
    logged_collective: np.ndarray,
    logged_sample_matches: np.ndarray,
) -> np.ndarray:
    """Recover missing thrust publications from their unsaturated motor output."""
    unsaturated = np.all(
        (motor_controls > _MOTOR_SATURATION_EPS) & (motor_controls < 1.0 - _MOTOR_SATURATION_EPS),
        axis=1,
    )
    recovered = -(motor_controls @ np.linalg.pinv(normalized_mix).T)[:, 3]
    return np.where(logged_sample_matches | ~unsaturated, logged_collective, recovered)


def _controller_handoff_indices(
    *,
    replay_indices: np.ndarray,
    current_attitude_collective: np.ndarray,
    previous_attitude_collective: np.ndarray,
    current_rates_collective: np.ndarray,
    previous_rates_collective: np.ndarray,
    allocator_collective: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join asynchronous PX4 controller work queues at adjacent publications."""
    indices = np.asarray(replay_indices, dtype=np.int64)
    current_attitude = np.asarray(current_attitude_collective, dtype=np.float64)
    previous_attitude = np.asarray(previous_attitude_collective, dtype=np.float64)
    current_rates = np.asarray(current_rates_collective, dtype=np.float64)
    previous_rates = np.asarray(previous_rates_collective, dtype=np.float64)
    allocator = np.asarray(allocator_collective, dtype=np.float64)
    if any(
        values.shape != indices.shape
        for values in (
            current_attitude,
            previous_attitude,
            current_rates,
            previous_rates,
            allocator,
        )
    ):
        raise ValueError("controller handoff samples must match the replay indices")
    # PX4 copies collective unchanged; allocator evidence can recover an unlogged rate publication.
    current_rates = np.where(
        (allocator == current_attitude)
        & (allocator != current_rates)
        & (allocator != previous_rates),
        allocator,
        current_rates,
    )
    attitude_delay = (
        np.abs(current_rates - current_attitude) > np.abs(current_rates - previous_attitude)
    ).astype(np.int64)
    allocator_delay = (
        np.abs(allocator - current_rates) > np.abs(allocator - previous_rates)
    ).astype(np.int64)
    joined_indices = np.maximum(indices - attitude_delay - allocator_delay, 0)
    return joined_indices, attitude_delay, allocator_delay


def _vertical_setpoint_delay_scan(
    *,
    target_acceleration_setpoint_grid_enu: np.ndarray,
    replay_indices: np.ndarray,
    px4_acceleration_setpoint_z_enu: np.ndarray,
    maximum_delay_steps: int,
) -> list[dict]:
    """Measure outer-loop Z setpoint timing before attitude conversion."""
    target = np.asarray(target_acceleration_setpoint_grid_enu, dtype=np.float64)
    indices = np.asarray(replay_indices, dtype=np.int64)
    px4 = np.asarray(px4_acceleration_setpoint_z_enu, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target acceleration setpoint grid must have shape (samples, 3)")
    if indices.ndim != 1 or px4.shape != indices.shape:
        raise ValueError("replay indices and PX4 acceleration setpoints must match")
    if maximum_delay_steps < 0 or np.any(indices < 0):
        raise ValueError("delay steps and replay indices must be non-negative")
    if indices.size and int(np.max(indices)) >= len(target):
        raise ValueError("replay index exceeds the target acceleration setpoint grid")
    results = []
    for delay_steps in range(maximum_delay_steps + 1):
        delayed_indices = np.maximum(indices - delay_steps, 0)
        error = target[delayed_indices, 2] - px4
        results.append(
            {
                "delay_steps": delay_steps,
                "rmse_mps2": _root_mean_square(values=error),
                "max_abs_mps2": float(np.max(np.abs(error))),
            }
        )
    return results


def _rate_input_metrics(
    *,
    target_angular_velocity_flu: np.ndarray,
    px4_angular_velocity_flu: np.ndarray,
    target_angular_acceleration_flu: np.ndarray,
    px4_angular_acceleration_flu: np.ndarray,
) -> dict:
    """Compare the measured-state inputs consumed by the rate PID."""
    target_rate = np.asarray(target_angular_velocity_flu, dtype=np.float64)
    px4_rate = np.asarray(px4_angular_velocity_flu, dtype=np.float64)
    target_acceleration = np.asarray(target_angular_acceleration_flu, dtype=np.float64)
    px4_acceleration = np.asarray(px4_angular_acceleration_flu, dtype=np.float64)
    expected_shape = target_rate.shape
    if (
        target_rate.ndim != 2
        or expected_shape[1] != 3
        or px4_rate.shape != expected_shape
        or target_acceleration.shape != expected_shape
        or px4_acceleration.shape != expected_shape
    ):
        raise ValueError("rate PID input arrays must have matching (samples, 3) shapes")
    return {
        "angular_velocity_input_rmse_radps": [
            _root_mean_square(values=target_rate[:, axis] - px4_rate[:, axis]) for axis in range(3)
        ],
        "angular_acceleration_input_rmse_radps2": [
            _root_mean_square(values=target_acceleration[:, axis] - px4_acceleration[:, axis])
            for axis in range(3)
        ],
    }


def _topic_time_us(*, topic: dict) -> np.ndarray:
    return np.asarray(topic["timestamp"], dtype=np.int64)


def _topic_vectors(*, topic: dict, field: str, length: int) -> np.ndarray:
    return np.column_stack(
        [np.asarray(topic[f"{field}[{index}]"], dtype=np.float64) for index in range(length)]
    )


def _bridge_actuator_samples(
    *,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actuator_times: list[int] = []
    rotor_speeds: list[list[float]] = []
    forces: list[list[float]] = []
    join_errors: list[int] = []
    body_torques: list[list[float]] = []
    with open_flight_evidence(path) as stream:
        for encoded in stream:
            line = encoded.decode("utf-8")
            dynamics_match = _BRIDGE_DYNAMICS_PATTERN.search(line)
            if dynamics_match is not None:
                actuator_times.append(int(dynamics_match.group("actuator_time_us")))
                rotor_speeds.append(list(ast.literal_eval(dynamics_match.group("rotor_speed"))))
                forces.append(list(ast.literal_eval(dynamics_match.group("forces"))))
                join_errors.append(int(dynamics_match.group("join_error_us")))
                body_torques.append(list(ast.literal_eval(dynamics_match.group("body_torque"))))
    if not actuator_times:
        raise ValueError(f"bridge log {path} has incomplete actuator diagnostics")
    return (
        np.asarray(actuator_times, dtype=np.int64),
        np.asarray(rotor_speeds, dtype=np.float64),
        np.asarray(forces, dtype=np.float64),
        np.column_stack(
            [
                np.asarray(join_errors, dtype=np.int64),
                np.asarray(body_torques, dtype=np.float64),
            ]
        ),
    )


def _bridge_state_samples(
    *,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times: list[int] = []
    positions: list[list[float]] = []
    attitudes: list[list[float]] = []
    velocities: list[list[float]] = []
    angular_velocities: list[list[float]] = []
    with open_flight_evidence(path) as stream:
        for encoded in stream:
            line = encoded.decode("utf-8")
            state_match = _BRIDGE_STATE_PATTERN.search(line)
            if state_match is None:
                continue
            sample_time = int(state_match.group("sim_time_us"))
            position = list(ast.literal_eval(state_match.group("position")))
            attitude = list(ast.literal_eval(state_match.group("attitude")))
            velocity = list(ast.literal_eval(state_match.group("velocity")))
            angular_velocity = list(ast.literal_eval(state_match.group("angular_velocity")))
            if times and sample_time == times[-1]:
                positions[-1] = position
                attitudes[-1] = attitude
                velocities[-1] = velocity
                angular_velocities[-1] = angular_velocity
                continue
            if times and sample_time < times[-1]:
                raise ValueError(f"bridge log {path} physical-state timestamps regress")
            times.append(sample_time)
            positions.append(position)
            attitudes.append(attitude)
            velocities.append(velocity)
            angular_velocities.append(angular_velocity)
    if not times:
        raise ValueError(f"bridge log {path} has incomplete physical-state diagnostics")
    time_array = np.asarray(times, dtype=np.int64)
    return (
        time_array,
        np.asarray(positions, dtype=np.float64),
        np.asarray(attitudes, dtype=np.float64),
        np.asarray(velocities, dtype=np.float64),
        np.asarray(angular_velocities, dtype=np.float64),
    )


def _profile_topic(
    *,
    topic: dict,
    values: np.ndarray,
    start_us: int,
    end_us: int,
) -> dict:
    time_us = _topic_time_us(topic=topic)
    selected = (time_us >= start_us) & (time_us <= end_us)
    return {
        "timestamp_us": time_us[selected].tolist(),
        "values": values[selected].tolist(),
    }


def _velocity_controller_derivative(*, ulog, local_position: dict) -> np.ndarray:
    """Reconstruct mc_pos_control's filtered velocity derivative, not EKF acceleration."""
    names = ("MPC_VEL_LP", "MPC_VEL_NF_FRQ", "MPC_VELD_LP")
    try:
        velocity_lp, notch_hz, derivative_hz = (
            float(ulog.initial_parameters[name]) for name in names
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("ULog is missing the PX4 velocity-controller filter parameters") from exc
    if not all(
        math.isfinite(value) and value >= 0.0 for value in (velocity_lp, notch_hz, derivative_hz)
    ):
        raise ValueError(
            "PX4 velocity-controller filter frequencies must be finite and non-negative"
        )
    if velocity_lp != 0.0 or notch_hz != 0.0:
        raise ValueError(
            "controller-boundary capture requires disabled MPC_VEL_LP and MPC_VEL_NF_FRQ"
        )
    if any(name in names for _timestamp, name, _value in ulog.changed_parameters):
        raise ValueError("controller-boundary capture requires constant velocity filter parameters")

    timestamps = np.asarray(local_position["timestamp_sample"], dtype=np.int64)
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("velocity-controller reconstruction requires increasing sample timestamps")
    velocity = np.column_stack([local_position[name] for name in ("vx", "vy", "vz")])
    valid = np.column_stack(
        [local_position[name] for name in ("v_xy_valid", "v_xy_valid", "v_z_valid")]
    ).astype(bool) & np.isfinite(velocity)
    valid[:, :2] = np.all(valid[:, :2], axis=1)[:, None]
    derivative = np.empty_like(velocity, dtype=np.float64)
    previous_velocity = np.zeros(3)
    filtered_derivative = np.zeros(3)
    # These are the pinned PX4 module's dt clamps, not experiment tunables.
    intervals = np.clip(
        np.diff(timestamps, prepend=timestamps[0] - (timestamps[1] - timestamps[0])) * 1e-6,
        0.002,
        0.04,
    )
    for index, dt in enumerate(intervals):
        alpha = 1.0 if derivative_hz == 0.0 else dt / (dt + 1.0 / (2.0 * math.pi * derivative_hz))
        acceleration = (velocity[index] - previous_velocity) / dt
        filtered_derivative = np.where(
            valid[index], filtered_derivative + alpha * (acceleration - filtered_derivative), 0.0
        )
        previous_velocity = np.where(valid[index], velocity[index], 0.0)
        derivative[index] = np.where(valid[index], filtered_derivative, np.nan)
    return derivative


def _controller_boundary(
    *,
    local_setpoint: dict,
    local_position: dict,
    velocity_derivative_ned: np.ndarray,
    attitude_setpoint: dict,
    rates_setpoint: dict,
    torque_setpoint: dict,
    angular_velocity: dict,
    rate_status: dict,
    hover: dict,
    timestamp_us: int,
) -> tuple[dict[str, np.ndarray | float], dict[str, int]]:
    def held_age_us(*, topic: dict) -> int:
        time_us = _topic_time_us(topic=topic)
        index = int(_hold_indices(source_t=time_us, query_t=np.array([timestamp_us]))[0])
        return timestamp_us - int(time_us[index])

    def interpolated_vector(*, topic: dict, fields: tuple[str, str, str]) -> np.ndarray:
        return _linear_interpolate(
            source_t=_topic_time_us(topic=topic).astype(np.float64),
            values=np.column_stack(
                [np.asarray(topic[field], dtype=np.float64) for field in fields]
            ),
            query_t=np.array([timestamp_us], dtype=np.float64),
        )[0]

    def held_vector(*, topic: dict, fields: tuple[str, ...]) -> np.ndarray:
        time_us = _topic_time_us(topic=topic)
        index = int(_hold_indices(source_t=time_us, query_t=np.array([timestamp_us]))[0])
        return np.asarray([topic[field][index] for field in fields], dtype=np.float64)

    local_setpoint_age_us = held_age_us(topic=local_setpoint)
    rate_age_us = held_age_us(topic=rate_status)
    acceleration_setpoint_ned = held_vector(
        topic=local_setpoint,
        fields=("acceleration[0]", "acceleration[1]", "acceleration[2]"),
    )
    velocity_setpoint_ned = held_vector(
        topic=local_setpoint,
        fields=("vx", "vy", "vz"),
    )
    linear_acceleration_ned = _linear_interpolate(
        source_t=_topic_time_us(topic=local_position).astype(np.float64),
        values=velocity_derivative_ned,
        query_t=np.array([timestamp_us], dtype=np.float64),
    )[0]
    angular_acceleration_frd = interpolated_vector(
        topic=angular_velocity,
        fields=(
            "xyz_derivative[0]",
            "xyz_derivative[1]",
            "xyz_derivative[2]",
        ),
    )
    estimated_angular_velocity_frd = interpolated_vector(
        topic=angular_velocity,
        fields=("xyz[0]", "xyz[1]", "xyz[2]"),
    )
    rate_integral_frd = held_vector(
        topic=rate_status,
        fields=("rollspeed_integ", "pitchspeed_integ", "yawspeed_integ"),
    )
    estimated_velocity_ned = interpolated_vector(
        topic=local_position,
        fields=("vx", "vy", "vz"),
    )
    hover_thrust_normalized = float(held_vector(topic=hover, fields=("hover_thrust",))[0])
    desired_attitude_frd_ned = held_vector(
        topic=attitude_setpoint,
        fields=("q_d[0]", "q_d[1]", "q_d[2]", "q_d[3]"),
    )
    desired_attitude_flu_enu, _desired_yaw = _px4_attitude_to_isaac(
        quaternion_frd_ned_wxyz=desired_attitude_frd_ned
    )
    desired_yaw_rate_radps = -float(
        held_vector(topic=attitude_setpoint, fields=("yaw_sp_move_rate",))[0]
    )
    desired_collective = -float(held_vector(topic=attitude_setpoint, fields=("thrust_body[2]",))[0])
    body_rate_setpoint_frd = held_vector(
        topic=rates_setpoint,
        fields=("roll", "pitch", "yaw"),
    )
    collective = -float(held_vector(topic=rates_setpoint, fields=("thrust_body[2]",))[0])
    torque_setpoint_frd = held_vector(
        topic=torque_setpoint,
        fields=("xyz[0]", "xyz[1]", "xyz[2]"),
    )
    vectors = {
        "initial_estimated_linear_velocity_enu_mps": _NED_TO_ENU @ estimated_velocity_ned,
        "initial_velocity_setpoint_enu_mps": _NED_TO_ENU @ velocity_setpoint_ned,
        "initial_estimated_angular_velocity_body_flu_radps": (
            _FLU_TO_FRD @ estimated_angular_velocity_frd
        ),
        "initial_acceleration_setpoint_enu_mps2": _NED_TO_ENU @ acceleration_setpoint_ned,
        "initial_linear_acceleration_enu_mps2": _NED_TO_ENU @ linear_acceleration_ned,
        "initial_angular_acceleration_body_flu_radps2": _FLU_TO_FRD @ angular_acceleration_frd,
        "initial_rate_integral_body_flu": _FLU_TO_FRD @ rate_integral_frd,
        "initial_hover_thrust_normalized": hover_thrust_normalized,
        "initial_desired_attitude_flu_enu_wxyz": desired_attitude_flu_enu,
        "initial_desired_yaw_rate_radps": desired_yaw_rate_radps,
        "initial_desired_collective_thrust_normalized": desired_collective,
        "initial_collective_thrust_normalized": collective,
        "initial_body_rate_setpoint_flu_radps": _FLU_TO_FRD @ body_rate_setpoint_frd,
        "initial_filtered_yaw_torque_normalized": -float(torque_setpoint_frd[2]),
    }
    if not all(np.all(np.isfinite(values)) for values in vectors.values()):
        raise ValueError("initial controller boundary contains non-finite values")
    if hover_thrust_normalized <= 0.0:
        raise ValueError("initial hover thrust estimate must be positive")
    return (
        vectors,
        {
            "vehicle_local_position_setpoint": local_setpoint_age_us,
            "vehicle_local_position": held_age_us(topic=local_position),
            "vehicle_attitude_setpoint": held_age_us(topic=attitude_setpoint),
            "vehicle_rates_setpoint": held_age_us(topic=rates_setpoint),
            "vehicle_torque_setpoint": held_age_us(topic=torque_setpoint),
            "vehicle_angular_velocity": held_age_us(topic=angular_velocity),
            "rate_ctrl_status": rate_age_us,
            "hover_thrust_estimate": held_age_us(topic=hover),
        },
    )


def _augment_reference_controller_state(
    *,
    ulog,
    reference_path: Path,
) -> dict:
    reference = json.loads(reference_path.read_text())
    if reference.get("schema_version") != 1:
        raise ValueError(f"{reference_path} must use alignment trace schema 1")
    raw_profiles = (
        [stage for stage in reference["stages"] if stage["kind"] == "controller-response"]
        if reference.get("artifact_kind") == "robotics-experiments-flight-recording"
        else reference.get("traces")
    )
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(f"{reference_path} has no alignment traces")

    local_setpoint = _dataset(ulog=ulog, name="vehicle_local_position_setpoint")
    local_position = _dataset(ulog=ulog, name="vehicle_local_position")
    velocity_derivative_ned = _velocity_controller_derivative(
        ulog=ulog, local_position=local_position
    )
    attitude_setpoint = _dataset(ulog=ulog, name="vehicle_attitude_setpoint")
    rates_setpoint = _dataset(ulog=ulog, name="vehicle_rates_setpoint")
    torque_setpoint = _dataset(ulog=ulog, name="vehicle_torque_setpoint")
    angular_velocity = _dataset(ulog=ulog, name="vehicle_angular_velocity")
    rate_status = _dataset(ulog=ulog, name="rate_ctrl_status")
    hover = _dataset(ulog=ulog, name="hover_thrust_estimate")

    profile_reports = []
    for raw_profile in raw_profiles:
        start_us = int(raw_profile["start_sim_time_us"])
        controller_vectors, source_age_us = _controller_boundary(
            local_setpoint=local_setpoint,
            local_position=local_position,
            velocity_derivative_ned=velocity_derivative_ned,
            attitude_setpoint=attitude_setpoint,
            rates_setpoint=rates_setpoint,
            torque_setpoint=torque_setpoint,
            angular_velocity=angular_velocity,
            rate_status=rate_status,
            hover=hover,
            timestamp_us=start_us,
        )
        raw_profile.update(
            {
                "initial_controller_state_valid": True,
                **{
                    name: values.tolist() if isinstance(values, np.ndarray) else values
                    for name, values in controller_vectors.items()
                },
                "initial_controller_state_source_age_us": source_age_us,
            }
        )
        profile_reports.append(
            {
                "profile": raw_profile.get("profile", raw_profile.get("name")),
                "start_sim_time_us": start_us,
                "source_age_us": raw_profile["initial_controller_state_source_age_us"],
            }
        )
    reference["controller_state_method"] = "px4-ulog-controller-boundary-v2"
    temporary_path = reference_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    temporary_path.replace(reference_path)
    return {
        "schema_version": 1,
        "method": reference["controller_state_method"],
        "reference": str(reference_path),
        "profiles": profile_reports,
    }


def _actuator_boundary_report(
    *,
    ulog,
    config_path: Path,
    reference: dict,
    bridge_log_path: Path,
) -> dict:
    config = load_config(config_path)
    vehicle = next(vehicle for vehicle in config.vehicles.values() if vehicle.bridge.enabled)
    airframe_name = Path(vehicle.spawn.usd_path).stem
    base_geometry, mass = _geometry_and_mass(airframe_name=airframe_name)
    bridge = vehicle.bridge
    geometry = MixerGeometry(
        positions_body=base_geometry.positions_body,
        rot_dir=base_geometry.rot_dir,
        rotor_constant=np.asarray(bridge.rotor_constant),
        rolling_moment_coefficient=np.asarray(bridge.rolling_moment_coefficient),
        min_rotor_velocity=np.asarray(bridge.min_rotor_velocity),
        max_rotor_velocity=np.asarray(bridge.max_rotor_velocity),
        input_offset=np.asarray(bridge.input_offset),
        input_scaling=np.asarray(bridge.input_scaling),
        zero_position_armed=np.asarray(bridge.zero_position_armed),
    )
    if reference.get("schema_version") != 1:
        raise ValueError("reference must use alignment trace schema 1")
    raw_profiles = reference.get("traces")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("reference has no alignment traces")

    thrust = _dataset(ulog=ulog, name="vehicle_thrust_setpoint")
    torque = _dataset(ulog=ulog, name="vehicle_torque_setpoint")
    motors = _dataset(ulog=ulog, name="actuator_motors")
    hover = _dataset(ulog=ulog, name="hover_thrust_estimate")
    rotor_count = len(bridge.input_scaling)
    thrust_values = _topic_vectors(topic=thrust, field="xyz", length=3)
    torque_values = _topic_vectors(topic=torque, field="xyz", length=3)
    motor_values = _topic_vectors(topic=motors, field="control", length=rotor_count)
    hover_values = np.asarray(hover["hover_thrust"], dtype=np.float64)[:, None]
    (
        bridge_time_us,
        bridge_rotor_speed,
        bridge_forces,
        bridge_join_and_torque,
    ) = _bridge_actuator_samples(path=bridge_log_path)
    bridge_dynamics_join_error_us = bridge_join_and_torque[:, 0].astype(np.int64)
    bridge_body_torque = bridge_join_and_torque[:, 1:]

    motor_time_us = _topic_time_us(topic=motors)
    bridge_index = _nearest_indices(
        source_t=bridge_time_us.astype(np.float64),
        query_t=motor_time_us.astype(np.float64),
    )
    motor_join_error_us = np.abs(bridge_time_us[bridge_index] - motor_time_us)
    input_scaling = np.asarray(bridge.input_scaling, dtype=np.float64)
    input_offset = np.asarray(bridge.input_offset, dtype=np.float64)
    armed_zero = np.asarray(bridge.zero_position_armed, dtype=np.float64)
    sampled_rotor_speed = bridge_rotor_speed[bridge_index]
    sampled_forces = bridge_forces[bridge_index]
    sampled_body_torque = bridge_body_torque[bridge_index]
    bridge_normalized_motor = (sampled_rotor_speed - armed_zero) / input_scaling - input_offset
    expected_rotor_speed = motor_values * input_scaling + armed_zero + input_offset * input_scaling
    expected_collective = np.sum(
        geometry.rotor_constant * np.square(expected_rotor_speed),
        axis=1,
    )
    bridge_collective = np.sum(sampled_forces, axis=1)
    bridge_torque = np.column_stack(
        [
            np.sum(geometry.positions_body[:, 1] * sampled_forces, axis=1),
            -np.sum(geometry.positions_body[:, 0] * sampled_forces, axis=1),
            sampled_body_torque[:, 2],
        ]
    )

    profile_reports = []
    audit_failures: list[str] = []
    for raw_profile in raw_profiles:
        profile_name = str(raw_profile["profile"])
        start_us = int(raw_profile["start_sim_time_us"])
        relative_time_s = raw_profile["time_s"]
        end_us = start_us + round(float(relative_time_s[-1]) * 1_000_000)
        bridge_selected = (motor_time_us >= start_us) & (motor_time_us <= end_us)
        topic_reports = {
            "vehicle_thrust_setpoint": _profile_topic(
                topic=thrust,
                values=thrust_values,
                start_us=start_us,
                end_us=end_us,
            ),
            "vehicle_torque_setpoint": _profile_topic(
                topic=torque,
                values=torque_values,
                start_us=start_us,
                end_us=end_us,
            ),
            "actuator_motors": _profile_topic(
                topic=motors,
                values=motor_values,
                start_us=start_us,
                end_us=end_us,
            ),
            "hover_thrust_estimate": _profile_topic(
                topic=hover,
                values=hover_values,
                start_us=start_us,
                end_us=end_us,
            ),
        }
        missing_topics = [
            name for name, topic_report in topic_reports.items() if not topic_report["timestamp_us"]
        ]
        if missing_topics or not np.any(bridge_selected):
            audit_failures.append(
                f"{profile_name}: incomplete topic coverage "
                f"{missing_topics + ([] if np.any(bridge_selected) else ['bridge'])}"
            )
        topic_coverage = {
            name: {
                "sample_count": len(topic_report["timestamp_us"]),
                "first_timestamp_us": (
                    topic_report["timestamp_us"][0] if topic_report["timestamp_us"] else None
                ),
                "last_timestamp_us": (
                    topic_report["timestamp_us"][-1] if topic_report["timestamp_us"] else None
                ),
            }
            for name, topic_report in topic_reports.items()
        }
        profile_motor_error = (
            bridge_normalized_motor[bridge_selected] - motor_values[bridge_selected]
        )
        profile_collective_error = (
            bridge_collective[bridge_selected] - expected_collective[bridge_selected]
        )
        normalized_motor_rmse = (
            _root_mean_square(values=profile_motor_error) if profile_motor_error.size else math.inf
        )
        collective_rmse_n = (
            _root_mean_square(values=profile_collective_error)
            if profile_collective_error.size
            else math.inf
        )
        max_join_error_us = int(
            max(
                np.max(motor_join_error_us[bridge_selected], initial=0),
                np.max(
                    bridge_dynamics_join_error_us[bridge_index][bridge_selected],
                    initial=0,
                ),
            )
        )
        if max_join_error_us > _PHYSICS_TICK_US:
            audit_failures.append(
                f"{profile_name}: timestamp join error {max_join_error_us} us exceeds "
                f"{_PHYSICS_TICK_US} us"
            )
        if normalized_motor_rmse > _MAX_NORMALIZED_MOTOR_RMSE:
            audit_failures.append(
                f"{profile_name}: normalized motor RMSE {normalized_motor_rmse:.6f} "
                f"exceeds {_MAX_NORMALIZED_MOTOR_RMSE:.6f}"
            )
        if collective_rmse_n > _MAX_COLLECTIVE_RMSE_N:
            audit_failures.append(
                f"{profile_name}: collective RMSE {collective_rmse_n:.6f} N "
                f"exceeds {_MAX_COLLECTIVE_RMSE_N:.6f} N"
            )
        profile_reports.append(
            {
                "profile": profile_name,
                "split": raw_profile["split"],
                "axis": raw_profile["axis"],
                "start_sim_time_us": start_us,
                "end_sim_time_us": end_us,
                "topic_coverage": topic_coverage,
                "bridge": {
                    "sample_count": int(np.count_nonzero(bridge_selected)),
                    "mean_mapped_rotor_speed_radps": np.mean(
                        sampled_rotor_speed[bridge_selected], axis=0
                    ).tolist(),
                    "mean_reconstructed_collective_thrust_n": float(
                        np.mean(bridge_collective[bridge_selected])
                    ),
                    "mean_reconstructed_body_torque_nm": np.mean(
                        bridge_torque[bridge_selected], axis=0
                    ).tolist(),
                    "saturated_motor_sample_count": int(
                        np.count_nonzero(
                            (bridge_normalized_motor[bridge_selected] <= 0.0)
                            | (bridge_normalized_motor[bridge_selected] >= 1.0)
                        )
                    ),
                },
                "metrics": {
                    "normalized_motor_rmse": normalized_motor_rmse,
                    "collective_thrust_rmse_n": collective_rmse_n,
                    "max_timestamp_join_error_us": max_join_error_us,
                },
            }
        )
    return {
        "schema_version": 1,
        "method": "px4-ulog-bridge-actuator-boundary-v1",
        "config": str(config_path),
        "bridge_log": str(bridge_log_path),
        "airframe": airframe_name,
        "composed_mass_kg": mass,
        "limits": {
            "timestamp_join_error_us": _PHYSICS_TICK_US,
            "normalized_motor_rmse": _MAX_NORMALIZED_MOTOR_RMSE,
            "collective_thrust_rmse_n": _MAX_COLLECTIVE_RMSE_N,
        },
        "profiles": profile_reports,
        "passed": not audit_failures,
        "failures": audit_failures,
    }


def _linear_scale_with_intercept(
    *,
    controller_output: np.ndarray,
    physical_output: np.ndarray,
) -> tuple[float, float, float]:
    design = np.column_stack([controller_output, np.ones(len(controller_output))])
    scale, intercept = np.linalg.lstsq(design, physical_output, rcond=None)[0]
    residual = physical_output - (scale * controller_output + intercept)
    return float(scale), float(intercept), _root_mean_square(values=residual)


def _report(
    *,
    ulog,
    config_path: Path,
    controller_artifact_path: Path,
    start_s: float,
    end_s: float,
    time_origin_us: int,
    command_trace: dict | None,
    physical_state: bool,
    bridge_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
) -> dict:
    config = load_config(config_path)
    vehicle = next(vehicle for vehicle in config.vehicles.values() if vehicle.bridge.enabled)
    airframe_name = Path(vehicle.spawn.usd_path).stem
    geometry, mass = _geometry_and_mass(airframe_name=airframe_name)
    bridge = vehicle.bridge
    geometry = MixerGeometry(
        positions_body=geometry.positions_body,
        rot_dir=geometry.rot_dir,
        rotor_constant=np.asarray(bridge.rotor_constant),
        rolling_moment_coefficient=np.asarray(bridge.rolling_moment_coefficient),
        min_rotor_velocity=np.asarray(bridge.min_rotor_velocity),
        max_rotor_velocity=np.asarray(bridge.max_rotor_velocity),
        input_offset=np.asarray(bridge.input_offset),
        input_scaling=np.asarray(bridge.input_scaling),
        zero_position_armed=np.asarray(bridge.zero_position_armed),
    )
    controller_settings = load_controller_artifact(controller_artifact_path).selected_settings
    if not physical_state:
        controller_settings = replace(
            controller_settings,
            velocity_observation_scale=(1.0, 1.0, 1.0),
            velocity_observation_hold_scale=(1.0, 1.0, 1.0),
            velocity_observation_bias_mps=(0.0, 0.0, 0.0),
            velocity_estimator_bias_decay_hz=0.0,
            velocity_estimator_bias_hold_decay_hz=(0.0, 0.0, 0.0),
            velocity_filter_hz=(0.0, 0.0, 0.0),
            angular_velocity_filter_hz=0.0,
        )
    control_mode = (
        str(command_trace["control_mode"]) if command_trace is not None else config.rl.action.kind
    )
    params = build_actuation_params(
        geometry=geometry,
        action_cfg=replace(config.rl.action, kind=control_mode),
        controller_settings=controller_settings,
        mass=mass,
    )

    motor = _dataset(ulog=ulog, name="actuator_motors")
    trajectory = _dataset(ulog=ulog, name="trajectory_setpoint")
    local_position = _dataset(ulog=ulog, name="vehicle_local_position")
    attitude = _dataset(ulog=ulog, name="vehicle_attitude")
    attitude_setpoint = _dataset(ulog=ulog, name="vehicle_attitude_setpoint")
    angular_velocity = _dataset(ulog=ulog, name="vehicle_angular_velocity")
    local_setpoint = _dataset(ulog=ulog, name="vehicle_local_position_setpoint")
    rate_status = _dataset(ulog=ulog, name="rate_ctrl_status")
    hover = _dataset(ulog=ulog, name="hover_thrust_estimate")
    rates_setpoint = _dataset(ulog=ulog, name="vehicle_rates_setpoint")
    thrust_setpoint = _dataset(ulog=ulog, name="vehicle_thrust_setpoint")
    torque_setpoint = _dataset(ulog=ulog, name="vehicle_torque_setpoint")
    time_origin = float(time_origin_us)

    motor_t = (np.asarray(motor["timestamp"], dtype=np.float64) - time_origin) / 1e6
    selected = (motor_t >= start_s) & (motor_t <= end_s)
    sample_t = motor_t[selected]
    if not sample_t.size:
        raise ValueError(f"no actuator_motors samples in [{start_s}, {end_s}] seconds")

    def relative_time(data: dict) -> np.ndarray:
        return (np.asarray(data["timestamp"], dtype=np.float64) - time_origin) / 1e6

    motor_index = np.flatnonzero(selected)
    controller_dt_s = 1.0 / config.sim.physics_hz
    replay_t, replay_relative_t = _replay_time_grid(
        start_s=start_s,
        end_s=end_s,
        controller_dt_s=controller_dt_s,
    )
    if command_trace is None:
        replay_actions = None
        trajectory_index = _hold_indices(source_t=relative_time(trajectory), query_t=replay_t)
        velocity_ned = np.column_stack(
            [
                np.asarray(trajectory[f"velocity[{index}]"], dtype=np.float64)[trajectory_index]
                for index in range(3)
            ]
        )
        yaw_speed = np.asarray(trajectory["yawspeed"], dtype=np.float64)[trajectory_index]
    elif control_mode == "velocity":
        command_time = np.asarray(command_trace["time_s"], dtype=np.float64)
        command_velocity = np.asarray(
            command_trace["command_velocity_body_flu_mps"], dtype=np.float64
        )
        command_yaw_rate = np.asarray(command_trace["command_yaw_rate_radps"], dtype=np.float64)
        if (
            command_time.ndim != 1
            or command_velocity.shape != (len(command_time), 3)
            or command_yaw_rate.shape != (len(command_time),)
        ):
            raise ValueError("profile command trace fields have inconsistent lengths")
        command_index = _hold_indices(
            source_t=command_time,
            query_t=replay_relative_t,
        )
        commanded_body_velocity = command_velocity[command_index]
        replay_actions = np.column_stack(
            [
                commanded_body_velocity / config.rl.action.max_body_velocity_mps,
                command_yaw_rate[command_index, None] / config.rl.action.max_yaw_rate_radps,
            ]
        )
        replay_actions = _delay_velocity_actions(
            actions=replay_actions,
            delay_steps=round(controller_settings.command_delay_s * config.sim.physics_hz),
        )
        replay_actions = _delay_transmitted_actions(actions=replay_actions)
        velocity_ned = np.zeros_like(commanded_body_velocity)
        velocity_ned[:, 2] = -(replay_actions[:, 2] * config.rl.action.max_body_velocity_mps)
        yaw_speed = -command_yaw_rate[command_index]
    elif control_mode == "ctbr":
        command_time = np.asarray(command_trace["time_s"], dtype=np.float64)
        command_collective = np.asarray(
            command_trace["command_collective_thrust_normalized"], dtype=np.float64
        )
        command_body_rate = np.asarray(
            command_trace["command_body_rate_flu_radps"], dtype=np.float64
        )
        if (
            command_time.ndim != 1
            or command_collective.shape != (len(command_time),)
            or command_body_rate.shape != (len(command_time), 3)
        ):
            raise ValueError("CTBR profile command trace fields have inconsistent lengths")
        command_index = _hold_indices(source_t=command_time, query_t=replay_relative_t)
        replay_actions = np.column_stack(
            [
                command_collective[command_index] / controller_settings.hover_thrust_normalized
                - 1.0,
                command_body_rate[command_index] / config.rl.action.max_body_rate_radps,
            ]
        )
        replay_actions = _delay_transmitted_actions(actions=replay_actions)
        velocity_ned = np.zeros((len(replay_t), 3), dtype=np.float64)
        yaw_speed = np.zeros(len(replay_t), dtype=np.float64)
    else:
        raise ValueError(f"unsupported profile control mode {control_mode!r}")
    local_velocity_ned = _linear_interpolate(
        source_t=relative_time(local_position),
        values=np.column_stack(
            [np.asarray(local_position[name], dtype=np.float64) for name in ("vx", "vy", "vz")]
        ),
        query_t=replay_t,
    )
    position_derivative_z_ned = _linear_interpolate(
        source_t=relative_time(local_position),
        values=np.asarray(local_position["z_deriv"], dtype=np.float64),
        query_t=replay_t,
    )
    try:
        landing_speed_mps = float(ulog.initial_parameters["MPC_LAND_SPEED"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("ULog is missing the finite MPC_LAND_SPEED controller parameter") from exc
    local_velocity_ned = _px4_vertical_velocity_state(
        velocity_setpoint_ned=velocity_ned,
        velocity_ned=local_velocity_ned,
        position_derivative_z_ned=position_derivative_z_ned,
        landing_speed_mps=landing_speed_mps,
    )
    quaternion_frd_ned = _interpolate_quaternions(
        source_t=relative_time(attitude),
        quaternions_wxyz=np.column_stack(
            [np.asarray(attitude[f"q[{index}]"], dtype=np.float64) for index in range(4)]
        ),
        query_t=replay_t,
    )
    angular_velocity_frd = _linear_interpolate(
        source_t=relative_time(angular_velocity),
        values=np.column_stack(
            [np.asarray(angular_velocity[f"xyz[{index}]"], dtype=np.float64) for index in range(3)]
        ),
        query_t=replay_t,
    )
    angular_acceleration_frd = _linear_interpolate(
        source_t=relative_time(angular_velocity),
        values=np.column_stack(
            [
                np.asarray(angular_velocity[f"xyz_derivative[{index}]"], dtype=np.float64)
                for index in range(3)
            ]
        ),
        query_t=replay_t,
    )
    replay_angular_acceleration_flu = angular_acceleration_frd @ _FLU_TO_FRD
    if not physical_state:
        replay_velocity_enu = local_velocity_ned @ _NED_TO_ENU
        replay_quaternion_flu_enu = np.asarray(
            [
                _px4_attitude_to_isaac(quaternion_frd_ned_wxyz=value)[0]
                for value in quaternion_frd_ned
            ],
            dtype=np.float64,
        )
        replay_angular_velocity_flu = angular_velocity_frd @ _FLU_TO_FRD
        replay_state_source = "px4-estimated-state"
    else:
        if command_trace is None or bridge_state is None:
            raise ValueError("physical-state replay requires command and bridge state traces")
        physical_time_us, _position, physical_attitude, physical_velocity, physical_rates = (
            bridge_state
        )
        physical_time = physical_time_us.astype(np.float64) / 1e6
        replay_velocity_enu = _linear_interpolate(
            source_t=physical_time,
            values=physical_velocity,
            query_t=replay_t,
        )
        replay_quaternion_flu_enu = _interpolate_quaternions(
            source_t=physical_time,
            quaternions_wxyz=physical_attitude,
            query_t=replay_t,
        )
        replay_angular_velocity_flu = _linear_interpolate(
            source_t=physical_time,
            values=physical_rates,
            query_t=replay_t,
        )
        replay_state_source = "captured-physical-state"
    hover_index = _hold_indices(source_t=relative_time(hover), query_t=replay_t)
    replay_hover_thrust = np.asarray(hover["hover_thrust"], dtype=np.float64)[hover_index]

    px4_omega_rows: list[np.ndarray] = []
    isaac_omega_rows: list[np.ndarray] = []
    px4_motor_rows: list[np.ndarray] = []
    isaac_motor_rows: list[np.ndarray] = []
    px4_wrench_rows: list[np.ndarray] = []
    isaac_wrench_rows: list[np.ndarray] = []
    isaac_rate_setpoint_rows: list[np.ndarray] = []
    isaac_desired_collective_rows: list[float] = []
    isaac_collective_setpoint_rows: list[float] = []
    isaac_acceleration_setpoint_rows: list[np.ndarray] = []
    isaac_velocity_integral_rows: list[np.ndarray] = []
    isaac_filtered_velocity_rows: list[np.ndarray] = []
    isaac_acceleration_rows: list[np.ndarray] = []
    isaac_velocity_setpoint_rows: list[np.ndarray] = []
    isaac_angular_velocity_rows: list[np.ndarray] = []
    isaac_angular_acceleration_rows: list[np.ndarray] = []
    used_t: list[float] = []
    used_replay_indices: list[int] = []
    used_actuator_replay_indices: list[int] = []
    used_attitude_handoff_delays: list[int] = []
    used_allocator_handoff_delays: list[int] = []
    rotor_count = len(bridge.input_scaling)
    controller_vectors, controller_source_age_us = _controller_boundary(
        local_setpoint=local_setpoint,
        local_position=local_position,
        velocity_derivative_ned=_velocity_controller_derivative(
            ulog=ulog, local_position=local_position
        ),
        attitude_setpoint=attitude_setpoint,
        rates_setpoint=rates_setpoint,
        torque_setpoint=torque_setpoint,
        angular_velocity=angular_velocity,
        rate_status=rate_status,
        hover=hover,
        timestamp_us=round(time_origin + start_s * 1e6),
    )
    controller_state = seed_actuation_controller_state(
        params=params,
        controller_dt_s=controller_dt_s,
        physical_linear_velocity_enu=replay_velocity_enu[0],
        filtered_linear_velocity_enu=(
            controller_vectors["initial_estimated_linear_velocity_enu_mps"]
            if physical_state
            else replay_velocity_enu[0]
        ),
        velocity_setpoint_enu_mps=controller_vectors["initial_velocity_setpoint_enu_mps"],
        angular_velocity_body_flu=controller_vectors[
            "initial_estimated_angular_velocity_body_flu_radps"
        ],
        acceleration_setpoint_enu_mps2=controller_vectors["initial_acceleration_setpoint_enu_mps2"],
        linear_acceleration_enu_mps2=controller_vectors["initial_linear_acceleration_enu_mps2"],
        angular_acceleration_body_flu_radps2=controller_vectors[
            "initial_angular_acceleration_body_flu_radps2"
        ],
        rate_integral_body_flu=controller_vectors["initial_rate_integral_body_flu"],
        hover_thrust_normalized=float(controller_vectors["initial_hover_thrust_normalized"]),
        outer_loop_elapsed_s=(
            controller_source_age_us["vehicle_local_position"] / 1e6 - controller_dt_s
        ),
        attitude_loop_elapsed_s=(controller_source_age_us["vehicle_rates_setpoint"] / 1e6),
        desired_rotation_enu_flu=_quat_to_matrix(
            quaternion_wxyz=np.asarray(
                controller_vectors["initial_desired_attitude_flu_enu_wxyz"],
                dtype=np.float64,
            )
        ),
        desired_yaw_rate_radps=float(controller_vectors["initial_desired_yaw_rate_radps"]),
        desired_collective_thrust_setpoint=float(
            controller_vectors["initial_desired_collective_thrust_normalized"]
        ),
        collective_thrust_setpoint=float(
            controller_vectors["initial_collective_thrust_normalized"]
        ),
        body_rate_setpoint=controller_vectors["initial_body_rate_setpoint_flu_radps"],
        filtered_yaw_torque=float(controller_vectors["initial_filtered_yaw_torque_normalized"]),
    )
    isaac_omega_grid: list[np.ndarray] = []
    isaac_rate_setpoint_grid: list[np.ndarray] = []
    isaac_desired_collective_grid: list[float] = []
    isaac_collective_setpoint_grid: list[float] = []
    isaac_acceleration_setpoint_grid: list[np.ndarray] = []
    isaac_velocity_integral_grid: list[np.ndarray] = []
    isaac_filtered_velocity_grid: list[np.ndarray] = []
    isaac_acceleration_grid: list[np.ndarray] = []
    isaac_velocity_setpoint_grid: list[np.ndarray] = []
    isaac_angular_velocity_grid: list[np.ndarray] = []
    isaac_angular_acceleration_grid: list[np.ndarray] = []
    for (
        replay_action,
        velocity_setpoint_ned,
        yaw_setpoint_rate,
        velocity_state_enu,
        quaternion,
        rates_flu,
        hover_thrust_normalized,
    ) in zip(
        replay_actions if replay_actions is not None else np.full((len(replay_t), 4), np.nan),
        velocity_ned,
        yaw_speed,
        replay_velocity_enu,
        replay_quaternion_flu_enu,
        replay_angular_velocity_flu,
        replay_hover_thrust,
        strict=True,
    ):
        if not (
            np.all(np.isfinite(velocity_setpoint_ned))
            and math.isfinite(float(yaw_setpoint_rate))
            and np.all(np.isfinite(velocity_state_enu))
            and np.all(np.isfinite(quaternion))
            and np.all(np.isfinite(rates_flu))
        ):
            raise ValueError("selected replay window contains non-finite controller inputs")
        quaternion_flu_enu = quaternion
        velocity_setpoint_enu = _NED_TO_ENU @ velocity_setpoint_ned

        state = ActuationState(
            attitude_flu_enu_wxyz=quaternion_flu_enu,
            linear_velocity_enu=velocity_state_enu,
            angular_velocity_body_flu=rates_flu,
        )
        controller_state = update_hover_thrust_estimate(
            params=params,
            controller_state=controller_state,
            hover_thrust_normalized=float(hover_thrust_normalized),
        )
        if not physical_state:
            px4_angular_acceleration = replay_angular_acceleration_flu[len(isaac_omega_grid)]
            controller_state = replace(
                controller_state,
                previous_angular_velocity_body_flu=(
                    rates_flu - px4_angular_acceleration * controller_dt_s
                ),
                angular_acceleration_body_flu=px4_angular_acceleration,
            )
        if replay_actions is None:
            command = np.array([*velocity_setpoint_enu, -float(yaw_setpoint_rate)])
        else:
            command = normalized_action_to_controller_command(
                params,
                np.clip(replay_action, -1.0, 1.0),
                quaternion_flu_enu,
            )
        controller_state, isaac_omega = controller_command_to_rotor_omega(
            params,
            controller_state,
            command,
            state,
            controller_dt_s,
        )
        isaac_omega_grid.append(isaac_omega)
        isaac_rate_setpoint_grid.append(
            command[1:4] if control_mode == "ctbr" else controller_state.body_rate_setpoint
        )
        isaac_desired_collective_grid.append(
            float(command[0])
            if control_mode == "ctbr"
            else controller_state.desired_collective_thrust_setpoint
        )
        isaac_collective_setpoint_grid.append(
            float(command[0])
            if control_mode == "ctbr"
            else controller_state.collective_thrust_setpoint
        )
        isaac_acceleration_setpoint_grid.append(controller_state.acceleration_setpoint_enu)
        isaac_velocity_integral_grid.append(controller_state.velocity_integral_enu)
        isaac_filtered_velocity_grid.append(controller_state.filtered_velocity_enu)
        isaac_acceleration_grid.append(controller_state.acceleration_enu)
        isaac_velocity_setpoint_grid.append(command[:3])
        isaac_angular_velocity_grid.append(controller_state.previous_angular_velocity_body_flu)
        isaac_angular_acceleration_grid.append(controller_state.angular_acceleration_body_flu)

    replay_index = _nearest_indices(source_t=replay_t, query_t=sample_t)
    thrust_sample_index = _nearest_indices(
        source_t=relative_time(thrust_setpoint),
        query_t=sample_t,
    )
    thrust_sample_collective = np.linalg.norm(
        np.column_stack(
            [
                np.asarray(thrust_setpoint[f"xyz[{index}]"], dtype=np.float64)[thrust_sample_index]
                for index in range(3)
            ]
        ),
        axis=1,
    )
    # A missing topic publication must not make a fresh motor output appear stale.
    # Unsaturated allocation is invertible at the exact motor timestamp.
    thrust_sample_collective = _allocator_collective_at_motor_samples(
        motor_controls=np.column_stack(
            [
                np.asarray(motor[f"control[{index}]"], dtype=np.float64)[motor_index]
                for index in range(rotor_count)
            ]
        ),
        normalized_mix=params.normalized_allocation_mix,
        logged_collective=thrust_sample_collective,
        logged_sample_matches=(
            thrust_setpoint["timestamp"][thrust_sample_index] == motor["timestamp"][motor_index]
        ),
    )
    rates_sample_index = _hold_indices(
        source_t=relative_time(rates_setpoint),
        query_t=sample_t,
    )
    rates_collective = -np.asarray(
        rates_setpoint["thrust_body[2]"],
        dtype=np.float64,
    )
    current_rates_collective = rates_collective[rates_sample_index]
    previous_rates_collective = rates_collective[np.maximum(rates_sample_index - 1, 0)]
    attitude_sample_index = _hold_indices(
        source_t=relative_time(attitude_setpoint),
        query_t=sample_t,
    )
    attitude_collective = -np.asarray(
        attitude_setpoint["thrust_body[2]"],
        dtype=np.float64,
    )
    current_attitude_collective = attitude_collective[attitude_sample_index]
    previous_attitude_collective = attitude_collective[np.maximum(attitude_sample_index - 1, 0)]
    if control_mode == "velocity":
        actuator_replay_indices, attitude_handoff_delays, allocator_handoff_delays = (
            _controller_handoff_indices(
                replay_indices=replay_index,
                current_attitude_collective=current_attitude_collective,
                previous_attitude_collective=previous_attitude_collective,
                current_rates_collective=current_rates_collective,
                previous_rates_collective=previous_rates_collective,
                allocator_collective=thrust_sample_collective,
            )
        )
    else:
        actuator_replay_indices = replay_index
        attitude_handoff_delays = np.zeros_like(replay_index)
        allocator_handoff_delays = np.zeros_like(replay_index)
    for (
        time_s,
        mi,
        replay_sample_index,
        actuator_replay_index,
        attitude_handoff_delay,
        allocator_handoff_delay,
    ) in zip(
        sample_t,
        motor_index,
        replay_index,
        actuator_replay_indices,
        attitude_handoff_delays,
        allocator_handoff_delays,
        strict=True,
    ):
        controls = np.array(
            [motor[f"control[{index}]"][mi] for index in range(rotor_count)],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(controls)):
            continue
        px4_omega = (
            controls * np.asarray(bridge.input_scaling)
            + np.asarray(bridge.zero_position_armed)
            + np.asarray(bridge.input_offset) * np.asarray(bridge.input_scaling)
        )
        isaac_omega = isaac_omega_grid[int(actuator_replay_index)]
        used_t.append(float(time_s))
        used_replay_indices.append(int(replay_sample_index))
        used_actuator_replay_indices.append(int(actuator_replay_index))
        used_attitude_handoff_delays.append(int(attitude_handoff_delay))
        used_allocator_handoff_delays.append(int(allocator_handoff_delay))
        px4_motor_rows.append(controls)
        isaac_motor_rows.append(
            (isaac_omega - np.asarray(bridge.zero_position_armed))
            / np.asarray(bridge.input_scaling)
            - np.asarray(bridge.input_offset)
        )
        px4_omega_rows.append(px4_omega)
        isaac_omega_rows.append(isaac_omega)
        px4_wrench_rows.append(_wrench(geometry=geometry, omega=px4_omega))
        isaac_wrench_rows.append(_wrench(geometry=geometry, omega=isaac_omega))
        isaac_rate_setpoint_rows.append(isaac_rate_setpoint_grid[replay_sample_index])
        isaac_desired_collective_rows.append(isaac_desired_collective_grid[replay_sample_index])
        isaac_collective_setpoint_rows.append(isaac_collective_setpoint_grid[replay_sample_index])
        isaac_acceleration_setpoint_rows.append(
            isaac_acceleration_setpoint_grid[replay_sample_index]
        )
        isaac_velocity_integral_rows.append(isaac_velocity_integral_grid[replay_sample_index])
        isaac_filtered_velocity_rows.append(isaac_filtered_velocity_grid[replay_sample_index])
        isaac_acceleration_rows.append(isaac_acceleration_grid[replay_sample_index])
        isaac_velocity_setpoint_rows.append(isaac_velocity_setpoint_grid[replay_sample_index])
        isaac_angular_velocity_rows.append(isaac_angular_velocity_grid[replay_sample_index])
        isaac_angular_acceleration_rows.append(isaac_angular_acceleration_grid[replay_sample_index])

    if not used_t:
        raise ValueError("selected window contains no finite velocity-setpoint/motor samples")

    px4_omega_array = np.stack(px4_omega_rows)
    isaac_omega_array = np.stack(isaac_omega_rows)
    px4_motor_array = np.stack(px4_motor_rows)
    isaac_motor_array = np.stack(isaac_motor_rows)
    px4_wrench_array = np.stack(px4_wrench_rows)
    isaac_wrench_array = np.stack(isaac_wrench_rows)
    isaac_rate_setpoint_array = np.stack(isaac_rate_setpoint_rows)
    isaac_desired_collective_array = np.asarray(
        isaac_desired_collective_rows,
        dtype=np.float64,
    )
    isaac_collective_setpoint_array = np.asarray(
        isaac_collective_setpoint_rows,
        dtype=np.float64,
    )
    isaac_acceleration_setpoint_array = np.stack(isaac_acceleration_setpoint_rows)
    isaac_velocity_integral_array = np.stack(isaac_velocity_integral_rows)
    isaac_filtered_velocity_array = np.stack(isaac_filtered_velocity_rows)
    isaac_acceleration_array = np.stack(isaac_acceleration_rows)
    isaac_velocity_setpoint_array = np.stack(isaac_velocity_setpoint_rows)
    isaac_angular_velocity_array = np.stack(isaac_angular_velocity_rows)
    isaac_angular_acceleration_array = np.stack(isaac_angular_acceleration_rows)
    omega_error = isaac_omega_array - px4_omega_array
    wrench_error = isaac_wrench_array - px4_wrench_array
    collective_error = wrench_error[:, 0]
    collective_peak_index = int(np.argmax(np.abs(collective_error)))
    collective_peak_start = max(collective_peak_index - 2, 0)
    collective_peak_end = min(collective_peak_index + 3, len(collective_error))
    max_omega = np.asarray(bridge.max_rotor_velocity)
    torque_index = _nearest_indices(
        source_t=relative_time(torque_setpoint),
        query_t=np.asarray(used_t),
    )
    torque_setpoint_flu = np.column_stack(
        [
            np.asarray(torque_setpoint[f"xyz[{index}]"], dtype=np.float64)[torque_index]
            for index in range(3)
        ]
    )
    torque_setpoint_flu *= np.array([1.0, -1.0, -1.0])
    thrust_index = _nearest_indices(
        source_t=relative_time(thrust_setpoint),
        query_t=np.asarray(used_t),
    )
    thrust_setpoint_body = np.column_stack(
        [
            np.asarray(thrust_setpoint[f"xyz[{index}]"], dtype=np.float64)[thrust_index]
            for index in range(3)
        ]
    )
    rates_setpoint_index = _hold_indices(
        source_t=relative_time(rates_setpoint),
        query_t=np.asarray(used_t),
    )
    px4_rate_setpoint_flu = np.column_stack(
        [
            np.asarray(rates_setpoint[field], dtype=np.float64)[rates_setpoint_index]
            for field in ("roll", "pitch", "yaw")
        ]
    )
    px4_rate_setpoint_flu *= np.array([1.0, -1.0, -1.0])
    px4_collective_setpoint = -np.asarray(
        rates_setpoint["thrust_body[2]"],
        dtype=np.float64,
    )[rates_setpoint_index]
    angular_velocity_index = _nearest_indices(
        source_t=relative_time(angular_velocity),
        query_t=np.asarray(used_t),
    )
    px4_angular_velocity_flu = np.column_stack(
        [
            np.asarray(angular_velocity[f"xyz[{index}]"], dtype=np.float64)[angular_velocity_index]
            for index in range(3)
        ]
    )
    px4_angular_velocity_flu *= np.array([1.0, -1.0, -1.0])
    px4_angular_acceleration_flu = np.column_stack(
        [
            np.asarray(angular_velocity[f"xyz_derivative[{index}]"], dtype=np.float64)[
                angular_velocity_index
            ]
            for index in range(3)
        ]
    )
    px4_angular_acceleration_flu *= np.array([1.0, -1.0, -1.0])
    boundary_metrics = _normalized_boundary_metrics(
        target_motor_controls=isaac_motor_array,
        px4_motor_controls=px4_motor_array,
        normalized_mix=params.normalized_allocation_mix,
        px4_collective=np.linalg.norm(thrust_setpoint_body, axis=1),
        px4_torque_flu=torque_setpoint_flu,
    )
    setpoint_metrics = _controller_setpoint_metrics(
        target_body_rate_flu=isaac_rate_setpoint_array,
        px4_body_rate_flu=px4_rate_setpoint_flu,
        target_collective=isaac_collective_setpoint_array,
        px4_collective=px4_collective_setpoint,
    )
    handoff_delay_scan = (
        _controller_setpoint_delay_scan(
            target_body_rate_grid_flu=np.stack(isaac_rate_setpoint_grid),
            target_collective_grid=np.asarray(isaac_collective_setpoint_grid),
            replay_indices=np.asarray(used_replay_indices),
            px4_body_rate_flu=px4_rate_setpoint_flu,
            px4_collective=px4_collective_setpoint,
            maximum_delay_steps=3,
        )
        if control_mode == "velocity"
        else []
    )
    allocator_handoff_delay_scan = [
        {
            "delay_steps": row["delay_steps"],
            "collective_setpoint_rmse": row["collective_setpoint_rmse"],
        }
        for row in _controller_setpoint_delay_scan(
            target_body_rate_grid_flu=np.zeros((len(isaac_collective_setpoint_grid), 3)),
            target_collective_grid=np.asarray(isaac_collective_setpoint_grid),
            replay_indices=np.asarray(used_replay_indices),
            px4_body_rate_flu=np.zeros((len(used_replay_indices), 3)),
            px4_collective=np.linalg.norm(thrust_setpoint_body, axis=1),
            maximum_delay_steps=3,
        )
    ]
    rate_input_metrics = _rate_input_metrics(
        target_angular_velocity_flu=isaac_angular_velocity_array,
        px4_angular_velocity_flu=px4_angular_velocity_flu,
        target_angular_acceleration_flu=isaac_angular_acceleration_array,
        px4_angular_acceleration_flu=px4_angular_acceleration_flu,
    )
    outer_loop_metrics: dict[str, float | list[float]] = {}
    vertical_setpoint_delay_scan: list[dict] = []
    if control_mode == "velocity":
        used_time = np.asarray(used_t)
        attitude_setpoint_index = _hold_indices(
            source_t=relative_time(attitude_setpoint),
            query_t=used_time,
        )
        hover_setpoint_index = _hold_indices(
            source_t=relative_time(hover),
            query_t=used_time,
        )
        px4_acceleration_setpoint_z = _vertical_acceleration_setpoint_enu(
            quaternion_frd_ned_wxyz=np.column_stack(
                [
                    np.asarray(attitude_setpoint[f"q_d[{index}]"], dtype=np.float64)[
                        attitude_setpoint_index
                    ]
                    for index in range(4)
                ]
            ),
            thrust_body_z=np.asarray(
                attitude_setpoint["thrust_body[2]"],
                dtype=np.float64,
            )[attitude_setpoint_index],
            hover_thrust_normalized=np.asarray(
                hover["hover_thrust"],
                dtype=np.float64,
            )[hover_setpoint_index],
        )
        acceleration_setpoint_z_error = (
            isaac_acceleration_setpoint_array[:, 2] - px4_acceleration_setpoint_z
        )
        px4_velocity_integral_z = (
            px4_acceleration_setpoint_z
            - params.kp_vel[2]
            * (isaac_velocity_setpoint_array[:, 2] - isaac_filtered_velocity_array[:, 2])
            + params.kd_vel[2] * isaac_acceleration_array[:, 2]
        )
        velocity_integral_z_error = isaac_velocity_integral_array[:, 2] - px4_velocity_integral_z
        px4_attitude_collective = -np.asarray(
            attitude_setpoint["thrust_body[2]"],
            dtype=np.float64,
        )[attitude_setpoint_index]
        px4_allocator_collective = np.linalg.norm(thrust_setpoint_body, axis=1)
        vertical_setpoint_delay_scan = _vertical_setpoint_delay_scan(
            target_acceleration_setpoint_grid_enu=np.stack(isaac_acceleration_setpoint_grid),
            replay_indices=np.asarray(used_replay_indices),
            px4_acceleration_setpoint_z_enu=px4_acceleration_setpoint_z,
            maximum_delay_steps=3,
        )
        velocity_input_error = (
            isaac_filtered_velocity_array
            - local_velocity_ned[np.asarray(used_replay_indices)] @ _NED_TO_ENU
        )
        outer_loop_metrics = {
            "linear_velocity_input_rmse_mps": [
                _root_mean_square(values=velocity_input_error[:, axis]) for axis in range(3)
            ],
            "linear_velocity_input_mean_error_mps": np.mean(velocity_input_error, axis=0).tolist(),
            **_collective_boundary_metrics(
                target_desired_collective=isaac_desired_collective_array,
                px4_attitude_collective=px4_attitude_collective,
                px4_rate_collective=px4_collective_setpoint,
                px4_allocator_collective=px4_allocator_collective,
            ),
            "vertical_acceleration_setpoint_rmse_mps2": _root_mean_square(
                values=acceleration_setpoint_z_error
            ),
            "vertical_acceleration_setpoint_max_abs_mps2": float(
                np.max(np.abs(acceleration_setpoint_z_error))
            ),
            "vertical_velocity_integral_rmse_mps2": _root_mean_square(
                values=velocity_integral_z_error
            ),
            "vertical_velocity_integral_max_abs_mps2": float(
                np.max(np.abs(velocity_integral_z_error))
            ),
        }
    scale_fits = [
        _linear_scale_with_intercept(
            controller_output=torque_setpoint_flu[:, axis],
            physical_output=px4_wrench_array[:, axis + 1],
        )
        for axis in range(3)
    ]
    return {
        "schema_version": 1,
        "method": "px4-ulog-state-replay-v1-physics-rate",
        "config": str(config_path),
        "airframe": airframe_name,
        "window_s": [start_s, end_s],
        "sample_count": len(used_t),
        "controller_sample_count": len(replay_t),
        "controller_rate_hz": config.sim.physics_hz,
        "controller_state_source": replay_state_source,
        "controller_state_source_age_us": controller_source_age_us,
        "physical_state_timestamp_join_error_us": (
            int(
                np.max(
                    np.abs(
                        physical_time_us[
                            _nearest_indices(
                                source_t=physical_time_us.astype(np.float64),
                                query_t=replay_t * 1e6,
                            )
                        ]
                        - np.rint(replay_t * 1e6).astype(np.int64)
                    )
                )
            )
            if physical_state and bridge_state is not None
            else 0
        ),
        "controller_setpoint_max_source_age_us": int(
            np.max((np.asarray(used_t) - relative_time(rates_setpoint)[rates_setpoint_index]) * 1e6)
        ),
        "actuator_handoff_join_error_us": int(
            np.max(
                np.abs(np.asarray(used_actuator_replay_indices) - np.asarray(used_replay_indices))
            )
            * controller_dt_s
            * 1e6
        ),
        "attitude_to_rate_handoff_join_error_us": int(
            np.max(used_attitude_handoff_delays) * controller_dt_s * 1e6
        ),
        "rate_to_allocator_handoff_join_error_us": int(
            np.max(used_allocator_handoff_delays) * controller_dt_s * 1e6
        ),
        "actuator_handoff_join_offset_counts": {
            str(offset): int(
                np.count_nonzero(
                    np.asarray(used_actuator_replay_indices) - np.asarray(used_replay_indices)
                    == offset
                )
            )
            for offset in (-2, -1, 0)
        },
        "rate_input_nearest_sample_error_us": int(
            np.max(
                np.abs(relative_time(angular_velocity)[angular_velocity_index] - np.asarray(used_t))
                * 1e6
            )
        ),
        "sample_time_s": [min(used_t), max(used_t)],
        "collective_thrust_peak_window": [
            {
                "time_s": float(used_t[index]),
                "px4_collective_thrust_n": float(px4_wrench_array[index, 0]),
                "target_collective_thrust_n": float(isaac_wrench_array[index, 0]),
                "target_desired_collective": float(isaac_desired_collective_array[index]),
                "target_rate_collective": float(isaac_collective_setpoint_array[index]),
                "px4_rate_collective": float(px4_collective_setpoint[index]),
                "px4_allocator_collective": float(np.linalg.norm(thrust_setpoint_body[index])),
            }
            for index in range(collective_peak_start, collective_peak_end)
        ],
        "outer_loop_handoff_delay_scan": handoff_delay_scan,
        "allocator_handoff_delay_scan": allocator_handoff_delay_scan,
        "vertical_setpoint_delay_scan": vertical_setpoint_delay_scan,
        "metrics": {
            **boundary_metrics,
            **setpoint_metrics,
            **rate_input_metrics,
            **outer_loop_metrics,
            "rotor_omega_rmse_radps": _root_mean_square(values=omega_error),
            "rotor_omega_normalized_rmse": _root_mean_square(values=omega_error / max_omega),
            "rotor_omega_max_abs_radps": float(np.max(np.abs(omega_error))),
            "collective_thrust_rmse_n": _root_mean_square(values=wrench_error[:, 0]),
            "collective_thrust_max_abs_n": float(np.abs(collective_error[collective_peak_index])),
            "collective_thrust_max_abs_time_s": float(used_t[collective_peak_index]),
            "body_torque_rmse_nm": [
                _root_mean_square(values=wrench_error[:, index]) for index in range(1, 4)
            ],
            "torque_effectiveness_scale": [fit[0] for fit in scale_fits],
            "torque_effectiveness_intercept_nm": [fit[1] for fit in scale_fits],
            "torque_effectiveness_fit_rmse_nm": [fit[2] for fit in scale_fits],
            "px4_mean_collective_thrust_n": float(np.mean(px4_wrench_array[:, 0])),
            "isaac_mean_collective_thrust_n": float(np.mean(isaac_wrench_array[:, 0])),
        },
    }


def _profile_windows(*, reference: dict) -> list[dict]:
    if reference.get("schema_version") != 1:
        raise ValueError("reference must use alignment trace schema 1")
    raw_profiles = reference.get("traces")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("reference has no alignment traces")
    windows = []
    for raw_profile in raw_profiles:
        relative_time_s = raw_profile.get("time_s")
        if not isinstance(relative_time_s, list) or not relative_time_s:
            raise ValueError(f"alignment profile {raw_profile.get('profile')!r} has no samples")
        start_s = int(raw_profile["start_sim_time_us"]) / 1_000_000.0
        windows.append(
            {
                "family": raw_profile["family"],
                "control_mode": raw_profile["control_mode"],
                "family_metric": raw_profile["family_metric"],
                "profile": raw_profile["profile"],
                "split": raw_profile["split"],
                "axis": raw_profile["axis"],
                "start_s": start_s,
                "end_s": start_s + float(relative_time_s[-1]),
                "command_trace": {
                    "time_s": raw_profile["command_time_s"],
                    "command_velocity_body_flu_mps": raw_profile["command_velocity_body_flu_mps"],
                    "command_yaw_rate_radps": raw_profile["command_yaw_rate_radps"],
                    "command_collective_thrust_normalized": raw_profile[
                        "command_collective_thrust_normalized"
                    ],
                    "command_body_rate_flu_radps": raw_profile["command_body_rate_flu_radps"],
                    "control_mode": raw_profile["control_mode"],
                    "attitude_flu_enu_wxyz": raw_profile["attitude_flu_enu_wxyz"],
                    "velocity_enu_mps": raw_profile["velocity_enu_mps"],
                    "angular_velocity_body_flu_radps": raw_profile[
                        "angular_velocity_body_flu_radps"
                    ],
                },
            }
        )
    return windows


def _profile_replay_report(
    *,
    ulog,
    config_path: Path,
    controller_artifact_path: Path,
    reference: dict,
    bridge_log_path: Path,
) -> dict:
    bridge_state = _bridge_state_samples(path=bridge_log_path)
    profiles = []
    failures = []
    for window in _profile_windows(reference=reference):
        core_replay = _report(
            ulog=ulog,
            config_path=config_path,
            controller_artifact_path=controller_artifact_path,
            start_s=window["start_s"],
            end_s=window["end_s"],
            time_origin_us=0,
            command_trace=window["command_trace"],
            physical_state=False,
            bridge_state=None,
        )
        physical_replay = _report(
            ulog=ulog,
            config_path=config_path,
            controller_artifact_path=controller_artifact_path,
            start_s=window["start_s"],
            end_s=window["end_s"],
            time_origin_us=0,
            command_trace=window["command_trace"],
            physical_state=True,
            bridge_state=bridge_state,
        )
        metrics = core_replay["metrics"]
        physical_metrics = physical_replay["metrics"]
        profile_name = window["profile"]
        if metrics["normalized_motor_rmse"] > _MAX_NORMALIZED_MOTOR_RMSE:
            failures.append(
                f"{profile_name}: normalized motor RMSE "
                f"{metrics['normalized_motor_rmse']:.6f} exceeds "
                f"{_MAX_NORMALIZED_MOTOR_RMSE:.6f}"
            )
        if metrics["collective_thrust_rmse_n"] > _MAX_COLLECTIVE_RMSE_N:
            failures.append(
                f"{profile_name}: collective RMSE "
                f"{metrics['collective_thrust_rmse_n']:.6f} N exceeds "
                f"{_MAX_COLLECTIVE_RMSE_N:.6f} N"
            )
        if physical_metrics["normalized_motor_rmse"] > _MAX_NORMALIZED_MOTOR_RMSE:
            failures.append(
                f"{profile_name}: physical-state normalized motor RMSE "
                f"{physical_metrics['normalized_motor_rmse']:.6f} exceeds "
                f"{_MAX_NORMALIZED_MOTOR_RMSE:.6f}"
            )
        physical_join_error_us = physical_replay["physical_state_timestamp_join_error_us"]
        actuator_join_error_us = core_replay["actuator_handoff_join_error_us"]
        attitude_handoff_join_error_us = core_replay["attitude_to_rate_handoff_join_error_us"]
        allocator_handoff_join_error_us = core_replay["rate_to_allocator_handoff_join_error_us"]
        setpoint_source_age_us = core_replay["controller_setpoint_max_source_age_us"]
        rate_input_sample_error_us = core_replay["rate_input_nearest_sample_error_us"]
        if physical_join_error_us > _PHYSICS_TICK_US:
            failures.append(
                f"{profile_name}: physical-state timestamp join error "
                f"{physical_join_error_us} us exceeds {_PHYSICS_TICK_US} us"
            )
        if actuator_join_error_us > _PHYSICS_TICK_US:
            failures.append(
                f"{profile_name}: actuator handoff join error "
                f"{actuator_join_error_us} us exceeds {_PHYSICS_TICK_US} us"
            )
        profiles.append(
            {
                "family": window["family"],
                "control_mode": window["control_mode"],
                "family_metric": window["family_metric"],
                "profile": window["profile"],
                "split": window["split"],
                "axis": window["axis"],
                "start_s": window["start_s"],
                "end_s": window["end_s"],
                "sample_count": core_replay["sample_count"],
                "controller_sample_count": core_replay["controller_sample_count"],
                "controller_state_source": core_replay["controller_state_source"],
                "controller_state_source_age_us": core_replay["controller_state_source_age_us"],
                "controller_setpoint_max_source_age_us": setpoint_source_age_us,
                "actuator_handoff_join_error_us": actuator_join_error_us,
                "attitude_to_rate_handoff_join_error_us": attitude_handoff_join_error_us,
                "rate_to_allocator_handoff_join_error_us": allocator_handoff_join_error_us,
                "actuator_handoff_join_offset_counts": core_replay[
                    "actuator_handoff_join_offset_counts"
                ],
                "rate_input_nearest_sample_error_us": rate_input_sample_error_us,
                "sample_time_s": core_replay["sample_time_s"],
                "collective_thrust_peak_window": core_replay["collective_thrust_peak_window"],
                "outer_loop_handoff_delay_scan": core_replay["outer_loop_handoff_delay_scan"],
                "allocator_handoff_delay_scan": core_replay["allocator_handoff_delay_scan"],
                "vertical_setpoint_delay_scan": core_replay["vertical_setpoint_delay_scan"],
                "metrics": metrics,
                "physical_state_metrics": physical_metrics,
                "physical_state_timestamp_join_error_us": physical_join_error_us,
            }
        )
    return {
        "schema_version": 1,
        "method": "physical-and-estimated-state-px4-profile-replay-v1-physics-rate",
        "timestamp_join_limit_us": _PHYSICS_TICK_US,
        "config": str(config_path),
        "controller_artifact": str(controller_artifact_path),
        "bridge_log": str(bridge_log_path),
        "limits": {
            "normalized_motor_rmse": _MAX_NORMALIZED_MOTOR_RMSE,
            "collective_thrust_rmse_n": _MAX_COLLECTIVE_RMSE_N,
        },
        "profiles": profiles,
        "passed": not failures,
        "failures": failures,
    }


def _dataset_report(
    *, reference_path: Path, config_path: Path, controller_artifact_path: Path | None, ulog_type
) -> dict:
    payload = load_alignment_reference(reference_path)
    descriptor = json.loads(reference_path.read_text())
    dataset, membership = load_flight_selection(reference_path.parent / descriptor["dataset"])
    reports = []
    for flight in dataset.flights:
        if flight.flight_index not in membership[descriptor["set_name"]]:
            continue
        reference = {
            **payload,
            "traces": [
                trace for trace in payload["traces"] if trace["flight_index"] == flight.flight_index
            ],
        }
        with open_flight_evidence(flight.directory / "flight.ulg.gz") as stream:
            ulog = ulog_type(stream)
        arguments = dict(
            ulog=ulog,
            config_path=config_path,
            reference=reference,
            bridge_log_path=flight.directory / "bridge.log.gz",
        )
        report = (
            _actuator_boundary_report(**arguments)
            if controller_artifact_path is None
            else _profile_replay_report(
                **arguments, controller_artifact_path=controller_artifact_path
            )
        )
        reports.append({"flight_index": flight.flight_index, **report})
    identity = (
        {}
        if controller_artifact_path is None
        else {
            "settings_fingerprint": load_controller_artifact(
                controller_artifact_path
            ).settings_fingerprint,
        }
    )
    return {
        **identity,
        "source_digest": dataset.artifact_digest,
        "schema_version": 1,
        "completed": True,
        "set_name": descriptor["set_name"],
        "passed": all(report["passed"] for report in reports),
        "failures": [
            f"flight {report['flight_index']}: {failure}"
            for report in reports
            for failure in report["failures"]
        ],
        "flights": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ulog_positional", nargs="?", type=Path)
    parser.add_argument("--ulog", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--controller-artifact", type=Path)
    parser.add_argument("--window", help="relative ULog seconds START,END")
    parser.add_argument("--augment-reference", type=Path)
    parser.add_argument("--audit-reference", type=Path)
    parser.add_argument("--profile-reference", type=Path)
    parser.add_argument("--bridge-log", type=Path)
    parser.add_argument("--dataset-reference", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    from pyulog import ULog

    if args.dataset_reference is not None:
        report = _dataset_report(
            reference_path=args.dataset_reference,
            config_path=args.config,
            controller_artifact_path=args.controller_artifact,
            ulog_type=ULog,
        )
        if args.json is None:
            parser.error("--dataset-reference requires --json")
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return
    ulog_path = args.ulog if args.ulog is not None else args.ulog_positional
    if ulog_path is None:
        parser.error("one ULog path is required")
    with open_flight_evidence(ulog_path) as stream:
        replay_ulog = ULog(stream)
    if args.augment_reference is not None:
        if (
            args.audit_reference is not None
            or args.profile_reference is not None
            or args.bridge_log is not None
        ):
            parser.error("--augment-reference cannot be combined with audit or profile options")
        report = _augment_reference_controller_state(
            ulog=replay_ulog,
            reference_path=args.augment_reference,
        )
    elif args.profile_reference is not None:
        if args.controller_artifact is None or args.bridge_log is None or args.window is not None:
            parser.error(
                "--profile-reference requires --controller-artifact and --bridge-log, "
                "and cannot use --window"
            )
        report = _profile_replay_report(
            ulog=replay_ulog,
            config_path=args.config,
            controller_artifact_path=args.controller_artifact,
            reference=json.loads(args.profile_reference.read_text()),
            bridge_log_path=args.bridge_log,
        )
    elif args.audit_reference is not None or args.bridge_log is not None:
        if args.audit_reference is None or args.bridge_log is None:
            parser.error("--audit-reference and --bridge-log must be provided together")
        report = _actuator_boundary_report(
            ulog=replay_ulog,
            config_path=args.config,
            reference=json.loads(args.audit_reference.read_text()),
            bridge_log_path=args.bridge_log,
        )
    else:
        if args.controller_artifact is None or args.window is None:
            parser.error("--controller-artifact and --window are required for controller replay")
        start_s, end_s = (float(value) for value in args.window.split(",", maxsplit=1))
        report = _report(
            ulog=replay_ulog,
            config_path=args.config,
            controller_artifact_path=args.controller_artifact,
            start_s=start_s,
            end_s=end_s,
            time_origin_us=replay_ulog.start_timestamp,
            command_trace=None,
            physical_state=False,
            bridge_state=None,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.audit_reference is not None and report.get("passed") is False:
        raise SystemExit("actuator-boundary audit failed")


if __name__ == "__main__":
    main()

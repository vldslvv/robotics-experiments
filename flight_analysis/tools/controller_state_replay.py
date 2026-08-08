"""Compare PX4 and Isaac velocity-controller outputs at matched replay states.

The PX4 ULog is the controller oracle. At each logged motor sample in the
selected window, this tool reconstructs the normalized body-FLU velocity action
from PX4's world-NED trajectory setpoint, converts the logged estimator state to
the Isaac controller's ENU/FLU convention, and evaluates the host numpy
actuation reference. Both outputs are reduced to the same physical boundary:
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

from robotics_experiments.config import load_config
from robotics_experiments.controller import (
    ActuationState,
    MixerGeometry,
    action_to_rotor_omega,
    build_actuation_params,
    load_controller_artifact,
    seed_actuation_controller_state,
    update_hover_thrust_estimate,
)
from robotics_experiments.drone._airframe import (
    compute_body_mass_properties,
    load_builder_config,
)

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
    for line in path.read_text().splitlines():
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
    for line in path.read_text().splitlines():
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


def _controller_boundary(
    *,
    local_setpoint: dict,
    local_position: dict,
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

    setpoint_age_us = held_age_us(topic=local_setpoint)
    rate_age_us = held_age_us(topic=rate_status)
    acceleration_setpoint_ned = interpolated_vector(
        topic=local_setpoint,
        fields=("acceleration[0]", "acceleration[1]", "acceleration[2]"),
    )
    linear_acceleration_ned = interpolated_vector(
        topic=local_position,
        fields=("ax", "ay", "az"),
    )
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
    rate_integral_frd = interpolated_vector(
        topic=rate_status,
        fields=("rollspeed_integ", "pitchspeed_integ", "yawspeed_integ"),
    )
    estimated_velocity_ned = interpolated_vector(
        topic=local_position,
        fields=("vx", "vy", "vz"),
    )
    hover_thrust_normalized = float(
        _linear_interpolate(
            source_t=_topic_time_us(topic=hover).astype(np.float64),
            values=np.asarray(hover["hover_thrust"], dtype=np.float64),
            query_t=np.array([timestamp_us], dtype=np.float64),
        )[0]
    )
    vectors = {
        "initial_estimated_linear_velocity_enu_mps": _NED_TO_ENU @ estimated_velocity_ned,
        "initial_estimated_angular_velocity_body_flu_radps": (
            _FLU_TO_FRD @ estimated_angular_velocity_frd
        ),
        "initial_acceleration_setpoint_enu_mps2": _NED_TO_ENU @ acceleration_setpoint_ned,
        "initial_linear_acceleration_enu_mps2": _NED_TO_ENU @ linear_acceleration_ned,
        "initial_angular_acceleration_body_flu_radps2": _FLU_TO_FRD @ angular_acceleration_frd,
        "initial_rate_integral_body_flu": _FLU_TO_FRD @ rate_integral_frd,
        "initial_hover_thrust_normalized": hover_thrust_normalized,
    }
    if not all(np.all(np.isfinite(values)) for values in vectors.values()):
        raise ValueError("initial controller boundary contains non-finite values")
    if hover_thrust_normalized <= 0.0:
        raise ValueError("initial hover thrust estimate must be positive")
    return (
        vectors,
        {
            "vehicle_local_position_setpoint": setpoint_age_us,
            "vehicle_local_position": held_age_us(topic=local_position),
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
    raw_profiles = reference.get("traces")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(f"{reference_path} has no alignment traces")

    local_setpoint = _dataset(ulog=ulog, name="vehicle_local_position_setpoint")
    local_position = _dataset(ulog=ulog, name="vehicle_local_position")
    angular_velocity = _dataset(ulog=ulog, name="vehicle_angular_velocity")
    rate_status = _dataset(ulog=ulog, name="rate_ctrl_status")
    hover = _dataset(ulog=ulog, name="hover_thrust_estimate")

    profile_reports = []
    for raw_profile in raw_profiles:
        start_us = int(raw_profile["start_sim_time_us"])
        controller_vectors, source_age_us = _controller_boundary(
            local_setpoint=local_setpoint,
            local_position=local_position,
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
                "profile": raw_profile["profile"],
                "start_sim_time_us": start_us,
                "source_age_us": raw_profile["initial_controller_state_source_age_us"],
            }
        )
    temporary_path = reference_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    temporary_path.replace(reference_path)
    return {
        "schema_version": 1,
        "method": "px4-ulog-controller-boundary-v1",
        "reference": str(reference_path),
        "profiles": profile_reports,
    }


def _actuator_boundary_report(
    *,
    ulog,
    config_path: Path,
    reference_path: Path,
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
    reference = json.loads(reference_path.read_text())
    if reference.get("schema_version") != 1:
        raise ValueError(f"{reference_path} must use alignment trace schema 1")
    raw_profiles = reference.get("traces")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(f"{reference_path} has no alignment traces")

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
        "reference": str(reference_path),
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
            velocity_estimator_bias_decay_hz=0.0,
            velocity_filter_hz=0.0,
            angular_velocity_filter_hz=0.0,
        )
    params = build_actuation_params(
        geometry=geometry,
        action_cfg=config.rl.action,
        controller_settings=controller_settings,
        mass=mass,
    )

    motor = _dataset(ulog=ulog, name="actuator_motors")
    trajectory = _dataset(ulog=ulog, name="trajectory_setpoint")
    local_position = _dataset(ulog=ulog, name="vehicle_local_position")
    attitude = _dataset(ulog=ulog, name="vehicle_attitude")
    angular_velocity = _dataset(ulog=ulog, name="vehicle_angular_velocity")
    local_setpoint = _dataset(ulog=ulog, name="vehicle_local_position_setpoint")
    rate_status = _dataset(ulog=ulog, name="rate_ctrl_status")
    hover = _dataset(ulog=ulog, name="hover_thrust_estimate")
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
    replay_t = np.arange(
        start_s,
        end_s + controller_dt_s * 0.5,
        controller_dt_s,
        dtype=np.float64,
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
    else:
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
            query_t=replay_t - start_s,
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
        velocity_ned = np.zeros_like(commanded_body_velocity)
        velocity_ned[:, 2] = -commanded_body_velocity[:, 2]
        yaw_speed = -command_yaw_rate[command_index]
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
    used_t: list[float] = []
    rotor_count = len(bridge.input_scaling)
    controller_vectors, controller_source_age_us = _controller_boundary(
        local_setpoint=local_setpoint,
        local_position=local_position,
        angular_velocity=angular_velocity,
        rate_status=rate_status,
        hover=hover,
        timestamp_us=round(time_origin + start_s * 1e6),
    )
    controller_state = seed_actuation_controller_state(
        params=params,
        physical_linear_velocity_enu=replay_velocity_enu[0],
        filtered_linear_velocity_enu=(
            controller_vectors["initial_estimated_linear_velocity_enu_mps"]
            if physical_state
            else replay_velocity_enu[0]
        ),
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
    )
    isaac_omega_grid = []

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
        _w, qx, qy, qz = quaternion_flu_enu
        yaw = math.atan2(
            2.0 * (_w * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        velocity_setpoint_enu = _NED_TO_ENU @ velocity_setpoint_ned
        heading = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        if replay_actions is None:
            action = np.zeros(4)
            action[:3] = heading.T @ velocity_setpoint_enu / config.rl.action.max_body_velocity_mps
            action[3] = -float(yaw_setpoint_rate) / config.rl.action.max_yaw_rate_radps
        else:
            action = replay_action
        action = np.clip(action, -1.0, 1.0)

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
        controller_state, isaac_omega = action_to_rotor_omega(
            params,
            controller_state,
            action,
            state,
            controller_dt_s,
        )
        isaac_omega_grid.append(isaac_omega)

    replay_index = _nearest_indices(source_t=replay_t, query_t=sample_t)
    for time_s, mi, replay_sample_index in zip(
        sample_t,
        motor_index,
        replay_index,
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
        isaac_omega = isaac_omega_grid[replay_sample_index]
        used_t.append(float(time_s))
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

    if not used_t:
        raise ValueError("selected window contains no finite velocity-setpoint/motor samples")

    px4_omega_array = np.stack(px4_omega_rows)
    isaac_omega_array = np.stack(isaac_omega_rows)
    px4_motor_array = np.stack(px4_motor_rows)
    isaac_motor_array = np.stack(isaac_motor_rows)
    px4_wrench_array = np.stack(px4_wrench_rows)
    isaac_wrench_array = np.stack(isaac_wrench_rows)
    omega_error = isaac_omega_array - px4_omega_array
    wrench_error = isaac_wrench_array - px4_wrench_array
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
    boundary_metrics = _normalized_boundary_metrics(
        target_motor_controls=isaac_motor_array,
        px4_motor_controls=px4_motor_array,
        normalized_mix=params.normalized_allocation_mix,
        px4_collective=np.linalg.norm(thrust_setpoint_body, axis=1),
        px4_torque_flu=torque_setpoint_flu,
    )
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
        "sample_time_s": [min(used_t), max(used_t)],
        "metrics": {
            **boundary_metrics,
            "rotor_omega_rmse_radps": _root_mean_square(values=omega_error),
            "rotor_omega_normalized_rmse": _root_mean_square(values=omega_error / max_omega),
            "rotor_omega_max_abs_radps": float(np.max(np.abs(omega_error))),
            "collective_thrust_rmse_n": _root_mean_square(values=wrench_error[:, 0]),
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


def _profile_windows(*, reference_path: Path) -> list[dict]:
    reference = json.loads(reference_path.read_text())
    if reference.get("schema_version") != 1:
        raise ValueError(f"{reference_path} must use alignment trace schema 1")
    raw_profiles = reference.get("traces")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(f"{reference_path} has no alignment traces")
    windows = []
    for raw_profile in raw_profiles:
        relative_time_s = raw_profile.get("time_s")
        if not isinstance(relative_time_s, list) or not relative_time_s:
            raise ValueError(f"alignment profile {raw_profile.get('profile')!r} has no samples")
        start_s = int(raw_profile["start_sim_time_us"]) / 1_000_000.0
        windows.append(
            {
                "profile": raw_profile["profile"],
                "split": raw_profile["split"],
                "axis": raw_profile["axis"],
                "start_s": start_s,
                "end_s": start_s + float(relative_time_s[-1]),
                "command_trace": {
                    "time_s": relative_time_s,
                    "command_velocity_body_flu_mps": raw_profile["command_velocity_body_flu_mps"],
                    "command_yaw_rate_radps": raw_profile["command_yaw_rate_radps"],
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
    reference_path: Path,
    bridge_log_path: Path,
) -> dict:
    bridge_state = _bridge_state_samples(path=bridge_log_path)
    profiles = []
    failures = []
    for window in _profile_windows(reference_path=reference_path):
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
        if physical_join_error_us > _PHYSICS_TICK_US:
            failures.append(
                f"{profile_name}: physical-state timestamp join error "
                f"{physical_join_error_us} us exceeds {_PHYSICS_TICK_US} us"
            )
        profiles.append(
            {
                "profile": window["profile"],
                "split": window["split"],
                "axis": window["axis"],
                "start_s": window["start_s"],
                "end_s": window["end_s"],
                "sample_count": core_replay["sample_count"],
                "controller_sample_count": core_replay["controller_sample_count"],
                "controller_state_source": core_replay["controller_state_source"],
                "controller_state_source_age_us": core_replay["controller_state_source_age_us"],
                "sample_time_s": core_replay["sample_time_s"],
                "metrics": metrics,
                "physical_state_metrics": physical_metrics,
                "physical_state_timestamp_join_error_us": physical_join_error_us,
            }
        )
    return {
        "schema_version": 1,
        "method": "physical-and-estimated-state-px4-profile-replay-v1-physics-rate",
        "config": str(config_path),
        "controller_artifact": str(controller_artifact_path),
        "reference": str(reference_path),
        "bridge_log": str(bridge_log_path),
        "limits": {
            "normalized_motor_rmse": _MAX_NORMALIZED_MOTOR_RMSE,
            "collective_thrust_rmse_n": _MAX_COLLECTIVE_RMSE_N,
        },
        "profiles": profiles,
        "passed": not failures,
        "failures": failures,
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
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    from pyulog import ULog

    ulog_path = args.ulog if args.ulog is not None else args.ulog_positional
    if ulog_path is None:
        parser.error("one ULog path is required")
    if args.augment_reference is not None:
        if (
            args.audit_reference is not None
            or args.profile_reference is not None
            or args.bridge_log is not None
        ):
            parser.error("--augment-reference cannot be combined with audit or profile options")
        report = _augment_reference_controller_state(
            ulog=ULog(str(ulog_path)),
            reference_path=args.augment_reference,
        )
    elif args.profile_reference is not None:
        if args.controller_artifact is None or args.bridge_log is None or args.window is not None:
            parser.error(
                "--profile-reference requires --controller-artifact and --bridge-log, "
                "and cannot use --window"
            )
        report = _profile_replay_report(
            ulog=ULog(str(ulog_path)),
            config_path=args.config,
            controller_artifact_path=args.controller_artifact,
            reference_path=args.profile_reference,
            bridge_log_path=args.bridge_log,
        )
    elif args.audit_reference is not None or args.bridge_log is not None:
        if args.audit_reference is None or args.bridge_log is None:
            parser.error("--audit-reference and --bridge-log must be provided together")
        report = _actuator_boundary_report(
            ulog=ULog(str(ulog_path)),
            config_path=args.config,
            reference_path=args.audit_reference,
            bridge_log_path=args.bridge_log,
        )
    else:
        if args.controller_artifact is None or args.window is None:
            parser.error("--controller-artifact and --window are required for controller replay")
        start_s, end_s = (float(value) for value in args.window.split(",", maxsplit=1))
        replay_ulog = ULog(str(ulog_path))
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

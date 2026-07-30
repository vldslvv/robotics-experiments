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
from pathlib import Path

import numpy as np

from robotics_experiments.alignment import load_alignment_artifact
from robotics_experiments.config import load_config
from robotics_experiments.drone._airframe import (
    compute_body_mass_properties,
    load_builder_config,
)
from robotics_experiments.rl import (
    ActuationState,
    MixerGeometry,
    action_to_rotor_omega,
    build_actuation_params,
    seed_actuation_controller_state,
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
_BRIDGE_DYNAMICS_PATTERN = re.compile(
    r"PX4 bridge dynamics: .* sim_time_us=(?P<sim_time_us>\d+) "
    r"last_actuator_us=(?P<actuator_time_us>\d+) "
    r"actuator_lag_us=(?P<join_error_us>\d+).* "
    r"commands=(?P<rotor_speed>\[[^\]]+\]) "
    r"forces_n=(?P<forces>\[[^\]]+\]) "
    r"body_torque_n_m=(?P<body_torque>\[[^\]]+\])"
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
    timestamp_us: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    def held_index(*, topic: dict) -> tuple[int, int]:
        time_us = _topic_time_us(topic=topic)
        index = int(_hold_indices(source_t=time_us, query_t=np.array([timestamp_us]))[0])
        return index, timestamp_us - int(time_us[index])

    def interpolated_vector(*, topic: dict, fields: tuple[str, str, str]) -> np.ndarray:
        return _linear_interpolate(
            source_t=_topic_time_us(topic=topic).astype(np.float64),
            values=np.column_stack(
                [np.asarray(topic[field], dtype=np.float64) for field in fields]
            ),
            query_t=np.array([timestamp_us], dtype=np.float64),
        )[0]

    setpoint_index, setpoint_age_us = held_index(topic=local_setpoint)
    rate_index, rate_age_us = held_index(topic=rate_status)
    acceleration_setpoint_ned = np.array(
        [local_setpoint[f"acceleration[{axis}]"][setpoint_index] for axis in range(3)],
        dtype=np.float64,
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
    rate_integral_frd = np.array(
        [
            rate_status["rollspeed_integ"][rate_index],
            rate_status["pitchspeed_integ"][rate_index],
            rate_status["yawspeed_integ"][rate_index],
        ],
        dtype=np.float64,
    )
    vectors = {
        "initial_acceleration_setpoint_enu_mps2": _NED_TO_ENU @ acceleration_setpoint_ned,
        "initial_linear_acceleration_enu_mps2": _NED_TO_ENU @ linear_acceleration_ned,
        "initial_angular_acceleration_body_flu_radps2": _FLU_TO_FRD @ angular_acceleration_frd,
        "initial_rate_integral_body_flu": _FLU_TO_FRD @ rate_integral_frd,
    }
    if not all(np.all(np.isfinite(values)) for values in vectors.values()):
        raise ValueError("initial controller boundary contains non-finite values")
    return (
        vectors,
        {
            "vehicle_local_position_setpoint": setpoint_age_us,
            "rate_ctrl_status": rate_age_us,
        },
    )


def _augment_reference_controller_state(
    *,
    ulog,
    reference_path: Path,
) -> dict:
    reference = json.loads(reference_path.read_text())
    if reference.get("schema_version") != 3:
        raise ValueError(f"{reference_path} must use alignment trace schema 3")
    raw_profiles = reference.get("traces")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError(f"{reference_path} has no alignment traces")

    local_setpoint = _dataset(ulog=ulog, name="vehicle_local_position_setpoint")
    local_position = _dataset(ulog=ulog, name="vehicle_local_position")
    angular_velocity = _dataset(ulog=ulog, name="vehicle_angular_velocity")
    rate_status = _dataset(ulog=ulog, name="rate_ctrl_status")

    profile_reports = []
    for raw_profile in raw_profiles:
        start_us = int(raw_profile["start_sim_time_us"])
        controller_vectors, source_age_us = _controller_boundary(
            local_setpoint=local_setpoint,
            local_position=local_position,
            angular_velocity=angular_velocity,
            rate_status=rate_status,
            timestamp_us=start_us,
        )
        raw_profile.update(
            {
                "initial_controller_state_valid": True,
                **{name: values.tolist() for name, values in controller_vectors.items()},
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
    reference["schema_version"] = 4
    reference_path.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
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
    if reference.get("schema_version") not in (3, 4):
        raise ValueError(f"{reference_path} must use alignment trace schema 3 or 4")
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
    motor_index = _nearest_indices(source_t=motor_time_us, query_t=bridge_time_us)
    motor_join_error_us = np.abs(motor_time_us[motor_index] - bridge_time_us)
    input_scaling = np.asarray(bridge.input_scaling, dtype=np.float64)
    input_offset = np.asarray(bridge.input_offset, dtype=np.float64)
    armed_zero = np.asarray(bridge.zero_position_armed, dtype=np.float64)
    bridge_normalized_motor = (bridge_rotor_speed - armed_zero) / input_scaling - input_offset
    expected_rotor_speed = (
        motor_values[motor_index] * input_scaling + armed_zero + input_offset * input_scaling
    )
    expected_collective = np.sum(
        geometry.rotor_constant * np.square(expected_rotor_speed),
        axis=1,
    )
    bridge_collective = np.sum(bridge_forces, axis=1)
    bridge_torque = np.column_stack(
        [
            np.sum(geometry.positions_body[:, 1] * bridge_forces, axis=1),
            -np.sum(geometry.positions_body[:, 0] * bridge_forces, axis=1),
            bridge_body_torque[:, 2],
        ]
    )

    profile_reports = []
    audit_failures: list[str] = []
    for raw_profile in raw_profiles:
        profile_name = str(raw_profile["profile"])
        start_us = int(raw_profile["start_sim_time_us"])
        relative_time_s = raw_profile["time_s"]
        end_us = start_us + round(float(relative_time_s[-1]) * 1_000_000)
        bridge_selected = (bridge_time_us >= start_us) & (bridge_time_us <= end_us)
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
        profile_motor_error = (
            bridge_normalized_motor[bridge_selected] - motor_values[motor_index][bridge_selected]
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
                np.max(bridge_dynamics_join_error_us[bridge_selected], initial=0),
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
                **topic_reports,
                "bridge": {
                    "actuator_timestamp_us": bridge_time_us[bridge_selected].tolist(),
                    "mapped_rotor_speed_radps": bridge_rotor_speed[bridge_selected].tolist(),
                    "reconstructed_collective_thrust_n": bridge_collective[
                        bridge_selected
                    ].tolist(),
                    "reconstructed_body_torque_nm": bridge_torque[bridge_selected].tolist(),
                    "timestamp_join_error_us": {
                        "actuator_motors": motor_join_error_us[bridge_selected].tolist(),
                        "dynamics": bridge_dynamics_join_error_us[bridge_selected].tolist(),
                    },
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
    alignment_artifact_path: Path,
    start_s: float,
    end_s: float,
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
    controller_settings = load_alignment_artifact(alignment_artifact_path).selected_settings
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
    torque_setpoint = _dataset(ulog=ulog, name="vehicle_torque_setpoint")
    start_timestamp = float(ulog.start_timestamp)

    motor_t = (np.asarray(motor["timestamp"], dtype=np.float64) - start_timestamp) / 1e6
    selected = (motor_t >= start_s) & (motor_t <= end_s)
    sample_t = motor_t[selected]
    if not sample_t.size:
        raise ValueError(f"no actuator_motors samples in [{start_s}, {end_s}] seconds")

    def relative_time(data: dict) -> np.ndarray:
        return (np.asarray(data["timestamp"], dtype=np.float64) - start_timestamp) / 1e6

    motor_index = np.flatnonzero(selected)
    controller_dt_s = 1.0 / config.sim.physics_hz
    replay_t = np.arange(
        start_s,
        end_s + controller_dt_s * 0.5,
        controller_dt_s,
        dtype=np.float64,
    )
    trajectory_index = _hold_indices(source_t=relative_time(trajectory), query_t=replay_t)
    velocity_ned = np.column_stack(
        [
            np.asarray(trajectory[f"velocity[{index}]"], dtype=np.float64)[trajectory_index]
            for index in range(3)
        ]
    )
    yaw_speed = np.asarray(trajectory["yawspeed"], dtype=np.float64)[trajectory_index]
    local_velocity_ned = _linear_interpolate(
        source_t=relative_time(local_position),
        values=np.column_stack(
            [np.asarray(local_position[name], dtype=np.float64) for name in ("vx", "vy", "vz")]
        ),
        query_t=replay_t,
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

    px4_omega_rows: list[np.ndarray] = []
    isaac_omega_rows: list[np.ndarray] = []
    px4_wrench_rows: list[np.ndarray] = []
    isaac_wrench_rows: list[np.ndarray] = []
    used_t: list[float] = []
    rotor_count = len(bridge.input_scaling)
    controller_vectors, controller_source_age_us = _controller_boundary(
        local_setpoint=local_setpoint,
        local_position=local_position,
        angular_velocity=angular_velocity,
        rate_status=rate_status,
        timestamp_us=round(start_timestamp + start_s * 1e6),
    )
    controller_state = seed_actuation_controller_state(
        params=params,
        linear_velocity_enu=_NED_TO_ENU @ local_velocity_ned[0],
        angular_velocity_body_flu=_FLU_TO_FRD @ angular_velocity_frd[0],
        acceleration_setpoint_enu_mps2=controller_vectors["initial_acceleration_setpoint_enu_mps2"],
        linear_acceleration_enu_mps2=controller_vectors["initial_linear_acceleration_enu_mps2"],
        angular_acceleration_body_flu_radps2=controller_vectors[
            "initial_angular_acceleration_body_flu_radps2"
        ],
        rate_integral_body_flu=controller_vectors["initial_rate_integral_body_flu"],
    )
    isaac_omega_grid = []

    for velocity_setpoint_ned, yaw_setpoint_rate, velocity_state_ned, quaternion, rates_frd in zip(
        velocity_ned,
        yaw_speed,
        local_velocity_ned,
        quaternion_frd_ned,
        angular_velocity_frd,
        strict=True,
    ):
        if not (
            np.all(np.isfinite(velocity_setpoint_ned))
            and math.isfinite(float(yaw_setpoint_rate))
            and np.all(np.isfinite(velocity_state_ned))
            and np.all(np.isfinite(quaternion))
            and np.all(np.isfinite(rates_frd))
        ):
            raise ValueError("selected replay window contains non-finite controller inputs")
        quaternion_flu_enu, yaw = _px4_attitude_to_isaac(quaternion_frd_ned_wxyz=quaternion)
        velocity_setpoint_enu = _NED_TO_ENU @ velocity_setpoint_ned
        heading = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        action = np.zeros(4)
        action[:3] = heading.T @ velocity_setpoint_enu / config.rl.action.max_body_velocity_mps
        action[3] = -float(yaw_setpoint_rate) / config.rl.action.max_yaw_rate_radps
        action = np.clip(action, -1.0, 1.0)

        state = ActuationState(
            attitude_flu_enu_wxyz=quaternion_flu_enu,
            linear_velocity_enu=_NED_TO_ENU @ velocity_state_ned,
            angular_velocity_body_flu=_FLU_TO_FRD @ rates_frd,
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
        px4_omega_rows.append(px4_omega)
        isaac_omega_rows.append(isaac_omega)
        px4_wrench_rows.append(_wrench(geometry=geometry, omega=px4_omega))
        isaac_wrench_rows.append(_wrench(geometry=geometry, omega=isaac_omega))

    if not used_t:
        raise ValueError("selected window contains no finite velocity-setpoint/motor samples")

    px4_omega_array = np.stack(px4_omega_rows)
    isaac_omega_array = np.stack(isaac_omega_rows)
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
    scale_fits = [
        _linear_scale_with_intercept(
            controller_output=torque_setpoint_flu[:, axis],
            physical_output=px4_wrench_array[:, axis + 1],
        )
        for axis in range(3)
    ]
    return {
        "schema_version": 3,
        "method": "px4-ulog-state-replay-v3-250hz",
        "config": str(config_path),
        "airframe": airframe_name,
        "window_s": [start_s, end_s],
        "sample_count": len(used_t),
        "controller_sample_count": len(replay_t),
        "controller_rate_hz": config.sim.physics_hz,
        "controller_state_source_age_us": controller_source_age_us,
        "sample_time_s": [min(used_t), max(used_t)],
        "metrics": {
            "rotor_omega_rmse_radps": _root_mean_square(values=omega_error),
            "rotor_omega_normalized_rmse": _root_mean_square(values=omega_error / max_omega),
            "rotor_omega_max_abs_radps": float(np.max(np.abs(omega_error))),
            "collective_thrust_rmse_n": _root_mean_square(values=wrench_error[:, 0]),
            "body_torque_rmse_nm": [
                _root_mean_square(values=wrench_error[:, index]) for index in range(1, 4)
            ],
            "rate_output_scale_nm_estimate": [fit[0] for fit in scale_fits],
            "rate_output_scale_intercept_nm": [fit[1] for fit in scale_fits],
            "rate_output_scale_fit_rmse_nm": [fit[2] for fit in scale_fits],
            "px4_mean_collective_thrust_n": float(np.mean(px4_wrench_array[:, 0])),
            "isaac_mean_collective_thrust_n": float(np.mean(isaac_wrench_array[:, 0])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ulog_positional", nargs="?", type=Path)
    parser.add_argument("--ulog", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignment-artifact", type=Path)
    parser.add_argument("--window", help="relative ULog seconds START,END")
    parser.add_argument("--augment-reference", type=Path)
    parser.add_argument("--audit-reference", type=Path)
    parser.add_argument("--bridge-log", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    from pyulog import ULog

    ulog_path = args.ulog if args.ulog is not None else args.ulog_positional
    if ulog_path is None:
        parser.error("one ULog path is required")
    if args.augment_reference is not None:
        if args.audit_reference is not None or args.bridge_log is not None:
            parser.error("--augment-reference cannot be combined with audit options")
        report = _augment_reference_controller_state(
            ulog=ULog(str(ulog_path)),
            reference_path=args.augment_reference,
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
        if args.alignment_artifact is None or args.window is None:
            parser.error("--alignment-artifact and --window are required for controller replay")
        start_s, end_s = (float(value) for value in args.window.split(",", maxsplit=1))
        report = _report(
            ulog=ULog(str(ulog_path)),
            config_path=args.config,
            alignment_artifact_path=args.alignment_artifact,
            start_s=start_s,
            end_s=end_s,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report.get("passed") is False:
        raise SystemExit("actuator-boundary audit failed")


if __name__ == "__main__":
    main()

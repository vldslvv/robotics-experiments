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
import json
import math
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
    initial_actuation_controller_state,
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
    start_timestamp = float(ulog.start_timestamp)

    motor_t = (np.asarray(motor["timestamp"], dtype=np.float64) - start_timestamp) / 1e6
    selected = (motor_t >= start_s) & (motor_t <= end_s)
    sample_t = motor_t[selected]
    if not sample_t.size:
        raise ValueError(f"no actuator_motors samples in [{start_s}, {end_s}] seconds")

    def relative_time(data: dict) -> np.ndarray:
        return (np.asarray(data["timestamp"], dtype=np.float64) - start_timestamp) / 1e6

    trajectory_index = _nearest_indices(source_t=relative_time(trajectory), query_t=sample_t)
    position_index = _nearest_indices(source_t=relative_time(local_position), query_t=sample_t)
    attitude_index = _nearest_indices(source_t=relative_time(attitude), query_t=sample_t)
    angular_index = _nearest_indices(source_t=relative_time(angular_velocity), query_t=sample_t)
    motor_index = np.flatnonzero(selected)

    px4_omega_rows: list[np.ndarray] = []
    isaac_omega_rows: list[np.ndarray] = []
    px4_wrench_rows: list[np.ndarray] = []
    isaac_wrench_rows: list[np.ndarray] = []
    used_t: list[float] = []
    rotor_count = len(bridge.input_scaling)
    controller_state = initial_actuation_controller_state()
    previous_time_s = math.nan

    for time_s, mi, ti, pi, ai, wi in zip(
        sample_t,
        motor_index,
        trajectory_index,
        position_index,
        attitude_index,
        angular_index,
        strict=True,
    ):
        velocity_ned = np.array(
            [
                trajectory["velocity[0]"][ti],
                trajectory["velocity[1]"][ti],
                trajectory["velocity[2]"][ti],
            ],
            dtype=np.float64,
        )
        controls = np.array(
            [motor[f"control[{index}]"][mi] for index in range(rotor_count)],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(velocity_ned)) or not np.all(np.isfinite(controls)):
            continue
        quaternion_frd_ned = np.array(
            [attitude[f"q[{index}]"][ai] for index in range(4)],
            dtype=np.float64,
        )
        quaternion_flu_enu, yaw = _px4_attitude_to_isaac(quaternion_frd_ned_wxyz=quaternion_frd_ned)
        velocity_setpoint_enu = _NED_TO_ENU @ velocity_ned
        heading = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        action = np.zeros(4)
        action[:3] = heading.T @ velocity_setpoint_enu / config.rl.action.max_body_velocity_mps
        yaw_speed = float(trajectory["yawspeed"][ti])
        action[3] = -yaw_speed / config.rl.action.max_yaw_rate_radps
        action = np.clip(action, -1.0, 1.0)

        state = ActuationState(
            attitude_flu_enu_wxyz=quaternion_flu_enu,
            linear_velocity_enu=_NED_TO_ENU
            @ np.array(
                [
                    local_position["vx"][pi],
                    local_position["vy"][pi],
                    local_position["vz"][pi],
                ]
            ),
            angular_velocity_body_flu=_FLU_TO_FRD
            @ np.array(
                [
                    angular_velocity["xyz[0]"][wi],
                    angular_velocity["xyz[1]"][wi],
                    angular_velocity["xyz[2]"][wi],
                ]
            ),
        )
        px4_omega = (
            controls * np.asarray(bridge.input_scaling)
            + np.asarray(bridge.zero_position_armed)
            + np.asarray(bridge.input_offset) * np.asarray(bridge.input_scaling)
        )
        dt_s = (
            params.controller_dt_s
            if math.isnan(previous_time_s)
            else float(time_s - previous_time_s)
        )
        controller_state, isaac_omega = action_to_rotor_omega(
            params,
            controller_state,
            action,
            state,
            dt_s,
        )
        previous_time_s = float(time_s)
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
    return {
        "schema_version": 1,
        "method": "px4-ulog-state-replay-v1",
        "config": str(config_path),
        "airframe": airframe_name,
        "window_s": [start_s, end_s],
        "sample_count": len(used_t),
        "sample_time_s": [min(used_t), max(used_t)],
        "metrics": {
            "rotor_omega_rmse_radps": _root_mean_square(values=omega_error),
            "rotor_omega_normalized_rmse": _root_mean_square(values=omega_error / max_omega),
            "rotor_omega_max_abs_radps": float(np.max(np.abs(omega_error))),
            "collective_thrust_rmse_n": _root_mean_square(values=wrench_error[:, 0]),
            "body_torque_rmse_nm": [
                _root_mean_square(values=wrench_error[:, index]) for index in range(1, 4)
            ],
            "px4_mean_collective_thrust_n": float(np.mean(px4_wrench_array[:, 0])),
            "isaac_mean_collective_thrust_n": float(np.mean(isaac_wrench_array[:, 0])),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ulog", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--alignment-artifact", type=Path, required=True)
    parser.add_argument("--window", required=True, help="relative ULog seconds START,END")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    start_s, end_s = (float(value) for value in args.window.split(",", maxsplit=1))

    from pyulog import ULog

    report = _report(
        ulog=ULog(str(args.ulog)),
        config_path=args.config,
        alignment_artifact_path=args.alignment_artifact,
        start_s=start_s,
        end_s=end_s,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

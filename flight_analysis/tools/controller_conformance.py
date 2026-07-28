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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
_ROUTINE_MARKER = "ROUTINE_STAGE "
_POLICY_MARKER = "PRESS_POLICY "
_TEMPORAL_LABEL = re.compile(r"^policy\[temporal-(?P<range>0\.\d{3})\]$")


@dataclass(frozen=True)
class _TemporalTrace:
    label: str
    nominal_entry_range_m: float
    samples: tuple[dict[str, Any], ...]


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


def _json_after_marker(*, line: str, marker: str) -> dict[str, Any] | None:
    marker_index = line.find(marker)
    if marker_index < 0:
        return None
    return json.loads(line[marker_index + len(marker) :])


def _load_temporal_traces(*, log_path: Path) -> tuple[_TemporalTrace, ...]:
    """Read policy-only temporal traces from one production runtime log."""
    traces: list[_TemporalTrace] = []
    label = ""
    nominal_range_m = math.nan
    samples: list[dict[str, Any]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        routine = _json_after_marker(line=line, marker=_ROUTINE_MARKER)
        if routine is not None and routine.get("kind") == "policy":
            event = routine.get("event")
            candidate = str(routine.get("label"))
            match = _TEMPORAL_LABEL.fullmatch(candidate)
            if match is None:
                continue
            if event == "start":
                if label:
                    raise ValueError(f"nested temporal policy stage {candidate!r} in {log_path}")
                label = candidate
                nominal_range_m = float(match.group("range"))
                samples = []
            elif event == "failed" and label == candidate:
                raise ValueError(f"temporal policy stage {candidate!r} failed in {log_path}")
            elif event == "complete" and label == candidate:
                if not samples:
                    raise ValueError(f"temporal policy stage {candidate!r} has no actor samples")
                first_us = int(samples[0]["sim_time_us"])
                aligned = tuple(
                    {
                        **sample,
                        "time_s": (int(sample["sim_time_us"]) - first_us) * 1e-6,
                    }
                    for sample in samples
                )
                traces.append(
                    _TemporalTrace(
                        label=label,
                        nominal_entry_range_m=nominal_range_m,
                        samples=aligned,
                    )
                )
                label = ""
                nominal_range_m = math.nan
                samples = []
            continue

        if not label:
            continue
        policy = _json_after_marker(line=line, marker=_POLICY_MARKER)
        if policy is None or policy.get("event") != "actor_evaluation":
            continue
        observation = np.asarray(policy["observation"], dtype=np.float64)
        action = np.asarray(policy["normalized_action"], dtype=np.float64)
        if observation.shape != (34,) or action.shape != (4,):
            raise ValueError(
                "temporal sample must contain a 34-value observation and four-value action"
            )
        setpoint = policy["setpoint"]
        if not isinstance(setpoint, dict):
            raise ValueError("temporal sample setpoint must be an object")
        samples.append(
            {
                "sim_time_us": int(policy["sim_time_us"]),
                "range_m": float(observation[0]),
                "closing_speed_mps": float(observation[10]),
                "linear_speed_mps": float(np.linalg.norm(observation[10:13])),
                "forward_command_mps": float(setpoint["forward_mps"]),
                "observation": [float(value) for value in observation],
                "action": [float(value) for value in action],
                "stable_goal_count": int(policy["stable_goal_count"]),
                "complete": bool(policy["complete"]),
            }
        )
    if label:
        raise ValueError(f"unterminated temporal policy stage {label!r} in {log_path}")
    expected = [0.460, 0.475, 0.490]
    actual = [trace.nominal_entry_range_m for trace in traces]
    if actual != expected:
        raise ValueError(f"temporal trace entries are {actual!r}, expected {expected!r}")
    return tuple(traces)


def _trace_metrics(*, trace: _TemporalTrace) -> dict[str, Any]:
    samples = trace.samples
    times = np.asarray([sample["time_s"] for sample in samples], dtype=np.float64)
    ranges = np.asarray([sample["range_m"] for sample in samples], dtype=np.float64)
    velocities = np.asarray(
        [sample["closing_speed_mps"] for sample in samples],
        dtype=np.float64,
    )
    commands = np.asarray(
        [sample["forward_command_mps"] for sample in samples],
        dtype=np.float64,
    )
    reversal_candidates = np.flatnonzero(commands <= 0.0)
    reversal_index = int(reversal_candidates[0]) if reversal_candidates.size else len(samples) - 1
    complete_candidates = np.flatnonzero(
        np.asarray([sample["complete"] for sample in samples], dtype=bool)
    )
    settling_time_s = float(times[complete_candidates[0]]) if complete_candidates.size else None
    cadence = np.diff(times)
    closest_index = int(np.argmin(ranges))
    return {
        "sample_count": len(samples),
        "entry_range_m": float(ranges[0]),
        "entry_closing_speed_mps": float(velocities[0]),
        "cadence_mean_s": float(np.mean(cadence)) if cadence.size else 0.0,
        "cadence_min_s": float(np.min(cadence)) if cadence.size else 0.0,
        "cadence_max_s": float(np.max(cadence)) if cadence.size else 0.0,
        "peak_closing_speed_mps": float(np.max(velocities)),
        "command_reversal_time_s": float(times[reversal_index]),
        "braking_distance_m": float(ranges[reversal_index] - ranges[closest_index]),
        "closest_range_m": float(ranges[closest_index]),
        "closest_range_time_s": float(times[closest_index]),
        "settling_time_s": settling_time_s,
    }


def _delayed_command(
    *,
    times: np.ndarray,
    commands: np.ndarray,
    query_time_s: float,
) -> float:
    index = int(np.searchsorted(times, query_time_s, side="right") - 1)
    return 0.0 if index < 0 else float(commands[index])


def _simulate_first_order_response(
    *,
    trace: _TemporalTrace,
    command_delay_s: float,
    response_time_constant_s: float,
) -> np.ndarray:
    times = np.asarray([sample["time_s"] for sample in trace.samples], dtype=np.float64)
    commands = np.asarray(
        [sample["forward_command_mps"] for sample in trace.samples],
        dtype=np.float64,
    )
    observed = np.asarray(
        [sample["closing_speed_mps"] for sample in trace.samples],
        dtype=np.float64,
    )
    predicted = np.empty_like(observed)
    predicted[0] = observed[0]
    for index in range(1, len(predicted)):
        dt = float(times[index] - times[index - 1])
        command = _delayed_command(
            times=times,
            commands=commands,
            query_time_s=float(times[index - 1] - command_delay_s),
        )
        alpha = 1.0 - math.exp(-dt / response_time_constant_s)
        predicted[index] = predicted[index - 1] + alpha * (command - predicted[index - 1])
    return predicted


def _response_rmse(
    *,
    traces: tuple[_TemporalTrace, ...],
    command_delay_s: float,
    response_time_constant_s: float,
) -> float:
    errors = []
    for trace in traces:
        observed = np.asarray(
            [sample["closing_speed_mps"] for sample in trace.samples],
            dtype=np.float64,
        )
        predicted = _simulate_first_order_response(
            trace=trace,
            command_delay_s=command_delay_s,
            response_time_constant_s=response_time_constant_s,
        )
        errors.append(predicted[1:] - observed[1:])
    return _root_mean_square(values=np.concatenate(errors))


def _fit_temporal_response(
    *,
    traces: tuple[_TemporalTrace, ...],
) -> dict[str, Any]:
    """Fit outer entry cases and hold the middle case out for validation."""
    by_range = {trace.nominal_entry_range_m: trace for trace in traces}
    training = (by_range[0.460], by_range[0.490])
    validation = (by_range[0.475],)
    best = (math.inf, 0.0, 0.02)
    for delay_s in np.arange(0.0, 0.404, 0.004):
        for tau_s in np.arange(0.02, 1.004, 0.004):
            rmse = _response_rmse(
                traces=training,
                command_delay_s=float(delay_s),
                response_time_constant_s=float(tau_s),
            )
            best = min(best, (rmse, float(delay_s), float(tau_s)))
    training_rmse, delay_s, tau_s = best
    return {
        "method": "outer-entry-first-order-delay-grid-v1",
        "fit_entry_ranges_m": [0.460, 0.490],
        "validation_entry_ranges_m": [0.475],
        "command_delay_s": delay_s,
        "response_time_constant_s": tau_s,
        "training_velocity_rmse_mps": training_rmse,
        "validation_velocity_rmse_mps": _response_rmse(
            traces=validation,
            command_delay_s=delay_s,
            response_time_constant_s=tau_s,
        ),
    }


def _trace_from_mapping(*, value: dict[str, Any]) -> _TemporalTrace:
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise ValueError("temporal trace JSON requires a samples list")
    return _TemporalTrace(
        label=str(value["label"]),
        nominal_entry_range_m=float(value["nominal_entry_range_m"]),
        samples=tuple(samples),
    )


def _aligned_trace_diagnostics(
    *,
    px4: _TemporalTrace,
    lab: _TemporalTrace,
) -> dict[str, Any]:
    if px4.nominal_entry_range_m != lab.nominal_entry_range_m:
        raise ValueError("PX4 and Lab temporal traces have different entry ranges")
    count = min(len(px4.samples), len(lab.samples))
    if count < 2:
        raise ValueError("PX4/Lab temporal comparison requires at least two samples")
    px4_velocity = np.asarray(
        [sample["closing_speed_mps"] for sample in px4.samples[:count]],
        dtype=np.float64,
    )
    lab_velocity = np.asarray(
        [sample["closing_speed_mps"] for sample in lab.samples[:count]],
        dtype=np.float64,
    )
    px4_metrics = _trace_metrics(trace=px4)
    lab_metrics = _trace_metrics(trace=lab)
    return {
        "aligned_sample_count": count,
        "velocity_response_rmse_mps": _root_mean_square(values=lab_velocity - px4_velocity),
        "braking_distance_delta_m": (
            float(lab_metrics["braking_distance_m"]) - float(px4_metrics["braking_distance_m"])
        ),
        "closest_range_delta_m": (
            float(lab_metrics["closest_range_m"]) - float(px4_metrics["closest_range_m"])
        ),
        "settling_time_delta_s": (
            None
            if px4_metrics["settling_time_s"] is None or lab_metrics["settling_time_s"] is None
            else float(lab_metrics["settling_time_s"]) - float(px4_metrics["settling_time_s"])
        ),
    }


def _aligned_velocity_rmse(
    *,
    px4_traces: tuple[_TemporalTrace, ...],
    lab_traces: tuple[_TemporalTrace, ...],
) -> float:
    errors = []
    for px4, lab in zip(px4_traces, lab_traces, strict=True):
        if px4.nominal_entry_range_m != lab.nominal_entry_range_m:
            raise ValueError("PX4 and Lab temporal traces have different entry ranges")
        count = min(len(px4.samples), len(lab.samples))
        if count < 2:
            raise ValueError("PX4/Lab temporal comparison requires at least two samples")
        px4_velocity = np.asarray(
            [sample["closing_speed_mps"] for sample in px4.samples[:count]],
            dtype=np.float64,
        )
        lab_velocity = np.asarray(
            [sample["closing_speed_mps"] for sample in lab.samples[:count]],
            dtype=np.float64,
        )
        errors.append(lab_velocity - px4_velocity)
    return _root_mean_square(values=np.concatenate(errors))


def _fit_lab_command_delay(
    *,
    px4_traces: tuple[_TemporalTrace, ...],
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], tuple[_TemporalTrace, ...]]:
    """Select a Lab delay on outer traces and hold the middle trace out."""
    if not candidates:
        raise ValueError("Lab temporal replay has no delay candidates")
    by_range = {trace.nominal_entry_range_m: trace for trace in px4_traces}
    px4_training = (by_range[0.460], by_range[0.490])
    px4_validation = (by_range[0.475],)
    scored = []
    replay_by_delay = {}
    for candidate in candidates:
        delay_s = float(candidate["controller_command_delay_s"])
        values = candidate.get("traces")
        if not isinstance(values, list):
            raise ValueError("Lab temporal replay candidate requires a traces list")
        traces = tuple(_trace_from_mapping(value=value) for value in values)
        lab_by_range = {trace.nominal_entry_range_m: trace for trace in traces}
        if set(lab_by_range) != set(by_range):
            raise ValueError("PX4 and Lab temporal reports have different entry ranges")
        training_rmse = _aligned_velocity_rmse(
            px4_traces=px4_training,
            lab_traces=(lab_by_range[0.460], lab_by_range[0.490]),
        )
        validation_rmse = _aligned_velocity_rmse(
            px4_traces=px4_validation,
            lab_traces=(lab_by_range[0.475],),
        )
        scored.append(
            {
                "controller_command_delay_s": delay_s,
                "training_velocity_rmse_mps": training_rmse,
                "validation_velocity_rmse_mps": validation_rmse,
            }
        )
        replay_by_delay[delay_s] = traces
    selected = min(
        scored,
        key=lambda value: (
            float(value["training_velocity_rmse_mps"]),
            float(value["controller_command_delay_s"]),
        ),
    )
    selected_delay_s = float(selected["controller_command_delay_s"])
    return (
        {
            "method": "outer-entry-isolated-lab-delay-grid-v1",
            "fit_entry_ranges_m": [0.460, 0.490],
            "validation_entry_ranges_m": [0.475],
            "selected_controller_command_delay_s": selected_delay_s,
            "training_velocity_rmse_mps": selected["training_velocity_rmse_mps"],
            "validation_velocity_rmse_mps": selected["validation_velocity_rmse_mps"],
            "candidates": scored,
        },
        replay_by_delay[selected_delay_s],
    )


def _temporal_report(
    *,
    px4_log_path: Path,
    lab_trace_path: Path | None,
) -> dict[str, object]:
    px4_traces = _load_temporal_traces(log_path=px4_log_path)
    report: dict[str, object] = {
        "schema_version": 2,
        "method": "press-policy-temporal-conformance-v1",
        "px4_log": str(px4_log_path),
        "fit": _fit_temporal_response(traces=px4_traces),
        "px4_traces": [
            {
                "label": trace.label,
                "nominal_entry_range_m": trace.nominal_entry_range_m,
                "metrics": _trace_metrics(trace=trace),
                "samples": list(trace.samples),
            }
            for trace in px4_traces
        ],
    }
    if lab_trace_path is not None:
        lab_payload = json.loads(lab_trace_path.read_text())
        values = lab_payload.get("candidates")
        if not isinstance(values, list):
            raise ValueError("Lab temporal replay requires a candidates list")
        lab_fit, lab_traces = _fit_lab_command_delay(
            px4_traces=px4_traces,
            candidates=values,
        )
        if len(lab_traces) != len(px4_traces):
            raise ValueError("PX4 and Lab temporal reports have different trace counts")
        report["lab_trace"] = str(lab_trace_path)
        report["lab_parameter_fit"] = lab_fit
        report["lab_traces"] = [
            {
                "label": trace.label,
                "nominal_entry_range_m": trace.nominal_entry_range_m,
                "metrics": _trace_metrics(trace=trace),
                "samples": list(trace.samples),
            }
            for trace in lab_traces
        ]
        report["alignment"] = [
            {
                "nominal_entry_range_m": px4.nominal_entry_range_m,
                **_aligned_trace_diagnostics(px4=px4, lab=lab),
            }
            for px4, lab in zip(px4_traces, lab_traces, strict=True)
        ]
    return report


def _report(
    *,
    ulog,
    config_path: Path,
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
    params = build_actuation_params(geometry, config.rl.action, mass)

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
        isaac_omega = action_to_rotor_omega(params, action, state)
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
    parser = argparse.ArgumentParser(
        description="Report instantaneous or temporal PX4/Isaac controller conformance"
    )
    parser.add_argument("ulog", type=Path, nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--window", help="relative ULog seconds START,END")
    parser.add_argument(
        "--temporal-log",
        type=Path,
        help="runtime log containing the three wall-free production-actor traces",
    )
    parser.add_argument(
        "--lab-trace",
        type=Path,
        help="optional isolated-Lab replay JSON to align with --temporal-log",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.temporal_log is not None:
        if args.ulog is not None or args.config is not None or args.window is not None:
            parser.error("--temporal-log cannot be combined with ULog/config/window arguments")
        report = _temporal_report(
            px4_log_path=args.temporal_log,
            lab_trace_path=args.lab_trace,
        )
    else:
        if args.lab_trace is not None:
            parser.error("--lab-trace requires --temporal-log")
        if args.ulog is None or args.config is None or args.window is None:
            parser.error("ULog mode requires ulog, --config, and --window")
        start_s, end_s = (float(value) for value in args.window.split(",", maxsplit=1))

        from pyulog import ULog

        report = _report(
            ulog=ULog(str(args.ulog)),
            config_path=args.config,
            start_s=start_s,
            end_s=end_s,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

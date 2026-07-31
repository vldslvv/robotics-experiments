"""Named alignment metric registry and policy-execution response diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from robotics_experiments.alignment._models import MetricFunction, MetricResult, ProbeTrace

__all__: list[str] = []

_V1_METRIC = "worst-axis-validation-velocity-nrmse-v1"
_V2_METRIC = "policy-execution-controller-quality-v2"
_V3_METRIC = "policy-execution-controller-quality-v3"
_STOP_SPEED_MPS = 0.01
_METRICS: dict[str, MetricFunction] = {}
_QUALITY_LIMITS = (
    ("braking_distance_error_m", 0.005),
    ("residual_speed_error_mps", 0.01),
    ("stop_time_error_s", 0.10),
    ("worst_axis_velocity_nrmse", 0.10),
    ("yaw_rate_nrmse", 0.10),
    ("heading_error_deg", 3.0),
    ("combined_endpoint_error_m", 0.01),
)


def _trace_map(*, traces: tuple[ProbeTrace, ...]) -> Mapping[str, ProbeTrace]:
    mapped = {trace.profile: trace for trace in traces}
    if len(mapped) != len(traces):
        raise ValueError("backend produced duplicate probe profile names")
    return mapped


def _matched_trace(
    *,
    reference: ProbeTrace,
    target: ProbeTrace,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        reference.split != target.split
        or reference.axis != target.axis
        or reference.time_s != target.time_s
        or reference.command_velocity_body_flu_mps != target.command_velocity_body_flu_mps
        or reference.command_yaw_rate_radps != target.command_yaw_rate_radps
    ):
        raise ValueError(
            f"target trace {target.profile!r} does not reproduce the reference cadence "
            "and command sequence"
        )
    sample_fields = (
        "position_enu_m",
        "attitude_flu_enu_wxyz",
        "velocity_body_flu_mps",
        "velocity_enu_mps",
        "angular_velocity_body_flu_radps",
    )
    for trace in (reference, target):
        if not trace.completed or trace.failure_reason is not None:
            raise ValueError(f"probe {trace.profile!r} did not complete cleanly")
        for field_name in sample_fields:
            values = np.asarray(getattr(trace, field_name), dtype=np.float64)
            expected_width = 4 if field_name == "attitude_flu_enu_wxyz" else 3
            if values.shape != (len(trace.time_s), expected_width):
                raise ValueError(f"probe trace {trace.profile!r} has invalid {field_name} shape")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"probe trace {trace.profile!r} contains non-finite {field_name}")
    return (
        np.asarray(reference.velocity_body_flu_mps, dtype=np.float64),
        np.asarray(target.velocity_body_flu_mps, dtype=np.float64),
    )


def _yaw_wxyz(attitude: np.ndarray, /) -> np.ndarray:
    w, x, y, z = (
        attitude[:, 0],
        attitude[:, 1],
        attitude[:, 2],
        attitude[:, 3],
    )
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _acceleration(*, trace: ProbeTrace) -> np.ndarray:
    time = np.asarray(trace.time_s, dtype=np.float64)
    velocity = np.asarray(trace.velocity_body_flu_mps, dtype=np.float64)
    if len(time) < 2:
        raise ValueError(f"probe trace {trace.profile!r} needs at least two samples")
    return np.gradient(velocity, time, axis=0)


def _stop_index(*, velocity: np.ndarray, start: int) -> int:
    below = np.flatnonzero(np.abs(velocity[start:]) <= _STOP_SPEED_MPS)
    return len(velocity) - 1 if not len(below) else start + int(below[0])


def _braking_events(
    *,
    reference: ProbeTrace,
    target: ProbeTrace,
) -> list[tuple[float, float, float, float, float]]:
    if reference.axis not in ("x", "y", "z"):
        return []
    axis = {"x": 0, "y": 1, "z": 2}[reference.axis]
    command = np.asarray(reference.command_velocity_body_flu_mps, dtype=np.float64)[:, axis]
    reference_velocity = np.asarray(reference.velocity_body_flu_mps, dtype=np.float64)[:, axis]
    target_velocity = np.asarray(target.velocity_body_flu_mps, dtype=np.float64)[:, axis]
    reference_position = np.asarray(reference.position_enu_m, dtype=np.float64)
    target_position = np.asarray(target.position_enu_m, dtype=np.float64)
    time = np.asarray(reference.time_s, dtype=np.float64)
    transitions = np.flatnonzero(
        (np.abs(command[:-1]) > _STOP_SPEED_MPS)
        & ((np.abs(command[1:]) <= _STOP_SPEED_MPS) | (command[:-1] * command[1:] < 0.0))
    )
    events = []
    for previous in transitions:
        start = int(previous + 1)
        reference_stop = _stop_index(velocity=reference_velocity, start=start)
        target_stop = _stop_index(velocity=target_velocity, start=start)
        reference_distance = float(
            np.linalg.norm(reference_position[reference_stop] - reference_position[start])
        )
        target_distance = float(
            np.linalg.norm(target_position[target_stop] - target_position[start])
        )
        reference_stop_ns = round(float(time[reference_stop]) * 1_000_000_000)
        target_stop_ns = round(float(time[target_stop]) * 1_000_000_000)
        events.append(
            (
                reference_distance,
                target_distance,
                abs(float(reference_velocity[-1])),
                abs(float(target_velocity[-1])),
                abs(target_stop_ns - reference_stop_ns) / 1_000_000_000,
            )
        )
    return events


def _split_metrics(
    *,
    reference_traces: tuple[ProbeTrace, ...],
    target_traces: tuple[ProbeTrace, ...],
    split: str,
) -> dict[str, float]:
    reference_by_name = _trace_map(traces=reference_traces)
    target_by_name = _trace_map(traces=target_traces)
    squared_errors: list[np.ndarray] = []
    squared_references: list[np.ndarray] = []
    acceleration_errors: list[np.ndarray] = []
    axis_errors: dict[str, list[np.ndarray]] = {"x": [], "y": [], "z": []}
    axis_references: dict[str, list[np.ndarray]] = {"x": [], "y": [], "z": []}
    yaw_errors: list[np.ndarray] = []
    yaw_references: list[np.ndarray] = []
    heading_errors: list[float] = []
    endpoint_errors: list[float] = []
    braking_distance_errors: list[float] = []
    residual_speed_errors: list[float] = []
    stop_time_errors: list[float] = []
    overshoots: list[float] = []
    settling_errors: list[float] = []
    for name, reference in reference_by_name.items():
        if reference.split != split:
            continue
        try:
            target = target_by_name[name]
        except KeyError as exc:
            raise ValueError(f"target backend omitted probe profile {name!r}") from exc
        reference_velocity, target_velocity = _matched_trace(
            reference=reference,
            target=target,
        )
        error = target_velocity - reference_velocity
        squared_errors.append(np.square(error))
        squared_references.append(np.square(reference_velocity))
        acceleration_errors.append(
            np.square(_acceleration(trace=target) - _acceleration(trace=reference))
        )
        if reference.axis in axis_errors:
            axis_index = {"x": 0, "y": 1, "z": 2}[reference.axis]
            axis_errors[reference.axis].append(np.square(error[:, axis_index]))
            axis_references[reference.axis].append(reference_velocity[:, axis_index])
            reference_peak = float(np.max(np.abs(reference_velocity[:, axis_index])))
            target_peak = float(np.max(np.abs(target_velocity[:, axis_index])))
            overshoots.append(max(target_peak - reference_peak, 0.0))
            tail = max(1, len(reference_velocity) // 10)
            settling_errors.append(float(np.sqrt(np.mean(np.square(error[-tail:, axis_index])))))
        reference_rate = np.asarray(
            reference.angular_velocity_body_flu_radps,
            dtype=np.float64,
        )[:, 2]
        target_rate = np.asarray(
            target.angular_velocity_body_flu_radps,
            dtype=np.float64,
        )[:, 2]
        if reference.axis in ("yaw", "combined"):
            yaw_errors.append(np.square(target_rate - reference_rate))
            yaw_references.append(reference_rate)
            reference_yaw = _yaw_wxyz(np.asarray(reference.attitude_flu_enu_wxyz, dtype=np.float64))
            target_yaw = _yaw_wxyz(np.asarray(target.attitude_flu_enu_wxyz, dtype=np.float64))
            heading_errors.append(
                float(np.max(np.abs((target_yaw - reference_yaw) * 180.0 / np.pi)))
            )
        if reference.axis == "combined":
            endpoint_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(target.position_enu_m[-1])
                        - np.asarray(reference.position_enu_m[-1])
                    )
                )
            )
        for (
            reference_distance,
            target_distance,
            reference_residual,
            target_residual,
            stop_time_error,
        ) in _braking_events(reference=reference, target=target):
            braking_distance_errors.append(abs(target_distance - reference_distance))
            residual_speed_errors.append(abs(target_residual - reference_residual))
            stop_time_errors.append(stop_time_error)
    if not squared_errors:
        raise ValueError(f"probe suite contains no {split!r} traces")
    aggregate_rmse = float(np.sqrt(np.mean(np.concatenate(squared_errors, axis=0))))
    reference_rms = float(np.sqrt(np.mean(np.concatenate(squared_references, axis=0))))
    metrics = {
        "velocity_rmse_mps": aggregate_rmse,
        "velocity_nrmse": aggregate_rmse / max(reference_rms, 1e-9),
        "acceleration_rmse_mps2": float(
            np.sqrt(np.mean(np.concatenate(acceleration_errors, axis=0)))
        ),
        "overshoot_mps": max(overshoots, default=0.0),
        "settling_rmse_mps": max(settling_errors, default=0.0),
        "braking_distance_error_m": max(braking_distance_errors, default=0.0),
        "residual_speed_error_mps": max(residual_speed_errors, default=0.0),
        "stop_time_error_s": max(stop_time_errors, default=0.0),
        "heading_error_deg": max(heading_errors, default=0.0),
        "combined_endpoint_error_m": max(endpoint_errors, default=0.0),
    }
    for axis in ("x", "y", "z"):
        if not axis_errors[axis]:
            raise ValueError(f"probe suite contains no {split!r} {axis}-axis trace")
        rmse = float(np.sqrt(np.mean(np.concatenate(axis_errors[axis]))))
        values = np.concatenate(axis_references[axis])
        scale = max(float(np.max(values) - np.min(values)), 1e-9)
        metrics[f"{axis}_velocity_rmse_mps"] = rmse
        metrics[f"{axis}_velocity_nrmse"] = rmse / scale
    metrics["worst_axis_velocity_nrmse"] = max(
        metrics[f"{axis}_velocity_nrmse"] for axis in ("x", "y", "z")
    )
    if yaw_errors:
        yaw_rmse = float(np.sqrt(np.mean(np.concatenate(yaw_errors))))
        yaw_values = np.concatenate(yaw_references)
        yaw_scale = max(float(np.max(yaw_values) - np.min(yaw_values)), 1e-9)
        metrics["yaw_rate_rmse_radps"] = yaw_rmse
        metrics["yaw_rate_nrmse"] = yaw_rmse / yaw_scale
    else:
        metrics["yaw_rate_rmse_radps"] = 0.0
        metrics["yaw_rate_nrmse"] = 0.0
    return metrics


def _metric_result(
    reference_traces: tuple[ProbeTrace, ...],
    target_traces: tuple[ProbeTrace, ...],
    /,
) -> MetricResult:
    reference_by_name = _trace_map(traces=reference_traces)
    target_by_name = _trace_map(traces=target_traces)
    for name, reference in reference_by_name.items():
        if reference.split != "stability":
            continue
        try:
            target = target_by_name[name]
        except KeyError as exc:
            raise ValueError(f"target backend omitted probe profile {name!r}") from exc
        _matched_trace(reference=reference, target=target)
    fit = _split_metrics(
        reference_traces=reference_traces,
        target_traces=target_traces,
        split="fit",
    )
    validation = _split_metrics(
        reference_traces=reference_traces,
        target_traces=target_traces,
        split="validation",
    )
    metrics = {
        **{f"fit_{name}": value for name, value in fit.items()},
        **{f"validation_{name}": value for name, value in validation.items()},
    }
    ranking = (
        validation["braking_distance_error_m"],
        validation["residual_speed_error_mps"],
        validation["worst_axis_velocity_nrmse"],
        validation["yaw_rate_nrmse"],
        validation["heading_error_deg"],
        validation["combined_endpoint_error_m"],
        validation["velocity_rmse_mps"],
        fit["velocity_rmse_mps"],
    )
    return MetricResult(
        metrics=metrics,
        ranking_values=ranking,
        fit_metrics=fit,
        validation_metrics=validation,
    )


def _feasibility_metric_result(
    reference_traces: tuple[ProbeTrace, ...],
    target_traces: tuple[ProbeTrace, ...],
    /,
) -> MetricResult:
    result = _metric_result(reference_traces, target_traces)
    validation = result.validation_metrics
    ratios = tuple(validation[name] / limit for name, limit in _QUALITY_LIMITS)
    excesses = tuple(max(ratio - 1.0, 0.0) for ratio in ratios)
    return MetricResult(
        metrics=result.metrics,
        ranking_values=(
            float(sum(excess > 0.0 for excess in excesses)),
            max(excesses),
            sum(excesses),
            *result.ranking_values,
        ),
        fit_metrics=result.fit_metrics,
        validation_metrics=validation,
        beam_values=ratios,
    )


def _worst_axis_velocity_nrmse(
    reference_traces: tuple[ProbeTrace, ...],
    target_traces: tuple[ProbeTrace, ...],
    /,
) -> MetricResult:
    result = _metric_result(reference_traces, target_traces)
    validation = result.validation_metrics
    fit = result.fit_metrics
    return MetricResult(
        metrics=result.metrics,
        ranking_values=(
            validation["worst_axis_velocity_nrmse"],
            validation["velocity_rmse_mps"],
            fit["velocity_rmse_mps"],
        ),
        fit_metrics=fit,
        validation_metrics=validation,
    )


def register_metric(*, name: str, metric: MetricFunction) -> None:
    """Register a unique metric name for selection and publication."""
    if not name or name in _METRICS:
        raise ValueError(f"alignment metric name must be unique and non-empty: {name!r}")
    _METRICS[name] = metric


def metric(*, name: str) -> MetricFunction:
    """Resolve a registered metric by its stable name."""
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported alignment metric {name!r}") from exc


def metric_names() -> tuple[str, ...]:
    """Return registered names in deterministic order."""
    return tuple(sorted(_METRICS))


def _finite_metric_result(*, result: MetricResult) -> bool:
    values = (
        *result.metrics.values(),
        *result.ranking_values,
        *result.beam_values,
        *result.fit_metrics.values(),
        *result.validation_metrics.values(),
    )
    return all(math.isfinite(value) for value in values)


register_metric(name=_V1_METRIC, metric=_worst_axis_velocity_nrmse)
register_metric(name=_V2_METRIC, metric=_metric_result)
register_metric(name=_V3_METRIC, metric=_feasibility_metric_result)

"""Generic probe-family metric registry and controller quality aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from robotics_experiments.alignment._models import (
    MetricFunction,
    MetricResult,
    ProbeTrace,
    QualityGateResult,
)

__all__: list[str] = []

_STOP_SPEED_MPS = 0.01
_MIN_BRAKING_RESPONSE_FRACTION = 0.5
_METRICS: dict[str, MetricFunction] = {}


def _trace_map(*, traces: tuple[ProbeTrace, ...]) -> Mapping[str, ProbeTrace]:
    mapped = {trace.profile: trace for trace in traces}
    if len(mapped) != len(traces):
        raise ValueError("backend produced duplicate probe profile names")
    return mapped


def _matched_trace(*, reference: ProbeTrace, target: ProbeTrace) -> None:
    identity = (
        "family",
        "control_mode",
        "family_metric",
        "split",
        "axis",
        "time_s",
        "command_velocity_body_flu_mps",
        "command_yaw_rate_radps",
        "command_collective_thrust_normalized",
        "command_body_rate_flu_radps",
    )
    if any(getattr(reference, name) != getattr(target, name) for name in identity):
        raise ValueError(
            f"target trace {target.profile!r} does not reproduce the reference family, "
            "cadence, and physical command sequence"
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
            width = 4 if field_name == "attitude_flu_enu_wxyz" else 3
            if values.shape != (len(trace.time_s), width):
                raise ValueError(f"probe trace {trace.profile!r} has invalid {field_name} shape")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"probe trace {trace.profile!r} contains non-finite {field_name}")


def _nrmse(*, errors: list[np.ndarray], references: list[np.ndarray], name: str) -> float:
    if not errors:
        raise ValueError(f"probe family has no {name} samples")
    error = float(np.sqrt(np.mean(np.concatenate(errors))))
    values = np.concatenate(references)
    scale = float(np.max(values) - np.min(values))
    if scale <= 1e-9:
        raise ValueError(f"probe family {name} reference has no excitation")
    return error / scale


def _acceleration(*, trace: ProbeTrace, axis: int) -> np.ndarray:
    time = np.asarray(trace.time_s, dtype=np.float64)
    velocity = np.asarray(trace.velocity_body_flu_mps, dtype=np.float64)[:, axis]
    if len(time) < 2:
        raise ValueError(f"probe trace {trace.profile!r} needs at least two samples")
    return np.gradient(velocity, time)


def _stop_index(*, velocity: np.ndarray, start: int) -> int:
    below = np.flatnonzero(np.abs(velocity[start:]) <= _STOP_SPEED_MPS)
    return len(velocity) - 1 if not len(below) else start + int(below[0])


def _braking_events(
    *, reference: ProbeTrace, target: ProbeTrace, axis: int
) -> list[tuple[float, float, float]]:
    command = np.asarray(reference.command_velocity_body_flu_mps, dtype=np.float64)[:, axis]
    reference_velocity = np.asarray(reference.velocity_body_flu_mps, dtype=np.float64)[:, axis]
    target_velocity = np.asarray(target.velocity_body_flu_mps, dtype=np.float64)[:, axis]
    time = np.asarray(reference.time_s, dtype=np.float64)
    transitions = np.flatnonzero(
        (np.abs(command[:-1]) > _STOP_SPEED_MPS)
        & ((np.abs(command[1:]) <= _STOP_SPEED_MPS) | (command[:-1] * command[1:] < 0.0))
    )
    events = []
    for previous in transitions:
        start = int(previous + 1)
        if abs(reference_velocity[start]) < (
            _MIN_BRAKING_RESPONSE_FRACTION * abs(command[previous])
        ):
            continue
        reference_stop = _stop_index(velocity=reference_velocity, start=start)
        target_stop = _stop_index(velocity=target_velocity, start=start)
        reference_distance = abs(
            float(
                np.trapz(
                    reference_velocity[start : reference_stop + 1], time[start : reference_stop + 1]
                )
            )
        )
        target_distance = abs(
            float(np.trapz(target_velocity[start : target_stop + 1], time[start : target_stop + 1]))
        )
        stop_time_error = round(abs(float(time[target_stop]) - float(time[reference_stop])), 9)
        events.append(
            (
                abs(target_distance - reference_distance),
                abs(abs(float(target_velocity[-1])) - abs(float(reference_velocity[-1]))),
                stop_time_error,
            )
        )
    return events


def _translation_split(
    *, reference_traces: tuple[ProbeTrace, ...], target_traces: tuple[ProbeTrace, ...], split: str
) -> dict[str, float]:
    reference_by_name = _trace_map(traces=reference_traces)
    target_by_name = _trace_map(traces=target_traces)
    axis_errors: dict[str, list[np.ndarray]] = {axis: [] for axis in ("x", "y", "z")}
    axis_references: dict[str, list[np.ndarray]] = {axis: [] for axis in ("x", "y", "z")}
    envelope_errors: dict[str, list[np.ndarray]] = {axis: [] for axis in ("x", "y", "z")}
    envelope_references: dict[str, list[np.ndarray]] = {axis: [] for axis in ("x", "y", "z")}
    braking: dict[str, list[float]] = defaultdict(list)
    residual: dict[str, list[float]] = defaultdict(list)
    stop_time: dict[str, list[float]] = defaultdict(list)
    endpoint_errors = []
    for name, reference in reference_by_name.items():
        if reference.split != split:
            continue
        target = target_by_name.get(name)
        if target is None:
            raise ValueError(f"target backend omitted probe profile {name!r}")
        _matched_trace(reference=reference, target=target)
        if reference.axis in axis_errors:
            axis = reference.axis
            index = {"x": 0, "y": 1, "z": 2}[axis]
            reference_velocity = np.asarray(reference.velocity_body_flu_mps, dtype=np.float64)[
                :, index
            ]
            target_velocity = np.asarray(target.velocity_body_flu_mps, dtype=np.float64)[:, index]
            error = np.square(target_velocity - reference_velocity)
            axis_errors[axis].append(error)
            axis_references[axis].append(reference_velocity)
            if "envelope" in reference.profile:
                envelope_errors[axis].append(error)
                envelope_references[axis].append(reference_velocity)
            else:
                for distance, speed, timing in _braking_events(
                    reference=reference, target=target, axis=index
                ):
                    braking[axis].append(distance)
                    residual[axis].append(speed)
                    stop_time[axis].append(timing)
        elif reference.axis == "combined":
            endpoint_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(target.position_enu_m[-1], dtype=np.float64)
                        - np.asarray(reference.position_enu_m[-1], dtype=np.float64)
                    )
                )
            )
    metrics: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        if split == "fit" and not envelope_errors[axis]:
            envelope_errors[axis] = axis_errors[axis]
            envelope_references[axis] = axis_references[axis]
        metrics[f"{axis}_velocity_nrmse"] = _nrmse(
            errors=axis_errors[axis], references=axis_references[axis], name=f"{split} {axis}"
        )
        metrics[f"{axis}_envelope_velocity_nrmse"] = _nrmse(
            errors=envelope_errors[axis],
            references=envelope_references[axis],
            name=f"{split} {axis} envelope",
        )
        metrics[f"{axis}_braking_distance_error_m"] = max(braking[axis], default=0.0)
        metrics[f"{axis}_residual_speed_error_mps"] = max(residual[axis], default=0.0)
        metrics[f"{axis}_stop_time_error_s"] = max(stop_time[axis], default=0.0)
    metrics["combined_endpoint_error_m"] = max(endpoint_errors, default=0.0)
    return metrics


def _gate(*, name: str, metrics: Mapping[str, float], limit: float) -> QualityGateResult:
    return QualityGateResult(name=name, value=metrics[name], limit=limit)


def _translation_quality(
    reference_traces: tuple[ProbeTrace, ...], target_traces: tuple[ProbeTrace, ...], /
) -> MetricResult:
    fit = _translation_split(
        reference_traces=reference_traces, target_traces=target_traces, split="fit"
    )
    validation = _translation_split(
        reference_traces=reference_traces, target_traces=target_traces, split="validation"
    )
    gates: list[QualityGateResult] = []
    for axis in ("x", "y", "z"):
        gates.extend(
            (
                _gate(name=f"{axis}_velocity_nrmse", metrics=validation, limit=0.10),
                _gate(name=f"{axis}_envelope_velocity_nrmse", metrics=validation, limit=0.10),
                _gate(name=f"{axis}_braking_distance_error_m", metrics=validation, limit=0.005),
                _gate(name=f"{axis}_residual_speed_error_mps", metrics=validation, limit=0.01),
                _gate(name=f"{axis}_stop_time_error_s", metrics=validation, limit=0.10),
            )
        )
    gates.append(_gate(name="combined_endpoint_error_m", metrics=validation, limit=0.01))
    ratios = tuple(gate.value / gate.limit for gate in gates)
    return MetricResult(
        metrics={**{f"fit_{key}": value for key, value in fit.items()}, **validation},
        ranking_values=(
            sum(validation[f"{axis}_residual_speed_error_mps"] for axis in ("x", "y", "z")),
            sum(validation[f"{axis}_velocity_nrmse"] for axis in ("x", "y", "z")),
            validation["combined_endpoint_error_m"],
        ),
        fit_metrics=fit,
        validation_metrics=validation,
        beam_values=ratios,
        quality_passed=all(gate.value <= gate.limit for gate in gates),
        gates=tuple(gates),
    )


def _euler_wxyz(attitude: np.ndarray, /) -> np.ndarray:
    w, x, y, z = attitude.T
    return np.column_stack(
        (
            np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        )
    )


def _ctbr_split(
    *, reference_traces: tuple[ProbeTrace, ...], target_traces: tuple[ProbeTrace, ...], split: str
) -> dict[str, float]:
    reference_by_name = _trace_map(traces=reference_traces)
    target_by_name = _trace_map(traces=target_traces)
    rate_errors: dict[str, list[np.ndarray]] = {axis: [] for axis in ("roll", "pitch", "yaw")}
    rate_references: dict[str, list[np.ndarray]] = {axis: [] for axis in ("roll", "pitch", "yaw")}
    envelope_errors: dict[str, list[np.ndarray]] = {axis: [] for axis in ("roll", "pitch", "yaw")}
    envelope_references: dict[str, list[np.ndarray]] = {
        axis: [] for axis in ("roll", "pitch", "yaw")
    }
    attitude_errors: dict[str, list[np.ndarray]] = {axis: [] for axis in ("roll", "pitch", "yaw")}
    collective_velocity_errors: list[np.ndarray] = []
    collective_velocity_references: list[np.ndarray] = []
    collective_acceleration_errors: list[np.ndarray] = []
    collective_acceleration_references: list[np.ndarray] = []
    for name, reference in reference_by_name.items():
        if reference.split != split:
            continue
        target = target_by_name.get(name)
        if target is None:
            raise ValueError(f"target backend omitted probe profile {name!r}")
        _matched_trace(reference=reference, target=target)
        if reference.axis in rate_errors:
            axis = reference.axis
            index = {"roll": 0, "pitch": 1, "yaw": 2}[axis]
            reference_rate = np.asarray(
                reference.angular_velocity_body_flu_radps, dtype=np.float64
            )[:, index]
            target_rate = np.asarray(target.angular_velocity_body_flu_radps, dtype=np.float64)[
                :, index
            ]
            error = np.square(target_rate - reference_rate)
            rate_errors[axis].append(error)
            rate_references[axis].append(reference_rate)
            if "envelope" in reference.profile:
                envelope_errors[axis].append(error)
                envelope_references[axis].append(reference_rate)
            reference_angle = _euler_wxyz(
                np.asarray(reference.attitude_flu_enu_wxyz, dtype=np.float64)
            )[:, index]
            target_angle = _euler_wxyz(np.asarray(target.attitude_flu_enu_wxyz, dtype=np.float64))[
                :, index
            ]
            delta = np.arctan2(
                np.sin(target_angle - reference_angle), np.cos(target_angle - reference_angle)
            )
            attitude_errors[axis].append(np.abs(delta) * 180.0 / np.pi)
        elif reference.axis == "collective":
            reference_velocity = np.asarray(reference.velocity_body_flu_mps, dtype=np.float64)[:, 2]
            target_velocity = np.asarray(target.velocity_body_flu_mps, dtype=np.float64)[:, 2]
            collective_velocity_errors.append(np.square(target_velocity - reference_velocity))
            collective_velocity_references.append(reference_velocity)
            reference_acceleration = _acceleration(trace=reference, axis=2)
            target_acceleration = _acceleration(trace=target, axis=2)
            collective_acceleration_errors.append(
                np.square(target_acceleration - reference_acceleration)
            )
            collective_acceleration_references.append(reference_acceleration)
    metrics: dict[str, float] = {}
    for axis in ("roll", "pitch", "yaw"):
        if split == "fit" and not envelope_errors[axis]:
            envelope_errors[axis] = rate_errors[axis]
            envelope_references[axis] = rate_references[axis]
        metrics[f"{axis}_body_rate_nrmse"] = _nrmse(
            errors=rate_errors[axis], references=rate_references[axis], name=f"{split} {axis}"
        )
        metrics[f"{axis}_envelope_body_rate_nrmse"] = _nrmse(
            errors=envelope_errors[axis],
            references=envelope_references[axis],
            name=f"{split} {axis} envelope",
        )
        metrics[f"{axis}_attitude_error_deg"] = float(np.max(np.concatenate(attitude_errors[axis])))
    if collective_velocity_errors:
        metrics["collective_velocity_nrmse"] = _nrmse(
            errors=collective_velocity_errors,
            references=collective_velocity_references,
            name=f"{split} collective velocity",
        )
        metrics["collective_acceleration_nrmse"] = _nrmse(
            errors=collective_acceleration_errors,
            references=collective_acceleration_references,
            name=f"{split} collective acceleration",
        )
    else:
        metrics["collective_velocity_nrmse"] = 0.0
        metrics["collective_acceleration_nrmse"] = 0.0
    return metrics


def _ctbr_quality(
    reference_traces: tuple[ProbeTrace, ...], target_traces: tuple[ProbeTrace, ...], /
) -> MetricResult:
    fit = _ctbr_split(reference_traces=reference_traces, target_traces=target_traces, split="fit")
    validation = _ctbr_split(
        reference_traces=reference_traces, target_traces=target_traces, split="validation"
    )
    gates: list[QualityGateResult] = []
    for axis in ("roll", "pitch", "yaw"):
        gates.extend(
            (
                _gate(name=f"{axis}_body_rate_nrmse", metrics=validation, limit=0.10),
                _gate(name=f"{axis}_envelope_body_rate_nrmse", metrics=validation, limit=0.10),
                _gate(name=f"{axis}_attitude_error_deg", metrics=validation, limit=3.0),
            )
        )
    gates.extend(
        (
            _gate(name="collective_velocity_nrmse", metrics=validation, limit=0.10),
            _gate(name="collective_acceleration_nrmse", metrics=validation, limit=0.10),
        )
    )
    ratios = tuple(gate.value / gate.limit for gate in gates)
    return MetricResult(
        metrics={**{f"fit_{key}": value for key, value in fit.items()}, **validation},
        ranking_values=(
            sum(validation[f"{axis}_body_rate_nrmse"] for axis in ("roll", "pitch", "yaw")),
            sum(validation[f"{axis}_attitude_error_deg"] for axis in ("roll", "pitch", "yaw")),
            validation["collective_velocity_nrmse"],
        ),
        fit_metrics=fit,
        validation_metrics=validation,
        beam_values=ratios,
        quality_passed=all(gate.value <= gate.limit for gate in gates),
        gates=tuple(gates),
    )


def _controller_quality(
    reference_traces: tuple[ProbeTrace, ...], target_traces: tuple[ProbeTrace, ...], /
) -> MetricResult:
    reference_by_profile = _trace_map(traces=reference_traces)
    target_by_profile = _trace_map(traces=target_traces)
    if reference_by_profile.keys() != target_by_profile.keys():
        raise ValueError("target backend did not reproduce every probe profile")
    for profile, reference in reference_by_profile.items():
        if reference.split == "stability":
            _matched_trace(reference=reference, target=target_by_profile[profile])
    reference_by_family: dict[str, list[ProbeTrace]] = defaultdict(list)
    target_by_family: dict[str, list[ProbeTrace]] = defaultdict(list)
    for trace in reference_traces:
        reference_by_family[trace.family].append(trace)
    for trace in target_traces:
        target_by_family[trace.family].append(trace)
    if reference_by_family.keys() != target_by_family.keys():
        raise ValueError("target backend did not reproduce the reference probe families")
    metrics: dict[str, float] = {}
    fit: dict[str, float] = {}
    validation: dict[str, float] = {}
    gates: list[QualityGateResult] = []
    ranking: list[float] = []
    beam: list[float] = []
    for family_name, reference_family in reference_by_family.items():
        metric_names = {trace.family_metric for trace in reference_family}
        if len(metric_names) != 1:
            raise ValueError(f"probe family {family_name!r} has inconsistent metrics")
        family_metric_name = next(iter(metric_names))
        if family_metric_name == "controller-quality":
            raise ValueError("a probe family cannot recursively use controller-quality")
        family_result = metric(name=family_metric_name)(
            tuple(reference_family), tuple(target_by_family[family_name])
        )
        for key, value in family_result.metrics.items():
            metrics[f"{family_name}.{key}"] = value
        for key, value in family_result.fit_metrics.items():
            fit[f"{family_name}.{key}"] = value
        for key, value in family_result.validation_metrics.items():
            validation[f"{family_name}.{key}"] = value
        gates.extend(
            QualityGateResult(name=f"{family_name}.{gate.name}", value=gate.value, limit=gate.limit)
            for gate in family_result.gates
        )
        ranking.extend(family_result.ranking_values)
        beam.extend(family_result.beam_values)
    excesses = tuple(max(gate.value / gate.limit - 1.0, 0.0) for gate in gates)
    return MetricResult(
        metrics=metrics,
        ranking_values=(
            float(sum(excess > 0.0 for excess in excesses)),
            max(excesses, default=0.0),
            sum(excesses),
            *ranking,
        ),
        fit_metrics=fit,
        validation_metrics=validation,
        beam_values=tuple(beam),
        quality_passed=not any(excesses),
        gates=tuple(gates),
    )


def register_metric(*, name: str, metric: MetricFunction) -> None:
    """Register a unique family or aggregate metric implementation."""
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
        *(value for gate in result.gates for value in (gate.value, gate.limit)),
    )
    return all(math.isfinite(value) for value in values)


register_metric(name="translational-controller-quality", metric=_translation_quality)
register_metric(name="ctbr-controller-quality", metric=_ctbr_quality)
register_metric(name="controller-quality", metric=_controller_quality)

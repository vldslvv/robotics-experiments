"""Named alignment metric registry and velocity-response diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from robotics_experiments.alignment._models import MetricFunction, MetricResult, ProbeTrace

__all__: list[str] = []

_DEFAULT_METRIC = "worst-axis-validation-velocity-nrmse-v1"
_METRICS: dict[str, MetricFunction] = {}


def _trace_map(*, traces: tuple[ProbeTrace, ...]) -> Mapping[str, ProbeTrace]:
    mapped = {trace.profile: trace for trace in traces}
    if len(mapped) != len(traces):
        raise ValueError("backend produced duplicate probe profile names")
    return mapped


def _matched_arrays(
    *,
    reference: ProbeTrace,
    target: ProbeTrace,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        reference.split != target.split
        or reference.axis != target.axis
        or reference.time_s != target.time_s
        or reference.command_velocity_body_flu_mps != target.command_velocity_body_flu_mps
    ):
        raise ValueError(
            f"target trace {target.profile!r} does not reproduce the reference cadence "
            "and command sequence"
        )
    reference_velocity = np.asarray(reference.velocity_body_flu_mps, dtype=np.float64)
    target_velocity = np.asarray(target.velocity_body_flu_mps, dtype=np.float64)
    if (
        reference_velocity.ndim != 2
        or reference_velocity.shape[1:] != (3,)
        or target_velocity.shape != reference_velocity.shape
        or reference_velocity.shape[0] != len(reference.time_s)
    ):
        raise ValueError(f"probe trace {reference.profile!r} has invalid velocity shape")
    if not np.all(np.isfinite(reference_velocity)) or not np.all(np.isfinite(target_velocity)):
        raise ValueError(f"probe trace {reference.profile!r} contains non-finite velocity")
    return reference_velocity, target_velocity


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
    axis_errors: dict[str, list[np.ndarray]] = {"x": [], "y": [], "z": []}
    axis_references: dict[str, list[np.ndarray]] = {"x": [], "y": [], "z": []}
    overshoots: list[float] = []
    settling_errors: list[float] = []
    for name, reference in reference_by_name.items():
        if reference.split != split:
            continue
        try:
            target = target_by_name[name]
        except KeyError as exc:
            raise ValueError(f"target backend omitted probe profile {name!r}") from exc
        if not reference.completed or reference.failure_reason is not None:
            raise ValueError(f"reference probe {name!r} did not complete cleanly")
        if not target.completed or target.failure_reason is not None:
            raise ValueError(f"target probe {name!r} did not complete cleanly")
        reference_velocity, target_velocity = _matched_arrays(
            reference=reference,
            target=target,
        )
        error = target_velocity - reference_velocity
        squared_errors.append(np.square(error))
        squared_references.append(np.square(reference_velocity))
        axis_index = {"x": 0, "y": 1, "z": 2}[reference.axis]
        axis_errors[reference.axis].append(np.square(error[:, axis_index]))
        axis_references[reference.axis].append(reference_velocity[:, axis_index])
        reference_peak = float(np.max(np.abs(reference_velocity[:, axis_index])))
        target_peak = float(np.max(np.abs(target_velocity[:, axis_index])))
        overshoots.append(max(target_peak - reference_peak, 0.0))
        tail = max(1, len(reference_velocity) // 10)
        settling_errors.append(float(np.sqrt(np.mean(np.square(error[-tail:, axis_index])))))
    if not squared_errors:
        raise ValueError(f"probe suite contains no {split!r} traces")
    aggregate_rmse = float(np.sqrt(np.mean(np.concatenate(squared_errors, axis=0))))
    reference_rms = float(np.sqrt(np.mean(np.concatenate(squared_references, axis=0))))
    aggregate_nrmse = aggregate_rmse / max(reference_rms, 1e-9)
    metrics = {
        "velocity_rmse_mps": aggregate_rmse,
        "velocity_nrmse": aggregate_nrmse,
        "overshoot_mps": max(overshoots),
        "settling_rmse_mps": max(settling_errors),
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
    return metrics


def _worst_axis_velocity_nrmse(
    reference_traces: tuple[ProbeTrace, ...],
    target_traces: tuple[ProbeTrace, ...],
    /,
) -> MetricResult:
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
        validation["worst_axis_velocity_nrmse"],
        validation["velocity_rmse_mps"],
        fit["velocity_rmse_mps"],
    )
    return MetricResult(
        metrics=metrics,
        ranking_values=ranking,
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
        *result.fit_metrics.values(),
        *result.validation_metrics.values(),
    )
    return all(math.isfinite(value) for value in values)


register_metric(name=_DEFAULT_METRIC, metric=_worst_axis_velocity_nrmse)

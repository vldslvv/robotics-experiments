"""Strict metric presentation registry for policy qualification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from robotics_experiments.rl.evaluation import PolicyEvaluation

__all__ = [
    "EVALUATION_METRICS",
    "METRIC_REGISTRY_REVISION",
    "MetricSpec",
    "evaluation_metric_names",
    "metric_registry_payload",
    "write_evaluation_scalars",
    "write_reset_scalars",
    "write_system_scalars",
]

METRIC_REGISTRY_REVISION = 18

_HOVER_SIGNAL_UNITS = {
    "position_error_enu_x": "m",
    "position_error_enu_y": "m",
    "position_error_enu_z": "m",
    "horizontal_position_error": "m",
    "position_error": "m",
    "attitude_error_roll": "rad",
    "attitude_error_pitch": "rad",
    "attitude_error_yaw": "rad",
    "tilt_error": "rad",
    "orientation_error": "rad",
}
_BINDING_STATISTICS = ("max", "rmse", "p95", "p99")


class _ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: Any, global_step: int, /, *args, **kwargs) -> None:
        """Write one scalar event."""


@dataclass(frozen=True)
class MetricSpec:
    """One semantic report metric and its stable TensorBoard presentation."""

    raw_name: str | None
    suffix: str
    unit: str
    description: str
    direction: str


EVALUATION_METRICS = (
    MetricSpec(
        None, "01_acceptance/all_checks_passed", "boolean", "All checks passed.", "maximize"
    ),
    MetricSpec(
        "success_rate",
        "01_acceptance/success_rate",
        "fraction",
        "Successful episodes.",
        "maximize",
    ),
    MetricSpec(
        "collision_rate",
        "01_acceptance/collision_rate",
        "fraction",
        "Collision episodes.",
        "minimize",
    ),
    MetricSpec(
        "other_flight_failure_rate",
        "01_acceptance/flight_failure_rate",
        "fraction",
        "Other flight failures.",
        "minimize",
    ),
    MetricSpec(
        "minimum_closest_range_m",
        "01_acceptance/minimum_closest_range_m",
        "m",
        "Minimum closest range.",
        "maximize",
    ),
    MetricSpec(
        "all_traces_finite",
        "01_acceptance/all_traces_finite",
        "boolean",
        "All traces are finite.",
        "maximize",
    ),
    MetricSpec(
        "hover_envelope_occupancy_rate",
        "02_hover/envelope_occupancy_rate",
        "fraction",
        "Samples inside the measured hover pose envelope.",
        "maximize",
    ),
    MetricSpec(
        "episode_longest_hover_duration_mean_s",
        "02_hover/episode_longest_duration_mean_s",
        "s",
        "Mean of each episode's longest continuous hover.",
        "maximize",
    ),
    MetricSpec(
        "episode_longest_hover_duration_max_s",
        "02_hover/episode_longest_duration_max_s",
        "s",
        "Longest continuous hover in any episode.",
        "maximize",
    ),
    MetricSpec(
        "hover_envelope_utilization_mean",
        "02_hover/envelope_utilization_mean",
        "normalized",
        "Mean maximum measured pose-envelope utilization; 1 is the boundary.",
        "minimize",
    ),
    MetricSpec(
        "position_envelope_utilization_mean",
        "02_hover/position_envelope_utilization_mean",
        "normalized",
        "Mean measured position-envelope utilization; 1 is the boundary.",
        "minimize",
    ),
    MetricSpec(
        "attitude_envelope_utilization_mean",
        "02_hover/attitude_envelope_utilization_mean",
        "normalized",
        "Mean measured attitude-envelope utilization; 1 is the boundary.",
        "minimize",
    ),
    MetricSpec(
        "range_undershoot_rate",
        "03_outcome/range_undershoot_rate",
        "fraction",
        "Range-undershoot episodes.",
        "minimize",
    ),
    MetricSpec(
        "timeout_rate",
        "03_outcome/timeout_rate",
        "fraction",
        "Timed-out episodes.",
        "minimize",
    ),
    MetricSpec(
        "completed_episodes",
        "03_outcome/completed_episodes",
        "count",
        "Completed episodes.",
        "maximize",
    ),
) + tuple(
    MetricSpec(
        f"flight_parameters.{signal_name}.{statistic}",
        f"04_flight_parameters/{signal_name}/{statistic}",
        unit,
        f"Measured hover {signal_name} {statistic}.",
        "minimize",
    )
    for signal_name, unit in _HOVER_SIGNAL_UNITS.items()
    for statistic in _BINDING_STATISTICS
)


def evaluation_metric_names() -> tuple[str, ...]:
    """Return the exact ordered evaluator metric contract."""
    return tuple(spec.raw_name for spec in EVALUATION_METRICS if spec.raw_name is not None)


def metric_registry_payload() -> dict[str, Any]:
    """Return the strict presentation registry recorded in campaign reports."""
    return {
        "revision": METRIC_REGISTRY_REVISION,
        "evaluation": [asdict(spec) for spec in EVALUATION_METRICS],
    }


def write_evaluation_scalars(
    *,
    writer: _ScalarWriter,
    role: str,
    evaluation: PolicyEvaluation,
    all_checks_passed: bool,
    step: int,
) -> tuple[str, ...]:
    """Write one validation or test record through the shared registry."""
    if role not in {"val", "test"}:
        raise ValueError("evaluation TensorBoard role must be 'val' or 'test'")
    names = tuple(metric.name for metric in evaluation.metrics)
    if names != evaluation_metric_names():
        raise ValueError(f"evaluation metric contract mismatch: {names!r}")
    values = {metric.name: metric.value for metric in evaluation.metrics}
    emitted = []
    for spec in EVALUATION_METRICS:
        value = float(all_checks_passed) if spec.raw_name is None else values[spec.raw_name]
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"evaluation metric {spec.raw_name!r} is not finite")
        if spec.raw_name is not None and spec.raw_name.endswith("_rate"):
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"rate metric {spec.raw_name!r} must be in [0, 1]")
        tag = f"{role}/{spec.suffix}"
        writer.add_scalar(tag, numeric, step, new_style=True, double_precision=True)
        emitted.append(tag)
    return tuple(emitted)


def write_reset_scalars(
    *, writer: _ScalarWriter, summary: dict[str, float], step: int
) -> tuple[str, ...]:
    """Write the fixed trace-backed reset distribution once per attempt."""
    emitted = []
    for name, value in summary.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"reset metric {name!r} is not finite")
        tag = f"train/01_reset/{name}"
        writer.add_scalar(tag, numeric, step, new_style=True, double_precision=True)
        emitted.append(tag)
    return tuple(emitted)


def write_system_scalars(
    *,
    writer: _ScalarWriter,
    role: str,
    summary: dict[str, float],
    step: int,
) -> tuple[str, ...]:
    """Write one interval's diagnostic-only telemetry summary."""
    if role not in {"train", "val", "test"}:
        raise ValueError("system telemetry role must be train, val, or test")
    fields = (
        ("01_gpu", "gpu_utilization_rate_mean"),
        ("01_gpu", "gpu_utilization_rate_p95"),
        ("01_gpu", "gpu_memory_used_gib_peak"),
        ("01_gpu", "gpu_memory_utilization_rate_peak"),
        ("01_gpu", "gpu_power_w_mean"),
        ("01_gpu", "gpu_temperature_c_max"),
        ("02_process", "process_resident_memory_gib_peak"),
        ("02_process", "process_cpu_cores_used_mean"),
        ("03_throughput", "duration_s"),
        ("03_throughput", "work_items_per_s"),
    )
    emitted = []
    for group, name in fields:
        value = float(summary[name])
        if not math.isfinite(value):
            raise ValueError(f"system telemetry metric {name!r} is not finite")
        if name.endswith(("_rate_mean", "_rate_p95", "_rate_peak")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"system telemetry rate {name!r} must be in [0, 1]")
        tag = f"system/{role}/{group}/{name}"
        writer.add_scalar(tag, value, step, new_style=True, double_precision=True)
        emitted.append(tag)
    return tuple(emitted)

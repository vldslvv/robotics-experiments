"""Strict metric presentation registry for policy qualification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from robotics_experiments.rl.training.progress import PolicyEvaluation

__all__ = [
    "EVALUATION_METRICS",
    "METRIC_REGISTRY_REVISION",
    "MetricSpec",
    "evaluation_metric_names",
    "metric_registry_payload",
    "write_evaluation_scalars",
    "write_system_scalars",
]

METRIC_REGISTRY_REVISION = 2


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
        None,
        "00_all_gates_passed",
        "boolean",
        "All qualification gates passed.",
        "maximize",
    ),
    MetricSpec(
        "approach_success_rate",
        "01_approach_success_rate",
        "fraction",
        "Fraction of episodes completing the stable approach.",
        "maximize",
    ),
    MetricSpec(
        "other_flight_failure_rate",
        "02_other_flight_failure_rate",
        "fraction",
        "Fraction of episodes ending in a non-collision flight failure.",
        "minimize",
    ),
    MetricSpec(
        "collision_rate",
        "03_collision_rate",
        "fraction",
        "Fraction of episodes ending in any collision.",
        "minimize",
    ),
    MetricSpec(
        "minimum_closest_range_m",
        "04_minimum_closest_range_m",
        "m",
        "Minimum closest wall-normal range over all episodes.",
        "maximize",
    ),
    MetricSpec(
        "all_traces_finite",
        "05_all_traces_finite",
        "boolean",
        "Whether all evaluated traces and aggregates are finite.",
        "maximize",
    ),
    MetricSpec(
        "held_out_task_cost",
        "06_held_out_task_cost",
        "cost",
        "Raw held-out task cost used for checkpoint progress and ranking.",
        "minimize",
    ),
    MetricSpec(
        "timeout_rate",
        "10_timeout_rate",
        "fraction",
        "Fraction of episodes ending at the configured timeout.",
        "minimize",
    ),
    MetricSpec(
        "body_collision_rate",
        "11_body_collision_rate",
        "fraction",
        "Fraction of episodes with a body collision.",
        "minimize",
    ),
    MetricSpec(
        "pusher_collision_rate",
        "12_pusher_collision_rate",
        "fraction",
        "Fraction of episodes with a pusher collision.",
        "minimize",
    ),
    MetricSpec(
        "mean_closest_range_m",
        "13_mean_closest_range_m",
        "m",
        "Mean closest wall-normal range across episodes.",
        "maximize",
    ),
    MetricSpec(
        "median_successful_completion_time_s",
        "14_median_successful_completion_time_s",
        "s",
        "Median completion time among successful episodes.",
        "minimize",
    ),
    MetricSpec(
        "mean_terminal_range_cost",
        "20_mean_terminal_range_cost",
        "cost",
        "Mean terminal-range cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_trajectory_alignment_cost",
        "21_mean_trajectory_alignment_cost",
        "cost",
        "Mean trajectory-alignment cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_velocity_tracking_cost",
        "22_mean_velocity_tracking_cost",
        "cost",
        "Mean velocity-tracking cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_upright_error_cost",
        "23_mean_upright_error_cost",
        "cost",
        "Mean upright-error cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_action_delta_cost",
        "24_mean_action_delta_cost",
        "cost",
        "Mean action-change cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_completion_time_fraction",
        "25_mean_completion_time_fraction",
        "fraction",
        "Mean episode completion-time fraction.",
        "minimize",
    ),
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
    all_gates_passed: bool,
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
        value = float(all_gates_passed) if spec.raw_name is None else values[spec.raw_name]
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"evaluation metric {spec.raw_name!r} is not finite")
        if spec.suffix.endswith("_rate") and not 0.0 <= numeric <= 1.0:
            raise ValueError(f"rate metric {spec.raw_name!r} must be in [0, 1]")
        tag = f"{role}/{spec.suffix}"
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
        ("10", "gpu_utilization_rate_mean"),
        ("11", "gpu_utilization_rate_p95"),
        ("12", "gpu_memory_used_gib_peak"),
        ("13", "gpu_memory_utilization_rate_peak"),
        ("14", "gpu_power_w_mean"),
        ("15", "gpu_temperature_c_max"),
        ("16", "process_resident_memory_gib_peak"),
        ("17", "process_cpu_cores_used_mean"),
        ("18", "duration_s"),
        ("19", "work_items_per_s"),
    )
    emitted = []
    for order, name in fields:
        value = float(summary[name])
        if not math.isfinite(value):
            raise ValueError(f"system telemetry metric {name!r} is not finite")
        if name.endswith("_rate_mean") or name.endswith("_rate_p95") or name.endswith("_rate_peak"):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"system telemetry rate {name!r} must be in [0, 1]")
        tag = f"system/{order}_{role}_{name}"
        writer.add_scalar(tag, value, step, new_style=True, double_precision=True)
        emitted.append(tag)
    return tuple(emitted)

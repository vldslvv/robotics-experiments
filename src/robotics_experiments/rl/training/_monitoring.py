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

METRIC_REGISTRY_REVISION = 20

_HOVER_SIGNALS = {
    "position_error_enu_x": ("East position error", "m"),
    "position_error_enu_y": ("North position error", "m"),
    "position_error_enu_z": ("Vertical position error", "m"),
    "horizontal_position_error": ("Horizontal position error", "m"),
    "position_error": ("Total position error", "m"),
    "attitude_error_roll": ("Roll error", "rad"),
    "attitude_error_pitch": ("Pitch error", "rad"),
    "attitude_error_yaw": ("Yaw error", "rad"),
    "tilt_error": ("Tilt error", "rad"),
    "orientation_error": ("Total orientation error", "rad"),
}
_BINDING_STATISTICS = {
    "max": "Maximum absolute",
    "rmse": "Root mean square",
    "p95": "95th percentile absolute",
    "p99": "99th percentile absolute",
}

_RESET_LABELS = {
    "sample_count": "Measured reset states (count)",
    "position_error_enu_x_mean_m": "Mean east position error (meters)",
    "position_error_enu_y_mean_m": "Mean north position error (meters)",
    "position_error_enu_z_mean_m": "Mean vertical position error (meters)",
    "position_envelope_utilization_mean": "Mean position error relative to hover limit (1 = limit)",
    "position_envelope_utilization_p95": (
        "95th percentile position error relative to hover limit (1 = limit)"
    ),
    "position_envelope_utilization_max": (
        "Maximum position error relative to hover limit (1 = limit)"
    ),
    "attitude_envelope_utilization_mean": "Mean attitude error relative to hover limit (1 = limit)",
    "attitude_envelope_utilization_p95": (
        "95th percentile attitude error relative to hover limit (1 = limit)"
    ),
    "attitude_envelope_utilization_max": (
        "Maximum attitude error relative to hover limit (1 = limit)"
    ),
    "linear_speed_mean_mps": "Mean linear speed (meters per second)",
    "linear_speed_p95_mps": "95th percentile linear speed (meters per second)",
    "angular_speed_mean_radps": "Mean angular speed (radians per second)",
    "angular_speed_p95_radps": "95th percentile angular speed (radians per second)",
}


class _ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: Any, global_step: int, /, *args, **kwargs) -> None:
        """Write one scalar event."""


@dataclass(frozen=True)
class MetricSpec:
    """One semantic report metric and its stable monitoring presentation."""

    raw_name: str | None
    suffix: str
    unit: str
    description: str
    direction: str


EVALUATION_METRICS = (
    MetricSpec(
        None,
        "All acceptance requirements passed (0 or 1)",
        "boolean",
        "All checks passed.",
        "maximize",
    ),
    MetricSpec(
        "success_rate",
        "Successful episodes (fraction)",
        "fraction",
        "Successful episodes.",
        "maximize",
    ),
    MetricSpec(
        "collision_rate",
        "Episodes ending in collision (fraction)",
        "fraction",
        "Collision episodes.",
        "minimize",
    ),
    MetricSpec(
        "other_flight_failure_rate",
        "Episodes ending in flight failure without collision (fraction)",
        "fraction",
        "Other flight failures.",
        "minimize",
    ),
    MetricSpec(
        "minimum_closest_range_m",
        "Closest approach to switch plane (meters)",
        "m",
        "Minimum closest range.",
        "maximize",
    ),
    MetricSpec(
        "all_traces_finite",
        "All recorded values are finite (0 or 1)",
        "boolean",
        "All traces are finite.",
        "maximize",
    ),
    MetricSpec(
        "hover_envelope_occupancy_rate",
        "Mean episode fraction inside hover limits",
        "fraction",
        "Samples inside the measured hover pose envelope.",
        "maximize",
    ),
    MetricSpec(
        "episode_longest_hover_duration_mean_s",
        "Mean longest continuous hover (seconds)",
        "s",
        "Mean of each episode's longest continuous hover.",
        "maximize",
    ),
    MetricSpec(
        "episode_longest_hover_duration_max_s",
        "Longest continuous hover in any episode (seconds)",
        "s",
        "Longest continuous hover in any episode.",
        "maximize",
    ),
    MetricSpec(
        "hover_envelope_utilization_mean",
        "Mean worst pose error relative to hover limit (1 = limit)",
        "normalized",
        "Mean maximum measured pose-envelope utilization; 1 is the boundary.",
        "minimize",
    ),
    MetricSpec(
        "position_envelope_utilization_mean",
        "Mean position error relative to hover limit (1 = limit)",
        "normalized",
        "Mean measured position-envelope utilization; 1 is the boundary.",
        "minimize",
    ),
    MetricSpec(
        "attitude_envelope_utilization_mean",
        "Mean attitude error relative to hover limit (1 = limit)",
        "normalized",
        "Mean measured attitude-envelope utilization; 1 is the boundary.",
        "minimize",
    ),
    MetricSpec(
        "range_undershoot_rate",
        "Episodes crossing the minimum range (fraction)",
        "fraction",
        "Range-undershoot episodes.",
        "minimize",
    ),
    MetricSpec(
        "timeout_rate",
        "Episodes reaching the time limit (fraction)",
        "fraction",
        "Timed-out episodes.",
        "minimize",
    ),
    MetricSpec(
        "completed_episodes",
        "Completed episodes (count)",
        "count",
        "Completed episodes.",
        "maximize",
    ),
) + tuple(
    MetricSpec(
        f"flight_parameters.{signal_name}.{statistic}",
        f"hover statistics/{label}/{statistic_label} ({'meters' if unit == 'm' else 'radians'})",
        unit,
        f"Measured hover {signal_name} {statistic}.",
        "minimize",
    )
    for signal_name, (label, unit) in _HOVER_SIGNALS.items()
    for statistic, statistic_label in _BINDING_STATISTICS.items()
)

_CONTACT_METRICS = (
    ("contact_progress_mean", "Mean contact task progress", "normalized", "maximize"),
    ("contact_rate", "Episodes with intended button contact (fraction)", "fraction", "maximize"),
    (
        "stable_first_contact_rate",
        "Stable first contacts among contacts (fraction)",
        "fraction",
        "maximize",
    ),
    ("toggle_rate", "Episodes with at least one toggle (fraction)", "fraction", "maximize"),
    (
        "retreat_rate",
        "Successful retreats among toggled episodes (fraction)",
        "fraction",
        "maximize",
    ),
    ("multiple_toggle_rate", "Episodes with multiple toggles (fraction)", "fraction", "minimize"),
    ("peak_compression_m", "Maximum pusher compression (meters)", "m", "minimize"),
)
EVALUATION_METRICS += tuple(
    MetricSpec(name, label, unit, label, direction)
    for name, label, unit, direction in _CONTACT_METRICS
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
        raise ValueError("evaluation monitoring role must be 'val' or 'test'")
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
        namespace = "validation" if role == "val" else role
        tag = f"{namespace}/{spec.suffix}"
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
        tag = f"training/reset/{_RESET_LABELS[name]}"
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
    fields = {
        "gpu_utilization_rate_mean": "Mean GPU utilization (fraction)",
        "gpu_utilization_rate_p95": "95th percentile GPU utilization (fraction)",
        "gpu_memory_used_gib_peak": "Peak GPU memory used (GiB)",
        "gpu_memory_utilization_rate_peak": "Peak GPU memory utilization (fraction)",
        "gpu_power_w_mean": "Mean GPU power (watts)",
        "gpu_temperature_c_max": "Maximum GPU temperature (degrees Celsius)",
        "process_resident_memory_gib_peak": "Peak process memory (GiB)",
        "process_cpu_cores_used_mean": "Mean CPU cores used",
        "duration_s": "Measurement interval duration (seconds)",
        "work_items_per_s": "Completed episodes per second",
    }
    emitted = []
    for name, label in fields.items():
        value = float(summary[name])
        if not math.isfinite(value):
            raise ValueError(f"system telemetry metric {name!r} is not finite")
        if name.endswith(("_rate_mean", "_rate_p95", "_rate_peak")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"system telemetry rate {name!r} must be in [0, 1]")
        namespace = {"train": "training", "val": "validation", "test": "test"}[role]
        if name == "work_items_per_s" and role == "train":
            label = "Training transitions per second"
        tag = f"system/{namespace}/{label}"
        writer.add_scalar(tag, value, step, new_style=True, double_precision=True)
        emitted.append(tag)
    return tuple(emitted)

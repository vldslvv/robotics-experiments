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

METRIC_REGISTRY_REVISION = 15

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
    MetricSpec(None, "acceptance/all_gates_passed", "boolean", "All gates passed.", "maximize"),
    MetricSpec(
        "approach_success_rate",
        "outcome/success_rate",
        "fraction",
        "Successful episodes.",
        "maximize",
    ),
    MetricSpec(
        "collision_rate", "outcome/collision_rate", "fraction", "Collision episodes.", "minimize"
    ),
    MetricSpec(
        "other_flight_failure_rate",
        "outcome/flight_failure_rate",
        "fraction",
        "Other flight failures.",
        "minimize",
    ),
    MetricSpec(
        "range_undershoot_rate",
        "outcome/range_undershoot_rate",
        "fraction",
        "Range-undershoot episodes.",
        "minimize",
    ),
    MetricSpec(
        "timeout_rate", "outcome/timeout_rate", "fraction", "Timed-out episodes.", "minimize"
    ),
    MetricSpec(
        "completed_episodes",
        "outcome/completed_episodes",
        "count",
        "Completed episodes.",
        "maximize",
    ),
    MetricSpec(
        "range_pass_rate", "goal/range_pass_rate", "fraction", "Strict range samples.", "maximize"
    ),
    MetricSpec(
        "alignment_pass_rate",
        "goal/alignment_pass_rate",
        "fraction",
        "Strict alignment samples.",
        "maximize",
    ),
    MetricSpec(
        "attitude_pass_rate",
        "goal/attitude_pass_rate",
        "fraction",
        "Strict attitude samples.",
        "maximize",
    ),
    MetricSpec(
        "linear_speed_pass_rate",
        "goal/linear_speed_pass_rate",
        "fraction",
        "Strict linear-speed samples.",
        "maximize",
    ),
    MetricSpec(
        "angular_speed_pass_rate",
        "goal/angular_speed_pass_rate",
        "fraction",
        "Strict angular-speed samples.",
        "maximize",
    ),
    MetricSpec(
        "contact_clear_pass_rate",
        "goal/contact_clear_pass_rate",
        "fraction",
        "Contact-free samples.",
        "maximize",
    ),
    MetricSpec(
        "complete_goal_pass_rate",
        "goal/complete_pass_rate",
        "fraction",
        "Complete strict-goal samples.",
        "maximize",
    ),
    MetricSpec(
        "mean_max_stable_goal_fraction",
        "goal/mean_dwell_fraction",
        "fraction",
        "Mean maximum dwell fraction.",
        "maximize",
    ),
    MetricSpec(
        "maximum_stable_goal_fraction",
        "goal/maximum_dwell_fraction",
        "fraction",
        "Maximum dwell fraction.",
        "maximize",
    ),
    MetricSpec(
        "mean_normalized_pose_error",
        "state/mean_normalized_pose_error",
        "normalized",
        "Mean normalized pose error.",
        "minimize",
    ),
    MetricSpec(
        "minimum_closest_range_m",
        "state/minimum_closest_range_m",
        "m",
        "Minimum closest range.",
        "maximize",
    ),
    MetricSpec(
        "mean_closest_range_m", "state/mean_closest_range_m", "m", "Mean closest range.", "maximize"
    ),
    MetricSpec(
        "all_traces_finite", "trace/all_finite", "boolean", "All traces are finite.", "maximize"
    ),
    MetricSpec("held_out_task_cost", "cost/total", "cost", "Total held-out task cost.", "minimize"),
    MetricSpec(
        "body_collision_rate",
        "outcome/body_collision_rate",
        "fraction",
        "Body-collision episodes.",
        "minimize",
    ),
    MetricSpec(
        "pusher_collision_rate",
        "outcome/pusher_collision_rate",
        "fraction",
        "Pusher-collision episodes.",
        "minimize",
    ),
    MetricSpec(
        "median_successful_completion_time_s",
        "outcome/median_successful_completion_time_s",
        "s",
        "Median successful completion time.",
        "minimize",
    ),
    MetricSpec(
        "mean_terminal_range_cost",
        "cost/terminal_range",
        "cost",
        "Terminal-range cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_trajectory_alignment_cost",
        "cost/trajectory_alignment",
        "cost",
        "Trajectory-alignment cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_relative_attitude_error_cost",
        "cost/relative_attitude_error",
        "cost",
        "Relative-attitude cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_velocity_tracking_cost",
        "cost/velocity_tracking",
        "cost",
        "Velocity-tracking cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_angular_velocity_cost",
        "cost/angular_velocity",
        "cost",
        "Angular-velocity cost.",
        "minimize",
    ),
    MetricSpec(
        "mean_upright_error_cost", "cost/upright_error", "cost", "Upright-error cost.", "minimize"
    ),
    MetricSpec(
        "mean_action_delta_cost", "cost/action_delta", "cost", "Action-change cost.", "minimize"
    ),
    MetricSpec(
        "mean_completion_time_fraction",
        "cost/completion_time_fraction",
        "fraction",
        "Completion-time fraction.",
        "minimize",
    ),
) + tuple(
    MetricSpec(
        f"flight_parameters.{signal_name}.{statistic}",
        f"flight_parameters/{signal_name}/{statistic}",
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
        "gpu_utilization_rate_mean",
        "gpu_utilization_rate_p95",
        "gpu_memory_used_gib_peak",
        "gpu_memory_utilization_rate_peak",
        "gpu_power_w_mean",
        "gpu_temperature_c_max",
        "process_resident_memory_gib_peak",
        "process_cpu_cores_used_mean",
        "duration_s",
        "work_items_per_s",
    )
    emitted = []
    for name in fields:
        value = float(summary[name])
        if not math.isfinite(value):
            raise ValueError(f"system telemetry metric {name!r} is not finite")
        if name.endswith("_rate_mean") or name.endswith("_rate_p95") or name.endswith("_rate_peak"):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"system telemetry rate {name!r} must be in [0, 1]")
        tag = f"system/{role}/{name}"
        writer.add_scalar(tag, value, step, new_style=True, double_precision=True)
        emitted.append(tag)
    return tuple(emitted)

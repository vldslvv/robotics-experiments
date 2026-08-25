"""Press-policy acceptance criteria and decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics_experiments.config import PressApproachAcceptanceConfig
from robotics_experiments.flight_parameters import FlightParameters, SignalStatistics
from robotics_experiments.rl.evaluation import PolicyEvaluation

__all__ = [
    "AcceptanceCheck",
    "AcceptanceResult",
    "PressAcceptanceCriteria",
    "create_press_acceptance_criteria",
    "evaluate_press_acceptance",
]

_BINDING_STATISTICS = (
    ("max", "maximum_absolute"),
    ("rmse", "rmse"),
    ("p95", "p95_absolute"),
    ("p99", "p99_absolute"),
)


@dataclass(frozen=True)
class PressAcceptanceCriteria:
    """Requirements fixed before a press-policy campaign starts."""

    success_rate_min: float
    collision_rate_max: float
    minimum_range_m: float
    hover_limits: tuple[SignalStatistics, ...]


@dataclass(frozen=True)
class AcceptanceCheck:
    """Result of applying one acceptance requirement."""

    name: str
    passed: bool
    actual: float | None
    comparison: str
    required: float


@dataclass(frozen=True)
class AcceptanceResult:
    """Acceptance decision for one completed final evaluation."""

    passed: bool
    checks: tuple[AcceptanceCheck, ...]
    reasons: tuple[str, ...]


def create_press_acceptance_criteria(
    *,
    config: PressApproachAcceptanceConfig,
    flight_parameters: FlightParameters,
) -> PressAcceptanceCriteria:
    """Combine task requirements with the pinned measured hover limits."""

    return PressAcceptanceCriteria(
        success_rate_min=float(config.success_rate_min),
        collision_rate_max=float(config.collision_rate_max),
        minimum_range_m=float(config.minimum_range_m),
        hover_limits=flight_parameters.hover.acceptance_limits,
    )


def evaluate_press_acceptance(
    *,
    criteria: PressAcceptanceCriteria,
    evaluation: PolicyEvaluation,
) -> AcceptanceResult:
    """Apply the campaign's fixed criteria to one raw policy evaluation."""

    requirements = (
        ("trace finiteness", evaluation.metric("all_traces_finite"), "==", 1.0),
        (
            "approach success rate",
            evaluation.metric("approach_success_rate"),
            ">=",
            criteria.success_rate_min,
        ),
        (
            "collision rate",
            evaluation.metric("collision_rate"),
            "<=",
            criteria.collision_rate_max,
        ),
        (
            "minimum closest range",
            evaluation.metric("minimum_closest_range_m"),
            ">=",
            criteria.minimum_range_m,
        ),
        (
            "other flight-failure rate",
            evaluation.metric("other_flight_failure_rate"),
            "==",
            0.0,
        ),
    ) + tuple(
        (
            f"flight parameters {limit.name} {metric_name}",
            next(
                (
                    float(metric.value)
                    for metric in evaluation.metrics
                    if metric.name == f"flight_parameters.{limit.name}.{metric_name}"
                    and metric.value is not None
                    and np.isfinite(metric.value)
                ),
                None,
            ),
            "<=",
            float(getattr(limit, field_name)),
        )
        for limit in criteria.hover_limits
        for metric_name, field_name in _BINDING_STATISTICS
    )
    checks = tuple(
        AcceptanceCheck(
            name=name,
            passed=(
                actual is not None
                and (
                    actual == required
                    if comparison == "=="
                    else actual >= required
                    if comparison == ">="
                    else actual <= required
                )
            ),
            actual=actual,
            comparison=comparison,
            required=required,
        )
        for name, actual, comparison, required in requirements
    )
    reasons = tuple(check.name for check in checks if not check.passed)
    return AcceptanceResult(passed=not reasons, checks=checks, reasons=reasons)

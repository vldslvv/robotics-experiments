"""Press-policy acceptance criteria and decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics_experiments.config import PressTaskConfig
from robotics_experiments.flight_parameters import FlightParameters
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
class _Requirement:
    name: str
    metric: str
    comparison: str
    required: float


@dataclass(frozen=True)
class PressAcceptanceCriteria:
    """Fixed physical requirements, independent of reward and stage labels."""

    requirements: tuple[_Requirement, ...]


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
    config: PressTaskConfig,
    flight_parameters: FlightParameters,
) -> PressAcceptanceCriteria:
    """Combine task requirements with the pinned measured hover limits."""

    acceptance = config.acceptance
    requirements = [
        _Requirement("trace finiteness", "all_traces_finite", "==", 1.0),
        _Requirement("success rate", "success_rate", ">=", acceptance.success_rate_min),
        _Requirement("collision rate", "collision_rate", "<=", acceptance.collision_rate_max),
        _Requirement(
            "minimum closest range", "minimum_closest_range_m", ">=", acceptance.minimum_range_m
        ),
        _Requirement("other flight-failure rate", "other_flight_failure_rate", "==", 0.0),
    ]
    if config.task in {"contact", "press"}:
        requirements.extend(
            (
                _Requirement(
                    "stable first contact",
                    "stable_first_contact_rate",
                    ">=",
                    config.contact.first_contact_rate_min,
                ),
                _Requirement(
                    "pusher compression",
                    "peak_compression_m",
                    "<=",
                    config.contact.compression_max_m,
                ),
                _Requirement("multiple toggles", "multiple_toggle_rate", "==", 0.0),
            )
        )
        if config.task == "press":
            requirements.append(
                _Requirement(
                    "retreat among toggles", "retreat_rate", ">=", config.contact.retreat_rate_min
                )
            )
    else:
        requirements.extend(
            _Requirement(
                f"flight parameters {limit.name} {metric_name}",
                f"flight_parameters.{limit.name}.{metric_name}",
                "<=",
                float(getattr(limit, field_name)),
            )
            for limit in flight_parameters.hover.acceptance_limits
            for metric_name, field_name in _BINDING_STATISTICS
        )
    return PressAcceptanceCriteria(requirements=tuple(requirements))


def evaluate_press_acceptance(
    *,
    criteria: PressAcceptanceCriteria,
    evaluation: PolicyEvaluation,
) -> AcceptanceResult:
    """Apply the campaign's fixed criteria to one raw policy evaluation."""

    metrics = {metric.name: metric.value for metric in evaluation.metrics}
    checks = []
    for requirement in criteria.requirements:
        actual = metrics.get(requirement.metric)
        passed = (
            actual is not None
            and np.isfinite(actual)
            and (
                actual == requirement.required
                if requirement.comparison == "=="
                else actual >= requirement.required
                if requirement.comparison == ">="
                else actual <= requirement.required
            )
        )
        checks.append(
            AcceptanceCheck(
                name=requirement.name,
                passed=bool(passed),
                actual=actual,
                comparison=requirement.comparison,
                required=requirement.required,
            )
        )
    reasons = tuple(check.name for check in checks if not check.passed)
    return AcceptanceResult(passed=not reasons, checks=tuple(checks), reasons=reasons)

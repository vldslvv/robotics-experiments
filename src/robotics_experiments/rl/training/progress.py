"""Evaluation-result and phase-progress contracts for scratch policy training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from robotics_experiments.config import RlPlateauCriteriaConfig

__all__ = [
    "AcceptanceResult",
    "EvaluationMetric",
    "PolicyEvaluation",
    "PlateauState",
    "PlateauUpdate",
    "RewardMean",
    "advance_plateau",
    "select_best_evaluation",
]


@dataclass(frozen=True)
class AcceptanceResult:
    """Final acceptance decision made from one selected evaluation record."""

    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationMetric:
    """One named scalar returned by a deterministic policy evaluator."""

    name: str
    value: float | None


@dataclass(frozen=True)
class RewardMean:
    """One training-reward diagnostic averaged over held-out transitions."""

    name: str
    mean: float


@dataclass(frozen=True)
class PolicyEvaluation:
    """Complete evaluator output with no plateau or acceptance decision."""

    iteration: int
    checkpoint_path: Path
    policy_digest: str
    metrics: tuple[EvaluationMetric, ...]
    reward_means: tuple[RewardMean, ...]
    source: str

    def metric(self, name: str, /) -> float:
        """Return one required finite scalar by its stable registered name."""
        match = next((metric.value for metric in self.metrics if metric.name == name), None)
        if match is None or not math.isfinite(match):
            raise ValueError(f"policy evaluation has no finite metric {name!r}")
        return match


@dataclass(frozen=True)
class PlateauState:
    """Phase-local material-progress reference and consecutive stale count."""

    reference_value: float
    stale_evaluations: int


@dataclass(frozen=True)
class PlateauUpdate:
    """Result of observing one evaluation with a declarative plateau rule."""

    state: PlateauState
    made_progress: bool
    plateau_reached: bool


def _improvement(
    *,
    direction: str,
    reference_value: float,
    current_value: float,
) -> float:
    if direction == "minimize":
        return reference_value - current_value
    if direction == "maximize":
        return current_value - reference_value
    raise ValueError(f"unsupported plateau direction {direction!r}")


def advance_plateau(
    *,
    criteria: RlPlateauCriteriaConfig,
    state: PlateauState,
    evaluation: PolicyEvaluation,
) -> PlateauUpdate:
    """Advance one phase uniformly using only its configured progress scalar."""
    current = evaluation.metric(criteria.metric_name)
    if math.isnan(state.reference_value):
        return PlateauUpdate(
            state=PlateauState(reference_value=current, stale_evaluations=0),
            made_progress=True,
            plateau_reached=False,
        )
    made_progress = (
        _improvement(
            direction=criteria.direction,
            reference_value=state.reference_value,
            current_value=current,
        )
        >= criteria.minimum_delta
    )
    next_state = PlateauState(
        reference_value=current if made_progress else state.reference_value,
        stale_evaluations=0 if made_progress else state.stale_evaluations + 1,
    )
    return PlateauUpdate(
        state=next_state,
        made_progress=made_progress,
        plateau_reached=next_state.stale_evaluations >= criteria.patience_evaluations,
    )


def select_best_evaluation(
    *,
    criteria: RlPlateauCriteriaConfig,
    evaluations: tuple[PolicyEvaluation, ...],
) -> PolicyEvaluation:
    """Select the run-wide best scalar, preferring the earlier exact tie."""
    if not evaluations:
        raise ValueError("policy selection requires at least one evaluation")
    multiplier = -1.0 if criteria.direction == "minimize" else 1.0
    return max(
        evaluations,
        key=lambda evaluation: (
            multiplier * evaluation.metric(criteria.metric_name),
            -evaluation.iteration,
        ),
    )

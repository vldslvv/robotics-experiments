"""Plateau and checkpoint-selection behavior for policy training."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robotics_experiments.config import RlPlateauCriteriaConfig
from robotics_experiments.rl.evaluation import PolicyEvaluation

__all__ = [
    "PlateauState",
    "PlateauUpdate",
    "advance_plateau",
    "select_best_evaluation",
]


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
    eligible_policy_digests: frozenset[str],
) -> PolicyEvaluation:
    """Rank eligible candidates first, then scalar, preserving exact ties."""
    if not evaluations:
        raise ValueError("policy selection requires at least one evaluation")
    multiplier = -1.0 if criteria.direction == "minimize" else 1.0
    return max(
        evaluations,
        key=lambda evaluation: (
            evaluation.policy_digest in eligible_policy_digests,
            multiplier * evaluation.metric(criteria.metric_name),
        ),
    )

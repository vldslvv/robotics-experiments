"""Task-neutral checkpoint enumeration, scoring, and selection."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CandidateEvaluation",
    "CheckpointCandidate",
    "CheckpointSelection",
    "CheckpointScorer",
    "MetricValue",
    "RewardMean",
    "enumerate_checkpoints",
    "publish_latest",
    "select_checkpoint",
]

_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt")


@dataclass(frozen=True)
class CheckpointCandidate:
    """One numerically identified rsl_rl checkpoint."""

    iteration: int
    path: Path


@dataclass(frozen=True)
class MetricValue:
    """One named task metric retained in a candidate result."""

    name: str
    value: float | None


@dataclass(frozen=True)
class RewardMean:
    """Mean value for one task reward term."""

    name: str
    mean: float


@dataclass(frozen=True)
class CandidateEvaluation:
    """Complete scorer output for one checkpoint candidate."""

    candidate: CheckpointCandidate
    accepted: bool
    metrics: tuple[MetricValue, ...]
    normalized_margins: tuple[float, float, float]
    ranking_values: tuple[float, ...]
    reward_means: tuple[RewardMean, ...]


@dataclass(frozen=True)
class CheckpointSelection:
    """All scorer results and the selected checkpoint."""

    selection_method: str
    evaluations: tuple[CandidateEvaluation, ...]
    selected: CheckpointCandidate


CheckpointScorer = Callable[
    [tuple[CheckpointCandidate, ...]],
    tuple[CandidateEvaluation, ...],
]


def enumerate_checkpoints(*, logs_dir: Path) -> tuple[CheckpointCandidate, ...]:
    """Return every numeric ``model_<iteration>.pt`` checkpoint in iteration order."""
    candidates = []
    for path in logs_dir.glob("model_*.pt"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None and path.is_file():
            candidates.append(CheckpointCandidate(iteration=int(match.group(1)), path=path))
    candidates.sort(key=lambda candidate: candidate.iteration)
    if not candidates:
        raise ValueError(f"no numeric model_<iteration>.pt checkpoints found in {logs_dir}")
    return tuple(candidates)


def select_checkpoint(
    *,
    policy_interface: str,
    candidates: tuple[CheckpointCandidate, ...],
    selection_method: str = "final-no-scorer",
    scorer: CheckpointScorer | None = None,
) -> CheckpointSelection:
    """Select from explicit scorer results, or use the final checkpoint."""
    if not candidates:
        raise ValueError("checkpoint selection requires at least one candidate")
    if scorer is None:
        final = candidates[-1]
        warnings.warn(
            f"no checkpoint scorer provided for policy_interface={policy_interface!r}; "
            f"selecting final checkpoint iteration {final.iteration}",
            RuntimeWarning,
            stacklevel=2,
        )
        return CheckpointSelection(
            selection_method="final-no-scorer",
            evaluations=(),
            selected=final,
        )

    evaluations = scorer(candidates)
    if tuple(evaluation.candidate for evaluation in evaluations) != candidates:
        raise ValueError("checkpoint scorer did not evaluate every candidate in order")
    accepted = tuple(evaluation for evaluation in evaluations if evaluation.accepted)
    ranked = accepted or evaluations
    selected = max(ranked, key=lambda evaluation: evaluation.ranking_values)
    return CheckpointSelection(
        selection_method=selection_method,
        evaluations=evaluations,
        selected=selected.candidate,
    )


def publish_latest(*, policy_root: Path, run_dir: Path) -> None:
    """Atomically advance the policy namespace's relative ``latest`` symlink."""
    link = policy_root / "latest"
    temp = policy_root / f".latest.{os.getpid()}.tmp"
    temp.unlink(missing_ok=True)
    temp.symlink_to(run_dir.name)
    temp.replace(link)

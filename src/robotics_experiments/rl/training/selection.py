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
    "MetricValue",
    "RewardMean",
    "SelectedCheckpoint",
    "enumerate_checkpoints",
    "publish_latest",
    "register_checkpoint_scorer",
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
class SelectedCheckpoint:
    """The checkpoint chosen for artifact export."""

    iteration: int
    path: Path


@dataclass(frozen=True)
class CheckpointSelection:
    """All scorer results and the selected checkpoint, when one is eligible."""

    selection_method: str
    candidates: tuple[CandidateEvaluation, ...]
    selected: SelectedCheckpoint | None


CheckpointScorer = Callable[
    [tuple[CheckpointCandidate, ...]],
    tuple[CandidateEvaluation, ...],
]

_SCORERS: dict[str, tuple[str, CheckpointScorer]] = {}


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


def register_checkpoint_scorer(
    *,
    policy_interface: str,
    selection_method: str,
    scorer: CheckpointScorer,
) -> None:
    """Register the checkpoint scorer for one artifact policy interface."""
    _SCORERS[policy_interface] = (selection_method, scorer)


def select_checkpoint(
    *,
    policy_interface: str,
    candidates: tuple[CheckpointCandidate, ...],
) -> CheckpointSelection:
    """Score candidates or select the final checkpoint when no scorer exists."""
    if not candidates:
        raise ValueError("checkpoint selection requires at least one candidate")
    registration = _SCORERS.get(policy_interface)
    if registration is None:
        final = candidates[-1]
        warnings.warn(
            f"no checkpoint scorer registered for policy_interface={policy_interface!r}; "
            f"selecting final checkpoint iteration {final.iteration}",
            RuntimeWarning,
            stacklevel=2,
        )
        return CheckpointSelection(
            selection_method="final-no-scorer",
            candidates=(),
            selected=SelectedCheckpoint(iteration=final.iteration, path=final.path),
        )

    selection_method, scorer = registration
    evaluations = scorer(candidates)
    if tuple(evaluation.candidate for evaluation in evaluations) != candidates:
        raise ValueError("checkpoint scorer did not evaluate every candidate in order")
    eligible = tuple(evaluation for evaluation in evaluations if evaluation.accepted)
    selected = max(eligible, key=lambda evaluation: evaluation.ranking_values) if eligible else None
    return CheckpointSelection(
        selection_method=selection_method,
        candidates=evaluations,
        selected=(
            SelectedCheckpoint(
                iteration=selected.candidate.iteration,
                path=selected.candidate.path,
            )
            if selected is not None
            else None
        ),
    )


def publish_latest(*, policy_root: Path, run_dir: Path) -> None:
    """Atomically advance the policy namespace's relative ``latest`` symlink."""
    link = policy_root / "latest"
    temp = policy_root / f".latest.{os.getpid()}.tmp"
    temp.unlink(missing_ok=True)
    temp.symlink_to(run_dir.name)
    temp.replace(link)

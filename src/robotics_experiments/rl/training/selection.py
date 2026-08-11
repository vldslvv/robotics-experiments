"""Task-neutral checkpoint enumeration, scoring, and selection."""

from __future__ import annotations

import json
import math
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
    "publish_policy_labels",
    "select_checkpoint",
]

_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt")
_SELECTION_REPORT = "checkpoint_selection.json"


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
    normalized_margins: tuple[float, ...]
    ranking_values: tuple[float, ...]
    reward_means: tuple[RewardMean, ...]
    source: str = "fresh"


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
    selected = max(evaluations, key=lambda evaluation: evaluation.ranking_values)
    return CheckpointSelection(
        selection_method=selection_method,
        evaluations=evaluations,
        selected=selected.candidate,
    )


def _publish_label(*, policy_root: Path, run_dir: Path, label: str) -> None:
    link = policy_root / label
    temp = policy_root / f".{label}.{os.getpid()}.tmp"
    temp.unlink(missing_ok=True)
    temp.symlink_to(run_dir.name)
    temp.replace(link)


def _comparable_best_score(
    *,
    policy_root: Path,
    comparison_fingerprint: str,
) -> float | None:
    best = policy_root / "best"
    if not best.is_symlink():
        return None
    target = Path(os.readlink(best))
    if target.is_absolute() or len(target.parts) != 1:
        return None
    report_path = policy_root / target / _SELECTION_REPORT
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("comparison_fingerprint") != comparison_fingerprint:
        return None
    score = report.get("selected_score")
    if not isinstance(score, int | float) or isinstance(score, bool):
        return None
    score = float(score)
    return score if math.isfinite(score) else None


def publish_policy_labels(
    *,
    policy_root: Path,
    run_dir: Path,
    comparison_fingerprint: str | None,
    score: float | None,
) -> bool:
    """Advance ``latest`` and conditionally advance comparable ``best``.

    Returns:
        Whether ``best`` advanced. Policy families without a comparison metric
        pass ``None`` for both comparison values and publish only ``latest``.
    """
    if (comparison_fingerprint is None) != (score is None):
        raise ValueError(
            "comparison fingerprint and score must either both be set or both be absent"
        )
    if score is not None and not math.isfinite(score):
        raise ValueError("policy comparison score must be finite")
    best_score = (
        None
        if comparison_fingerprint is None
        else _comparable_best_score(
            policy_root=policy_root,
            comparison_fingerprint=comparison_fingerprint,
        )
    )
    advance_best = score is not None and (best_score is None or score > best_score)
    _publish_label(policy_root=policy_root, run_dir=run_dir, label="latest")
    if advance_best:
        _publish_label(policy_root=policy_root, run_dir=run_dir, label="best")
    return advance_best

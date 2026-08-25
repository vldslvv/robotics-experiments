"""Task-neutral policy evaluation records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

__all__ = ["EvaluationMetric", "PolicyEvaluation", "RewardMean"]


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
    """Complete evaluator output without selection or acceptance decisions."""

    iteration: int
    checkpoint_path: Path
    policy_digest: str
    metrics: tuple[EvaluationMetric, ...]
    reward_means: tuple[RewardMean, ...]
    source: str

    def metric(self, name: str, /) -> float:
        """Return one required finite scalar by its registered name."""

        match = next((metric.value for metric in self.metrics if metric.name == name), None)
        if match is None or not math.isfinite(match):
            raise ValueError(f"policy evaluation has no finite metric {name!r}")
        return match

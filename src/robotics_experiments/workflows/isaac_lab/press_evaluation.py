"""Deterministic approach-policy rollout shared by selection and publication."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Protocol

import numpy as np
import torch

from robotics_experiments.rl.press import PressApproachResult, PressApproachRewardTerms
from robotics_experiments.rl.training.selection import RewardMean

__all__ = [
    "PressApproachEpisodeMetrics",
    "PressApproachEvaluationResult",
    "PressApproachEvaluationSummary",
    "evaluate_press_approach_policy",
    "summarize_press_approach_episodes",
]


@dataclass(frozen=True)
class PressApproachEpisodeMetrics:
    """Outcome and diagnostic values retained for one completed episode."""

    approach_success: bool
    contact_failure: bool
    other_flight_failure: bool
    timeout: bool
    closest_range_m: float
    completion_time_s: float | None


@dataclass(frozen=True)
class PressApproachEvaluationSummary:
    """Aggregated approach acceptance and ranking metrics."""

    episodes: int
    approach_success_rate: float
    contact_failure_rate: float
    other_flight_failure_rate: float
    timeout_rate: float
    mean_closest_range_m: float
    median_completion_time_s: float | None


@dataclass(frozen=True)
class PressApproachEvaluationResult:
    """Complete deterministic evaluation result for one actor."""

    summary: PressApproachEvaluationSummary
    reward_means: tuple[RewardMean, ...]


class _PressApproachPolicy(Protocol):
    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """Return one normalized action batch."""
        ...


class _PressApproachEvaluationEnvironment(Protocol):
    device: str | torch.device
    max_episode_length: int
    step_dt: float
    press_approach_result: PressApproachResult[torch.Tensor] | None

    def reset(
        self,
        *,
        seed: int,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset the vector environment with a deterministic seed."""
        ...

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Advance the vector environment by one policy step."""
        ...


def summarize_press_approach_episodes(
    *,
    episodes: tuple[PressApproachEpisodeMetrics, ...],
) -> PressApproachEvaluationSummary:
    """Aggregate Stage 1 metrics over their all-episode denominators."""
    total = len(episodes)
    if total == 0:
        raise ValueError("press approach evaluation requires at least one episode")
    completion_times = tuple(
        episode.completion_time_s for episode in episodes if episode.completion_time_s is not None
    )
    return PressApproachEvaluationSummary(
        episodes=total,
        approach_success_rate=sum(episode.approach_success for episode in episodes) / total,
        contact_failure_rate=sum(episode.contact_failure for episode in episodes) / total,
        other_flight_failure_rate=(
            sum(episode.other_flight_failure for episode in episodes) / total
        ),
        timeout_rate=sum(episode.timeout for episode in episodes) / total,
        mean_closest_range_m=float(np.mean([episode.closest_range_m for episode in episodes])),
        median_completion_time_s=(float(np.median(completion_times)) if completion_times else None),
    )


def _evaluate_seed(
    *,
    env: _PressApproachEvaluationEnvironment,
    policy: _PressApproachPolicy,
    seed: int,
    episode_count: int,
    report: Callable[[str], None],
) -> tuple[tuple[PressApproachEpisodeMetrics, ...], dict[str, tuple[float, int]]]:
    reward_names = tuple(field.name for field in fields(PressApproachRewardTerms)) + ("total",)
    reward_sums = torch.zeros(len(reward_names), dtype=torch.float32, device=env.device)
    reward_sample_count = torch.zeros((), dtype=torch.long, device=env.device)
    active = torch.ones(episode_count, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(active)
    contact_failure = torch.zeros_like(active)
    other_flight_failure = torch.zeros_like(active)
    timeout = torch.zeros_like(active)
    closest_range = torch.full(
        (episode_count,),
        float("inf"),
        dtype=torch.float32,
        device=env.device,
    )
    completion_time = torch.full_like(closest_range, float("nan"))
    reset_result = env.reset(seed=seed)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    horizon = int(env.max_episode_length)
    progress_interval = max(1, horizon // 10)
    started_at = time.monotonic()
    report(
        f"press approach evaluation seed={seed} START "
        f"episodes={episode_count} horizon={horizon}"
    )
    for step in range(horizon):
        with torch.no_grad():
            action = policy(observation["policy"]).clamp(-1.0, 1.0)
        observation, _reward, terminated, truncated, _info = env.step(action)
        result = env.press_approach_result
        if result is None:
            raise RuntimeError("press approach environment did not publish a transition result")
        termination = result.termination
        closest_range = torch.minimum(
            closest_range,
            torch.where(active, result.state.closest_range_m, closest_range),
        )
        newly_successful = active & termination.success
        completion_time = torch.where(
            newly_successful,
            torch.full_like(completion_time, (step + 1) * float(env.step_dt)),
            completion_time,
        )
        success |= newly_successful
        contact_failure |= active & termination.contact
        other_flight_failure |= active & termination.flight_failure
        timeout |= active & termination.time_out
        reward_values = torch.stack(
            [getattr(result.rewards, name) for name in reward_names],
            dim=0,
        )
        reward_sums += reward_values[:, active].sum(dim=1)
        reward_sample_count += active.sum()
        active &= ~(terminated | truncated)
        completed_steps = step + 1
        if completed_steps % progress_interval == 0 or completed_steps == horizon:
            counts = (
                torch.stack(
                    (
                        active.sum(),
                        success.sum(),
                        contact_failure.sum(),
                        other_flight_failure.sum(),
                        timeout.sum(),
                    )
                )
                .cpu()
                .tolist()
            )
            report(
                f"press approach evaluation seed={seed} "
                f"progress={completed_steps}/{horizon} "
                f"elapsed_s={time.monotonic() - started_at:.1f} active={counts[0]} "
                f"success={counts[1]} contact={counts[2]} "
                f"flight_failure={counts[3]} timeout={counts[4]}"
            )
    if bool(torch.any(active)):
        raise RuntimeError("press approach evaluation exceeded the configured episode horizon")

    episodes = tuple(
        PressApproachEpisodeMetrics(
            approach_success=bool(success[index].item()),
            contact_failure=bool(contact_failure[index].item()),
            other_flight_failure=bool(other_flight_failure[index].item()),
            timeout=bool(timeout[index].item()),
            closest_range_m=float(closest_range[index].item()),
            completion_time_s=(
                float(completion_time[index].item()) if bool(success[index].item()) else None
            ),
        )
        for index in range(episode_count)
    )
    reward_sum_values = reward_sums.cpu().tolist()
    sample_count = int(reward_sample_count.item())
    summary = summarize_press_approach_episodes(episodes=episodes)
    report(
        f"press approach evaluation seed={seed} COMPLETE "
        f"elapsed_s={time.monotonic() - started_at:.1f} "
        f"success={summary.approach_success_rate:.3f} "
        f"contact_failure={summary.contact_failure_rate:.3f} "
        f"other_flight_failure={summary.other_flight_failure_rate:.3f} "
        f"timeout={summary.timeout_rate:.3f}"
    )
    return episodes, {
        name: (float(value), sample_count)
        for name, value in zip(reward_names, reward_sum_values, strict=True)
    }


def evaluate_press_approach_policy(
    *,
    env: _PressApproachEvaluationEnvironment,
    policy: _PressApproachPolicy,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    report: Callable[[str], None],
) -> PressApproachEvaluationResult:
    """Evaluate one deterministic approach actor on all configured seeds."""
    episodes: list[PressApproachEpisodeMetrics] = []
    rewards: dict[str, tuple[float, int]] = {}
    for seed in seeds:
        seed_episodes, seed_rewards = _evaluate_seed(
            env=env,
            policy=policy,
            seed=seed,
            episode_count=episodes_per_seed,
            report=report,
        )
        episodes.extend(seed_episodes)
        for name, (value, count) in seed_rewards.items():
            previous_value, previous_count = rewards.get(name, (0.0, 0))
            rewards[name] = previous_value + value, previous_count + count
    return PressApproachEvaluationResult(
        summary=summarize_press_approach_episodes(episodes=tuple(episodes)),
        reward_means=tuple(
            RewardMean(name=name, mean=value / count) for name, (value, count) in rewards.items()
        ),
    )

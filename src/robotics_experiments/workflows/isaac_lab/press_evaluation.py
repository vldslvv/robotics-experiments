"""Deterministic press-policy rollout and aggregation shared by selection and gates.

This Isaac-Lab workflow leaf imports Torch at module scope. It stays behind the
lazy :mod:`robotics_experiments.workflows.isaac_lab` package boundary so core
imports remain usable without the training dependency.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from robotics_experiments.rl.press import PressRewardTerms
from robotics_experiments.rl.training.selection import RewardMean

__all__ = [
    "PressEpisodeMetrics",
    "PressEvaluationResult",
    "PressEvaluationSummary",
    "evaluate_press_policy",
    "summarize_press_episodes",
]


@dataclass(frozen=True)
class PressEpisodeMetrics:
    """Acceptance observations retained for one completed episode."""

    contacted: bool
    first_contact_stable: bool
    toggled: bool
    retreated: bool
    toggle_time_s: float
    peak_compression_overshoot_m: float


@dataclass(frozen=True)
class PressEvaluationSummary:
    """Aggregated episode count and acceptance metrics."""

    episodes: int
    confirmed_toggle_rate: float
    stable_first_contact_rate: float
    post_toggle_retreat_rate: float
    full_completion_rate: float
    peak_compression_overshoot_m: float
    median_toggle_time_s: float | None


@dataclass(frozen=True)
class PressEvaluationResult:
    """Complete deterministic evaluation result for one policy."""

    summary: PressEvaluationSummary
    reward_means: tuple[RewardMean, ...]


def summarize_press_episodes(
    *, episodes: tuple[PressEpisodeMetrics, ...]
) -> PressEvaluationSummary:
    """Aggregate press acceptance metrics with their contract denominators."""
    total = len(episodes)
    if total == 0:
        raise ValueError("press evaluation requires at least one episode")
    contacted = tuple(episode for episode in episodes if episode.contacted)
    toggled = tuple(episode for episode in episodes if episode.toggled)
    toggle_times = tuple(episode.toggle_time_s for episode in toggled)
    return PressEvaluationSummary(
        episodes=total,
        confirmed_toggle_rate=sum(episode.toggled for episode in episodes) / total,
        stable_first_contact_rate=(
            sum(episode.first_contact_stable for episode in contacted) / len(contacted)
            if contacted
            else 0.0
        ),
        post_toggle_retreat_rate=(
            sum(episode.retreated for episode in toggled) / len(toggled) if toggled else 0.0
        ),
        full_completion_rate=sum(episode.retreated for episode in episodes) / total,
        median_toggle_time_s=(float(np.median(toggle_times)) if toggle_times else None),
        peak_compression_overshoot_m=max(
            (episode.peak_compression_overshoot_m for episode in episodes),
            default=0.0,
        ),
    )


def _evaluate_seed(
    *,
    env: Any,
    policy: Any,
    seed: int,
    episode_count: int,
    report: Callable[[str], None],
) -> tuple[tuple[PressEpisodeMetrics, ...], dict[str, tuple[float, int]]]:
    reward_names = tuple(field.name for field in fields(PressRewardTerms))
    reward_sums = torch.zeros(len(reward_names), dtype=torch.float32, device=env.device)
    reward_sample_count = torch.zeros((), dtype=torch.long, device=env.device)
    active = torch.ones(episode_count, dtype=torch.bool, device=env.device)
    contacted = torch.zeros_like(active)
    first_contact_stable = torch.zeros_like(active)
    toggled = torch.zeros_like(active)
    retreated = torch.zeros_like(active)
    toggle_time = torch.full((episode_count,), float("nan"), dtype=torch.float32, device=env.device)
    peak_overshoot = torch.zeros(episode_count, dtype=torch.float32, device=env.device)
    reset_result = env.reset(seed=seed)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    horizon = int(env.max_episode_length)
    progress_interval = max(1, horizon // 10)
    started_at = time.monotonic()
    report(f"press evaluation seed={seed} START episodes={episode_count} horizon={horizon}")
    for step in range(horizon):
        with torch.no_grad():
            action = policy(observation["policy"]).clamp(-1.0, 1.0)
        observation, _reward, terminated, truncated, _info = env.step(action)
        result = env.press_task_result
        if result is None:
            raise RuntimeError("press environment did not publish a transition result")
        metrics = result.metrics
        contacted |= metrics.contacted & active
        first_contact_stable |= metrics.first_contact_stable & active
        first_toggle = metrics.toggled_this_step & active & ~toggled
        toggle_time = torch.where(first_toggle, metrics.toggle_time_s, toggle_time)
        toggled |= metrics.toggled & active
        retreated |= metrics.retreated & active
        peak_overshoot = torch.maximum(
            peak_overshoot,
            torch.where(active, metrics.peak_compression_overshoot_m, 0.0),
        )
        reward_values = torch.stack(
            [getattr(result.rewards, name) for name in reward_names],
            dim=0,
        )
        reward_sums += reward_values[:, active].sum(dim=1)
        reward_sample_count += active.sum()
        active &= ~(terminated | truncated)
        completed_steps = step + 1
        if completed_steps % progress_interval == 0 or completed_steps == horizon:
            active_count, contact_count, toggle_count, retreat_count = (
                torch.stack((active.sum(), contacted.sum(), toggled.sum(), retreated.sum()))
                .cpu()
                .tolist()
            )
            report(
                f"press evaluation seed={seed} progress={completed_steps}/{horizon} "
                f"elapsed_s={time.monotonic() - started_at:.1f} active={active_count} "
                f"contacted={contact_count} toggled={toggle_count} retreated={retreat_count}"
            )
    if bool(torch.any(active)):
        raise RuntimeError("press evaluation exceeded the configured episode horizon")

    episodes = tuple(
        PressEpisodeMetrics(
            contacted=bool(contacted[index].item()),
            first_contact_stable=bool(first_contact_stable[index].item()),
            toggled=bool(toggled[index].item()),
            retreated=bool(retreated[index].item()),
            toggle_time_s=float(toggle_time[index].item()),
            peak_compression_overshoot_m=float(peak_overshoot[index].item()),
        )
        for index in range(episode_count)
    )
    reward_sum_values = reward_sums.cpu().tolist()
    sample_count = int(reward_sample_count.item())
    summary = summarize_press_episodes(episodes=episodes)
    report(
        f"press evaluation seed={seed} COMPLETE elapsed_s={time.monotonic() - started_at:.1f} "
        f"success={summary.confirmed_toggle_rate:.3f} "
        f"first_contact_stability={summary.stable_first_contact_rate:.3f} "
        f"retreat={summary.post_toggle_retreat_rate:.3f}"
    )
    return episodes, {
        name: (float(value), sample_count)
        for name, value in zip(reward_names, reward_sum_values, strict=True)
    }


def evaluate_press_policy(
    *,
    env: Any,
    policy: Any,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    report: Callable[[str], None],
) -> PressEvaluationResult:
    """Evaluate one deterministic actor on all configured seeds."""
    episodes: list[PressEpisodeMetrics] = []
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
    return PressEvaluationResult(
        summary=summarize_press_episodes(episodes=tuple(episodes)),
        reward_means=tuple(
            RewardMean(name=name, mean=value / count) for name, (value, count) in rewards.items()
        ),
    )

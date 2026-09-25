"""Deterministic approach-policy rollout shared by selection and publication."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from robotics_experiments.flight_parameters import (
    FlightParameters,
    SignalStatistics,
    summarize_signal,
)
from robotics_experiments.rl.evaluation import EvaluationMetric, PolicyEvaluation, RewardMean
from robotics_experiments.rl.press import (
    PressApproachResult,
    PressApproachRewardTerms,
)
from robotics_experiments.rl.training._monitoring import evaluation_metric_names

__all__ = [
    "PressApproachEpisodeMetrics",
    "PressApproachEvaluationResult",
    "PressApproachEvaluationSummary",
    "evaluate_press_approach_policy",
    "press_policy_evaluation",
    "summarize_press_approach_episodes",
]


@dataclass(frozen=True)
class PressApproachEpisodeMetrics:
    """Outcome and diagnostic values retained for one completed episode."""

    success: bool
    collision: bool
    other_flight_failure: bool
    range_undershoot: bool
    timeout: bool
    hover_envelope_occupancy_rate: float
    hover_envelope_utilization_mean: float
    position_envelope_utilization_mean: float
    attitude_envelope_utilization_mean: float
    longest_hover_duration_s: float
    closest_range_m: float


@dataclass(frozen=True)
class PressApproachEvaluationSummary:
    """Aggregated approach acceptance and ranking metrics."""

    episodes: int
    success_rate: float
    collision_rate: float
    other_flight_failure_rate: float
    range_undershoot_rate: float
    timeout_rate: float
    hover_envelope_occupancy_rate: float
    hover_envelope_utilization_mean: float
    position_envelope_utilization_mean: float
    attitude_envelope_utilization_mean: float
    episode_longest_hover_duration_mean_s: float
    episode_longest_hover_duration_max_s: float
    min_closest_range_m: float


@dataclass(frozen=True)
class PressApproachEvaluationResult:
    """Complete deterministic evaluation result for one actor."""

    summary: PressApproachEvaluationSummary
    reward_means: tuple[RewardMean, ...]
    hover_statistics: tuple[SignalStatistics, ...]


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


_HOVER_SIGNAL_UNITS = {
    "position_error_enu_x": "m",
    "position_error_enu_y": "m",
    "position_error_enu_z": "m",
    "horizontal_position_error": "m",
    "position_error": "m",
    "attitude_error_roll": "rad",
    "attitude_error_pitch": "rad",
    "attitude_error_yaw": "rad",
    "tilt_error": "rad",
    "orientation_error": "rad",
}
_BINDING_STATISTICS = (
    ("max", "maximum_absolute"),
    ("rmse", "rmse"),
    ("p95", "p95_absolute"),
    ("p99", "p99_absolute"),
)


def _hover_signal_values(*, result: PressApproachResult[torch.Tensor]) -> dict[str, torch.Tensor]:
    hover = result.hover_evaluation
    return {
        "position_error_enu_x": hover.position_error_enu_m[:, 0],
        "position_error_enu_y": hover.position_error_enu_m[:, 1],
        "position_error_enu_z": hover.position_error_enu_m[:, 2],
        "horizontal_position_error": hover.horizontal_position_error_m,
        "position_error": hover.position_error_m,
        "attitude_error_roll": hover.attitude_error_rpy_rad[:, 0],
        "attitude_error_pitch": hover.attitude_error_rpy_rad[:, 1],
        "attitude_error_yaw": hover.attitude_error_rpy_rad[:, 2],
        "tilt_error": hover.tilt_error_rad,
        "orientation_error": hover.orientation_error_rad,
    }


def _require_finite(
    *, values: dict[str, torch.Tensor], active: torch.Tensor, seed: int, step: int
) -> None:
    samples = torch.cat(
        [value.reshape(len(active), -1) for value in values.values()],
        dim=1,
    )
    invalid = ~torch.isfinite(samples) & active[:, None]
    if bool(invalid.any()):
        episode, field = invalid.nonzero()[0].tolist()
        names = tuple(
            name for name, value in values.items() for _ in range(value.numel() // len(active))
        )
        name = names[field]
        raise ValueError(f"non-finite {name}: seed={seed} episode={episode} step={step}")


def summarize_press_approach_episodes(
    *,
    episodes: tuple[PressApproachEpisodeMetrics, ...],
) -> PressApproachEvaluationSummary:
    """Aggregate Stage 1 metrics over their all-episode denominators."""
    total = len(episodes)
    if total == 0:
        raise ValueError("press approach evaluation requires at least one episode")
    return PressApproachEvaluationSummary(
        episodes=total,
        success_rate=sum(episode.success for episode in episodes) / total,
        collision_rate=sum(episode.collision for episode in episodes) / total,
        other_flight_failure_rate=(
            sum(episode.other_flight_failure for episode in episodes) / total
        ),
        range_undershoot_rate=sum(episode.range_undershoot for episode in episodes) / total,
        timeout_rate=sum(episode.timeout for episode in episodes) / total,
        hover_envelope_occupancy_rate=float(
            np.mean([episode.hover_envelope_occupancy_rate for episode in episodes])
        ),
        hover_envelope_utilization_mean=float(
            np.mean([episode.hover_envelope_utilization_mean for episode in episodes])
        ),
        position_envelope_utilization_mean=float(
            np.mean([episode.position_envelope_utilization_mean for episode in episodes])
        ),
        attitude_envelope_utilization_mean=float(
            np.mean([episode.attitude_envelope_utilization_mean for episode in episodes])
        ),
        episode_longest_hover_duration_mean_s=float(
            np.mean([episode.longest_hover_duration_s for episode in episodes])
        ),
        episode_longest_hover_duration_max_s=max(
            episode.longest_hover_duration_s for episode in episodes
        ),
        min_closest_range_m=min(episode.closest_range_m for episode in episodes),
    )


def _report_worst_episode_trace(
    *,
    seed: int,
    closest_range: torch.Tensor,
    active_trace: list[torch.Tensor],
    observation_trace: list[torch.Tensor],
    action_trace: list[torch.Tensor],
    range_trace: list[torch.Tensor],
    body_collision_trace: list[torch.Tensor],
    pusher_collision_trace: list[torch.Tensor],
    step_dt: float,
    report: Callable[[str], None],
) -> None:
    """Report the complete active window for one seed's closest episode."""
    episode = int(torch.argmin(closest_range).item())
    report(
        f"press approach worst trace seed={seed} episode={episode} "
        f"closest_range_m={float(closest_range[episode].item()):.6f}"
    )
    for step, active in enumerate(active_trace):
        if not bool(active[episode].item()):
            continue
        observation = observation_trace[step][episode]
        action = action_trace[step][episode]
        goal_offset_body = observation[:3].tolist()
        body_velocity = observation[3:6].tolist()
        body_angular_velocity = observation[6:9].tolist()
        report(
            f"press approach worst trace seed={seed} episode={episode} step={step} "
            f"time_s={(step + 1) * step_dt:.3f} "
            f"goal_offset_body_m={goal_offset_body} "
            f"range_after_m={float(range_trace[step][episode].item()):.6f} "
            f"body_velocity_mps={body_velocity} "
            f"body_angular_velocity_radps={body_angular_velocity} "
            f"action={action.tolist()} "
            f"body_collision={bool(body_collision_trace[step][episode].item())} "
            f"pusher_collision={bool(pusher_collision_trace[step][episode].item())}"
        )


def _evaluate_seed(
    *,
    env: _PressApproachEvaluationEnvironment,
    policy: _PressApproachPolicy,
    seed: int,
    episode_count: int,
    hold_steps: int,
    policy_warmup_steps: int,
    report: Callable[[str], None],
    trace_worst_episode: bool,
) -> tuple[
    tuple[PressApproachEpisodeMetrics, ...],
    dict[str, tuple[float, int]],
    dict[str, tuple[np.ndarray, ...]],
]:
    reward_names = tuple(field.name for field in fields(PressApproachRewardTerms)) + ("total",)
    reward_sums = torch.zeros(len(reward_names), dtype=torch.float32, device=env.device)
    reward_sample_count = torch.zeros((), dtype=torch.long, device=env.device)
    active = torch.ones(episode_count, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(active)
    collision = torch.zeros_like(active)
    body_collision = torch.zeros_like(active)
    pusher_collision = torch.zeros_like(active)
    other_flight_failure = torch.zeros_like(active)
    range_undershoot = torch.zeros_like(active)
    timeout = torch.zeros_like(active)
    longest_hover_steps = torch.zeros(episode_count, dtype=torch.long, device=env.device)
    closest_range = torch.full(
        (episode_count,),
        float("inf"),
        dtype=torch.float32,
        device=env.device,
    )
    trajectory_sample_count = torch.zeros(
        episode_count,
        dtype=torch.long,
        device=env.device,
    )
    hover_occupancy_sums = torch.zeros(episode_count, dtype=torch.float32, device=env.device)
    envelope_utilization_sums = torch.zeros(
        (3, episode_count),
        dtype=torch.float32,
        device=env.device,
    )
    active_trace: list[torch.Tensor] = []
    observation_trace: list[torch.Tensor] = []
    action_trace: list[torch.Tensor] = []
    range_trace: list[torch.Tensor] = []
    body_collision_trace: list[torch.Tensor] = []
    pusher_collision_trace: list[torch.Tensor] = []
    reset_result = env.reset(seed=seed)
    observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    horizon = int(env.max_episode_length)
    hover_history = torch.empty(
        (horizon, episode_count, len(_HOVER_SIGNAL_UNITS)),
        dtype=torch.float32,
        device=env.device,
    )
    progress_interval = max(1, horizon // 10)
    started_at = time.monotonic()
    report(
        f"press approach evaluation seed={seed} START "
        f"episodes={episode_count} horizon={horizon}"
    )
    for step in range(horizon):
        policy_observation = observation["policy"]
        _require_finite(
            values={"policy_observation": policy_observation}, active=active, seed=seed, step=step
        )
        with torch.no_grad():
            action = policy(policy_observation)
        _require_finite(values={"raw_action": action}, active=active, seed=seed, step=step)
        action = action.clamp(-1.0, 1.0)
        if trace_worst_episode:
            active_trace.append(active.detach().cpu().clone())
            observation_trace.append(policy_observation.detach().cpu())
            action_trace.append(action.detach().cpu())
        trajectory_sample_count += active
        observation, _reward, terminated, truncated, _info = env.step(action)
        result = env.press_approach_result
        if result is None:
            raise RuntimeError("press approach environment did not publish a transition result")
        termination = result.termination
        hover_values = _hover_signal_values(result=result)
        reward_terms = {f"reward.{name}": getattr(result.rewards, name) for name in reward_names}
        _require_finite(
            values={
                **hover_values,
                **reward_terms,
                "reward": _reward,
                "range_m": result.range_m,
                "closest_range_m": result.state.closest_range_m,
                "normalized_pose_error": result.state.normalized_pose_error,
                "position_utilization": result.hover_evaluation.position_utilization,
                "attitude_utilization": result.hover_evaluation.attitude_utilization,
                "longest_stable_goal_steps": result.state.longest_stable_goal_steps,
            },
            active=active,
            seed=seed,
            step=step,
        )
        hover_history[step] = torch.stack(tuple(hover_values.values()), dim=-1).detach()
        current_range = result.range_m
        hover_occupancy_sums += torch.where(active, result.strict_goal.float(), 0.0)
        utilization = torch.stack(
            (
                result.state.normalized_pose_error,
                result.hover_evaluation.position_utilization,
                result.hover_evaluation.attitude_utilization,
            ),
            dim=0,
        )
        envelope_utilization_sums += torch.where(active[None, :], utilization, 0.0)
        longest_hover_steps = torch.maximum(
            longest_hover_steps,
            torch.where(
                active,
                result.state.longest_stable_goal_steps,
                longest_hover_steps,
            ),
        )
        if trace_worst_episode:
            range_trace.append(current_range.detach().cpu())
            body_collision_trace.append(termination.body_collision.detach().cpu())
            pusher_collision_trace.append(termination.pusher_collision.detach().cpu())
        closest_range = torch.minimum(
            closest_range,
            torch.where(active, result.state.closest_range_m, closest_range),
        )
        newly_successful = active & termination.success
        success |= newly_successful
        collision |= active & termination.collision
        body_collision |= active & termination.body_collision
        pusher_collision |= active & termination.pusher_collision
        other_flight_failure |= active & termination.flight_failure & ~termination.collision
        range_undershoot |= active & termination.range_undershoot
        timeout |= active & termination.time_out
        reward_values = torch.stack(
            tuple(reward_terms.values()),
            dim=0,
        )
        reward_sums += reward_values[:, active].sum(dim=1)
        reward_sample_count += active.sum()
        active &= ~(terminated | truncated)
        if not bool(torch.any(active)):
            break
        completed_steps = step + 1
        if completed_steps % progress_interval == 0 or completed_steps == horizon:
            counts = (
                torch.stack(
                    (
                        active.sum(),
                        success.sum(),
                        collision.sum(),
                        body_collision.sum(),
                        pusher_collision.sum(),
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
                f"success={counts[1]} collision={counts[2]} "
                f"body_collision={counts[3]} pusher_collision={counts[4]} "
                f"flight_failure={counts[5]} timeout={counts[6]}"
            )
    if bool(torch.any(active)):
        raise RuntimeError("press approach evaluation exceeded the configured episode horizon")

    episodes = tuple(
        PressApproachEpisodeMetrics(
            success=bool(success[index].item()),
            collision=bool(collision[index].item()),
            other_flight_failure=bool(other_flight_failure[index].item()),
            range_undershoot=bool(range_undershoot[index].item()),
            timeout=bool(timeout[index].item()),
            hover_envelope_occupancy_rate=float(
                hover_occupancy_sums[index].item() / trajectory_sample_count[index].item()
            ),
            hover_envelope_utilization_mean=float(
                envelope_utilization_sums[0, index].item() / trajectory_sample_count[index].item()
            ),
            position_envelope_utilization_mean=float(
                envelope_utilization_sums[1, index].item() / trajectory_sample_count[index].item()
            ),
            attitude_envelope_utilization_mean=float(
                envelope_utilization_sums[2, index].item() / trajectory_sample_count[index].item()
            ),
            longest_hover_duration_s=float(longest_hover_steps[index].item()) * float(env.step_dt),
            closest_range_m=float(closest_range[index].item()),
        )
        for index in range(episode_count)
    )
    reward_sum_values = reward_sums.cpu().tolist()
    sample_count = int(reward_sample_count.item())
    summary = summarize_press_approach_episodes(episodes=episodes)
    if trace_worst_episode:
        _report_worst_episode_trace(
            seed=seed,
            closest_range=closest_range.detach().cpu(),
            active_trace=active_trace,
            observation_trace=observation_trace,
            action_trace=action_trace,
            range_trace=range_trace,
            body_collision_trace=body_collision_trace,
            pusher_collision_trace=pusher_collision_trace,
            step_dt=float(env.step_dt),
            report=report,
        )
    report(
        f"press approach evaluation seed={seed} COMPLETE "
        f"elapsed_s={time.monotonic() - started_at:.1f} "
        f"success={summary.success_rate:.3f} "
        f"mean_longest_hover_s={summary.episode_longest_hover_duration_mean_s:.3f} "
        f"collision={summary.collision_rate:.3f} "
        f"other_flight_failure={summary.other_flight_failure_rate:.3f} "
        f"timeout={summary.timeout_rate:.3f}"
    )
    # Each episode owns its completion time; recovery and later auto-reset
    # samples do not belong to its successful continuous hold.
    history = hover_history[: step + 1].cpu().numpy()
    window_steps = hold_steps - policy_warmup_steps
    episode_step_counts = trajectory_sample_count.cpu().tolist()
    successful_windows = [
        history[end - window_steps : end, index]
        for index, (episode, end) in enumerate(zip(episodes, episode_step_counts, strict=True))
        if episode.success
    ]
    successful_values = (
        np.stack(successful_windows, axis=1)
        if successful_windows
        else np.empty((0, 0, len(_HOVER_SIGNAL_UNITS)))
    )
    successful_hover_samples = {
        name: (successful_values[:, :, signal].reshape(-1),) if successful_windows else ()
        for signal, name in enumerate(_HOVER_SIGNAL_UNITS)
    }
    return (
        episodes,
        {
            name: (float(value), sample_count)
            for name, value in zip(reward_names, reward_sum_values, strict=True)
        },
        successful_hover_samples,
    )


def evaluate_press_approach_policy(
    *,
    env: _PressApproachEvaluationEnvironment,
    policy: _PressApproachPolicy,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    hold_steps: int,
    flight_parameters: FlightParameters,
    report: Callable[[str], None],
    trace_worst_episode: bool,
) -> PressApproachEvaluationResult:
    """Evaluate one deterministic approach actor on all configured seeds."""
    episodes: list[PressApproachEpisodeMetrics] = []
    rewards: dict[str, tuple[float, int]] = {}
    hover_samples: dict[str, list[np.ndarray]] = {name: [] for name in _HOVER_SIGNAL_UNITS}
    policy_warmup_steps = round(flight_parameters.policy_warmup_s / float(env.step_dt))
    if policy_warmup_steps >= hold_steps:
        raise ValueError("flight-parameters policy warmup must be shorter than the hold")
    for seed in seeds:
        seed_episodes, seed_rewards, seed_hover_samples = _evaluate_seed(
            env=env,
            policy=policy,
            seed=seed,
            episode_count=episodes_per_seed,
            hold_steps=hold_steps,
            policy_warmup_steps=policy_warmup_steps,
            report=report,
            trace_worst_episode=trace_worst_episode,
        )
        episodes.extend(seed_episodes)
        for name, (value, count) in seed_rewards.items():
            previous_value, previous_count = rewards.get(name, (0.0, 0))
            rewards[name] = previous_value + value, previous_count + count
        for name, samples in seed_hover_samples.items():
            hover_samples[name].extend(samples)
    return PressApproachEvaluationResult(
        summary=summarize_press_approach_episodes(episodes=tuple(episodes)),
        reward_means=tuple(
            RewardMean(name=name, mean=value / count) for name, (value, count) in rewards.items()
        ),
        hover_statistics=tuple(
            summarize_signal(
                name=name,
                unit=_HOVER_SIGNAL_UNITS[name],
                values=np.concatenate(samples),
            )
            for name, samples in hover_samples.items()
            if samples
        ),
    )


def press_policy_evaluation(
    *,
    iteration: int,
    checkpoint_path: Path,
    policy_digest: str,
    result: PressApproachEvaluationResult,
    source: str,
) -> PolicyEvaluation:
    """Convert one rollout result into the evaluator's decision-free contract."""
    summary = result.summary
    finite_values = {field.name: getattr(summary, field.name) for field in fields(summary)}
    finite_values.update({f"reward.{reward.name}": reward.mean for reward in result.reward_means})
    finite_values.update(
        {
            f"flight_parameters.{statistics.name}.{field.name}": getattr(statistics, field.name)
            for statistics in result.hover_statistics
            for field in fields(statistics)
            if field.name not in {"name", "unit"}
        }
    )
    for name, value in finite_values.items():
        if not np.isfinite(value):
            raise ValueError(f"non-finite evaluation {name}: source={source} iteration={iteration}")
    hover_statistics = {statistics.name: statistics for statistics in result.hover_statistics}
    evaluation = PolicyEvaluation(
        iteration=iteration,
        checkpoint_path=checkpoint_path,
        policy_digest=policy_digest,
        metrics=(
            EvaluationMetric("success_rate", summary.success_rate),
            EvaluationMetric("collision_rate", summary.collision_rate),
            EvaluationMetric("other_flight_failure_rate", summary.other_flight_failure_rate),
            EvaluationMetric("minimum_closest_range_m", summary.min_closest_range_m),
            EvaluationMetric("all_traces_finite", 1.0),
            EvaluationMetric(
                "hover_envelope_occupancy_rate", summary.hover_envelope_occupancy_rate
            ),
            EvaluationMetric(
                "episode_longest_hover_duration_mean_s",
                summary.episode_longest_hover_duration_mean_s,
            ),
            EvaluationMetric(
                "episode_longest_hover_duration_max_s",
                summary.episode_longest_hover_duration_max_s,
            ),
            EvaluationMetric(
                "hover_envelope_utilization_mean",
                summary.hover_envelope_utilization_mean,
            ),
            EvaluationMetric(
                "position_envelope_utilization_mean",
                summary.position_envelope_utilization_mean,
            ),
            EvaluationMetric(
                "attitude_envelope_utilization_mean",
                summary.attitude_envelope_utilization_mean,
            ),
            EvaluationMetric("range_undershoot_rate", summary.range_undershoot_rate),
            EvaluationMetric("timeout_rate", summary.timeout_rate),
            EvaluationMetric("completed_episodes", float(summary.episodes)),
        )
        + tuple(
            EvaluationMetric(
                f"flight_parameters.{signal_name}.{metric_name}",
                getattr(statistics, field_name) if statistics is not None else None,
            )
            for signal_name in _HOVER_SIGNAL_UNITS
            for metric_name, field_name in _BINDING_STATISTICS
            for statistics in (hover_statistics.get(signal_name),)
        ),
        reward_means=result.reward_means,
        source=source,
    )
    metrics = {metric.name: metric for metric in evaluation.metrics}
    return replace(
        evaluation,
        metrics=tuple(
            metrics.get(name, EvaluationMetric(name, None)) for name in evaluation_metric_names()
        ),
    )

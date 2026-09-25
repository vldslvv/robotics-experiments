"""Fresh contact-task rollouts; only physical outcomes determine qualification."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from robotics_experiments.rl.evaluation import EvaluationMetric, PolicyEvaluation, RewardMean
from robotics_experiments.rl.training._monitoring import evaluation_metric_names

__all__: list[str] = []


def _evaluate_contact_policy(
    *,
    env: Any,
    policy: Callable,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    iteration: int,
    checkpoint_path: Path,
    policy_digest: str,
    source: str,
    report: Callable[[str], None],
    reset_policy: Callable[[], None] | None = None,
) -> PolicyEvaluation:
    totals = dict.fromkeys(
        (
            "success",
            "collision",
            "body_collision",
            "pad_collision",
            "failure",
            "timeout",
            "contact",
            "stable",
            "toggle",
            "multiple",
        ),
        0,
    )
    categories = (
        "button_contact",
        "housing_collision",
        "wall_collision",
        "other_pad_collision",
        "body_collision",
    )
    category_totals = dict.fromkeys((*categories, "simultaneous_contact"), 0)
    rewards: dict[str, float] = {}
    reward_samples = 0
    closest_range = float("inf")
    peak_compression = 0.0
    progress_sum = 0.0
    for seed in seeds:
        observation = env.reset(seed=seed)
        if isinstance(observation, tuple):
            observation = observation[0]
        if reset_policy is not None:
            reset_policy()
        active = torch.ones(episodes_per_seed, dtype=torch.bool, device=env.device)
        category_seen = torch.zeros((episodes_per_seed, 5), dtype=torch.bool, device=env.device)
        simultaneous_seen = torch.zeros(episodes_per_seed, dtype=torch.bool, device=env.device)
        best_progress = torch.zeros(episodes_per_seed, device=env.device)
        report(f"contact evaluation seed={seed} START episodes={episodes_per_seed}")
        for step in range(env.max_episode_length):
            with torch.no_grad():
                action = policy(observation["policy"])
            if not bool(
                torch.isfinite(observation["policy"][active]).all()
                & torch.isfinite(action[active]).all()
            ):
                raise ValueError(f"non-finite contact actor trace seed={seed} step={step}")
            observation, reward, terminated, truncated, _info = env.step(action.clamp(-1, 1))
            result = env.press_contact_result
            if result is None:
                raise RuntimeError("contact environment did not return its transition")
            state = result.state
            values = torch.stack(
                (
                    reward,
                    result.progress,
                    state.closest_range_m,
                    state.maximum_button_m,
                    state.peak_compression_m,
                )
            )
            if not bool(torch.isfinite(values[:, active]).all()):
                raise ValueError(f"non-finite contact physical trace seed={seed} step={step}")
            closest_range = min(closest_range, float(state.closest_range_m[active].min()))
            peak_compression = max(peak_compression, float(state.peak_compression_m[active].max()))
            best_progress = torch.maximum(best_progress, torch.where(active, result.progress, 0))
            reward_samples += int(active.sum())
            for name, value in result.rewards.items():
                rewards[name] = rewards.get(name, 0.0) + float(value[active].sum())
            category_seen |= result.contact_categories & active[:, None]
            simultaneous_seen |= result.simultaneous_contact & active
            ended = active & (terminated | truncated)
            flight = env.press_approach_result
            for name, value in {
                "success": result.success,
                "collision": result.collision,
                "body_collision": flight.termination.body_collision,
                "pad_collision": flight.termination.pusher_collision,
                "failure": result.failure & ~result.collision,
                "timeout": result.timeout,
                "contact": state.contact_seen,
                "stable": state.contact_seen & state.first_contact_stable,
                "toggle": state.toggle_count > 0,
                "multiple": state.toggle_count > 1,
            }.items():
                totals[name] += int((ended & value).sum())
            for index, name in enumerate(categories):
                category_totals[name] += int((ended & category_seen[:, index]).sum())
            category_totals["simultaneous_contact"] += int((ended & simultaneous_seen).sum())
            active &= ~ended
            if not bool(active.any()):
                break
            if (step + 1) % max(1, env.max_episode_length // 10) == 0:
                report(f"contact evaluation seed={seed} step={step + 1} active={int(active.sum())}")
                flight = env.press_approach_result
                report(
                    "contact active means: "
                    f"range={float(flight.range_m[active].mean()):.5f} "
                    f"alignment={float(flight.alignment_error_m[active].mean()):.5f} "
                    f"attitude={float(flight.attitude_error_rad[active].mean()):.5f} "
                    f"speed={float(flight.linear_speed_mps[active].mean()):.5f} "
                    f"angular_speed={float(flight.angular_speed_radps[active].mean()):.5f} "
                    f"stable_steps={float(state.stable_contact_steps[active].float().mean()):.3f} "
                    f"button_max={float(state.maximum_button_m[active].max()):.6f} "
                    f"toggled={int((state.toggle_count[active] > 0).sum())} "
                    f"retreat_steps={int(state.longest_retreat_steps[active].max())} "
                    f"peak_compression={float(state.peak_compression_m[active].max()):.6f}"
                )
        if bool(active.any()):
            raise RuntimeError("contact evaluation exceeded its configured episode horizon")
        progress_sum += float(best_progress.sum())
        report(f"contact evaluation seed={seed} COMPLETE cumulative={totals}")
    count = len(seeds) * episodes_per_seed
    metrics = {
        "success_rate": totals["success"] / count,
        "collision_rate": totals["collision"] / count,
        "other_flight_failure_rate": totals["failure"] / count,
        "minimum_closest_range_m": closest_range,
        "all_traces_finite": 1.0,
        "timeout_rate": totals["timeout"] / count,
        "completed_episodes": float(count),
        "contact_progress_mean": progress_sum / count,
        "contact_rate": totals["contact"] / count,
        "stable_first_contact_rate": totals["stable"] / totals["contact"]
        if totals["contact"]
        else None,
        "toggle_rate": totals["toggle"] / count,
        "retreat_rate": totals["success"] / totals["toggle"] if totals["toggle"] else None,
        "multiple_toggle_rate": totals["multiple"] / count,
        "peak_compression_m": peak_compression,
    }
    metrics.update({f"{name}_rate": value / count for name, value in category_totals.items()})
    return PolicyEvaluation(
        iteration=iteration,
        checkpoint_path=checkpoint_path,
        policy_digest=policy_digest,
        metrics=tuple(
            EvaluationMetric(name, metrics.get(name)) for name in evaluation_metric_names()
        ),
        reward_means=tuple(
            RewardMean(name, value / reward_samples) for name, value in rewards.items()
        ),
        source=source,
    )

"""Evaluate the exported F3 press policy against deterministic acceptance metrics."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robotics_experiments.config import ExperimentConfig, load_config

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"


@dataclass(frozen=True)
class _EpisodeMetrics:
    contacted: bool
    first_contact_stable: bool
    toggled: bool
    retreated: bool
    toggle_time_s: float
    peak_compression_overshoot_m: float


@dataclass(frozen=True)
class _EvaluationSummary:
    episodes: int
    press_success_rate: float
    first_contact_stability: float
    retreat_rate: float
    median_time_to_toggle_s: float
    peak_compression_overshoot_m: float


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _summarize_episodes(episodes: list[_EpisodeMetrics]) -> _EvaluationSummary:
    total = len(episodes)
    contacted = [episode for episode in episodes if episode.contacted]
    toggled = [episode for episode in episodes if episode.toggled]
    toggle_times = [episode.toggle_time_s for episode in toggled]
    return _EvaluationSummary(
        episodes=total,
        press_success_rate=sum(episode.toggled for episode in episodes) / total,
        first_contact_stability=(
            sum(episode.first_contact_stable for episode in contacted) / len(contacted)
            if contacted
            else 0.0
        ),
        retreat_rate=(
            sum(episode.retreated for episode in toggled) / len(toggled) if toggled else 0.0
        ),
        median_time_to_toggle_s=(float(np.median(toggle_times)) if toggle_times else float("nan")),
        peak_compression_overshoot_m=max(
            (episode.peak_compression_overshoot_m for episode in episodes),
            default=0.0,
        ),
    )


def _run_seed(
    *,
    env: Any,
    policy: Any,
    seed: int,
    episode_count: int,
) -> tuple[list[_EpisodeMetrics], dict[str, tuple[float, int]]]:
    from robotics_experiments.rl.press import PressRewardTerms

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
    _report(f"press evaluation seed={seed} START episodes={episode_count} horizon={horizon}")
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
            _report(
                f"press evaluation seed={seed} progress={completed_steps}/{horizon} "
                f"elapsed_s={time.monotonic() - started_at:.1f} active={active_count} "
                f"contacted={contact_count} toggled={toggle_count} retreated={retreat_count}"
            )
    if bool(torch.any(active)):
        raise RuntimeError("press evaluation exceeded the configured episode horizon")

    episodes = [
        _EpisodeMetrics(
            contacted=bool(contacted[index].item()),
            first_contact_stable=bool(first_contact_stable[index].item()),
            toggled=bool(toggled[index].item()),
            retreated=bool(retreated[index].item()),
            toggle_time_s=float(toggle_time[index].item()),
            peak_compression_overshoot_m=float(peak_overshoot[index].item()),
        )
        for index in range(episode_count)
    ]
    reward_sum_values = reward_sums.cpu().tolist()
    sample_count = int(reward_sample_count.item())
    summary = _summarize_episodes(episodes)
    _report(
        f"press evaluation seed={seed} COMPLETE elapsed_s={time.monotonic() - started_at:.1f} "
        f"success={summary.press_success_rate:.3f} "
        f"first_contact_stability={summary.first_contact_stability:.3f} "
        f"retreat={summary.retreat_rate:.3f}"
    )
    return episodes, {
        name: (float(value), sample_count)
        for name, value in zip(reward_names, reward_sum_values, strict=True)
    }


def _run(*, config: ExperimentConfig, artifact_dir: Path) -> None:
    from robotics_experiments.rl.press import (
        POLICY_FILENAME,
        load_policy_contract,
        validate_policy_artifact,
    )
    from robotics_experiments.workflows.isaac_lab import make_isaac_lab_env

    evaluation = config.rl.press.evaluation
    seeds = tuple(int(seed) for seed in evaluation.seeds)
    contract = load_policy_contract(artifact_dir=artifact_dir)
    validate_policy_artifact(
        artifact_dir=artifact_dir,
        obs_schema_version=contract.obs_schema_version,
        obs_dim=contract.obs_dim,
        policy_interface=contract.policy_interface,
    )
    env, env_cfg = make_isaac_lab_env(
        task_id=config.rl.training.task_id,
        experiment_config=config,
        seed=seeds[0],
        num_envs=evaluation.episodes_per_seed,
        policy_contract=contract,
    )
    episodes: list[_EpisodeMetrics] = []
    rewards: dict[str, tuple[float, int]] = {}
    try:
        if (
            int(env_cfg.observation_schema_version),
            int(env_cfg.observation_space),
            str(env_cfg.policy_interface),
        ) != (
            contract.obs_schema_version,
            contract.obs_dim,
            contract.policy_interface,
        ):
            raise ValueError("Isaac Lab evaluation environment changed the artifact contract")
        policy = torch.jit.load(str(artifact_dir / POLICY_FILENAME), map_location=env.device)
        policy.eval()
        for seed in seeds:
            seed_episodes, seed_rewards = _run_seed(
                env=env,
                policy=policy,
                seed=seed,
                episode_count=evaluation.episodes_per_seed,
            )
            episodes.extend(seed_episodes)
            for name, (value, count) in seed_rewards.items():
                previous_value, previous_count = rewards.get(name, (0.0, 0))
                rewards[name] = previous_value + value, previous_count + count
    finally:
        env.close()

    summary = _summarize_episodes(episodes)
    _report(
        "press evaluation: "
        f"episodes={summary.episodes} "
        f"success={summary.press_success_rate:.3f} "
        f"first_contact_stability={summary.first_contact_stability:.3f} "
        f"retreat={summary.retreat_rate:.3f} "
        f"median_toggle_s={summary.median_time_to_toggle_s:.3f} "
        f"peak_compression_overshoot_m={summary.peak_compression_overshoot_m:.5f}"
    )
    for name, (value, count) in rewards.items():
        _report(f"reward_mean {name}={value / count:.6f}")

    failures = []
    if summary.press_success_rate < evaluation.success_rate_min:
        failures.append("press success")
    if summary.first_contact_stability < evaluation.first_contact_stability_min:
        failures.append("first-contact stability")
    if summary.retreat_rate < evaluation.retreat_rate_min:
        failures.append("post-toggle retreat")
    if failures:
        raise AssertionError("press acceptance failed: " + ", ".join(failures))
    _report("press evaluation PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=_repo_root() / ".build" / "training" / "press" / "latest",
    )
    parser.add_argument("--scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        config_path = os.environ[_CONFIG_PATH_ENV]
        _run(config=load_config(config_path), artifact_dir=args.artifact_dir)
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

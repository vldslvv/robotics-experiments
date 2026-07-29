"""Independently evaluate an already selected F3 press policy artifact."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

from robotics_experiments.config import ExperimentConfig, load_config
from robotics_experiments.rl import POLICY_FILENAME, validate_policy_artifact
from robotics_experiments.workflows.isaac_lab import (
    PressApproachEvaluationResult,
    evaluate_press_approach_policy,
    make_isaac_lab_env,
)

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run(*, config: ExperimentConfig, artifact_dir: Path) -> PressApproachEvaluationResult:
    evaluation = config.rl.press.evaluation
    seeds = tuple(int(seed) for seed in evaluation.seeds)
    metadata = validate_policy_artifact(artifact_dir=artifact_dir)
    contract = metadata.contract
    if contract.curriculum_stage != config.rl.press.curriculum_stage:
        raise ValueError(
            f"press artifact stage {contract.curriculum_stage!r} does not match "
            f"configured stage {config.rl.press.curriculum_stage!r}"
        )
    env, env_cfg = make_isaac_lab_env(
        task_id=config.rl.training.task_id,
        experiment_config=config,
        controller_settings=metadata.controller_settings,
        seed=seeds[0],
        num_envs=evaluation.episodes_per_seed,
        policy_contract=contract,
    )
    try:
        if (
            int(env_cfg.observation_schema_version),
            int(env_cfg.observation_space),
            str(env_cfg.policy_interface),
        ) != (
            contract.obs_schema_version,
            contract.obs_dim,
            contract.policy_interface.value,
        ):
            raise ValueError("Isaac Lab evaluation environment changed the artifact contract")
        policy = torch.jit.load(str(artifact_dir / POLICY_FILENAME), map_location=env.device)
        policy.eval()
        result = evaluate_press_approach_policy(
            env=env,
            policy=policy,
            seeds=seeds,
            episodes_per_seed=evaluation.episodes_per_seed,
            goal_config=config.rl.press.goal,
            report=_report,
        )
    finally:
        env.close()

    summary = result.summary
    _report(
        "press approach evaluation: "
        f"episodes={summary.episodes} "
        f"success={summary.approach_success_rate:.3f} "
        f"collision={summary.collision_rate:.3f} "
        f"body_collision={summary.body_collision_rate:.3f} "
        f"pusher_collision={summary.pusher_collision_rate:.3f} "
        f"other_flight_failure={summary.other_flight_failure_rate:.3f} "
        f"timeout={summary.timeout_rate:.3f} "
        f"mean_closest_range_m={summary.mean_closest_range_m:.5f} "
        f"min_closest_range_m={summary.min_closest_range_m:.5f} "
        f"median_completion_s={summary.median_completion_time_s!s} "
        f"terminal_range_cost={summary.mean_terminal_range_cost:.5f} "
        f"alignment_cost={summary.mean_trajectory_alignment_cost:.5f} "
        f"velocity_tracking_cost={summary.mean_velocity_tracking_cost:.5f} "
        f"upright_cost={summary.mean_upright_error_cost:.5f} "
        f"action_delta_cost={summary.mean_action_delta_cost:.5f} "
        f"completion_fraction={summary.mean_completion_time_fraction:.5f}"
    )
    for reward in result.reward_means:
        _report(f"reward_mean {reward.name}={reward.mean:.6f}")

    finite_trace_values = (
        summary.approach_success_rate,
        summary.collision_rate,
        summary.body_collision_rate,
        summary.pusher_collision_rate,
        summary.other_flight_failure_rate,
        summary.timeout_rate,
        summary.mean_closest_range_m,
        summary.min_closest_range_m,
        summary.mean_terminal_range_cost,
        summary.mean_trajectory_alignment_cost,
        summary.mean_velocity_tracking_cost,
        summary.mean_upright_error_cost,
        summary.mean_action_delta_cost,
        summary.mean_completion_time_fraction,
    )
    failures = []
    if not all(math.isfinite(value) for value in finite_trace_values) or (
        summary.median_completion_time_s is not None
        and not math.isfinite(summary.median_completion_time_s)
    ):
        failures.append("non-finite traces")
    if summary.approach_success_rate < evaluation.success_rate_min:
        failures.append("approach success")
    if summary.collision_rate > evaluation.collision_rate_max:
        failures.append("simulated collisions")
    if summary.min_closest_range_m < config.rl.press.goal.range_min_m:
        failures.append("range undershoot")
    if summary.other_flight_failure_rate > 0.0:
        failures.append("flight-envelope failure")
    if failures:
        raise AssertionError("press acceptance failed: " + ", ".join(failures))
    _report("press approach evaluation PASSED")
    return result


def _resolve_artifact_dir(
    *,
    explicit_artifact_dir: Path | None,
    config: ExperimentConfig,
) -> Path:
    if explicit_artifact_dir is not None:
        return explicit_artifact_dir
    policy_root = _repo_root() / ".build" / "training" / config.rl.training.policy_name
    run_id = os.environ.get(_RUN_ID_ENV)
    return policy_root / run_id if run_id else policy_root / "latest"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        config_path = os.environ[_CONFIG_PATH_ENV]
        config = load_config(config_path)
        _run(
            config=config,
            artifact_dir=_resolve_artifact_dir(
                explicit_artifact_dir=args.artifact_dir,
                config=config,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

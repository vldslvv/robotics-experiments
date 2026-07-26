"""Independently evaluate an already selected F3 press policy artifact."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from robotics_experiments.config import ExperimentConfig, load_config
from robotics_experiments.rl import POLICY_FILENAME, validate_policy_artifact
from robotics_experiments.workflows.isaac_lab import (
    PressEvaluationResult,
    evaluate_press_policy,
    make_isaac_lab_env,
)

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run(*, config: ExperimentConfig, artifact_dir: Path) -> PressEvaluationResult:
    evaluation = config.rl.press.evaluation
    seeds = tuple(int(seed) for seed in evaluation.seeds)
    contract = validate_policy_artifact(artifact_dir=artifact_dir).contract
    env, env_cfg = make_isaac_lab_env(
        task_id=config.rl.training.task_id,
        experiment_config=config,
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
        result = evaluate_press_policy(
            env=env,
            policy=policy,
            seeds=seeds,
            episodes_per_seed=evaluation.episodes_per_seed,
            report=_report,
        )
    finally:
        env.close()

    summary = result.summary
    _report(
        "press evaluation: "
        f"episodes={summary.episodes} "
        f"success={summary.confirmed_toggle_rate:.3f} "
        f"first_contact_stability={summary.stable_first_contact_rate:.3f} "
        f"retreat={summary.post_toggle_retreat_rate:.3f} "
        f"full_completion={summary.full_completion_rate:.3f} "
        f"median_toggle_s={summary.median_toggle_time_s!s} "
        f"peak_compression_overshoot_m={summary.peak_compression_overshoot_m:.5f}"
    )
    for reward in result.reward_means:
        _report(f"reward_mean {reward.name}={reward.mean:.6f}")

    failures = []
    if summary.confirmed_toggle_rate < evaluation.success_rate_min:
        failures.append("press success")
    if summary.stable_first_contact_rate < evaluation.first_contact_stability_min:
        failures.append("first-contact stability")
    if summary.post_toggle_retreat_rate < evaluation.retreat_rate_min:
        failures.append("post-toggle retreat")
    if failures:
        raise AssertionError("press acceptance failed: " + ", ".join(failures))
    _report("press evaluation PASSED")
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

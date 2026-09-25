"""Independently evaluate an already selected F3 press policy artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from robotics_experiments.config import ExperimentConfig, load_config, load_curriculum_config
from robotics_experiments.rl import (
    POLICY_FILENAME,
    PolicyInputs,
    prepare_policy_inputs,
    validate_policy_artifact,
)
from robotics_experiments.rl.press import (
    create_press_acceptance_criteria,
    evaluate_press_acceptance,
    press_policy_contract,
)
from robotics_experiments.rl.press._fingerprint import _press_comparison_fingerprint
from robotics_experiments.rl.workflow import _load_pinned_policy_inputs
from robotics_experiments.workflows.isaac_lab.training import _config_values

if TYPE_CHECKING:
    from robotics_experiments.workflows.isaac_lab import PressApproachEvaluationResult

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _HoverFeedback:
    """Position/yaw feedback using only the production schema-3 observation."""

    def __init__(self, *, config: ExperimentConfig) -> None:
        self._action = config.rl.action
        self._evaluation = config.rl.press.evaluation

    def __call__(self, observation: torch.Tensor, /) -> torch.Tensor:
        body_velocity = self._evaluation.feasibility_position_gain_per_s * observation[:, :3]
        body_velocity -= self._evaluation.feasibility_velocity_damping * observation[:, 3:6]
        # Transpose the goal-to-body rotation to recover drone yaw in the goal frame.
        yaw_error = torch.atan2(observation[:, 15], observation[:, 12])
        yaw_rate = -self._evaluation.feasibility_yaw_gain_per_s * yaw_error
        return torch.cat(
            (
                body_velocity / self._action.max_body_velocity_mps,
                (yaw_rate / self._action.max_yaw_rate_radps).unsqueeze(-1),
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)


def _run_feasibility(*, config: ExperimentConfig, output_dir: Path) -> None:
    from robotics_experiments.workflows.isaac_lab import (
        evaluate_press_approach_policy,
        make_isaac_lab_env,
        press_policy_evaluation,
    )

    if config.rl.press.task != "hover" or config.rl.action.kind != "velocity":
        raise ValueError("hover feasibility requires the press-hover velocity task")
    inputs = prepare_policy_inputs(config=config, repo_root=_repo_root())
    contract = press_policy_contract(config=config, controller=inputs.controller)
    seeds = tuple(config.rl.press.evaluation.validation_seeds)
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "mode": "hover-feasibility",
        "set_name": "validation",
        "seeds": seeds,
        "episodes_per_seed": config.rl.press.evaluation.episodes_per_seed,
        "episode_length_s": config.rl.episode_length_s,
        "controller_path": str(inputs.controller_path),
        "controller_digest": inputs.controller_digest,
        "controller_run_id": inputs.controller.run_id,
        "flight_parameters_path": str(inputs.flight_parameters_path),
        "flight_parameters_digest": inputs.flight_parameters_digest,
        "flight_parameters_run_id": inputs.flight_parameters.run_id,
        "feedback": {
            name: getattr(config.rl.press.evaluation, name)
            for name in (
                "feasibility_position_gain_per_s",
                "feasibility_velocity_damping",
                "feasibility_yaw_gain_per_s",
            )
        },
    }
    report_path = output_dir / "feasibility.json"
    report_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        env, _env_cfg = make_isaac_lab_env(
            task_id=config.rl.training.task_id,
            training=False,
            experiment_config=config,
            controller_settings=inputs.controller.selected_settings,
            seed=seeds[0],
            num_envs=config.rl.press.evaluation.episodes_per_seed,
            policy_contract=contract,
            flight_parameters=inputs.flight_parameters,
            hover_reset_samples=inputs.hover_reset_samples,
        )
        try:
            result = evaluate_press_approach_policy(
                env=env,
                policy=_HoverFeedback(config=config),
                seeds=seeds,
                episodes_per_seed=config.rl.press.evaluation.episodes_per_seed,
                hold_steps=config.rl.press.goal.stable_steps,
                flight_parameters=inputs.flight_parameters,
                report=_report,
                trace_worst_episode=True,
            )
        finally:
            env.close()
        evaluated = press_policy_evaluation(
            iteration=0,
            checkpoint_path=report_path,
            policy_digest="",
            result=result,
            source="hover-feasibility",
        )
        acceptance = evaluate_press_acceptance(
            criteria=create_press_acceptance_criteria(
                config=config.rl.press.acceptance,
                flight_parameters=inputs.flight_parameters,
            ),
            evaluation=evaluated,
        )
        record.update(
            result=asdict(result), acceptance=asdict(acceptance), passed=acceptance.passed
        )
        report_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        if not acceptance.passed:
            raise AssertionError("hover feasibility failed: " + ", ".join(acceptance.reasons))
        _report(f"hover feasibility PASSED; report={report_path}")
    except Exception as exc:
        record.update(error=str(exc), passed=False)
        report_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        raise


def _run(
    *, config: ExperimentConfig, artifact_dir: Path, inputs: PolicyInputs
) -> PressApproachEvaluationResult:
    # AppLauncher must start before the Isaac Lab workflow import.
    from robotics_experiments.workflows.isaac_lab import (
        evaluate_press_approach_policy,
        make_isaac_lab_env,
        press_policy_evaluation,
    )

    evaluation = config.rl.press.evaluation
    seeds = tuple(int(seed) for seed in evaluation.test_seeds)
    metadata = validate_policy_artifact(artifact_dir=artifact_dir)
    contract = metadata.contract
    if contract.curriculum_stage != config.rl.press.task:
        raise ValueError(
            f"press artifact stage {contract.curriculum_stage!r} does not match "
            f"configured stage {config.rl.press.task!r}"
        )
    comparison_fingerprint = _press_comparison_fingerprint(
        config=config,
        contract=contract,
        flight_parameters_fingerprint=metadata.flight_parameters.artifact_fingerprint,
    )
    if metadata.values.get("comparison_fingerprint") != comparison_fingerprint:
        raise ValueError("published policy does not match the selected press configuration")
    if inputs.flight_parameters != metadata.flight_parameters:
        raise ValueError("configured flight parameters differ from policy metadata")
    env, env_cfg = make_isaac_lab_env(
        task_id=config.rl.training.task_id,
        training=False,
        experiment_config=config,
        controller_settings=metadata.controller_settings,
        seed=seeds[0],
        num_envs=evaluation.episodes_per_seed,
        policy_contract=contract,
        flight_parameters=metadata.flight_parameters,
        hover_reset_samples=inputs.hover_reset_samples,
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
            hold_steps=config.rl.press.goal.stable_steps,
            flight_parameters=metadata.flight_parameters,
            report=_report,
            trace_worst_episode=True,
        )
    finally:
        env.close()

    summary = result.summary
    _report(
        "press approach evaluation: "
        f"episodes={summary.episodes} "
        f"success={summary.success_rate:.3f} "
        f"collision={summary.collision_rate:.3f} "
        f"other_flight_failure={summary.other_flight_failure_rate:.3f} "
        f"timeout={summary.timeout_rate:.3f} "
        f"min_closest_range_m={summary.min_closest_range_m:.5f} "
        f"longest_hover_mean_s={summary.episode_longest_hover_duration_mean_s:.3f} "
        f"hover_occupancy_rate={summary.hover_envelope_occupancy_rate:.3f} "
        f"hover_utilization_mean={summary.hover_envelope_utilization_mean:.3f}"
    )
    for reward in result.reward_means:
        _report(f"reward_mean {reward.name}={reward.mean:.6f}")
    digest = hashlib.sha256((artifact_dir / POLICY_FILENAME).read_bytes()).hexdigest()
    if metadata.values.get("selected_policy_digest") != digest:
        raise ValueError("published actor digest does not match policy metadata")
    recorded_test = metadata.values.get("test_evaluation")
    expected_workload = {
        "role": "test",
        "seeds": list(seeds),
        "episodes_per_seed": evaluation.episodes_per_seed,
        "episode_count": len(seeds) * evaluation.episodes_per_seed,
    }
    if (
        not isinstance(recorded_test, dict)
        or recorded_test.get("policy_digest") != digest
        or recorded_test.get("workload") != expected_workload
    ):
        raise ValueError("published policy does not record the configured test workload")
    evaluated = press_policy_evaluation(
        iteration=int(metadata.values["selected_checkpoint_iteration"]),
        checkpoint_path=artifact_dir / POLICY_FILENAME,
        policy_digest=digest,
        result=result,
        source="independent",
    )
    acceptance = evaluate_press_acceptance(
        criteria=create_press_acceptance_criteria(
            config=config.rl.press.acceptance,
            flight_parameters=metadata.flight_parameters,
        ),
        evaluation=evaluated,
    )
    if not acceptance.passed:
        raise AssertionError("press acceptance failed: " + ", ".join(acceptance.reasons))
    _report("press approach evaluation PASSED")
    return result


def _resolve_artifact_dir(*, explicit_artifact_dir: Path | None, config_path: Path) -> Path:
    if explicit_artifact_dir is not None:
        return explicit_artifact_dir.resolve(strict=True)
    curriculum = load_curriculum_config(config_path)
    policy_root = _repo_root() / ".build/training" / curriculum.policy_name
    run_id = os.environ.get(_RUN_ID_ENV)
    if not run_id:
        return (policy_root / "latest").resolve(strict=True)
    report = json.loads((policy_root / run_id / "curriculum_report.json").read_text())
    published = [stage for stage in report["stages"] if stage["published"]]
    if not published:
        raise ValueError("curriculum has no published stage")
    return Path(published[-1]["report_path"]).resolve(strict=True).parent


def _producer_config(artifact_dir: Path) -> tuple[ExperimentConfig, PolicyInputs]:
    report = json.loads((artifact_dir / "training_report.json").read_text())
    if report["schema_version"] != 28 or report["report_kind"] != "training":
        raise ValueError("unsupported producer training report")
    provenance = report["provenance"]
    config = load_config(provenance["config_path"])
    if _config_values(config) != provenance["resolved_config"]:
        raise ValueError("producer stage configuration has changed")
    report = json.loads((artifact_dir / "training_report.json").read_text())
    evidence = report["provenance"]["inputs"]
    inputs = _load_pinned_policy_inputs(
        controller_path=Path(evidence["controller"]["path"]),
        controller_digest=evidence["controller"]["sha256"],
        flight_parameters_path=Path(evidence["flight_parameters"]["path"]),
        flight_parameters_digest=evidence["flight_parameters"]["sha256"],
    )
    return config, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hover-feasibility", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        config_path = Path(os.environ[_CONFIG_PATH_ENV])
        if args.hover_feasibility:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            _run_feasibility(
                config=load_config(config_path),
                output_dir=args.output_dir
                or _repo_root() / ".build" / "hover-feasibility" / run_id,
            )
        else:
            artifact_dir = _resolve_artifact_dir(
                explicit_artifact_dir=args.artifact_dir, config_path=config_path
            )
            config, inputs = _producer_config(artifact_dir)
            _run(config=config, inputs=inputs, artifact_dir=artifact_dir)
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

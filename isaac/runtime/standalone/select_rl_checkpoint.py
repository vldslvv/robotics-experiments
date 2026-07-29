"""Evaluate a training run's checkpoints and publish the selected artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import torch

import robotics_experiments.rl.training.rsl_rl as training
from robotics_experiments.config import (
    ExperimentConfig,
    PressApproachEvaluationConfig,
    PressApproachGoalConfig,
    load_config,
)
from robotics_experiments.rl import (
    METADATA_FILENAME,
    POLICY_FILENAME,
    PolicyContract,
    PolicyInterface,
    load_policy_metadata,
)
from robotics_experiments.rl.press import PressApproachResult
from robotics_experiments.rl.training.selection import (
    CandidateEvaluation,
    CheckpointCandidate,
    CheckpointSelection,
    MetricValue,
    enumerate_checkpoints,
    publish_policy_labels,
    select_checkpoint,
)
from robotics_experiments.workflows.isaac_lab import (
    PressApproachEvaluationResult,
    evaluate_press_approach_policy,
    make_isaac_lab_env,
)

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"
_REPORT_FILENAME = "checkpoint_selection.json"
_REPORT_SCHEMA_VERSION = 6


class _ActorCritic(Protocol):
    actor: torch.nn.Module


class _Algorithm(Protocol):
    policy: _ActorCritic
    actor_critic: _ActorCritic


class _Runner(Protocol):
    alg: _Algorithm
    log_dir: str

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = ...) -> None:
        """Run the training loop."""
        ...

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ) -> dict[str, Any]:
        """Restore a runner checkpoint."""
        ...


class _SelectionEnvironment(Protocol):
    device: str | torch.device
    max_episode_length: int
    press_approach_result: PressApproachResult[torch.Tensor] | None

    def reset(
        self, *, seed: int
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset the vector environment."""
        ...

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Advance the vector environment."""
        ...

    def close(self) -> None:
        """Release environment resources."""
        ...


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _press_evaluation(
    *,
    candidate: CheckpointCandidate,
    runner: _Runner,
    env: _SelectionEnvironment,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    evaluation_config: PressApproachEvaluationConfig,
    goal_config: PressApproachGoalConfig,
    device: str,
) -> CandidateEvaluation:
    _report(f"checkpoint selection iteration={candidate.iteration} START")
    training.load_checkpoint(
        runner=runner,
        checkpoint_path=candidate.path,
        device=device,
    )
    actor = training.deterministic_actor(runner=runner)
    result = evaluate_press_approach_policy(
        env=env,
        policy=actor,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        goal_config=goal_config,
        report=_report,
    )
    evaluation = _score_press_result(
        candidate=candidate,
        result=result,
        evaluation_config=evaluation_config,
        goal_range_min_m=float(goal_config.range_min_m),
    )
    summary = result.summary
    _report(
        f"checkpoint selection iteration={candidate.iteration} COMPLETE "
        f"advisory_accepted={str(evaluation.accepted).lower()} "
        f"score={evaluation.ranking_values[0]:.6f} "
        f"success={summary.approach_success_rate:.3f} "
        f"collision={summary.collision_rate:.3f} "
        f"body_collision={summary.body_collision_rate:.3f} "
        f"pusher_collision={summary.pusher_collision_rate:.3f} "
        f"other_flight_failure={summary.other_flight_failure_rate:.3f} "
        f"timeout={summary.timeout_rate:.3f}"
    )
    return evaluation


def _score_press_result(
    *,
    candidate: CheckpointCandidate,
    result: PressApproachEvaluationResult,
    evaluation_config: PressApproachEvaluationConfig,
    goal_range_min_m: float,
) -> CandidateEvaluation:
    summary = result.summary
    success_threshold = float(evaluation_config.success_rate_min)
    collision_threshold = float(evaluation_config.collision_rate_max)
    success_margin = summary.approach_success_rate / success_threshold
    collision_margin = (
        1.0 + (collision_threshold - summary.collision_rate) / collision_threshold
        if collision_threshold > 0.0
        else (1.0 if summary.collision_rate == 0.0 else -summary.collision_rate)
    )
    normalized_margins = (success_margin, collision_margin)
    accepted = (
        summary.approach_success_rate >= success_threshold
        and summary.collision_rate <= collision_threshold
        and summary.min_closest_range_m >= goal_range_min_m
    )
    quality_cost = (
        float(evaluation_config.collision_penalty) * summary.collision_rate
        + float(evaluation_config.flight_failure_penalty) * summary.other_flight_failure_rate
        + float(evaluation_config.timeout_penalty) * summary.timeout_rate
        + float(evaluation_config.non_success_penalty) * (1.0 - summary.approach_success_rate)
        + float(evaluation_config.terminal_range_weight) * summary.mean_terminal_range_cost
        + float(evaluation_config.trajectory_alignment_weight)
        * summary.mean_trajectory_alignment_cost
        + float(evaluation_config.closing_speed_overshoot_weight)
        * summary.mean_closing_speed_overshoot_cost
        + float(evaluation_config.upright_error_weight) * summary.mean_upright_error_cost
        + float(evaluation_config.action_delta_weight) * summary.mean_action_delta_cost
        + float(evaluation_config.completion_time_weight) * summary.mean_completion_time_fraction
    )
    score = -quality_cost
    if not math.isfinite(score):
        raise ValueError(f"checkpoint {candidate.path} produced a non-finite quality score")
    ranking_values = (score, -candidate.iteration)
    return CandidateEvaluation(
        candidate=candidate,
        accepted=accepted,
        metrics=(
            MetricValue("approach_success_rate", summary.approach_success_rate),
            MetricValue("collision_rate", summary.collision_rate),
            MetricValue("body_collision_rate", summary.body_collision_rate),
            MetricValue("pusher_collision_rate", summary.pusher_collision_rate),
            MetricValue("other_flight_failure_rate", summary.other_flight_failure_rate),
            MetricValue("timeout_rate", summary.timeout_rate),
            MetricValue("mean_closest_range_m", summary.mean_closest_range_m),
            MetricValue("min_closest_range_m", summary.min_closest_range_m),
            MetricValue("median_completion_time_s", summary.median_completion_time_s),
            MetricValue("terminal_range_cost", summary.mean_terminal_range_cost),
            MetricValue(
                "trajectory_alignment_cost",
                summary.mean_trajectory_alignment_cost,
            ),
            MetricValue(
                "closing_speed_overshoot_cost",
                summary.mean_closing_speed_overshoot_cost,
            ),
            MetricValue("upright_error_cost", summary.mean_upright_error_cost),
            MetricValue("action_delta_cost", summary.mean_action_delta_cost),
            MetricValue(
                "completion_time_fraction",
                summary.mean_completion_time_fraction,
            ),
            MetricValue("held_out_task_quality", score),
        ),
        normalized_margins=normalized_margins,
        ranking_values=ranking_values,
        reward_means=result.reward_means,
    )


def _comparison_fingerprint(
    *,
    config: ExperimentConfig,
    contract: PolicyContract,
) -> str | None:
    if contract.policy_interface is not PolicyInterface.PRESS_V1:
        return None
    vehicle_label, vehicle = next(
        (label, item)
        for label, item in config.vehicles.items()
        if item.id == config.rl.training.vehicle_id
    )
    payload = {
        "selection_metric": config.rl.press.evaluation.selection_metric,
        "evaluation": asdict(config.rl.press.evaluation),
        "goal": asdict(config.rl.press.goal),
        "handoff": asdict(config.rl.press.handoff),
        "reset": asdict(config.rl.press.reset),
        "termination": asdict(config.rl.press.termination),
        "scene": config.scene,
        "switches": asdict(config.switches),
        "vehicle": vehicle_label,
        "airframe": vehicle.spawn.usd_path,
        "lab_vehicle": asdict(config.rl.lab_vehicle),
        "physics_hz": config.sim.physics_hz,
        "episode_length_s": config.rl.episode_length_s,
        "task_id": config.rl.training.task_id,
        "policy_interface": contract.policy_interface.value,
        "curriculum_stage": contract.curriculum_stage,
        "observation_schema_version": contract.obs_schema_version,
        "observation_dimension": contract.obs_dim,
        "policy_rate_hz": contract.policy_rate_hz,
        "action_kind": contract.action_kind,
        "action_dimension": contract.action_dim,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selection_report(
    *,
    selection: CheckpointSelection,
    run_dir: Path,
    policy_name: str,
    policy_interface: str,
    curriculum_stage: str | None,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    evaluation_config: PressApproachEvaluationConfig,
    goal_range_min_m: float,
    all_candidates: tuple[CheckpointCandidate, ...],
    comparison_fingerprint: str | None,
) -> dict[str, Any]:
    thresholds = (
        {
            "approach_success_rate_min": float(evaluation_config.success_rate_min),
            "collision_rate_max": float(evaluation_config.collision_rate_max),
            "min_closest_range_m": goal_range_min_m,
        }
        if policy_interface == "press-v1"
        else None
    )
    evaluated = {evaluation.candidate.iteration: evaluation for evaluation in selection.evaluations}
    selected_evaluation = next(
        (
            evaluation
            for evaluation in selection.evaluations
            if evaluation.candidate.iteration == selection.selected.iteration
        ),
        None,
    )
    candidates = []
    for candidate in all_candidates:
        evaluation = evaluated.get(candidate.iteration)
        record: dict[str, Any] = {
            "iteration": candidate.iteration,
            "checkpoint": str(candidate.path.relative_to(run_dir)),
            "evaluated": evaluation is not None,
        }
        if evaluation is not None:
            record |= {
                "advisory_accepted": evaluation.accepted,
                "metrics": {metric.name: metric.value for metric in evaluation.metrics},
                "normalized_margins": list(evaluation.normalized_margins),
                "ranking_values": list(evaluation.ranking_values),
                "reward_means": {reward.name: reward.mean for reward in evaluation.reward_means},
            }
        candidates.append(record)
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "artifact_run_id": run_dir.name,
        "policy_name": policy_name,
        "policy_interface": policy_interface,
        "curriculum_stage": curriculum_stage,
        "selection_method": selection.selection_method,
        "comparison_fingerprint": comparison_fingerprint,
        "seeds": list(seeds),
        "episodes_per_seed": episodes_per_seed,
        "thresholds": thresholds,
        "candidates": candidates,
        "selected_iteration": selection.selected.iteration,
        "selected_score": (
            selected_evaluation.ranking_values[0] if selected_evaluation is not None else None
        ),
        "selected_advisory_accepted": (
            selected_evaluation.accepted if selected_evaluation is not None else None
        ),
    }


def _select_candidates(
    *,
    policy_interface: str,
    candidates: tuple[CheckpointCandidate, ...],
    runner: _Runner,
    env: _SelectionEnvironment,
    seeds: tuple[int, ...],
    evaluation_config: PressApproachEvaluationConfig,
    goal_config: PressApproachGoalConfig,
    device: str,
) -> CheckpointSelection:
    scorer = None
    selection_method = "final-no-scorer"
    if policy_interface == "press-v1":

        def score_press(
            scorer_candidates: tuple[CheckpointCandidate, ...],
        ) -> tuple[CandidateEvaluation, ...]:
            return tuple(
                _press_evaluation(
                    candidate=candidate,
                    runner=runner,
                    env=env,
                    seeds=seeds,
                    episodes_per_seed=evaluation_config.episodes_per_seed,
                    evaluation_config=evaluation_config,
                    goal_config=goal_config,
                    device=device,
                )
                for candidate in scorer_candidates
            )

        scorer = score_press
        selection_method = evaluation_config.selection_metric
    return select_checkpoint(
        policy_interface=policy_interface,
        candidates=candidates,
        selection_method=selection_method,
        scorer=scorer,
    )


def _publish_selection(
    *,
    selection: CheckpointSelection,
    report: dict[str, Any],
    metadata: dict[str, Any],
    runner: _Runner,
    run_dir: Path,
    policy_root: Path,
    device: str,
) -> bool:
    selected_metadata = metadata | {
        "selected_checkpoint_iteration": selection.selected.iteration,
        "selection_method": selection.selection_method,
    }
    selected_evaluation = next(
        (
            evaluation
            for evaluation in selection.evaluations
            if evaluation.candidate.iteration == selection.selected.iteration
        ),
        None,
    )
    if selected_evaluation is not None:
        selected_metadata["checkpoint_evaluation"] = {
            "selection_metric": selection.selection_method,
            "comparison_fingerprint": report["comparison_fingerprint"],
            "score": selected_evaluation.ranking_values[0],
            "advisory_accepted": selected_evaluation.accepted,
            "seeds": report["seeds"],
            "episodes_per_seed": report["episodes_per_seed"],
            "thresholds": report["thresholds"],
            "metrics": {metric.name: metric.value for metric in selected_evaluation.metrics},
            "normalized_margins": list(selected_evaluation.normalized_margins),
            "ranking_values": list(selected_evaluation.ranking_values),
            "reward_means": {
                reward.name: reward.mean for reward in selected_evaluation.reward_means
            },
        }
    training.load_checkpoint(
        runner=runner,
        checkpoint_path=selection.selected.path,
        device=device,
    )
    with tempfile.TemporaryDirectory(prefix=".checkpoint-selection-", dir=run_dir) as stage_name:
        stage_dir = Path(stage_name)
        training.export_policy(runner, stage_dir, metadata=selected_metadata)
        (stage_dir / _REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        for filename in (POLICY_FILENAME, METADATA_FILENAME, _REPORT_FILENAME):
            (stage_dir / filename).replace(run_dir / filename)
    return publish_policy_labels(
        policy_root=policy_root,
        run_dir=run_dir,
        comparison_fingerprint=report["comparison_fingerprint"],
        score=report["selected_score"],
    )


def _selected_evaluation(selection: CheckpointSelection) -> CandidateEvaluation | None:
    return next(
        (
            evaluation
            for evaluation in selection.evaluations
            if evaluation.candidate.iteration == selection.selected.iteration
        ),
        None,
    )


def _run(*, config: ExperimentConfig, run_id: str) -> CheckpointSelection:
    training_config = config.rl.training
    policy_root = _repo_root() / ".build" / "training" / training_config.policy_name
    run_dir = policy_root / run_id
    if not run_dir.is_dir() or run_dir.parent != policy_root:
        raise ValueError(f"RL_RUN_ID does not identify a checkpoint run: {run_id!r}")
    policy_metadata = load_policy_metadata(artifact_dir=run_dir)
    metadata = dict(policy_metadata.values)
    contract = policy_metadata.contract
    comparison_fingerprint = _comparison_fingerprint(config=config, contract=contract)
    if (
        contract.policy_interface is PolicyInterface.PRESS_V1
        and contract.curriculum_stage != config.rl.press.curriculum_stage
    ):
        raise ValueError(
            f"press checkpoint run stage {contract.curriculum_stage!r} does not match "
            f"configured stage {config.rl.press.curriculum_stage!r}"
        )
    candidates = enumerate_checkpoints(logs_dir=run_dir / "logs")
    evaluation = config.rl.press.evaluation
    is_press = contract.policy_interface is PolicyInterface.PRESS_V1
    seeds = tuple(int(seed) for seed in evaluation.seeds) if is_press else ()
    env_seed = seeds[0] if seeds else int(training_config.seed)
    num_envs = evaluation.episodes_per_seed if is_press else 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env, env_cfg = make_isaac_lab_env(
        task_id=training_config.task_id,
        experiment_config=config,
        controller_settings=policy_metadata.controller_settings,
        seed=env_seed,
        num_envs=num_envs,
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
            raise ValueError("Isaac Lab selection environment changed the stored contract")
        runner = training.build_runner(
            env=env,
            training_config=training_config,
            policy_config=env_cfg.policy_config,
            log_dir=str(run_dir / "selection"),
            device=device,
        )
        selection = _select_candidates(
            policy_interface=contract.policy_interface.value,
            candidates=candidates,
            runner=runner,
            env=env,
            seeds=seeds,
            evaluation_config=evaluation,
            goal_config=config.rl.press.goal,
            device=device,
        )
        report = _selection_report(
            selection=selection,
            run_dir=run_dir,
            policy_name=training_config.policy_name,
            policy_interface=contract.policy_interface.value,
            curriculum_stage=contract.curriculum_stage,
            seeds=seeds,
            episodes_per_seed=evaluation.episodes_per_seed if is_press else 0,
            evaluation_config=evaluation,
            goal_range_min_m=config.rl.press.goal.range_min_m,
            all_candidates=candidates,
            comparison_fingerprint=comparison_fingerprint,
        )
        best_updated = _publish_selection(
            selection=selection,
            report=report,
            metadata=metadata,
            runner=runner,
            run_dir=run_dir,
            policy_root=policy_root,
            device=device,
        )
    finally:
        env.close()

    selected_evaluation = _selected_evaluation(selection)
    outcome = (
        f"score={selected_evaluation.ranking_values[0]:.6f} "
        f"advisory_accepted={str(selected_evaluation.accepted).lower()}"
        if selected_evaluation is not None
        else "unscored"
    )
    _report(
        f"checkpoint selection COMPLETE run={run_id} "
        f"iteration={selection.selected.iteration} {outcome} "
        f"method={selection.selection_method} latest={policy_root / 'latest'} "
        f"best_updated={str(best_updated).lower()}"
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=os.environ.get(_RUN_ID_ENV))
    parser.add_argument("--scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()
    if not args.run_id:
        raise SystemExit("RL_RUN_ID or --run-id is required")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        config_path = os.environ[_CONFIG_PATH_ENV]
        _run(config=load_config(config_path), run_id=args.run_id)
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

"""Evaluate a training run's checkpoints and publish the selected artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from robotics_experiments.config import ExperimentConfig, load_config
from robotics_experiments.rl.training.selection import (
    CandidateEvaluation,
    CheckpointCandidate,
    CheckpointSelection,
    MetricValue,
    enumerate_checkpoints,
    publish_latest,
    register_checkpoint_scorer,
    select_checkpoint,
)

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"
_REPORT_FILENAME = "checkpoint_selection.json"
_REPORT_SCHEMA_VERSION = 2
_PRESS_SELECTION_METHOD = "press-gated-maximin-v2"


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _stored_metadata(*, run_dir: Path) -> dict[str, Any]:
    from robotics_experiments.rl.press import METADATA_FILENAME

    path = run_dir / METADATA_FILENAME
    if not path.is_file():
        raise ValueError(f"checkpoint run {run_dir} is missing {METADATA_FILENAME}")
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint run {run_dir} metadata must be a JSON object")
    return metadata


def _press_evaluation(
    *,
    candidate: CheckpointCandidate,
    runner: Any,
    env: Any,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    evaluation_config: Any,
    device: str,
) -> CandidateEvaluation:
    import robotics_experiments.rl.training.rsl_rl as training
    from robotics_experiments.workflows.isaac_lab import evaluate_press_policy

    _report(f"checkpoint selection iteration={candidate.iteration} START")
    training.load_checkpoint(
        runner=runner,
        checkpoint_path=candidate.path,
        device=device,
    )
    actor = training.deterministic_actor(runner=runner)
    result = evaluate_press_policy(
        env=env,
        policy=actor,
        seeds=seeds,
        episodes_per_seed=episodes_per_seed,
        report=_report,
    )
    evaluation = _score_press_result(
        candidate=candidate,
        result=result,
        evaluation_config=evaluation_config,
    )
    summary = result.summary
    _report(
        f"checkpoint selection iteration={candidate.iteration} COMPLETE "
        f"accepted={str(evaluation.accepted).lower()} "
        f"success={summary.confirmed_toggle_rate:.3f} "
        f"first_contact_stability={summary.stable_first_contact_rate:.3f} "
        f"retreat={summary.post_toggle_retreat_rate:.3f}"
    )
    return evaluation


def _score_press_result(
    *,
    candidate: CheckpointCandidate,
    result: Any,
    evaluation_config: Any,
) -> CandidateEvaluation:
    summary = result.summary
    thresholds = (
        float(evaluation_config.success_rate_min),
        float(evaluation_config.first_contact_stability_min),
        float(evaluation_config.retreat_rate_min),
    )
    acceptance_rates = (
        summary.confirmed_toggle_rate,
        summary.stable_first_contact_rate,
        summary.post_toggle_retreat_rate,
    )
    normalized_margins = tuple(
        rate / threshold for rate, threshold in zip(acceptance_rates, thresholds, strict=True)
    )
    accepted = all(
        rate >= threshold for rate, threshold in zip(acceptance_rates, thresholds, strict=True)
    )
    median_toggle_time_s = summary.median_toggle_time_s
    ranking_values = (
        *sorted(normalized_margins),
        summary.full_completion_rate,
        -summary.peak_compression_overshoot_m,
        -(median_toggle_time_s if median_toggle_time_s is not None else sys.float_info.max),
        -candidate.iteration,
    )
    return CandidateEvaluation(
        candidate=candidate,
        accepted=accepted,
        metrics=(
            MetricValue("confirmed_toggle_rate", summary.confirmed_toggle_rate),
            MetricValue("stable_first_contact_rate", summary.stable_first_contact_rate),
            MetricValue("post_toggle_retreat_rate", summary.post_toggle_retreat_rate),
            MetricValue("full_completion_rate", summary.full_completion_rate),
            MetricValue(
                "peak_compression_overshoot_m",
                summary.peak_compression_overshoot_m,
            ),
            MetricValue("median_toggle_time_s", summary.median_toggle_time_s),
        ),
        normalized_margins=normalized_margins,
        ranking_values=ranking_values,
        reward_means=result.reward_means,
    )


def _selection_report(
    *,
    selection: CheckpointSelection,
    run_dir: Path,
    policy_name: str,
    policy_interface: str,
    seeds: tuple[int, ...],
    episodes_per_seed: int,
    evaluation_config: Any,
    all_candidates: tuple[CheckpointCandidate, ...],
) -> dict[str, Any]:
    thresholds = (
        {
            "confirmed_toggle_rate_min": float(evaluation_config.success_rate_min),
            "stable_first_contact_rate_min": float(evaluation_config.first_contact_stability_min),
            "post_toggle_retreat_rate_min": float(evaluation_config.retreat_rate_min),
        }
        if policy_interface == "press-v1"
        else None
    )
    evaluated = {evaluation.candidate.iteration: evaluation for evaluation in selection.candidates}
    selected_evaluation = next(
        (
            evaluation
            for evaluation in selection.candidates
            if selection.selected is not None
            and evaluation.candidate.iteration == selection.selected.iteration
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
                "accepted": evaluation.accepted,
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
        "selection_method": selection.selection_method,
        "seeds": list(seeds),
        "episodes_per_seed": episodes_per_seed,
        "thresholds": thresholds,
        "candidates": candidates,
        "selected_iteration": (
            selection.selected.iteration if selection.selected is not None else None
        ),
        "selected_accepted": (
            selected_evaluation.accepted if selected_evaluation is not None else None
        ),
    }


def _select_candidates(
    *,
    policy_interface: str,
    candidates: tuple[CheckpointCandidate, ...],
    runner: Any,
    env: Any,
    seeds: tuple[int, ...],
    evaluation_config: Any,
    device: str,
) -> CheckpointSelection:
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
                    device=device,
                )
                for candidate in scorer_candidates
            )

        register_checkpoint_scorer(
            policy_interface=policy_interface,
            selection_method=_PRESS_SELECTION_METHOD,
            scorer=score_press,
        )
    return select_checkpoint(
        policy_interface=policy_interface,
        candidates=candidates,
    )


def _publish_selection(
    *,
    selection: CheckpointSelection,
    report: dict[str, Any],
    metadata: dict[str, Any],
    runner: Any,
    run_dir: Path,
    policy_root: Path,
    device: str,
) -> None:
    import robotics_experiments.rl.training.rsl_rl as training
    from robotics_experiments.rl.press import METADATA_FILENAME, POLICY_FILENAME

    if selection.selected is None:
        raise ValueError("cannot publish a checkpoint selection with no winner")
    selected_metadata = metadata | {
        "selected_checkpoint_iteration": selection.selected.iteration,
        "selection_method": selection.selection_method,
    }
    selected_evaluation = next(
        (
            evaluation
            for evaluation in selection.candidates
            if evaluation.candidate.iteration == selection.selected.iteration
        ),
        None,
    )
    if selected_evaluation is not None:
        selected_metadata["checkpoint_evaluation"] = {
            "accepted": selected_evaluation.accepted,
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
    publish_latest(policy_root=policy_root, run_dir=run_dir)


def _run(*, config: ExperimentConfig, run_id: str) -> CheckpointSelection:
    import robotics_experiments.rl.training.rsl_rl as training
    from robotics_experiments.rl.press import load_policy_contract
    from robotics_experiments.workflows.isaac_lab import make_isaac_lab_env

    training_config = config.rl.training
    policy_root = _repo_root() / ".build" / "training" / training_config.policy_name
    run_dir = policy_root / run_id
    if not run_dir.is_dir() or run_dir.parent != policy_root:
        raise ValueError(f"RL_RUN_ID does not identify a checkpoint run: {run_id!r}")
    metadata = _stored_metadata(run_dir=run_dir)
    contract = load_policy_contract(artifact_dir=run_dir)
    candidates = enumerate_checkpoints(logs_dir=run_dir / "logs")
    evaluation = config.rl.press.evaluation
    is_press = contract.policy_interface == "press-v1"
    seeds = tuple(int(seed) for seed in evaluation.seeds) if is_press else ()
    env_seed = seeds[0] if seeds else int(training_config.seed)
    num_envs = evaluation.episodes_per_seed if is_press else 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env, env_cfg = make_isaac_lab_env(
        task_id=training_config.task_id,
        experiment_config=config,
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
            contract.policy_interface,
        ):
            raise ValueError("Isaac Lab selection environment changed the stored contract")
        runner = training.build_runner(
            env,
            training_config,
            log_dir=str(run_dir / "selection"),
            device=device,
        )
        selection = _select_candidates(
            policy_interface=contract.policy_interface,
            candidates=candidates,
            runner=runner,
            env=env,
            seeds=seeds,
            evaluation_config=evaluation,
            device=device,
        )
        report = _selection_report(
            selection=selection,
            run_dir=run_dir,
            policy_name=training_config.policy_name,
            policy_interface=contract.policy_interface,
            seeds=seeds,
            episodes_per_seed=evaluation.episodes_per_seed if is_press else 0,
            evaluation_config=evaluation,
            all_candidates=candidates,
        )
        _publish_selection(
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

    if selection.selected is None:
        raise AssertionError("checkpoint selector returned no winner")
    selected_accepted = next(
        (
            evaluation.accepted
            for evaluation in selection.candidates
            if evaluation.candidate.iteration == selection.selected.iteration
        ),
        None,
    )
    outcome = (
        "accepted"
        if selected_accepted
        else "fallback"
        if selected_accepted is not None
        else "selected"
    )
    _report(
        f"checkpoint selection COMPLETE run={run_id} "
        f"iteration={selection.selected.iteration} outcome={outcome} "
        f"method={selection.selection_method} latest={policy_root / 'latest'}"
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

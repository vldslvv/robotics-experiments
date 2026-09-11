"""Run a deterministic multi-attempt policy-training campaign.

Each attempt owns a fresh Isaac process, environment, runner, and optimizer.
Validation ranks exact exported actors on one fixed workload. Later phases
restore the attempt-wide best resumable checkpoint before refinement. One
disjoint, fresh test workload accepts the immutable selected actor before
publication.

Run inside the Isaac Lab container (make train-rl):
    /isaac-sim/python.sh /isaac-sim/proj/isaac/runtime/standalone/train_rl.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import robotics_experiments.rl as rl
import robotics_experiments.rl.training.rsl_rl as training
from robotics_experiments.config import (
    ExperimentConfig,
    RlPlateauCriteriaConfig,
    RlTrainingPhaseConfig,
    load_config,
)
from robotics_experiments.controller import ControllerArtifact
from robotics_experiments.flight_parameters import FlightParameters, load_flight_parameters_bundle
from robotics_experiments.rl import METADATA_FILENAME, POLICY_FILENAME, HoverResetSamples
from robotics_experiments.rl.evaluation import EvaluationMetric, PolicyEvaluation, RewardMean
from robotics_experiments.rl.press import (
    AcceptanceCheck,
    AcceptanceResult,
    PressAcceptanceCriteria,
    create_press_acceptance_criteria,
    evaluate_press_acceptance,
    press_policy_contract,
)
from robotics_experiments.rl.press._fingerprint import _press_comparison_fingerprint
from robotics_experiments.rl.training._monitoring import (
    metric_registry_payload,
    write_evaluation_scalars,
    write_reset_scalars,
    write_system_scalars,
)
from robotics_experiments.rl.training._telemetry import TelemetrySampler, probe_telemetry
from robotics_experiments.rl.training.progress import (
    PlateauState,
    advance_plateau,
    select_best_evaluation,
)
from robotics_experiments.rl.training.publication import publish_policy_labels
from robotics_experiments.rl.workflow import _load_pinned_policy_inputs

_POLICY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"
_CAMPAIGN_RUN_DIR_ENV = "RL_CAMPAIGN_RUN_DIR"
_REPORT_FILENAME = "training_report.json"
_ATTEMPT_REPORT_FILENAME = "attempt_report.json"
_TELEMETRY_FILENAME = "telemetry.json"
_REPORT_SCHEMA_VERSION = 25


class _TrackioWriter:
    """Write one campaign attempt to its local Trackio database."""

    def __init__(
        self,
        *,
        monitoring_dir: Path,
        project: str,
        run_name: str,
        campaign_run_id: str,
        training_seed: int,
        resume: str,
    ) -> None:
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRACKIO_DIR"] = str(monitoring_dir)
        import trackio

        self.log_dir = str(monitoring_dir)
        self._finish = trackio.finish
        self._run = trackio.init(
            project=project,
            name=run_name,
            config={
                "campaign_run_id": campaign_run_id,
                "training_seed": training_seed,
            },
            resume=resume,
            embed=False,
            auto_log_gpu=False,
            auto_log_cpu=False,
        )

    def add_scalar(
        self,
        tag: str,
        scalar_value: Any,
        global_step: int,
        /,
        *args,
        **kwargs,
    ) -> None:
        del args, kwargs
        self._run.log({tag: float(scalar_value)}, step=int(global_step))

    def flush(self) -> None:
        """Allow Trackio's live background writer to flush its queue."""

    def close(self) -> None:
        self._finish()


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _raise_keyboard_interrupt(_signum: int, _frame: Any, /) -> None:
    raise KeyboardInterrupt


def _install_interrupt_handlers() -> None:
    for signal_number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, _raise_keyboard_interrupt)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _make_run_dir(*, out_root: Path, run_id: str | None) -> Path:
    resolved_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{os.getpid()}"
    if _RUN_ID_PATTERN.fullmatch(resolved_id) is None:
        raise ValueError(
            "RL_RUN_ID must be a non-empty single path component using only "
            "letters, digits, '.', '_' or '-'"
        )
    run_dir = out_root / resolved_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _validate_policy_name(policy_name: str) -> str:
    if _POLICY_NAME_PATTERN.fullmatch(policy_name) is None:
        raise ValueError(
            "rl.training.policy_name must be a non-empty single path component "
            "using only letters, digits, '_' or '-'"
        )
    return policy_name


def _campaign_phases(*, config: ExperimentConfig) -> tuple[RlTrainingPhaseConfig, ...]:
    if config.rl.training.task_id == "Isaac-Switch-Press-Direct-v1":
        return tuple(config.rl.press.selected_stage().phases)
    return tuple(config.rl.training.phases)


def _campaign_plateau(*, config: ExperimentConfig) -> RlPlateauCriteriaConfig:
    if config.rl.training.task_id == "Isaac-Switch-Press-Direct-v1":
        return config.rl.press.selected_stage().plateau
    return config.rl.training.plateau


def _git_source_metadata(*, repo_root: Path) -> dict[str, str | bool]:
    git = ["git", "-c", f"safe.directory={repo_root.resolve()}"]
    commit = subprocess.run(
        [*git, "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"source_commit": commit, "source_dirty": bool(status)}


def _build_env(
    *,
    task_id: str,
    training: bool,
    seed: int,
    num_envs: int | None,
    experiment_config: ExperimentConfig,
    controller_settings,
    flight_parameters: FlightParameters,
    hover_reset_samples: HoverResetSamples,
    policy_contract=None,
):
    # AppLauncher must start before the Isaac Lab workflow import.
    from robotics_experiments.workflows.isaac_lab import make_isaac_lab_env

    return make_isaac_lab_env(
        task_id=task_id,
        training=training,
        experiment_config=experiment_config,
        controller_settings=controller_settings,
        seed=seed,
        num_envs=num_envs,
        policy_contract=policy_contract,
        flight_parameters=flight_parameters,
        hover_reset_samples=hover_reset_samples,
    )


def _artifact_metadata(
    *,
    config: ExperimentConfig,
    env_cfg,
    source_metadata: dict[str, str | bool],
    artifact_run_id: str,
    controller_artifact: ControllerArtifact,
    controller_digest: str,
    flight_parameters: FlightParameters,
    flight_parameters_digest: str,
    hover_reset_samples: HoverResetSamples,
) -> dict[str, Any]:
    metadata = rl.policy_metadata(
        action_cfg=config.rl.action,
        controller_artifact=controller_artifact,
        controller_digest=controller_digest,
        flight_parameters=flight_parameters,
        flight_parameters_digest=flight_parameters_digest,
        obs_schema_version=int(env_cfg.observation_schema_version),
        obs_dim=int(env_cfg.observation_space),
        policy_rate_hz=config.rl.policy_rate_hz,
        policy_interface=str(env_cfg.policy_interface),
        curriculum_stage=(
            config.rl.press.selected_stage().name
            if str(env_cfg.policy_interface) == rl.PolicyInterface.PRESS.value
            else None
        ),
    )
    metadata["hover_reset_distribution"] = {
        "source": "flight_parameters_policy_window",
        **hover_reset_samples.summary(flight_parameters=flight_parameters),
    }
    return metadata | source_metadata | {"artifact_run_id": artifact_run_id}


def _write_json(*, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _record_telemetry(*, path: Path, interval: dict[str, Any], target_rate: float) -> None:
    summary = interval.get("summary")
    if interval.get("status") == "complete" and isinstance(summary, dict) and summary:
        measured = float(summary["gpu_utilization_rate_mean"])
        interval["diagnostic"] = {
            "gpu_utilization_target_rate": target_rate,
            "gpu_utilization_target_met": measured >= target_rate,
            "measured_bottleneck": (
                "none_observed"
                if measured >= target_rate
                else "gpu_utilization_below_diagnostic_target"
            ),
        }
    if path.is_file():
        payload = json.loads(path.read_text())
    else:
        payload = {
            "schema_version": 1,
            "diagnostic_gpu_utilization_target_rate": target_rate,
            "intervals": [],
        }
    intervals = payload.get("intervals")
    if not isinstance(intervals, list):
        raise ValueError(f"telemetry artifact is invalid: {path}")
    intervals.append(interval)
    _write_json(path=path, payload=payload)


def _require_telemetry_interval(*, interval: dict[str, Any]) -> dict[str, float]:
    summary = interval.get("summary")
    if interval.get("status") != "complete" or not isinstance(summary, dict) or not summary:
        raise RuntimeError(str(interval.get("error") or "telemetry interval failed"))
    return {str(name): float(value) for name, value in summary.items()}


def _telemetry_report_summary(*, summary: dict[str, float], target_rate: float) -> dict[str, Any]:
    measured = summary["gpu_utilization_rate_mean"]
    return {
        **summary,
        "gpu_utilization_target_rate": target_rate,
        "gpu_utilization_target_met": measured >= target_rate,
        "measured_bottleneck": (
            "none_observed"
            if measured >= target_rate
            else "gpu_utilization_below_diagnostic_target"
        ),
    }


def _write_artifact_metadata(*, run_dir: Path, metadata: dict[str, Any]) -> None:
    _write_json(path=run_dir / METADATA_FILENAME, payload=metadata)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _previous_stage_checkpoint(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    contract: rl.PolicyContract,
    controller_digest: str,
    flight_parameters_digest: str,
) -> tuple[Path, dict[str, Any]] | None:
    stages = tuple(config.rl.press.curriculum.stages)
    active_index = next(
        index
        for index, stage in enumerate(stages)
        if stage.name == config.rl.press.curriculum.active_stage
    )
    if active_index == 0:
        return None

    previous = stages[active_index - 1]
    run_dir = (repo_root / ".build" / "training" / f"press-{previous.name}" / "best").resolve(
        strict=True
    )
    metadata = rl.validate_policy_artifact(artifact_dir=run_dir)
    if replace(metadata.contract, curriculum_stage=contract.curriculum_stage) != contract:
        raise ValueError("previous press stage has an incompatible policy contract")
    if metadata.contract.curriculum_stage != previous.name:
        raise ValueError("previous press stage artifact does not match curriculum order")
    if metadata.controller_digest != controller_digest:
        raise ValueError("previous press stage uses a different controller artifact")
    if metadata.flight_parameters_digest != flight_parameters_digest:
        raise ValueError("previous press stage uses a different flight parameters")

    report = json.loads((run_dir / _REPORT_FILENAME).read_text(encoding="utf-8"))
    acceptance = report.get("acceptance")
    if (
        report.get("status") != "accepted"
        or not isinstance(acceptance, dict)
        or not acceptance.get("passed")
    ):
        raise ValueError("previous press stage has no accepted campaign")
    selected_checkpoint = report.get("selected_checkpoint")
    if not isinstance(selected_checkpoint, str):
        raise ValueError("previous press stage report has no selected checkpoint")
    checkpoint = (run_dir / selected_checkpoint).resolve(strict=True)
    if not checkpoint.is_relative_to(run_dir):
        raise ValueError("previous press stage checkpoint escapes its campaign")
    return checkpoint, {
        "method": "previous-accepted-stage",
        "stage": previous.name,
        "run_id": run_dir.name,
        "checkpoint_sha256": _file_sha256(checkpoint),
        "policy_digest": metadata.values["selected_policy_digest"],
    }


def _procedural_smoke_phases(
    *, phases: tuple[RlTrainingPhaseConfig, ...]
) -> tuple[RlTrainingPhaseConfig, ...]:
    """Retain every phase boundary while limiting each branch to one update."""
    return tuple(replace(phase, max_iterations=1, evaluation_interval=1) for phase in phases)


def _evaluation_from_payload(
    *,
    payload: dict[str, Any],
    iteration: int,
    checkpoint_path: Path,
    policy_digest: str,
    source: str,
) -> PolicyEvaluation:
    metrics = payload.get("metrics")
    reward_means = payload.get("reward_means")
    if not isinstance(metrics, dict) or not isinstance(reward_means, dict):
        raise ValueError("policy evaluation report has an invalid metric contract")
    return PolicyEvaluation(
        iteration=iteration,
        checkpoint_path=checkpoint_path,
        policy_digest=policy_digest,
        metrics=tuple(
            EvaluationMetric(name=str(name), value=value) for name, value in metrics.items()
        ),
        reward_means=tuple(
            RewardMean(name=str(name), mean=float(value)) for name, value in reward_means.items()
        ),
        source=source,
    )


def _evaluation_worker_main(
    *,
    config_path: str,
    run_dir: str,
    flight_parameters_path: str,
    flight_parameters_digest: str,
    task_queue,
    result_queue,
    workload_role: str,
) -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        config = load_config(config_path)
        metadata = rl.load_policy_metadata(artifact_dir=Path(run_dir))
        flight_parameters_bundle = load_flight_parameters_bundle(path=flight_parameters_path)
        if flight_parameters_bundle.flight_parameters_digest != flight_parameters_digest:
            raise ValueError("pinned flight parameters changed before evaluation")
        if flight_parameters_bundle.flight_parameters != metadata.flight_parameters:
            raise ValueError("evaluation flight parameters differ from policy metadata")
        hover_reset_samples = rl.load_hover_reset_samples(bundle=flight_parameters_bundle)
        evaluation_config = config.rl.press.evaluation
        if workload_role == "validation":
            seeds = tuple(int(seed) for seed in evaluation_config.validation_seeds)
        elif workload_role == "test":
            seeds = tuple(int(seed) for seed in evaluation_config.test_seeds)
        else:
            raise ValueError(f"unsupported evaluation workload role {workload_role!r}")
        env, _env_cfg = _build_env(
            task_id=config.rl.training.task_id,
            training=False,
            seed=seeds[0],
            num_envs=evaluation_config.episodes_per_seed,
            experiment_config=config,
            controller_settings=metadata.controller_settings,
            flight_parameters=metadata.flight_parameters,
            hover_reset_samples=hover_reset_samples,
            policy_contract=metadata.contract,
        )
        telemetry = TelemetrySampler(
            sample_interval_s=config.rl.training.telemetry_sample_interval_s
        )
        try:
            result_queue.put(("ready", None))
            while True:
                request = task_queue.get()
                if request is None:
                    break
                iteration, checkpoint_name, artifact_name, policy_digest = request
                artifact_dir = Path(artifact_name)
                candidate_metadata = rl.validate_policy_artifact(artifact_dir=artifact_dir)
                if candidate_metadata.contract != metadata.contract:
                    raise ValueError("evaluation candidate changed the policy contract")
                policy = torch.jit.load(
                    str(artifact_dir / POLICY_FILENAME),
                    map_location=env.device,
                )
                policy.eval()
                # AppLauncher must start before the Isaac Lab workflow import.
                from robotics_experiments.workflows.isaac_lab import (
                    evaluate_press_approach_policy,
                    press_policy_evaluation,
                )

                telemetry.start(role="val" if workload_role == "validation" else "test")
                try:
                    rollout = evaluate_press_approach_policy(
                        env=env,
                        policy=policy,
                        seeds=seeds,
                        episodes_per_seed=evaluation_config.episodes_per_seed,
                        hold_steps=round(
                            config.rl.press.selected_stage().hold_s * config.rl.policy_rate_hz
                        ),
                        flight_parameters=candidate_metadata.flight_parameters,
                        report=_report,
                        trace_worst_episode=False,
                    )
                finally:
                    telemetry_interval = telemetry.finish(
                        work_items=len(seeds) * evaluation_config.episodes_per_seed
                    )
                evaluation = press_policy_evaluation(
                    iteration=int(iteration),
                    checkpoint_path=Path(checkpoint_name),
                    policy_digest=str(policy_digest),
                    result=rollout,
                    source=f"fresh-{workload_role}",
                )
                result_queue.put(("evaluation", (evaluation, telemetry_interval)))
        finally:
            env.close()
    except Exception:  # noqa: BLE001 - propagate the complete worker failure
        result_queue.put(("error", traceback.format_exc()))
        raise
    finally:
        simulation_app.close()


class _EvaluationWorker:
    """Persistent exact held-out evaluator with one request/response contract."""

    def __init__(
        self,
        *,
        config_path: Path,
        run_dir: Path,
        flight_parameters_path: Path,
        flight_parameters_digest: str,
        timeout_s: float,
        workload_role: str,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._tasks = context.Queue()
        self._results = context.Queue()
        self._timeout_s = timeout_s
        self._process = context.Process(
            target=_evaluation_worker_main,
            kwargs={
                "config_path": str(config_path),
                "run_dir": str(run_dir),
                "flight_parameters_path": str(flight_parameters_path),
                "flight_parameters_digest": flight_parameters_digest,
                "task_queue": self._tasks,
                "result_queue": self._results,
                "workload_role": workload_role,
            },
            name="rl-held-out-evaluator",
        )
        self._process.start()
        kind, payload = self._receive()
        if kind != "ready":
            raise RuntimeError(f"held-out evaluator failed to start: {payload}")

    def _receive(self) -> tuple[str, Any]:
        try:
            message = self._results.get(timeout=self._timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("held-out evaluator exceeded its configured timeout") from exc
        if not isinstance(message, tuple) or len(message) != 2:
            raise RuntimeError("held-out evaluator returned an invalid response")
        return message

    def evaluate(
        self,
        *,
        iteration: int,
        checkpoint_path: Path,
        artifact_dir: Path,
        policy_digest: str,
    ) -> tuple[PolicyEvaluation, dict[str, Any]]:
        self._tasks.put((iteration, str(checkpoint_path), str(artifact_dir), policy_digest))
        kind, payload = self._receive()
        if kind == "error":
            raise RuntimeError(f"held-out evaluator failed:\n{payload}")
        if (
            kind != "evaluation"
            or not isinstance(payload, tuple)
            or len(payload) != 2
            or not isinstance(payload[0], PolicyEvaluation)
            or not isinstance(payload[1], dict)
        ):
            raise RuntimeError("held-out evaluator returned an invalid evaluation")
        return payload

    def close(self) -> None:
        self._tasks.put(None)
        self._process.join(timeout=self._timeout_s)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
            raise RuntimeError("held-out evaluator did not shut down before its timeout")
        if self._process.exitcode != 0:
            raise RuntimeError(f"held-out evaluator exited with status {self._process.exitcode}")


def _export_candidate(
    *,
    runner,
    artifact_dir: Path,
    metadata: dict[str, Any],
) -> tuple[Path, str]:
    training.export_policy(runner, artifact_dir, metadata=metadata)
    policy_path = artifact_dir / POLICY_FILENAME
    return artifact_dir, _file_sha256(policy_path)


def _report_evaluation(
    *,
    evaluation: PolicyEvaluation,
    phase_index: int,
    phase_iteration: int,
    plateau_state: PlateauState,
    run_dir: Path,
    artifact_dir: Path,
    validation,
) -> dict[str, Any]:
    metrics = {metric.name: metric.value for metric in evaluation.metrics}
    _report(
        f"evaluation phase={phase_index + 1} phase_iteration={phase_iteration} "
        f"iteration={evaluation.iteration} source={evaluation.source} "
        f"longest_hover_mean_s={metrics['episode_longest_hover_duration_mean_s']:.3f} "
        f"success={metrics['success_rate']:.3f} "
        f"collision={metrics['collision_rate']:.3f} "
        f"other_flight_failure={metrics['other_flight_failure_rate']:.3f} "
        f"timeout={metrics['timeout_rate']:.3f}"
    )
    return {
        "phase_index": phase_index,
        "phase_iteration": phase_iteration,
        "iteration": evaluation.iteration,
        "checkpoint": str(evaluation.checkpoint_path.relative_to(run_dir)),
        "policy": str((artifact_dir / POLICY_FILENAME).relative_to(run_dir)),
        "policy_digest": evaluation.policy_digest,
        "source": evaluation.source,
        "workload_role": "validation",
        "metrics": metrics,
        "reward_means": {reward.name: reward.mean for reward in evaluation.reward_means},
        "checks": [asdict(check) for check in validation.checks],
        "validation_passed": validation.passed,
        "plateau": {
            "made_progress": False,
            "reference_value": plateau_state.reference_value,
            "stale_evaluations": plateau_state.stale_evaluations,
            "reached": False,
        },
    }


def _report_plateau(
    *,
    evaluation: PolicyEvaluation,
    made_progress: bool,
    plateau_state: PlateauState,
    plateau_reached: bool,
) -> None:
    _report(
        f"plateau iteration={evaluation.iteration} "
        f"progress={str(made_progress).lower()} "
        f"reference={plateau_state.reference_value:.6f} "
        f"stale={plateau_state.stale_evaluations} "
        f"reached={str(plateau_reached).lower()}"
    )


def _workload_payload(*, role: str, config: ExperimentConfig) -> dict[str, Any]:
    evaluation = config.rl.press.evaluation
    seeds = evaluation.validation_seeds if role == "validation" else evaluation.test_seeds
    return {
        "role": role,
        "seeds": list(seeds),
        "episodes_per_seed": evaluation.episodes_per_seed,
        "episode_count": len(seeds) * evaluation.episodes_per_seed,
    }


def _acceptance_payload(*, acceptance) -> dict[str, Any]:
    return {
        "passed": acceptance.passed,
        "checks": [asdict(check) for check in acceptance.checks],
        "reasons": list(acceptance.reasons),
    }


def _acceptance_from_payload(*, payload: dict[str, Any]) -> AcceptanceResult:
    checks = payload.get("checks")
    reasons = payload.get("reasons")
    if not isinstance(checks, list) or not isinstance(reasons, list):
        raise ValueError("attempt acceptance payload is incomplete")
    return AcceptanceResult(
        passed=bool(payload.get("passed")),
        checks=tuple(
            AcceptanceCheck(
                name=str(check["name"]),
                passed=bool(check["passed"]),
                actual=None if check["actual"] is None else float(check["actual"]),
                comparison=str(check["comparison"]),
                required=float(check["required"]),
            )
            for check in checks
        ),
        reasons=tuple(str(reason) for reason in reasons),
    )


def _selection_from_report(
    *,
    run_dir: Path,
    report: dict[str, Any],
) -> tuple[int, _AttemptSelection]:
    attempts = report.get("attempts")
    selected_attempt_index = report.get("selected_attempt_index")
    selected_iteration = report.get("selected_iteration")
    selected_checkpoint = report.get("selected_checkpoint")
    selected_policy = report.get("selected_policy")
    selected_digest = report.get("selected_policy_digest")
    if (
        not isinstance(attempts, list)
        or len(attempts) != 1
        or not isinstance(selected_attempt_index, int)
        or not isinstance(selected_iteration, int)
        or not isinstance(selected_checkpoint, str)
        or not isinstance(selected_policy, str)
        or not isinstance(selected_digest, str)
    ):
        raise ValueError("attempt worker report has no unique selected policy")
    attempt = attempts[0]
    evaluations = attempt.get("evaluations")
    validation = attempt.get("validation")
    if not isinstance(evaluations, list) or not isinstance(validation, dict):
        raise ValueError("attempt worker report is missing evaluation evidence")
    selected_record = next(
        (
            record
            for record in evaluations
            if record.get("iteration") == selected_iteration
            and record.get("policy_digest") == selected_digest
        ),
        None,
    )
    if not isinstance(selected_record, dict):
        raise ValueError("attempt worker selected policy has no evaluation record")
    evaluation = _evaluation_from_payload(
        payload=selected_record,
        iteration=selected_iteration,
        checkpoint_path=run_dir / selected_checkpoint,
        policy_digest=selected_digest,
        source=str(selected_record.get("source")),
    )
    return selected_attempt_index, _AttemptSelection(
        evaluation=evaluation,
        artifact_dir=(run_dir / selected_policy).parent,
        validation=_acceptance_from_payload(payload=validation),
    )


def _training_report(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    phases: tuple[RlTrainingPhaseConfig, ...],
    comparison_fingerprint: str,
    attempts: list[dict[str, Any]],
    selected: PolicyEvaluation | None,
    selected_attempt_index: int | None,
    selected_policy_path: Path | None,
    test_evaluation: PolicyEvaluation | None,
    acceptance,
    status: str,
    acceptance_criteria: PressAcceptanceCriteria | None = None,
    active_attempt: dict[str, Any] | None = None,
    telemetry_probe: dict[str, float] | None = None,
    hover_reset_distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    criteria = _campaign_plateau(config=config)
    reset_distribution = hover_reset_distribution
    if reset_distribution is None and attempts:
        reset_distribution = attempts[0].get("hover_reset_distribution")
    rollout_steps = config.rl.training.num_steps_per_env
    rollout_s = rollout_steps / config.rl.policy_rate_hz
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_kind": "campaign",
        "artifact_run_id": run_dir.name,
        "policy_name": config.rl.training.policy_name,
        "policy_interface": rl.PolicyInterface.PRESS.value,
        "comparison_fingerprint": comparison_fingerprint,
        "acceptance_criteria": (
            asdict(acceptance_criteria) if acceptance_criteria is not None else None
        ),
        "reward_configuration": asdict(config.rl.press.reward),
        "rollout_configuration": {
            "num_envs": config.rl.num_envs,
            "num_steps_per_env": rollout_steps,
            "policy_rate_hz": config.rl.policy_rate_hz,
            "simulated_seconds_per_env_per_update": rollout_s,
            "transitions_per_update": config.rl.num_envs * rollout_steps,
            "num_learning_epochs": config.rl.training.num_learning_epochs,
            "num_mini_batches": config.rl.training.num_mini_batches,
            "phase_maximum_simulated_seconds_per_env": [
                phase.max_iterations * rollout_s for phase in phases
            ],
        },
        "phases": [asdict(phase) for phase in phases],
        "plateau_criteria": asdict(criteria),
        "metric_registry": metric_registry_payload(),
        "hover_reset_distribution": reset_distribution,
        "telemetry": {
            "sample_interval_s": config.rl.training.telemetry_sample_interval_s,
            "diagnostic_gpu_utilization_target_rate": (
                config.rl.training.gpu_utilization_target_rate
            ),
            "policy_decision_input": False,
            "startup_probe": telemetry_probe,
        },
        "attempt_execution": "isolated_process",
        "training_attempt_seeds": list(config.rl.training.attempt_seeds),
        "workloads": {
            "validation": _workload_payload(role="validation", config=config),
            "test": _workload_payload(role="test", config=config),
        },
        "attempts": attempts,
        "active_attempt": active_attempt,
        "status": status,
        "selected_attempt_index": selected_attempt_index,
        "selected_iteration": selected.iteration if selected is not None else None,
        "selected_checkpoint": (
            str(selected.checkpoint_path.relative_to(run_dir)) if selected is not None else None
        ),
        "selected_policy": (
            str(selected_policy_path.relative_to(run_dir))
            if selected_policy_path is not None
            else None
        ),
        "selected_policy_digest": selected.policy_digest if selected is not None else None,
        "selected_metric": (
            {
                "name": criteria.metric_name,
                "direction": criteria.direction,
                "value": selected.metric(criteria.metric_name),
            }
            if selected is not None
            else None
        ),
        "test_evaluation": (
            {
                "policy_digest": test_evaluation.policy_digest,
                "source": test_evaluation.source,
                "metrics": {metric.name: metric.value for metric in test_evaluation.metrics},
                "reward_means": {
                    reward.name: reward.mean for reward in test_evaluation.reward_means
                },
            }
            if test_evaluation is not None
            else None
        ),
        "acceptance": _acceptance_payload(acceptance=acceptance) if acceptance else None,
    }


def _attempt_report(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    phases: tuple[RlTrainingPhaseConfig, ...],
    comparison_fingerprint: str,
    attempt: dict[str, Any],
    selected: PolicyEvaluation | None,
    selected_policy_path: Path | None,
    status: str,
) -> dict[str, Any]:
    payload = _training_report(
        run_dir=run_dir,
        config=config,
        phases=phases,
        comparison_fingerprint=comparison_fingerprint,
        attempts=[attempt],
        selected=selected,
        selected_attempt_index=int(attempt["attempt_index"]) if selected is not None else None,
        selected_policy_path=selected_policy_path,
        test_evaluation=None,
        acceptance=None,
        status=status,
    )
    payload["report_kind"] = "attempt"
    return payload


def _final_phase_failed_to_improve(
    *, attempt: dict[str, Any], selection: _AttemptSelection, phase_count: int
) -> bool:
    selected_record = next(
        record
        for record in attempt["evaluations"]
        if record["policy_digest"] == selection.evaluation.policy_digest
    )
    return int(selected_record["phase_index"]) < phase_count - 1


def _publish_selected(
    *,
    run_dir: Path,
    policy_root: Path,
    metadata: dict[str, Any],
    report: dict[str, Any],
    selected: PolicyEvaluation,
    selected_policy: Path,
    test_evaluation: PolicyEvaluation,
    acceptance,
    comparison_fingerprint: str,
    config: ExperimentConfig,
) -> bool:
    criteria = _campaign_plateau(config=config)
    if not acceptance.passed:
        raise ValueError("a failed test evaluation cannot be published")
    if selected.policy_digest != test_evaluation.policy_digest:
        raise ValueError("selected and tested policy digests differ")
    if report.get("selected_policy_digest") != selected.policy_digest:
        raise ValueError("training report selected policy digest differs")
    reported_test = report.get("test_evaluation")
    if not isinstance(reported_test, dict) or (
        reported_test.get("policy_digest") != test_evaluation.policy_digest
    ):
        raise ValueError("training report test policy digest differs")
    if _file_sha256(selected_policy) != selected.policy_digest:
        raise ValueError("selected actor bytes changed after validation")
    selected_metadata = metadata | {
        "selected_checkpoint_iteration": selected.iteration,
        "selected_policy_digest": selected.policy_digest,
        "selection_method": criteria.metric_name,
        "training_attempt_seeds": list(config.rl.training.attempt_seeds),
        "validation_selection": {
            "comparison_fingerprint": comparison_fingerprint,
            "policy_digest": selected.policy_digest,
            "metrics": {metric.name: metric.value for metric in selected.metrics},
            "reward_means": {reward.name: reward.mean for reward in selected.reward_means},
            "workload": _workload_payload(role="validation", config=config),
        },
        "test_evaluation": {
            "policy_digest": test_evaluation.policy_digest,
            "metrics": {metric.name: metric.value for metric in test_evaluation.metrics},
            "reward_means": {reward.name: reward.mean for reward in test_evaluation.reward_means},
            "workload": _workload_payload(role="test", config=config),
        },
        "acceptance": _acceptance_payload(acceptance=acceptance),
    }
    with tempfile.TemporaryDirectory(prefix=".training-publication-", dir=run_dir) as stage_name:
        stage_dir = Path(stage_name)
        shutil.copy2(selected_policy, stage_dir / POLICY_FILENAME)
        if _file_sha256(stage_dir / POLICY_FILENAME) != test_evaluation.policy_digest:
            raise RuntimeError("publication copy changed the tested actor bytes")
        (stage_dir / METADATA_FILENAME).write_text(
            json.dumps(selected_metadata, indent=2, allow_nan=False) + "\n"
        )
        (stage_dir / _REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        for filename in (POLICY_FILENAME, METADATA_FILENAME, _REPORT_FILENAME):
            (stage_dir / filename).replace(run_dir / filename)
    return publish_policy_labels(
        policy_root=policy_root,
        run_dir=run_dir,
        comparison_fingerprint=comparison_fingerprint,
        metric_name=criteria.metric_name,
        metric_value=selected.metric(criteria.metric_name),
        direction=criteria.direction,
    )


@dataclass(frozen=True)
class _AttemptSelection:
    evaluation: PolicyEvaluation
    artifact_dir: Path
    validation: Any


def _select_attempt_candidate(
    *,
    criteria,
    candidates: tuple[_AttemptSelection, ...],
) -> _AttemptSelection:
    eligible = frozenset(
        candidate.evaluation.policy_digest
        for candidate in candidates
        if candidate.validation.passed
    )
    selected = select_best_evaluation(
        criteria=criteria,
        evaluations=tuple(candidate.evaluation for candidate in candidates),
        eligible_policy_digests=eligible,
    )
    return next(candidate for candidate in candidates if candidate.evaluation is selected)


def _restore_phase_branch(
    *,
    runner,
    env,
    parent_checkpoint: Path,
    phase: RlTrainingPhaseConfig,
    phase_number: int,
    device: str,
) -> int:
    """Restore a learner, apply refinement settings, then reset all environments."""
    _report(f"refinement restore START checkpoint={parent_checkpoint}")
    restored_iteration = training.restore_checkpoint(
        runner=runner,
        checkpoint_path=parent_checkpoint,
        device=device,
    )
    _report(f"refinement restore COMPLETE iteration={restored_iteration}")
    training.configure_phase(runner=runner, phase=phase, phase_number=phase_number)
    _report(
        f"refinement configuration COMPLETE learning_rate={phase.learning_rate}; "
        "resetting environments"
    )
    with torch.inference_mode():
        env.reset()
    _report("refinement environment reset COMPLETE")
    return restored_iteration


def _run_final_test(
    *,
    config_path: Path,
    config: ExperimentConfig,
    selection: _AttemptSelection,
    flight_parameters_path: Path,
    flight_parameters_digest: str,
) -> tuple[PolicyEvaluation, dict[str, Any]]:
    """Run one fresh final test of the exact validation-selected actor."""

    selected = selection.evaluation
    selected_policy = selection.artifact_dir / POLICY_FILENAME
    if _file_sha256(selected_policy) != selected.policy_digest:
        raise RuntimeError("validation-selected actor changed before final test")
    test_worker = _EvaluationWorker(
        config_path=config_path,
        run_dir=selection.artifact_dir,
        flight_parameters_path=flight_parameters_path,
        flight_parameters_digest=flight_parameters_digest,
        timeout_s=config.rl.training.evaluation_worker_timeout_s,
        workload_role="test",
    )
    try:
        test_evaluation, telemetry_interval = test_worker.evaluate(
            iteration=selected.iteration,
            checkpoint_path=selected.checkpoint_path,
            artifact_dir=selection.artifact_dir,
            policy_digest=selected.policy_digest,
        )
    finally:
        test_worker.close()
    if test_evaluation.source != "fresh-test":
        raise RuntimeError("final test evaluation was not executed freshly")
    if test_evaluation.policy_digest != selected.policy_digest:
        raise RuntimeError("final test evaluated a different actor digest")
    return test_evaluation, telemetry_interval


def _record_final_test(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    phases: tuple[RlTrainingPhaseConfig, ...],
    attempts: list[dict[str, Any]],
    selection: _AttemptSelection,
    selected_attempt_index: int,
    comparison_fingerprint: str,
    writer,
    telemetry_probe: dict[str, float],
    acceptance_criteria: PressAcceptanceCriteria,
    test_evaluation: PolicyEvaluation,
    telemetry_interval: dict[str, Any],
    acceptance: AcceptanceResult,
) -> dict[str, Any]:
    """Record final-test metrics, acceptance, and telemetry."""

    selected = selection.evaluation
    selected_policy = selection.artifact_dir / POLICY_FILENAME
    attempt_dir = selection.artifact_dir.parents[1]
    attempt_record = attempts[selected_attempt_index]
    monitoring_step = int(attempt_record["attempted_updates"])
    telemetry_interval["monitoring_step"] = monitoring_step
    _record_telemetry(
        path=attempt_dir / _TELEMETRY_FILENAME,
        interval=telemetry_interval,
        target_rate=config.rl.training.gpu_utilization_target_rate,
    )
    try:
        telemetry_summary = _require_telemetry_interval(interval=telemetry_interval)
    except RuntimeError:
        telemetry_failed_report = _training_report(
            run_dir=run_dir,
            config=config,
            phases=phases,
            comparison_fingerprint=comparison_fingerprint,
            attempts=attempts,
            selected=selected,
            selected_attempt_index=selected_attempt_index,
            selected_policy_path=selected_policy,
            test_evaluation=test_evaluation,
            acceptance=None,
            status="telemetry_failed",
            acceptance_criteria=acceptance_criteria,
            telemetry_probe=telemetry_probe,
        )
        _write_json(path=run_dir / _REPORT_FILENAME, payload=telemetry_failed_report)
        raise
    write_evaluation_scalars(
        writer=writer,
        role="test",
        evaluation=test_evaluation,
        all_checks_passed=acceptance.passed,
        step=monitoring_step,
    )
    write_system_scalars(
        writer=writer,
        role="test",
        summary=telemetry_summary,
        step=monitoring_step,
    )
    attempt_record["telemetry"]["summaries"].append(
        {
            "role": "test",
            "monitoring_step": monitoring_step,
            **_telemetry_report_summary(
                summary=telemetry_summary,
                target_rate=config.rl.training.gpu_utilization_target_rate,
            ),
        }
    )
    writer.flush()
    final_report = _training_report(
        run_dir=run_dir,
        config=config,
        phases=phases,
        comparison_fingerprint=comparison_fingerprint,
        attempts=attempts,
        selected=selected,
        selected_attempt_index=selected_attempt_index,
        selected_policy_path=selected_policy,
        test_evaluation=test_evaluation,
        acceptance=acceptance,
        status="accepted" if acceptance.passed else "test_failed",
        acceptance_criteria=acceptance_criteria,
        telemetry_probe=telemetry_probe,
    )
    _write_json(path=run_dir / _REPORT_FILENAME, payload=final_report)
    return final_report


def _train_and_validate_attempt(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    config_path: Path,
    attempt_seeds: tuple[int, ...],
    repo_root: Path,
    device: str,
    inputs: rl.PolicyInputs,
    acceptance_criteria: PressAcceptanceCriteria,
    comparison_fingerprint: str,
    run_dir: Path,
    attempt_index_offset: int = 0,
) -> None:
    """Train one isolated attempt and validate its exported checkpoints."""

    if not args.attempt_worker or len(attempt_seeds) != 1:
        raise ValueError("a training worker must own exactly one isolated attempt")
    training_config = config.rl.training
    plateau_criteria = _campaign_plateau(config=config)
    controller_artifact = inputs.controller
    controller_digest = inputs.controller_digest
    flight_parameters = inputs.flight_parameters
    flight_parameters_digest = inputs.flight_parameters_digest
    phases = (
        _procedural_smoke_phases(phases=_campaign_phases(config=config))
        if args.procedural_smoke
        else _campaign_phases(config=config)
    )
    total_maximum = sum(phase.max_iterations for phase in phases)
    if not run_dir.is_dir():
        raise ValueError(f"campaign run directory does not exist: {run_dir}")
    _report(
        f"training {training_config.task_id} for at most {len(attempt_seeds)} attempts "
        f"and {total_maximum} updates per attempt "
        f"(policy={training_config.policy_name}, seeds={attempt_seeds}, device={device}, "
        f"start=scratch-neutral-output) -> {run_dir}"
    )
    writer = None
    validation_worker: _EvaluationWorker | None = None
    metadata: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    partial_attempt_report_path: Path | None = None
    partial_attempt_record: dict[str, Any] | None = None
    partial_candidates: list[_AttemptSelection] = []
    try:
        for attempt_index, seed in enumerate(attempt_seeds, start=attempt_index_offset):
            _seed_everything(seed)
            attempt_dir = run_dir / f"attempt_{attempt_index + 1:02d}_seed_{seed}"
            attempt_dir.mkdir()
            attempt_report_path = attempt_dir / _ATTEMPT_REPORT_FILENAME
            partial_attempt_report_path = attempt_report_path
            _write_json(
                path=attempt_report_path,
                payload={
                    "schema_version": _REPORT_SCHEMA_VERSION,
                    "report_kind": "attempt",
                    "status": "initializing",
                    "artifact_run_id": run_dir.name,
                    "attempt_index": attempt_index,
                    "training_seed": seed,
                },
            )
            attempt_writer = _TrackioWriter(
                monitoring_dir=run_dir / "monitoring",
                project=training_config.policy_name,
                run_name=f"attempt_{attempt_index + 1:02d}",
                campaign_run_id=run_dir.name,
                training_seed=seed,
                resume="never",
            )
            writer = attempt_writer
            write_reset_scalars(
                writer=attempt_writer,
                summary=inputs.hover_reset_samples.summary(
                    flight_parameters=inputs.flight_parameters
                ),
                step=0,
            )
            setup_started = time.monotonic()
            _report(f"attempt setup START attempt={attempt_index + 1} seed={seed}")
            env, env_cfg = _build_env(
                task_id=training_config.task_id,
                training=True,
                seed=seed,
                num_envs=config.rl.num_envs,
                experiment_config=config,
                controller_settings=controller_artifact.selected_settings,
                flight_parameters=flight_parameters,
                hover_reset_samples=inputs.hover_reset_samples,
            )
            runner: Any = None
            evaluate_candidate: Any = None
            write_training_progress: Any = None
            try:
                if metadata is None:
                    metadata = _artifact_metadata(
                        config=config,
                        env_cfg=env_cfg,
                        source_metadata=_git_source_metadata(repo_root=repo_root),
                        artifact_run_id=run_dir.name,
                        controller_artifact=controller_artifact,
                        controller_digest=controller_digest,
                        flight_parameters=flight_parameters,
                        flight_parameters_digest=flight_parameters_digest,
                        hover_reset_samples=inputs.hover_reset_samples,
                    )
                    if metadata["policy_interface"] != rl.PolicyInterface.PRESS.value:
                        raise ValueError("campaign requires the press deterministic evaluator")
                    metadata["training_phases"] = [asdict(phase) for phase in phases]
                    metadata["plateau_criteria"] = asdict(plateau_criteria)
                _write_artifact_metadata(run_dir=attempt_dir, metadata=metadata)
                contract = rl.load_policy_metadata(artifact_dir=attempt_dir).contract
                attempt_fingerprint = _press_comparison_fingerprint(
                    config=config,
                    contract=contract,
                    flight_parameters_fingerprint=flight_parameters.artifact_fingerprint,
                )
                if attempt_fingerprint is None:
                    raise ValueError("press campaign did not produce a comparison fingerprint")
                if attempt_fingerprint != comparison_fingerprint:
                    raise ValueError("training attempt changed the comparison fingerprint")
                metadata["comparison_fingerprint"] = comparison_fingerprint
                _write_artifact_metadata(run_dir=attempt_dir, metadata=metadata)
                attempt_metadata = metadata
                previous_stage = _previous_stage_checkpoint(
                    config=config,
                    repo_root=repo_root,
                    contract=contract,
                    controller_digest=controller_digest,
                    flight_parameters_digest=flight_parameters_digest,
                )
                metadata["initialization"] = (
                    {"method": env_cfg.policy_config.actor.output_initialization}
                    if previous_stage is None
                    else previous_stage[1]
                )
                _write_artifact_metadata(run_dir=attempt_dir, metadata=metadata)
                runner = training.build_runner(
                    env=env,
                    training_config=training_config,
                    phase=phases[0],
                    policy_config=env_cfg.policy_config,
                    log_dir=None,
                    device=device,
                )
                training.configure_monitoring(
                    runner=runner,
                    writer=writer,
                    phase=phases[0],
                    phase_number=1,
                )
                if previous_stage is None:
                    training.initialize_actor_output(
                        runner=runner,
                        method=env_cfg.policy_config.actor.output_initialization,
                    )
                else:
                    training.restore_checkpoint(
                        runner=runner,
                        checkpoint_path=previous_stage[0],
                        device=device,
                    )
                checkpoints_dir = attempt_dir / "checkpoints"
                checkpoints_dir.mkdir()
                initial_checkpoint = checkpoints_dir / "phase_01_model_0.pt"
                training.save_checkpoint(
                    runner=runner,
                    checkpoint_path=initial_checkpoint,
                    iteration=0,
                )
                telemetry = TelemetrySampler(
                    sample_interval_s=training_config.telemetry_sample_interval_s
                )
                _write_json(
                    path=attempt_dir / _TELEMETRY_FILENAME,
                    payload={
                        "schema_version": 1,
                        "diagnostic_gpu_utilization_target_rate": (
                            training_config.gpu_utilization_target_rate
                        ),
                        "intervals": [],
                    },
                )
                candidates: list[_AttemptSelection] = []
                partial_candidates = candidates
                evaluation_records: list[dict[str, Any]] = []
                phase_results: list[dict[str, Any]] = []
                attempt_record: dict[str, Any] = {
                    "attempt_index": attempt_index,
                    "training_seed": seed,
                    "worker_pid": os.getpid(),
                    "hover_reset_distribution": metadata["hover_reset_distribution"],
                    "evaluations": evaluation_records,
                    "phase_results": phase_results,
                    "attempted_updates": 0,
                    "telemetry": {
                        "artifact": str((attempt_dir / _TELEMETRY_FILENAME).relative_to(run_dir)),
                        "summaries": [],
                    },
                    "selected_policy_digest": None,
                    "selected_checkpoint": None,
                    "validation": None,
                }
                attempts.append(attempt_record)
                partial_attempt_record = attempt_record
                evaluation_index = 0

                _report(
                    f"attempt setup COMPLETE attempt={attempt_index + 1} seed={seed} "
                    f"elapsed_s={time.monotonic() - setup_started:.1f}"
                )

                def evaluate_candidate(
                    *,
                    iteration: int,
                    checkpoint_path: Path,
                    phase_index: int,
                    phase_iteration: int,
                    plateau_state: PlateauState,
                    attempt_dir: Path = attempt_dir,
                    runner=runner,
                    metadata: dict[str, Any] = attempt_metadata,
                    candidates: list[_AttemptSelection] = candidates,
                    evaluation_records: list[dict[str, Any]] = evaluation_records,
                    attempt_index: int = attempt_index,
                    writer=attempt_writer,
                    attempt_record: dict[str, Any] = attempt_record,
                ) -> PolicyEvaluation:
                    nonlocal evaluation_index, validation_worker
                    artifact_dir = attempt_dir / "evaluated" / f"candidate_{evaluation_index:03d}"
                    artifact_dir, policy_digest = _export_candidate(
                        runner=runner,
                        artifact_dir=artifact_dir,
                        metadata=metadata,
                    )
                    if validation_worker is None:
                        validation_worker = _EvaluationWorker(
                            config_path=config_path,
                            run_dir=attempt_dir,
                            flight_parameters_path=inputs.flight_parameters_path,
                            flight_parameters_digest=inputs.flight_parameters_digest,
                            timeout_s=training_config.evaluation_worker_timeout_s,
                            workload_role="validation",
                        )
                    evaluation, telemetry_interval = validation_worker.evaluate(
                        iteration=iteration,
                        checkpoint_path=checkpoint_path,
                        artifact_dir=artifact_dir,
                        policy_digest=policy_digest,
                    )
                    monitoring_step = training.monitoring_step(runner=runner)
                    telemetry_interval["monitoring_step"] = monitoring_step
                    _record_telemetry(
                        path=attempt_dir / _TELEMETRY_FILENAME,
                        interval=telemetry_interval,
                        target_rate=training_config.gpu_utilization_target_rate,
                    )
                    telemetry_summary = _require_telemetry_interval(interval=telemetry_interval)
                    write_system_scalars(
                        writer=writer,
                        role="val",
                        summary=telemetry_summary,
                        step=monitoring_step,
                    )
                    attempt_record["telemetry"]["summaries"].append(
                        {
                            "role": "val",
                            "monitoring_step": monitoring_step,
                            **_telemetry_report_summary(
                                summary=telemetry_summary,
                                target_rate=training_config.gpu_utilization_target_rate,
                            ),
                        }
                    )
                    validation = evaluate_press_acceptance(
                        criteria=acceptance_criteria,
                        evaluation=evaluation,
                    )
                    candidates.append(
                        _AttemptSelection(
                            evaluation=evaluation,
                            artifact_dir=artifact_dir,
                            validation=validation,
                        )
                    )
                    reporting_plateau_state = plateau_state
                    if math.isnan(reporting_plateau_state.reference_value):
                        reporting_plateau_state = PlateauState(
                            reference_value=evaluation.metric(plateau_criteria.metric_name),
                            stale_evaluations=0,
                        )
                    evaluation_records.append(
                        _report_evaluation(
                            evaluation=evaluation,
                            phase_index=phase_index,
                            phase_iteration=phase_iteration,
                            plateau_state=reporting_plateau_state,
                            run_dir=run_dir,
                            artifact_dir=artifact_dir,
                            validation=validation,
                        )
                    )
                    monitoring_step = training.monitoring_step(runner=runner)
                    evaluation_records[-1]["monitoring_step"] = monitoring_step
                    write_evaluation_scalars(
                        writer=writer,
                        role="val",
                        evaluation=evaluation,
                        all_checks_passed=validation.passed,
                        step=monitoring_step,
                    )
                    writer.flush()

                    evaluation_index += 1
                    return evaluation

                def write_training_progress(
                    *,
                    candidates: list[_AttemptSelection] = candidates,
                    attempt_record: dict[str, Any] = attempt_record,
                    attempt_index: int = attempt_index,
                    runner=runner,
                    attempt_report_path: Path = attempt_report_path,
                ) -> None:
                    current_selection = _select_attempt_candidate(
                        criteria=plateau_criteria,
                        candidates=tuple(candidates),
                    )
                    attempt_record.update(
                        {
                            "selected_policy_digest": current_selection.evaluation.policy_digest,
                            "selected_checkpoint": str(
                                current_selection.evaluation.checkpoint_path.relative_to(run_dir)
                            ),
                            "validation": _acceptance_payload(
                                acceptance=current_selection.validation
                            ),
                        }
                    )
                    attempt_record["attempted_updates"] = training.monitoring_step(runner=runner)
                    _write_json(
                        path=attempt_report_path,
                        payload=_attempt_report(
                            run_dir=run_dir,
                            config=config,
                            phases=phases,
                            comparison_fingerprint=comparison_fingerprint,
                            attempt=attempt_record,
                            selected=current_selection.evaluation,
                            selected_policy_path=(current_selection.artifact_dir / POLICY_FILENAME),
                            status="training",
                        ),
                    )

                initial_plateau = PlateauState(reference_value=math.nan, stale_evaluations=0)
                initial = evaluate_candidate(
                    iteration=0,
                    checkpoint_path=initial_checkpoint,
                    phase_index=0,
                    phase_iteration=0,
                    plateau_state=initial_plateau,
                )
                evaluation_records[-1]["plateau"] = {
                    "made_progress": True,
                    "reference_value": initial.metric(plateau_criteria.metric_name),
                    "stale_evaluations": 0,
                    "reached": False,
                }
                _report_plateau(
                    evaluation=initial,
                    made_progress=True,
                    plateau_state=PlateauState(
                        reference_value=initial.metric(plateau_criteria.metric_name),
                        stale_evaluations=0,
                    ),
                    plateau_reached=False,
                )
                write_training_progress()
                branch_iteration = 0
                randomize_episode_lengths = config.rl.press.selected_stage().name != "hover"
                for phase_index, phase in enumerate(phases):
                    parent = _select_attempt_candidate(
                        criteria=plateau_criteria,
                        candidates=tuple(candidates),
                    )
                    if phase_index > 0:
                        branch_iteration = _restore_phase_branch(
                            runner=runner,
                            env=env,
                            parent_checkpoint=parent.evaluation.checkpoint_path,
                            phase=phase,
                            phase_number=phase_index + 1,
                            device=device,
                        )
                    else:
                        training.configure_phase(
                            runner=runner,
                            phase=phase,
                            phase_number=phase_index + 1,
                        )
                        _report(
                            "initial phase configuration COMPLETE "
                            f"learning_rate={phase.learning_rate}; "
                            "resetting environments"
                        )
                        with torch.inference_mode():
                            env.reset()
                        _report("initial phase environment reset COMPLETE")
                    plateau_state = PlateauState(
                        reference_value=parent.evaluation.metric(plateau_criteria.metric_name),
                        stale_evaluations=0,
                    )
                    phase_iteration = 0
                    phase_reason = "maximum_iterations"
                    while phase_iteration < phase.max_iterations:
                        chunk = min(
                            phase.evaluation_interval,
                            phase.max_iterations - phase_iteration,
                        )
                        telemetry.start(role="train")
                        try:
                            result = training.train_chunk(
                                runner=runner,
                                completed_iterations=branch_iteration,
                                iterations=chunk,
                                randomize_initial_episode_lengths=randomize_episode_lengths,
                            )
                        finally:
                            telemetry_interval = telemetry.finish(
                                work_items=(
                                    chunk
                                    * int(training_config.num_steps_per_env)
                                    * int(env.num_envs)
                                )
                            )
                        monitoring_step = training.monitoring_step(runner=runner)
                        telemetry_interval["monitoring_step"] = monitoring_step
                        _record_telemetry(
                            path=attempt_dir / _TELEMETRY_FILENAME,
                            interval=telemetry_interval,
                            target_rate=training_config.gpu_utilization_target_rate,
                        )
                        telemetry_summary = _require_telemetry_interval(interval=telemetry_interval)
                        write_system_scalars(
                            writer=writer,
                            role="train",
                            summary=telemetry_summary,
                            step=monitoring_step,
                        )
                        attempt_record["telemetry"]["summaries"].append(
                            {
                                "role": "train",
                                "monitoring_step": monitoring_step,
                                **_telemetry_report_summary(
                                    summary=telemetry_summary,
                                    target_rate=training_config.gpu_utilization_target_rate,
                                ),
                            }
                        )
                        randomize_episode_lengths = False
                        branch_iteration += result.iterations
                        phase_iteration += result.iterations
                        checkpoint_path = checkpoints_dir / (
                            f"phase_{phase_index + 1:02d}_model_{branch_iteration}.pt"
                        )
                        training.save_checkpoint(
                            runner=runner,
                            checkpoint_path=checkpoint_path,
                            iteration=branch_iteration,
                        )
                        provisional = PlateauState(
                            reference_value=plateau_state.reference_value,
                            stale_evaluations=plateau_state.stale_evaluations,
                        )
                        evaluation = evaluate_candidate(
                            iteration=branch_iteration,
                            checkpoint_path=checkpoint_path,
                            phase_index=phase_index,
                            phase_iteration=phase_iteration,
                            plateau_state=provisional,
                        )
                        plateau = advance_plateau(
                            criteria=plateau_criteria,
                            state=plateau_state,
                            evaluation=evaluation,
                        )
                        plateau_state = plateau.state
                        evaluation_records[-1]["plateau"] = {
                            "made_progress": plateau.made_progress,
                            "reference_value": plateau.state.reference_value,
                            "stale_evaluations": plateau.state.stale_evaluations,
                            "reached": plateau.plateau_reached,
                        }
                        _report_plateau(
                            evaluation=evaluation,
                            made_progress=plateau.made_progress,
                            plateau_state=plateau.state,
                            plateau_reached=plateau.plateau_reached,
                        )
                        write_training_progress()
                        if plateau.plateau_reached:
                            phase_reason = "plateau"
                            break
                    phase_results.append(
                        {
                            "phase_index": phase_index,
                            "parent_checkpoint": str(
                                parent.evaluation.checkpoint_path.relative_to(run_dir)
                            ),
                            "parent_policy_digest": parent.evaluation.policy_digest,
                            "restored_iteration": parent.evaluation.iteration,
                            "environment_reset_after_restore": phase_index > 0,
                            "phase_configuration": asdict(phase),
                            "completed_iterations": phase_iteration,
                            "terminal_reason": phase_reason,
                        }
                    )
                attempt_selection = _select_attempt_candidate(
                    criteria=plateau_criteria,
                    candidates=tuple(candidates),
                )
                attempt_record.update(
                    {
                        "selected_policy_digest": attempt_selection.evaluation.policy_digest,
                        "selected_checkpoint": str(
                            attempt_selection.evaluation.checkpoint_path.relative_to(run_dir)
                        ),
                        "validation": _acceptance_payload(acceptance=attempt_selection.validation),
                    }
                )
                attempt_record["attempted_updates"] = training.monitoring_step(runner=runner)
                _write_json(
                    path=attempt_report_path,
                    payload=_attempt_report(
                        run_dir=run_dir,
                        config=config,
                        phases=phases,
                        comparison_fingerprint=comparison_fingerprint,
                        attempt=attempt_record,
                        selected=attempt_selection.evaluation,
                        selected_policy_path=attempt_selection.artifact_dir / POLICY_FILENAME,
                        status="validation_passed"
                        if attempt_selection.validation.passed
                        else "training",
                    ),
                )
            finally:
                teardown_started = time.monotonic()
                _report(f"attempt teardown START attempt={attempt_index + 1} seed={seed}")
                evaluate_candidate = None
                write_training_progress = None
                runner = None
                gc.collect()
                env.close()
                env = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _report(
                    f"attempt teardown COMPLETE attempt={attempt_index + 1} seed={seed} "
                    f"elapsed_s={time.monotonic() - teardown_started:.1f}"
                )

        if validation_worker is not None:
            validation_worker.close()
            validation_worker = None
        worker_report = _attempt_report(
            run_dir=run_dir,
            config=config,
            phases=phases,
            comparison_fingerprint=comparison_fingerprint or "",
            attempt=attempt_record,
            selected=attempt_selection.evaluation,
            selected_policy_path=attempt_selection.artifact_dir / POLICY_FILENAME,
            status=(
                "procedural_smoke_complete"
                if args.procedural_smoke
                else (
                    "validation_passed"
                    if attempt_selection.validation.passed
                    else "validation_failed"
                )
            ),
        )
        _write_json(path=attempt_report_path, payload=worker_report)
        _report(
            f"attempt worker COMPLETE attempt={attempt_index + 1} "
            f"validation={str(attempt_selection.validation.passed).lower()}"
        )
    except (Exception, KeyboardInterrupt) as exc:
        if partial_attempt_report_path is not None and partial_attempt_record is not None:
            partial_selection = (
                _select_attempt_candidate(
                    criteria=plateau_criteria,
                    candidates=tuple(partial_candidates),
                )
                if partial_candidates
                else None
            )
            _write_json(
                path=partial_attempt_report_path,
                payload=_attempt_report(
                    run_dir=run_dir,
                    config=config,
                    phases=phases,
                    comparison_fingerprint=comparison_fingerprint or "",
                    attempt=partial_attempt_record,
                    selected=(
                        partial_selection.evaluation if partial_selection is not None else None
                    ),
                    selected_policy_path=(
                        partial_selection.artifact_dir / POLICY_FILENAME
                        if partial_selection is not None
                        else None
                    ),
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                ),
            )
        raise
    finally:
        if writer is not None:
            writer.close()
        if validation_worker is not None:
            validation_worker.close()


def _attempt_worker_command(
    *,
    args: argparse.Namespace,
    attempt_index: int,
    controller_artifact_digest: str,
    controller_artifact_path: Path,
    flight_parameters_artifact_digest: str,
    flight_parameters_artifact_path: Path,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--attempt-worker",
        "--attempt-index",
        str(attempt_index),
        "--attempt-seed",
        str(seed),
        "--controller-artifact",
        str(controller_artifact_path),
        "--controller-artifact-digest",
        controller_artifact_digest,
        "--flight-parameters-artifact",
        str(flight_parameters_artifact_path),
        "--flight-parameters-artifact-digest",
        flight_parameters_artifact_digest,
    ]
    if args.procedural_smoke:
        command.append("--procedural-smoke")
    if args.scene is not None:
        command.extend(("--scene", str(args.scene)))
    if args.strict:
        command.append("--strict")
    return command


def _run_policy_workflow(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    config_path: Path,
    inputs: rl.PolicyInputs,
    acceptance_criteria: PressAcceptanceCriteria,
    comparison_fingerprint: str,
    attempt_seeds: tuple[int, ...],
    policy_root: Path,
    repo_root: Path,
) -> None:
    """Train and validate attempts, select one actor, test it, then publish it."""

    plateau_criteria = _campaign_plateau(config=config)
    phases = (
        _procedural_smoke_phases(phases=_campaign_phases(config=config))
        if args.procedural_smoke
        else _campaign_phases(config=config)
    )
    telemetry_probe = probe_telemetry()
    run_dir = _make_run_dir(out_root=policy_root, run_id=os.environ.get(_RUN_ID_ENV))
    hover_reset_distribution = {
        "source": "flight_parameters_policy_window",
        **inputs.hover_reset_samples.summary(flight_parameters=inputs.flight_parameters),
    }
    attempts: list[dict[str, Any]] = []
    selections: list[tuple[int, _AttemptSelection]] = []
    worker_pids: set[int] = set()
    _write_json(
        path=run_dir / _REPORT_FILENAME,
        payload=_training_report(
            run_dir=run_dir,
            config=config,
            phases=phases,
            comparison_fingerprint=comparison_fingerprint,
            attempts=attempts,
            selected=None,
            selected_attempt_index=None,
            selected_policy_path=None,
            test_evaluation=None,
            acceptance=None,
            status="training",
            acceptance_criteria=acceptance_criteria,
            telemetry_probe=telemetry_probe,
            hover_reset_distribution=hover_reset_distribution,
        ),
    )
    _report(
        f"campaign START run={run_dir} attempts={len(attempt_seeds)} " "execution=isolated_process"
    )
    selected_attempt_index: int | None = None
    campaign_selection: _AttemptSelection | None = None
    for attempt_index, seed in enumerate(attempt_seeds):
        worker_environment = os.environ.copy()
        worker_environment[_CAMPAIGN_RUN_DIR_ENV] = str(run_dir)
        worker_environment[_RUN_ID_ENV] = run_dir.name
        started = time.monotonic()
        attempt_dir = run_dir / f"attempt_{attempt_index + 1:02d}_seed_{seed}"
        attempt_report_path = attempt_dir / _ATTEMPT_REPORT_FILENAME
        active_attempt = {
            "attempt_index": attempt_index,
            "training_seed": seed,
            "report": str(attempt_report_path.relative_to(run_dir)),
        }
        current_selection = (
            _select_attempt_candidate(
                criteria=plateau_criteria,
                candidates=tuple(candidate for _, candidate in selections),
            )
            if selections
            else None
        )
        current_index = (
            next(index for index, candidate in selections if candidate is current_selection)
            if current_selection is not None
            else None
        )
        _write_json(
            path=run_dir / _REPORT_FILENAME,
            payload=_training_report(
                run_dir=run_dir,
                config=config,
                phases=phases,
                comparison_fingerprint=comparison_fingerprint,
                attempts=attempts,
                selected=(current_selection.evaluation if current_selection is not None else None),
                selected_attempt_index=current_index,
                selected_policy_path=(
                    current_selection.artifact_dir / POLICY_FILENAME
                    if current_selection is not None
                    else None
                ),
                test_evaluation=None,
                acceptance=None,
                status="training",
                acceptance_criteria=acceptance_criteria,
                active_attempt=active_attempt,
                telemetry_probe=telemetry_probe,
                hover_reset_distribution=hover_reset_distribution,
            ),
        )
        _report(
            f"attempt process START attempt={attempt_index + 1} seed={seed} "
            f"completed_attempts={len(attempts)}"
        )
        try:
            subprocess.run(
                _attempt_worker_command(
                    args=args,
                    attempt_index=attempt_index,
                    controller_artifact_digest=inputs.controller_digest,
                    controller_artifact_path=inputs.controller_path,
                    flight_parameters_artifact_digest=inputs.flight_parameters_digest,
                    flight_parameters_artifact_path=inputs.flight_parameters_path,
                    seed=seed,
                ),
                cwd=repo_root,
                env=worker_environment,
                check=True,
            )
        except KeyboardInterrupt:
            interrupted = json.loads((run_dir / _REPORT_FILENAME).read_text())
            interrupted["status"] = "interrupted"
            _write_json(path=run_dir / _REPORT_FILENAME, payload=interrupted)
            raise
        except subprocess.CalledProcessError:
            failed = json.loads((run_dir / _REPORT_FILENAME).read_text())
            failed["status"] = "attempt_failed"
            _write_json(path=run_dir / _REPORT_FILENAME, payload=failed)
            raise
        if not attempt_report_path.is_file():
            _write_json(
                path=attempt_report_path,
                payload={
                    "schema_version": _REPORT_SCHEMA_VERSION,
                    "report_kind": "attempt",
                    "status": "failed",
                    "artifact_run_id": run_dir.name,
                    "attempt_index": attempt_index,
                    "training_seed": seed,
                },
            )
            failed = json.loads((run_dir / _REPORT_FILENAME).read_text())
            failed["status"] = "attempt_failed"
            _write_json(path=run_dir / _REPORT_FILENAME, payload=failed)
            raise RuntimeError("attempt worker exited without an attempt report")
        payload = json.loads(attempt_report_path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _REPORT_SCHEMA_VERSION
            or payload.get("report_kind") != "attempt"
        ):
            raise ValueError("attempt worker wrote an invalid attempt report")
        if payload.get("status") not in {
            "validation_passed",
            "validation_failed",
            "procedural_smoke_complete",
        }:
            failed = json.loads((run_dir / _REPORT_FILENAME).read_text())
            failed["status"] = "attempt_failed"
            _write_json(path=run_dir / _REPORT_FILENAME, payload=failed)
            raise RuntimeError("attempt worker did not complete successfully")
        attempt_payloads = payload.get("attempts")
        if not isinstance(attempt_payloads, list) or len(attempt_payloads) != 1:
            raise ValueError("attempt worker did not report exactly one isolated attempt")
        attempt = attempt_payloads[0]
        if (
            not isinstance(attempt, dict)
            or attempt.get("attempt_index") != attempt_index
            or attempt.get("training_seed") != seed
        ):
            raise ValueError("attempt worker report identity does not match its launch")
        worker_pid = attempt.get("worker_pid")
        if (
            not isinstance(worker_pid, int)
            or worker_pid == os.getpid()
            or worker_pid in worker_pids
        ):
            raise ValueError("training attempts did not execute in distinct worker processes")
        worker_pids.add(worker_pid)
        reported_index, selection = _selection_from_report(run_dir=run_dir, report=payload)
        if reported_index != attempt_index:
            raise ValueError("attempt worker selected index does not match its launch")
        worker_fingerprint = payload.get("comparison_fingerprint")
        if not isinstance(worker_fingerprint, str) or not worker_fingerprint:
            raise ValueError("attempt worker report has no comparison fingerprint")
        if comparison_fingerprint and worker_fingerprint != comparison_fingerprint:
            raise ValueError("training attempt changed the comparison fingerprint")
        comparison_fingerprint = worker_fingerprint
        attempts.append(attempt)
        selections.append((attempt_index, selection))
        current_selection = _select_attempt_candidate(
            criteria=plateau_criteria,
            candidates=tuple(candidate for _, candidate in selections),
        )
        current_index = next(
            index for index, candidate in selections if candidate is current_selection
        )
        status = "validation_passed" if selection.validation.passed else "training"
        _write_json(
            path=run_dir / _REPORT_FILENAME,
            payload=_training_report(
                run_dir=run_dir,
                config=config,
                phases=phases,
                comparison_fingerprint=comparison_fingerprint,
                attempts=attempts,
                selected=current_selection.evaluation,
                selected_attempt_index=current_index,
                selected_policy_path=current_selection.artifact_dir / POLICY_FILENAME,
                test_evaluation=None,
                acceptance=None,
                status=status,
                acceptance_criteria=acceptance_criteria,
                telemetry_probe=telemetry_probe,
            ),
        )
        _report(
            f"attempt process COMPLETE attempt={attempt_index + 1} seed={seed} "
            f"worker_pid={worker_pid} validation="
            f"{str(selection.validation.passed).lower()} "
            f"elapsed_s={time.monotonic() - started:.1f}"
        )
        if not args.procedural_smoke and selection.validation.passed:
            selected_attempt_index = attempt_index
            campaign_selection = selection
            break
        if not args.procedural_smoke and _final_phase_failed_to_improve(
            attempt=attempt,
            selection=selection,
            phase_count=len(phases),
        ):
            attempt["early_stop_reason"] = (
                "the final refinement phase did not improve the selected earlier-phase actor"
            )
            _report(
                "campaign EARLY STOP: final refinement did not improve the selected actor; "
                "another independent seed would repeat an unpromising curriculum"
            )
            break

    if args.procedural_smoke:
        smoke_selection = _select_attempt_candidate(
            criteria=plateau_criteria,
            candidates=tuple(selection for _, selection in selections),
        )
        smoke_attempt_index = next(
            index for index, selection in selections if selection is smoke_selection
        )
        smoke_report = _training_report(
            run_dir=run_dir,
            config=config,
            phases=phases,
            comparison_fingerprint=comparison_fingerprint,
            attempts=attempts,
            selected=smoke_selection.evaluation,
            selected_attempt_index=smoke_attempt_index,
            selected_policy_path=smoke_selection.artifact_dir / POLICY_FILENAME,
            test_evaluation=None,
            acceptance=None,
            status="procedural_smoke_complete",
            acceptance_criteria=acceptance_criteria,
            telemetry_probe=telemetry_probe,
        )
        _write_json(path=run_dir / _REPORT_FILENAME, payload=smoke_report)
        _report(
            "procedural smoke COMPLETE: every configured attempt ran in a distinct "
            "process; final test was not run"
        )
        return

    if campaign_selection is None or selected_attempt_index is None:
        failed_report = _training_report(
            run_dir=run_dir,
            config=config,
            phases=phases,
            comparison_fingerprint=comparison_fingerprint,
            attempts=attempts,
            selected=None,
            selected_attempt_index=None,
            selected_policy_path=None,
            test_evaluation=None,
            acceptance=None,
            status="validation_failed",
            acceptance_criteria=acceptance_criteria,
            telemetry_probe=telemetry_probe,
        )
        _write_json(path=run_dir / _REPORT_FILENAME, payload=failed_report)
        raise RuntimeError("all training attempts failed validation")

    metadata_payload = json.loads((campaign_selection.artifact_dir / METADATA_FILENAME).read_text())
    if not isinstance(metadata_payload, dict):
        raise ValueError("selected attempt metadata is not an object")
    writer = _TrackioWriter(
        monitoring_dir=run_dir / "monitoring",
        project=config.rl.training.policy_name,
        run_name=f"attempt_{selected_attempt_index + 1:02d}",
        campaign_run_id=run_dir.name,
        training_seed=int(attempts[selected_attempt_index]["training_seed"]),
        resume="must",
    )
    try:
        test_evaluation, telemetry_interval = _run_final_test(
            config_path=config_path,
            config=config,
            selection=campaign_selection,
            flight_parameters_path=inputs.flight_parameters_path,
            flight_parameters_digest=inputs.flight_parameters_digest,
        )
        acceptance = evaluate_press_acceptance(
            criteria=acceptance_criteria,
            evaluation=test_evaluation,
        )
        final_report = _record_final_test(
            run_dir=run_dir,
            config=config,
            phases=phases,
            attempts=attempts,
            selection=campaign_selection,
            selected_attempt_index=selected_attempt_index,
            comparison_fingerprint=comparison_fingerprint,
            writer=writer,
            telemetry_probe=telemetry_probe,
            acceptance_criteria=acceptance_criteria,
            test_evaluation=test_evaluation,
            telemetry_interval=telemetry_interval,
            acceptance=acceptance,
        )
        if not acceptance.passed:
            raise RuntimeError("final policy test failed: " + ", ".join(acceptance.reasons))
        selected = campaign_selection.evaluation
        selected_policy = campaign_selection.artifact_dir / POLICY_FILENAME
        best_updated = _publish_selected(
            run_dir=run_dir,
            policy_root=policy_root,
            metadata=metadata_payload,
            report=final_report,
            selected=selected,
            selected_policy=selected_policy,
            test_evaluation=test_evaluation,
            acceptance=acceptance,
            comparison_fingerprint=comparison_fingerprint,
            config=config,
        )
        _report(
            f"train-rl COMPLETE run={run_dir} attempts={len(attempts)} "
            f"selected={selected.iteration} metric={plateau_criteria.metric_name} "
            f"value={selected.metric(plateau_criteria.metric_name):.6f} "
            f"acceptance=true latest={policy_root / 'latest'} "
            f"best_updated={str(best_updated).lower()}"
        )
    finally:
        writer.close()


def _run(*, args: argparse.Namespace, config: ExperimentConfig, config_path: Path) -> None:
    training_config = config.rl.training
    attempt_seeds: tuple[int, ...]
    repo_root = _repo_root()
    policy_root = (
        repo_root / ".build" / "training" / _validate_policy_name(training_config.policy_name)
    )
    try:
        if args.attempt_worker:
            inputs = _load_pinned_policy_inputs(
                controller_path=args.controller_artifact,
                controller_digest=args.controller_artifact_digest,
                flight_parameters_path=args.flight_parameters_artifact,
                flight_parameters_digest=args.flight_parameters_artifact_digest,
            )
            attempt_seeds = (int(args.attempt_seed),)
        else:
            inputs = rl.prepare_policy_inputs(config=config, repo_root=repo_root)
            attempt_seeds = tuple(int(seed) for seed in training_config.attempt_seeds)
        acceptance_criteria = create_press_acceptance_criteria(
            config=config.rl.press.acceptance,
            flight_parameters=inputs.flight_parameters,
        )
        comparison_fingerprint = _press_comparison_fingerprint(
            config=config,
            contract=press_policy_contract(config=config, controller=inputs.controller),
            flight_parameters_fingerprint=inputs.flight_parameters.artifact_fingerprint,
        )
        if comparison_fingerprint is None:
            raise ValueError("press training did not produce a comparison fingerprint")
    except (OSError, ValueError) as exc:
        _report(f"input artifact validation FAILED: {exc}")
        raise SystemExit(2) from None
    if args.attempt_worker:
        if args.attempt_index is None:
            raise ValueError("attempt worker has no attempt index")
        run_dir_value = os.environ.get(_CAMPAIGN_RUN_DIR_ENV)
        if run_dir_value is None:
            raise ValueError(f"{_CAMPAIGN_RUN_DIR_ENV} is required for an attempt worker")
        run_dir = Path(run_dir_value)
        if not run_dir.is_absolute():
            raise ValueError(f"{_CAMPAIGN_RUN_DIR_ENV} must be an absolute path")
        _train_and_validate_attempt(
            args=args,
            config=config,
            config_path=config_path,
            attempt_seeds=attempt_seeds,
            repo_root=repo_root,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            inputs=inputs,
            acceptance_criteria=acceptance_criteria,
            comparison_fingerprint=comparison_fingerprint,
            run_dir=run_dir,
            attempt_index_offset=args.attempt_index,
        )
        return
    _report(
        "input artifact validation PASSED: "
        f"controller={inputs.controller_path} "
        f"flight_parameters={inputs.flight_parameters_path}"
    )
    _run_policy_workflow(
        args=args,
        config=config,
        config_path=config_path,
        inputs=inputs,
        acceptance_criteria=acceptance_criteria,
        comparison_fingerprint=comparison_fingerprint,
        attempt_seeds=attempt_seeds,
        policy_root=policy_root,
        repo_root=repo_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--procedural-smoke",
        action="store_true",
        help="run one update per phase and stop before the reserved final test",
    )
    parser.add_argument("--attempt-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--controller-artifact", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--controller-artifact-digest", help=argparse.SUPPRESS)
    parser.add_argument("--flight-parameters-artifact", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--flight-parameters-artifact-digest", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-index", type=int, default=None, help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()
    if args.attempt_worker and (
        args.attempt_seed is None
        or args.attempt_index is None
        or args.controller_artifact is None
        or args.controller_artifact_digest is None
        or args.flight_parameters_artifact is None
        or args.flight_parameters_artifact_digest is None
    ):
        raise SystemExit(
            "--attempt-worker requires --attempt-seed, --attempt-index, exact controller and "
            "flight-parameters artifacts, and their digests"
        )
    if args.attempt_index is not None and args.attempt_index < 0:
        raise SystemExit("--attempt-index must be >= 0")

    config_path = Path(os.environ[_CONFIG_PATH_ENV])
    config = load_config(config_path)
    simulation_app = None
    try:
        if args.attempt_worker:
            from isaaclab.app import AppLauncher

            app_launcher = AppLauncher(headless=True)
            simulation_app = app_launcher.app
        _install_interrupt_handlers()
        _run(args=args, config=config, config_path=config_path)
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()

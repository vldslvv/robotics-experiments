"""Stage training: isolated attempts, validation selection and fresh acceptance tests.

The curriculum coordinator calls these entrypoints and owns publication order.
Complete learner serialization and restoration live in ``rl.training.checkpoint``.
Isaac imports stay inside environment and evaluation worker startup.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
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
    create_press_acceptance_criteria,
    evaluate_press_acceptance,
    press_policy_contract,
)
from robotics_experiments.rl.press._fingerprint import _press_comparison_fingerprint
from robotics_experiments.rl.training._monitoring import (
    write_evaluation_scalars,
    write_reset_scalars,
    write_system_scalars,
)
from robotics_experiments.rl.training._telemetry import TelemetrySampler, probe_telemetry
from robotics_experiments.rl.training.checkpoint import (
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from robotics_experiments.rl.training.progress import (
    PlateauState,
    advance_plateau,
    select_best_evaluation,
)
from robotics_experiments.rl.training.publication import (
    _REPORT_SCHEMA_VERSION,
    publish_policy_labels,
)
from robotics_experiments.switches import SwitchAsset, load_switch_asset
from robotics_experiments.workflows.isaac_lab._contact_evaluation import _evaluate_contact_policy

__all__ = [
    "LearnerCheckpoint",
    "TrainingResult",
    "pin_learner",
    "validate_learner",
    "run_training",
    "run_attempt",
    "publish_training",
]
_REPORT_FILENAME = "training_report.json"
_ATTEMPT_REPORT_FILENAME = "attempt_report.json"
_TELEMETRY_FILENAME = "telemetry.json"
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_INITIAL_LEARNER_ENV = "RL_INITIAL_LEARNER"
_CAMPAIGN_RUN_DIR_ENV = "RL_CAMPAIGN_RUN_DIR"


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
        update_offset: int,
    ) -> None:
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TRACKIO_DIR"] = str(monitoring_dir)
        import trackio

        self.log_dir = str(monitoring_dir)
        self._update_offset = update_offset
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
        step = self._update_offset + int(global_step)
        self._run.log(
            {tag: float(scalar_value), "training/Completed curriculum updates": step}, step=step
        )

    def flush(self) -> None:
        """Allow Trackio's live background writer to flush its queue."""

    def close(self) -> None:
        self._finish()


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _close_resource(resource, /) -> None:
    """Close owned resources while retaining an exception already being reported."""
    unwinding = sys.exc_info()[0] is not None
    try:
        resource.close()
    except Exception as exc:  # noqa: BLE001 - cleanup must retain the primary error
        if not unwinding:
            raise
        _report(f"WARNING: resource cleanup failed while handling an error: {exc}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _config_values(value: Any, /) -> Any:
    if not isinstance(value, type) and is_dataclass(value):
        return {field.name: _config_values(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _config_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_config_values(item) for item in value]
    return value


def _experiment_provenance(
    *,
    config: ExperimentConfig,
    config_path: Path,
    inputs: rl.PolicyInputs,
    procedural_smoke: bool,
) -> dict[str, Any]:
    """Snapshot the experiment inputs before starting its work."""
    return {
        "config_path": str(config_path.resolve(strict=True)),
        "resolved_config": _config_values(config),
        "procedural_smoke": procedural_smoke,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "rsl_rl": importlib.metadata.version("rsl-rl-lib"),
            "cuda": torch.version.cuda,
        },
        "inputs": {
            "switch_asset": inputs.switch_asset.record(),
            "controller": {
                "path": str(inputs.controller_path),
                "run_id": inputs.controller.run_id,
                "sha256": inputs.controller_digest,
                "comparison_fingerprint": inputs.controller.comparison_fingerprint,
            },
            "flight_parameters": {
                "path": str(inputs.flight_parameters_path),
                "run_id": inputs.flight_parameters.run_id,
                "sha256": inputs.flight_parameters_digest,
                "fingerprint": inputs.flight_parameters.artifact_fingerprint,
            },
        },
    }


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
    switch_asset: SwitchAsset,
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
        switch_asset=switch_asset,
    )


def _artifact_metadata(
    *,
    config: ExperimentConfig,
    env_cfg,
    artifact_run_id: str,
    controller_artifact: ControllerArtifact,
    controller_digest: str,
    flight_parameters: FlightParameters,
    flight_parameters_digest: str,
    hover_reset_samples: HoverResetSamples,
    switch_asset: SwitchAsset,
) -> dict[str, Any]:
    metadata = rl.policy_metadata(
        action_cfg=config.rl.action,
        controller_artifact=controller_artifact,
        controller_digest=controller_digest,
        flight_parameters=flight_parameters,
        flight_parameters_digest=flight_parameters_digest,
        obs_schema_version=int(env_cfg.observation_schema_version),
        obs_dim=int(env_cfg.observation_space),
        policy_rate_hz=config.rates.command_rate_hz,
        policy_interface=str(env_cfg.policy_interface),
        curriculum_stage=(
            config.rl.press.task
            if str(env_cfg.policy_interface) == rl.PolicyInterface.PRESS.value
            else None
        ),
    )
    metadata["switch_asset"] = switch_asset.record()
    metadata["hover_reset_distribution"] = {
        "source": "flight_parameters_policy_window",
        **hover_reset_samples.summary(flight_parameters=flight_parameters),
    }
    return metadata | {"artifact_run_id": artifact_run_id}


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


def _write_artifact_metadata(*, run_dir: Path, metadata: dict[str, Any]) -> None:
    _write_json(path=run_dir / METADATA_FILENAME, payload=metadata)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LearnerCheckpoint:
    """Canonical producer paths and digests for a complete selected learner."""

    checkpoint: str
    artifact_dir: str
    report_path: str
    checkpoint_digest: str
    policy_digest: str
    metadata_digest: str
    report_digest: str


@dataclass(frozen=True)
class TrainingResult:
    """Selected learner, fresh-test decision, completed work and detailed evidence."""

    selected_checkpoint: LearnerCheckpoint | None
    accepted: bool
    completed_updates: int
    report_path: Path


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
    switch_asset: SwitchAsset,
    task_queue,
    result_queue,
    workload_role: str,
) -> None:
    config = load_config(config_path)
    load_switch_asset(model=config.switches.model, expected=switch_asset)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
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
            switch_asset=switch_asset,
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
                    if config.rl.press.task in {"contact", "press"}:
                        evaluation = _evaluate_contact_policy(
                            env=env,
                            policy=policy,
                            seeds=seeds,
                            episodes_per_seed=evaluation_config.episodes_per_seed,
                            iteration=int(iteration),
                            checkpoint_path=Path(checkpoint_name),
                            policy_digest=str(policy_digest),
                            source=f"fresh-{workload_role}",
                            report=_report,
                        )
                    else:
                        rollout = evaluate_press_approach_policy(
                            env=env,
                            policy=policy,
                            seeds=seeds,
                            episodes_per_seed=evaluation_config.episodes_per_seed,
                            hold_steps=config.rl.press.goal.stable_steps,
                            flight_parameters=candidate_metadata.flight_parameters,
                            report=_report,
                            trace_worst_episode=False,
                        )
                        evaluation = press_policy_evaluation(
                            iteration=int(iteration),
                            checkpoint_path=Path(checkpoint_name),
                            policy_digest=str(policy_digest),
                            result=rollout,
                            source=f"fresh-{workload_role}",
                        )
                finally:
                    telemetry_interval = telemetry.finish(
                        work_items=len(seeds) * evaluation_config.episodes_per_seed
                    )
                result_queue.put(("evaluation", (evaluation, telemetry_interval)))
        finally:
            _close_resource(env)
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
        switch_asset: SwitchAsset,
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
                "switch_asset": switch_asset,
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
    criteria: RlPlateauCriteriaConfig,
    evaluation: PolicyEvaluation,
    made_progress: bool,
    plateau_state: PlateauState,
    plateau_reached: bool,
) -> None:
    _report(
        f"plateau iteration={evaluation.iteration} "
        f"metric={criteria.metric_name} direction={criteria.direction} "
        f"current={evaluation.metric(criteria.metric_name):.6f} "
        f"progress={str(made_progress).lower()} "
        f"reference={plateau_state.reference_value:.6f} "
        f"stale={plateau_state.stale_evaluations} "
        f"reached={str(plateau_reached).lower()}"
    )


def _acceptance_payload(*, acceptance) -> dict[str, Any]:
    return {
        "passed": acceptance.passed,
        "checks": [asdict(check) for check in acceptance.checks],
    }


def _acceptance_from_payload(*, payload: dict[str, Any]) -> AcceptanceResult:
    checks = payload.get("checks")
    if not isinstance(checks, list):
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
        reasons=tuple(str(check["name"]) for check in checks if not check["passed"]),
    )


@dataclass(frozen=True)
class _AttemptSelection:
    evaluation: PolicyEvaluation
    artifact_dir: Path
    validation: AcceptanceResult
    attempt_index: int
    evaluation_index: int


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
    by_checkpoint = {candidate.evaluation.checkpoint_path: candidate for candidate in candidates}
    return by_checkpoint[selected.checkpoint_path]


def _restore_phase_branch(
    *,
    runner,
    env,
    parent_checkpoint: Path,
    phase: RlTrainingPhaseConfig,
    phase_number: int,
) -> int:
    """Restore a learner, apply refinement settings, then reset all environments."""
    _report(f"refinement restore START checkpoint={parent_checkpoint}")
    progress = restore_checkpoint(
        runner=runner,
        checkpoint_path=parent_checkpoint,
    )
    restored_iteration = progress.iteration
    _report(f"refinement restore COMPLETE iteration={restored_iteration}")
    training.configure_phase(runner=runner, phase=phase, phase_number=phase_number)
    _report(
        f"refinement configuration COMPLETE learning_rate={phase.learning_rate}; "
        f"update_observation_normalization={phase.update_observation_normalization}; "
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
    switch_asset: SwitchAsset,
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
        switch_asset=switch_asset,
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
        _close_resource(test_worker)
    if test_evaluation.source != "fresh-test":
        raise RuntimeError("final test evaluation was not executed freshly")
    if test_evaluation.policy_digest != selected.policy_digest:
        raise RuntimeError("final test evaluated a different actor digest")
    return test_evaluation, telemetry_interval


def _attempt_report(
    *, run_dir: Path, attempt: dict[str, Any], selection: _AttemptSelection | None, status: str
) -> dict[str, Any]:
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_kind": "attempt",
        "artifact_run_id": run_dir.name,
        **attempt,
        "status": status,
        "selected_evaluation_index": selection.evaluation_index if selection else None,
    }


def _selection_from_report(*, run_dir: Path, report: dict[str, Any]) -> _AttemptSelection | None:
    if (
        report["schema_version"] != _REPORT_SCHEMA_VERSION
        or report["report_kind"] != "attempt"
        or report["artifact_run_id"] != run_dir.name
    ):
        raise ValueError("invalid attempt report identity or schema")
    index = report.get("selected_evaluation_index")
    if index is None:
        return None
    record = report["evaluations"][index]
    return _AttemptSelection(
        attempt_index=report["attempt_index"],
        evaluation_index=index,
        evaluation=_evaluation_from_payload(
            payload=record,
            iteration=record["iteration"],
            checkpoint_path=run_dir / record["checkpoint"],
            policy_digest=record["policy_digest"],
            source=record["source"],
        ),
        artifact_dir=(run_dir / record["policy"]).parent,
        validation=_acceptance_from_payload(
            payload={"passed": record["validation_passed"], "checks": record["checks"]}
        ),
    )


def _selected_payload(
    *, selection: _AttemptSelection, run_dir: Path, criteria: RlPlateauCriteriaConfig
) -> dict[str, Any]:
    selected = selection.evaluation
    return {
        "attempt_index": selection.attempt_index,
        "evaluation_index": selection.evaluation_index,
        "checkpoint": str(selected.checkpoint_path.relative_to(run_dir)),
        "checkpoint_digest": _file_sha256(selected.checkpoint_path),
        "policy": str((selection.artifact_dir / POLICY_FILENAME).relative_to(run_dir)),
        "policy_digest": selected.policy_digest,
        "metadata_digest": _file_sha256(selection.artifact_dir / METADATA_FILENAME),
        "metric": {
            "name": criteria.metric_name,
            "direction": criteria.direction,
            "value": selected.metric(criteria.metric_name),
        },
    }


def _attempt_worker_command(
    *,
    inputs: rl.PolicyInputs,
    attempt_index: int,
    procedural_smoke: bool,
    monitoring_dir: Path,
    update_offset: int,
) -> list[str]:
    command = [
        sys.executable,
        str(_repo_root() / "isaac/runtime/standalone/train_rl.py"),
        "--attempt-worker",
        "--attempt-index",
        str(attempt_index),
        "--controller-artifact",
        str(inputs.controller_path),
        "--controller-artifact-digest",
        inputs.controller_digest,
        "--flight-parameters-artifact",
        str(inputs.flight_parameters_path),
        "--flight-parameters-artifact-digest",
        inputs.flight_parameters_digest,
        "--switch-asset",
        json.dumps(inputs.switch_asset.record()),
        "--monitoring-dir",
        str(monitoring_dir),
        "--update-offset",
        str(update_offset),
    ]
    if procedural_smoke:
        command.append("--procedural-smoke")
    return command


def pin_learner(*, report_path: Path) -> LearnerCheckpoint:
    report_path = report_path.resolve(strict=True)
    report = json.loads(report_path.read_text())
    run_dir = report_path.parent
    if report["artifact_run_id"] != run_dir.name:
        raise ValueError("initial learner report does not identify its canonical producer run")
    if report["schema_version"] != _REPORT_SCHEMA_VERSION or report["report_kind"] != "training":
        raise ValueError("unsupported training report")
    selected = report["selected"]
    checkpoint = (run_dir / selected["checkpoint"]).resolve(strict=True)
    artifact_dir = (run_dir / selected["policy"]).resolve(strict=True).parent
    if not checkpoint.is_relative_to(run_dir) or not artifact_dir.is_relative_to(run_dir):
        raise ValueError("selected learner escapes its producer run")
    checkpoint_digest = _file_sha256(checkpoint)
    if selected["checkpoint_digest"] != checkpoint_digest:
        raise ValueError("selected checkpoint digest does not match its report")
    policy_digest = _file_sha256(artifact_dir / POLICY_FILENAME)
    if policy_digest != selected["policy_digest"]:
        raise ValueError("selected actor digest does not match its report")
    if _file_sha256(artifact_dir / METADATA_FILENAME) != selected["metadata_digest"]:
        raise ValueError("selected policy metadata changed")
    return LearnerCheckpoint(
        checkpoint=str(checkpoint),
        artifact_dir=str(artifact_dir),
        report_path=str(report_path),
        checkpoint_digest=checkpoint_digest,
        policy_digest=policy_digest,
        metadata_digest=_file_sha256(artifact_dir / METADATA_FILENAME),
        report_digest=_file_sha256(report_path),
    )


def validate_learner(
    *,
    learner: LearnerCheckpoint,
    config: ExperimentConfig,
    inputs: rl.PolicyInputs,
) -> None:
    """Check complete learner identity and compatibility, regardless of qualification."""
    if pin_learner(report_path=Path(learner.report_path)) != learner:
        raise ValueError("pinned learner differs from its producer report")
    artifact_dir = Path(learner.artifact_dir)
    for path, digest in (
        (Path(learner.checkpoint), learner.checkpoint_digest),
        (artifact_dir / POLICY_FILENAME, learner.policy_digest),
        (artifact_dir / METADATA_FILENAME, learner.metadata_digest),
        (Path(learner.report_path), learner.report_digest),
    ):
        if _file_sha256(path) != digest:
            raise ValueError(f"pinned learner input changed: {path}")
    metadata = rl.validate_policy_artifact(artifact_dir=artifact_dir)
    contract = press_policy_contract(config=config, controller=inputs.controller)
    if replace(metadata.contract, curriculum_stage=contract.curriculum_stage) != contract:
        raise ValueError("initial learner has an incompatible policy contract")
    if (
        metadata.controller_digest != inputs.controller_digest
        or metadata.flight_parameters_digest != inputs.flight_parameters_digest
    ):
        raise ValueError("initial learner uses different pinned controller or flight evidence")
    # Architecture comes from the pinned producer report, independent of pipeline labels.
    source_policy = json.loads(Path(learner.report_path).read_text())["provenance"][
        "resolved_config"
    ]["rl"]["press"]["policy"]
    target_policy = _config_values(config.rl.press.policy)
    for key in ("activation", "normalize_observations", "critic"):
        if source_policy[key] != target_policy[key]:
            raise ValueError(f"initial learner has incompatible policy {key}")
    if source_policy["actor"]["hidden_dims"] != target_policy["actor"]["hidden_dims"]:
        raise ValueError("initial learner has incompatible actor dimensions")
    payload = load_checkpoint(checkpoint_path=Path(learner.checkpoint))
    state = payload["model_state_dict"]
    actor_state = {
        (
            "0." + key.removeprefix("actor_obs_normalizer.")
            if key.startswith("actor_obs_normalizer.")
            else "1." + key.removeprefix("actor.")
        ): value
        for key, value in state.items()
        if key.startswith(("actor_obs_normalizer.", "actor."))
    }
    exported = torch.jit.load(str(artifact_dir / POLICY_FILENAME), map_location="cpu").state_dict()
    if actor_state.keys() != exported.keys() or any(
        not torch.equal(actor_state[key], exported[key]) for key in exported
    ):
        raise ValueError("initial learner checkpoint does not match its selected actor")


def run_attempt(
    *,
    config: ExperimentConfig,
    config_path: Path,
    inputs: rl.PolicyInputs,
    initial_learner: LearnerCheckpoint | None,
    run_dir: Path,
    monitoring_dir: Path,
    update_offset: int,
    attempt_index: int,
    procedural_smoke: bool,
    device: str,
) -> None:
    """Train exactly one attempt in this process, with fresh simulator state."""
    seed = config.rl.training.attempt_seeds[attempt_index]
    acceptance_criteria = create_press_acceptance_criteria(
        config=config.rl.press, flight_parameters=inputs.flight_parameters
    )
    comparison_fingerprint = _press_comparison_fingerprint(
        config=config,
        contract=press_policy_contract(config=config, controller=inputs.controller),
        flight_parameters_fingerprint=inputs.flight_parameters.artifact_fingerprint,
    )
    training_config = config.rl.training
    plateau_criteria = config.rl.training.plateau
    controller_artifact = inputs.controller
    controller_digest = inputs.controller_digest
    flight_parameters = inputs.flight_parameters
    flight_parameters_digest = inputs.flight_parameters_digest
    phases = (
        _procedural_smoke_phases(phases=tuple(config.rl.training.phases))
        if procedural_smoke
        else tuple(config.rl.training.phases)
    )
    if initial_learner is not None:
        validate_learner(learner=initial_learner, config=config, inputs=inputs)
    recorded = json.loads((run_dir / _REPORT_FILENAME).read_text())
    if recorded["provenance"]["resolved_config"] != _config_values(config):
        raise ValueError("stage configuration changed before worker startup")
    if not run_dir.is_dir():
        raise ValueError(f"campaign run directory does not exist: {run_dir}")
    writer = None
    validation_worker: _EvaluationWorker | None = None
    partial_attempt_report_path: Path | None = None
    partial_attempt_record: dict[str, Any] | None = None
    partial_candidates: list[_AttemptSelection] = []
    try:
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
            monitoring_dir=monitoring_dir,
            project=run_dir.parent.name,
            run_name=f"{run_dir.name}.attempt_{attempt_index + 1:02d}",
            campaign_run_id=run_dir.name,
            training_seed=seed,
            resume="never",
            update_offset=update_offset,
        )
        writer = attempt_writer
        write_reset_scalars(
            writer=attempt_writer,
            summary=inputs.hover_reset_samples.summary(flight_parameters=inputs.flight_parameters),
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
            switch_asset=inputs.switch_asset,
        )
        runner: Any = None
        evaluate_candidate: Any = None
        write_training_progress: Any = None
        try:
            metadata = _artifact_metadata(
                config=config,
                env_cfg=env_cfg,
                artifact_run_id=run_dir.name,
                controller_artifact=controller_artifact,
                controller_digest=controller_digest,
                flight_parameters=flight_parameters,
                flight_parameters_digest=flight_parameters_digest,
                hover_reset_samples=inputs.hover_reset_samples,
                switch_asset=inputs.switch_asset,
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
            metadata["initialization"] = (
                {"method": env_cfg.policy_config.actor.output_initialization}
                if initial_learner is None
                else {"method": "checkpoint", **asdict(initial_learner)}
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
            if initial_learner is None:
                training.initialize_actor_output(
                    runner=runner,
                    method=env_cfg.policy_config.actor.output_initialization,
                )
            else:
                restore_checkpoint(runner=runner, checkpoint_path=Path(initial_learner.checkpoint))
            checkpoints_dir = attempt_dir / "checkpoints"
            checkpoints_dir.mkdir()
            initial_checkpoint = checkpoints_dir / "phase_01_model_0.pt"
            save_checkpoint(
                runner=runner,
                checkpoint_path=initial_checkpoint,
                iteration=0,
                curriculum_updates=update_offset + runner.completed_updates,
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
                "evaluations": evaluation_records,
                "phase_results": phase_results,
                "progress": {"completed_updates": 0, "update_offset": update_offset},
                "telemetry_path": str((attempt_dir / _TELEMETRY_FILENAME).relative_to(run_dir)),
                "selected_evaluation_index": None,
            }
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
                        switch_asset=inputs.switch_asset,
                        timeout_s=training_config.evaluation_worker_timeout_s,
                        workload_role="validation",
                    )
                evaluation, telemetry_interval = validation_worker.evaluate(
                    iteration=iteration,
                    checkpoint_path=checkpoint_path,
                    artifact_dir=artifact_dir,
                    policy_digest=policy_digest,
                )
                monitoring_step = runner.completed_updates
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

                validation = evaluate_press_acceptance(
                    criteria=acceptance_criteria,
                    evaluation=evaluation,
                )
                candidates.append(
                    _AttemptSelection(
                        evaluation=evaluation,
                        artifact_dir=artifact_dir,
                        validation=validation,
                        attempt_index=attempt_index,
                        evaluation_index=evaluation_index,
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
                monitoring_step = runner.completed_updates
                evaluation_records[-1].update(
                    monitoring_step=monitoring_step,
                    learner_updates=runner.learner_updates,
                    curriculum_updates=update_offset + monitoring_step,
                )
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

            def write_training_progress() -> None:
                current_selection = _select_attempt_candidate(
                    criteria=plateau_criteria,
                    candidates=tuple(candidates),
                )
                attempt_record["selected_evaluation_index"] = current_selection.evaluation_index
                attempt_record["progress"]["completed_updates"] = runner.completed_updates
                _write_json(
                    path=attempt_report_path,
                    payload=_attempt_report(
                        run_dir=run_dir,
                        attempt=attempt_record,
                        selection=current_selection,
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
                criteria=plateau_criteria,
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
            randomize_episode_lengths = config.rl.press.task == "approach"
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
                        "update_observation_normalization="
                        f"{phase.update_observation_normalization}; "
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
                                chunk * int(training_config.num_steps_per_env) * int(env.num_envs)
                            )
                        )
                    monitoring_step = runner.completed_updates
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

                    randomize_episode_lengths = False
                    branch_iteration += result
                    phase_iteration += result
                    checkpoint_path = checkpoints_dir / (
                        f"phase_{phase_index + 1:02d}_model_{branch_iteration}.pt"
                    )
                    save_checkpoint(
                        runner=runner,
                        checkpoint_path=checkpoint_path,
                        iteration=branch_iteration,
                        curriculum_updates=update_offset + runner.completed_updates,
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
                        criteria=plateau_criteria,
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
                        "completed_iterations": phase_iteration,
                        "terminal_reason": phase_reason,
                    }
                )
            attempt_selection = _select_attempt_candidate(
                criteria=plateau_criteria,
                candidates=tuple(candidates),
            )
            attempt_record["selected_evaluation_index"] = attempt_selection.evaluation_index
            attempt_record["progress"]["completed_updates"] = runner.completed_updates
            _write_json(
                path=attempt_report_path,
                payload=_attempt_report(
                    run_dir=run_dir,
                    attempt=attempt_record,
                    selection=attempt_selection,
                    status="validation_passed"
                    if attempt_selection.validation.passed
                    else "training",
                ),
            )
        finally:
            if runner is not None and partial_attempt_record is not None:
                partial_attempt_record["progress"]["completed_updates"] = runner.completed_updates
            teardown_started = time.monotonic()
            _report(f"attempt teardown START attempt={attempt_index + 1} seed={seed}")
            evaluate_candidate = None
            write_training_progress = None
            runner = None
            gc.collect()
            _close_resource(env)
            env = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _report(
                f"attempt teardown COMPLETE attempt={attempt_index + 1} seed={seed} "
                f"elapsed_s={time.monotonic() - teardown_started:.1f}"
            )

        if validation_worker is not None:
            _close_resource(validation_worker)
            validation_worker = None
        worker_report = _attempt_report(
            run_dir=run_dir,
            attempt=attempt_record,
            selection=attempt_selection,
            status="procedural_smoke_complete"
            if procedural_smoke
            else "validation_passed"
            if attempt_selection.validation.passed
            else "validation_failed",
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
                    attempt=partial_attempt_record,
                    selection=partial_selection,
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                ),
            )
        if partial_attempt_report_path is not None and partial_attempt_report_path.is_file():
            failure = json.loads(partial_attempt_report_path.read_text())
            failure.update(
                status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=str(exc),
            )
            _write_json(path=partial_attempt_report_path, payload=failure)
        raise
    finally:
        if writer is not None:
            _close_resource(writer)
        if validation_worker is not None:
            # evaluate_candidate assigns this nonlocal before failures reach cleanup.
            _close_resource(validation_worker)  # type: ignore[unreachable]


def run_training(
    *,
    config: ExperimentConfig,
    config_path: Path,
    inputs: rl.PolicyInputs,
    initial_learner: LearnerCheckpoint | None,
    run_dir: Path,
    monitoring_dir: Path,
    update_offset: int,
    procedural_smoke: bool,
) -> TrainingResult:
    """Own one attempt campaign, validation selection and one fresh acceptance test."""
    if initial_learner is not None:
        validate_learner(learner=initial_learner, config=config, inputs=inputs)
    criteria = create_press_acceptance_criteria(
        config=config.rl.press, flight_parameters=inputs.flight_parameters
    )
    fingerprint = _press_comparison_fingerprint(
        config=config,
        contract=press_policy_contract(config=config, controller=inputs.controller),
        flight_parameters_fingerprint=inputs.flight_parameters.artifact_fingerprint,
    )
    phases = (
        _procedural_smoke_phases(phases=tuple(config.rl.training.phases))
        if procedural_smoke
        else tuple(config.rl.training.phases)
    )
    provenance = _experiment_provenance(
        config=config, config_path=config_path, inputs=inputs, procedural_smoke=procedural_smoke
    )
    provenance["initial_learner"] = asdict(initial_learner) if initial_learner else None
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / _REPORT_FILENAME
    report: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_kind": "training",
        "artifact_run_id": run_dir.name,
        "policy_name": run_dir.parent.name,
        "provenance": provenance,
        "comparison_fingerprint": fingerprint,
        "acceptance_criteria": asdict(criteria),
        "effective_phases": [asdict(p) for p in phases],
        "hover_reset_distribution": inputs.hover_reset_samples.summary(
            flight_parameters=inputs.flight_parameters
        ),
        "attempts": [],
        "progress": {"completed_updates": 0, "update_offset": update_offset},
        "status": "training",
        "selected": None,
        "test_evaluation": None,
        "acceptance": None,
        "telemetry_path": _TELEMETRY_FILENAME,
    }
    selections: list[_AttemptSelection] = []
    selected: _AttemptSelection | None = None
    worker_pids: set[int] = set()

    def collect_attempts() -> None:
        nonlocal selected
        selections.clear()
        completed_updates = 0
        for reference in report["attempts"]:
            path = run_dir / reference
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            completed_updates += payload.get("progress", {}).get("completed_updates", 0)
            candidate = _selection_from_report(run_dir=run_dir, report=payload)
            if candidate is not None:
                selections.append(candidate)
        report["progress"]["completed_updates"] = completed_updates
        if selections:
            selected = _select_attempt_candidate(
                criteria=config.rl.training.plateau, candidates=tuple(selections)
            )
            report["selected"] = _selected_payload(
                selection=selected, run_dir=run_dir, criteria=config.rl.training.plateau
            )

    _write_json(path=report_path, payload=report)
    try:
        # Diagnostics are evidence in their own sidecar, never selection inputs.
        _write_json(
            path=run_dir / _TELEMETRY_FILENAME,
            payload={"startup_probe": probe_telemetry(), "intervals": []},
        )
        for attempt_index, seed in enumerate(config.rl.training.attempt_seeds):
            attempt_dir = run_dir / f"attempt_{attempt_index + 1:02d}_seed_{seed}"
            attempt_report_path = attempt_dir / _ATTEMPT_REPORT_FILENAME
            report["attempts"].append(str(attempt_report_path.relative_to(run_dir)))
            _write_json(path=report_path, payload=report)
            worker_environment = os.environ.copy()
            worker_environment[_CAMPAIGN_RUN_DIR_ENV] = str(run_dir)
            worker_environment[_CONFIG_PATH_ENV] = str(config_path)
            worker_environment.pop(_INITIAL_LEARNER_ENV, None)
            if initial_learner:
                worker_environment[_INITIAL_LEARNER_ENV] = json.dumps(asdict(initial_learner))
            command = _attempt_worker_command(
                inputs=inputs,
                attempt_index=attempt_index,
                procedural_smoke=procedural_smoke,
                monitoring_dir=monitoring_dir,
                update_offset=update_offset + report["progress"]["completed_updates"],
            )
            _report(
                f"attempt process START run={run_dir.name} attempt={attempt_index + 1} seed={seed}"
            )
            with subprocess.Popen(command, cwd=_repo_root(), env=worker_environment) as process:
                try:
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=config.rl.training.evaluation_worker_timeout_s)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if returncode:
                raise RuntimeError(f"attempt worker exited with status {returncode}")
            collect_attempts()
            if not attempt_report_path.is_file():
                raise RuntimeError("attempt worker exited without an attempt report")
            attempt = json.loads(attempt_report_path.read_text())
            if attempt["attempt_index"] != attempt_index or attempt["training_seed"] != seed:
                raise ValueError("attempt worker identity differs from its launch")
            pid = attempt["worker_pid"]
            if pid == os.getpid() or pid in worker_pids:
                raise ValueError("attempts must run in distinct worker processes")
            worker_pids.add(pid)
            if attempt["status"] not in {
                "validation_passed",
                "validation_failed",
                "procedural_smoke_complete",
            }:
                raise RuntimeError("attempt worker did not complete successfully")
            candidate = _selection_from_report(run_dir=run_dir, report=attempt)
            if candidate is None:
                raise RuntimeError("completed attempt has no evaluated candidate")
            _write_json(path=report_path, payload=report)
            if not procedural_smoke:
                if candidate.validation.passed:
                    break
                if (
                    attempt["evaluations"][candidate.evaluation_index]["phase_index"]
                    < len(phases) - 1
                ):
                    report["early_stop_reason"] = (
                        "final refinement did not improve the selected earlier-phase actor"
                    )
                    break
        if selected is None:
            raise RuntimeError("training produced no evaluated candidate")
        if procedural_smoke:
            report["status"] = "procedural_smoke_complete"
        elif not selected.validation.passed:
            report["status"] = "validation_failed"
        else:
            report["status"] = "testing"
            _write_json(path=report_path, payload=report)
            evaluation, telemetry = _run_final_test(
                config_path=config_path,
                config=config,
                selection=selected,
                flight_parameters_path=inputs.flight_parameters_path,
                flight_parameters_digest=inputs.flight_parameters_digest,
                switch_asset=inputs.switch_asset,
            )
            acceptance = evaluate_press_acceptance(criteria=criteria, evaluation=evaluation)
            report["test_evaluation"] = {
                "policy_digest": evaluation.policy_digest,
                "source": evaluation.source,
                "metrics": {metric.name: metric.value for metric in evaluation.metrics},
                "reward_means": {reward.name: reward.mean for reward in evaluation.reward_means},
            }
            report["acceptance"] = _acceptance_payload(acceptance=acceptance)
            _record_telemetry(
                path=run_dir / _TELEMETRY_FILENAME,
                interval=telemetry,
                target_rate=config.rl.training.gpu_utilization_target_rate,
            )
            summary = _require_telemetry_interval(interval=telemetry)
            attempt = json.loads((run_dir / report["attempts"][selected.attempt_index]).read_text())
            writer = _TrackioWriter(
                monitoring_dir=monitoring_dir,
                project=run_dir.parent.name,
                run_name=f"{run_dir.name}.attempt_{selected.attempt_index + 1:02d}",
                campaign_run_id=run_dir.name,
                training_seed=attempt["training_seed"],
                resume="must",
                update_offset=attempt["progress"]["update_offset"],
            )
            try:
                step = attempt["progress"]["completed_updates"]
                write_evaluation_scalars(
                    writer=writer,
                    role="test",
                    evaluation=evaluation,
                    all_checks_passed=acceptance.passed,
                    step=step,
                )
                write_system_scalars(writer=writer, role="test", summary=summary, step=step)
            finally:
                _close_resource(writer)
            report["status"] = "accepted" if acceptance.passed else "test_failed"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = str(exc)
        try:
            collect_attempts()
        except Exception as evidence_error:  # noqa: BLE001 - retain the original runtime failure
            report["evidence_error"] = str(evidence_error)
        raise
    finally:
        _write_json(path=report_path, payload=report)
    return TrainingResult(
        selected_checkpoint=pin_learner(report_path=report_path) if selected else None,
        accepted=report["status"] == "accepted",
        completed_updates=report["progress"]["completed_updates"],
        report_path=report_path,
    )


def publish_training(
    *, result: TrainingResult, curriculum_run_id: str, passed_stages: list[dict[str, str]]
) -> None:
    """Publish an accepted stage's exact tested actor without changing its report."""
    learner = result.selected_checkpoint
    report = json.loads(result.report_path.read_text())
    if (
        not result.accepted
        or learner is None
        or report["status"] != "accepted"
        or not report["acceptance"]["passed"]
    ):
        raise ValueError("only an accepted stage can be published")
    if (
        pin_learner(report_path=result.report_path) != learner
        or report["test_evaluation"]["policy_digest"] != learner.policy_digest
    ):
        raise ValueError("published learner must match the exact accepted actor")
    run_dir = result.report_path.parent
    selected = report["selected"]
    attempt = json.loads((run_dir / report["attempts"][selected["attempt_index"]]).read_text())
    evaluation = attempt["evaluations"][selected["evaluation_index"]]
    metadata = json.loads((Path(learner.artifact_dir) / METADATA_FILENAME).read_text())
    evaluation_config = report["provenance"]["resolved_config"]["rl"]["press"]["evaluation"]

    def workload(role: str) -> dict[str, Any]:
        seeds = evaluation_config[f"{role}_seeds"]
        return {
            "role": role,
            "seeds": seeds,
            "episodes_per_seed": evaluation_config["episodes_per_seed"],
            "episode_count": len(seeds) * evaluation_config["episodes_per_seed"],
        }

    metadata.update(
        {
            "selected_checkpoint_iteration": evaluation["iteration"],
            "selected_policy_digest": learner.policy_digest,
            "selection_method": selected["metric"]["name"],
            "validation_selection": {
                "comparison_fingerprint": report["comparison_fingerprint"],
                "policy_digest": learner.policy_digest,
                "metrics": evaluation["metrics"],
                "reward_means": evaluation["reward_means"],
                "workload": workload("validation"),
            },
            "test_evaluation": {**report["test_evaluation"], "workload": workload("test")},
            "acceptance": report["acceptance"],
            "curriculum": {"run_id": curriculum_run_id, "passed_stages": passed_stages},
        }
    )
    with tempfile.TemporaryDirectory(prefix=".training-publication-", dir=run_dir) as temporary:
        staging = Path(temporary)
        shutil.copy2(Path(learner.artifact_dir) / POLICY_FILENAME, staging / POLICY_FILENAME)
        if _file_sha256(staging / POLICY_FILENAME) != learner.policy_digest:
            raise RuntimeError("publication changed the tested actor bytes")
        _write_json(path=staging / METADATA_FILENAME, payload=metadata)
        rl.validate_policy_artifact(artifact_dir=staging)
        for filename in (POLICY_FILENAME, METADATA_FILENAME):
            (staging / filename).replace(run_dir / filename)
    metric = selected["metric"]
    publish_policy_labels(
        policy_root=run_dir.parent,
        run_dir=run_dir,
        comparison_fingerprint=report["comparison_fingerprint"],
        metric_name=metric["name"],
        metric_value=metric["value"],
        direction=metric["direction"],
    )

"""Train, evaluate, select, and publish one configured scratch RL policy.

One invocation owns one fresh runner and optimizer. Declarative phases train in
bounded chunks, a persistent held-out worker evaluates the exact exported actor
after each chunk, and a phase-local plateau rule decides when to advance. Final
acceptance is applied only after training to the run-wide best evaluation.

Run inside the Isaac Lab container (make train-rl):
    /isaac-sim/python.sh /isaac-sim/proj/isaac/runtime/standalone/train_rl.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import robotics_experiments.rl as rl
import robotics_experiments.rl.training.rsl_rl as training
from robotics_experiments.config import (
    ExperimentConfig,
    RlTrainingPhaseConfig,
    load_config,
)
from robotics_experiments.controller import (
    ControllerArtifact,
    load_controller_artifact,
    resolve_controller_path,
)
from robotics_experiments.rl import METADATA_FILENAME, POLICY_FILENAME
from robotics_experiments.rl.training._fingerprint import _press_comparison_fingerprint
from robotics_experiments.rl.training.progress import (
    EvaluationMetric,
    PlateauState,
    PolicyEvaluation,
    RewardMean,
    advance_plateau,
    select_best_evaluation,
)
from robotics_experiments.rl.training.selection import publish_policy_labels

_POLICY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"
_CONTROLLER_ARTIFACT_ENV = "CONTROLLER_ARTIFACT"
_REPORT_FILENAME = "training_report.json"
_REPORT_SCHEMA_VERSION = 1
_CACHE_SCHEMA_VERSION = 2
_EVALUATOR_REVISION = 1


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


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
    seed: int,
    num_envs: int | None,
    experiment_config: ExperimentConfig,
    controller_settings,
    policy_contract=None,
    use_training_reset_distribution: bool = False,
):
    # AppLauncher must start before the Isaac Lab workflow import.
    from robotics_experiments.workflows.isaac_lab import make_isaac_lab_env

    return make_isaac_lab_env(
        task_id=task_id,
        experiment_config=experiment_config,
        controller_settings=controller_settings,
        seed=seed,
        num_envs=num_envs,
        policy_contract=policy_contract,
        use_training_reset_distribution=use_training_reset_distribution,
    )


def _artifact_metadata(
    *,
    config: ExperimentConfig,
    env_cfg,
    source_metadata: dict[str, str | bool],
    artifact_run_id: str,
    controller_artifact: ControllerArtifact,
    controller_digest: str,
) -> dict[str, Any]:
    metadata = rl.policy_metadata(
        action_cfg=config.rl.action,
        controller_artifact=controller_artifact,
        controller_digest=controller_digest,
        obs_schema_version=int(env_cfg.observation_schema_version),
        obs_dim=int(env_cfg.observation_space),
        policy_rate_hz=config.rl.policy_rate_hz,
        policy_interface=str(env_cfg.policy_interface),
        curriculum_stage=(
            config.rl.press.curriculum_stage
            if str(env_cfg.policy_interface) == rl.PolicyInterface.PRESS_APPROACH.value
            else None
        ),
    )
    return metadata | source_metadata | {"artifact_run_id": artifact_run_id}


def _write_json(*, path: Path, payload: dict[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_artifact_metadata(*, run_dir: Path, metadata: dict[str, Any]) -> None:
    _write_json(path=run_dir / METADATA_FILENAME, payload=metadata)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_content_sha256(path: Path) -> str:
    """Hash TorchScript executable content without its generated archive ID."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist() if not name.endswith("/.data/serialization_id")
        )
        for name in members:
            encoded_name = name.encode()
            content = archive.read(name)
            digest.update(len(encoded_name).to_bytes(8, "big"))
            digest.update(encoded_name)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def _load_training_controller(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    repo_root: Path,
) -> tuple[ControllerArtifact, str]:
    authored = (
        args.controller_artifact
        or os.environ.get(_CONTROLLER_ARTIFACT_ENV)
        or config.rl.training.controller_artifact
    )
    artifact_path = Path(authored)
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    artifact = load_controller_artifact(artifact_path)
    vehicle_id = config.rl.training.vehicle_id
    try:
        vehicle_label, vehicle = next(
            (label, item) for label, item in config.vehicles.items() if item.id == vehicle_id
        )
    except StopIteration as exc:
        raise ValueError(f"rl.training.vehicle_id={vehicle_id} is not in the experiment") from exc
    action = config.rl.action
    mismatches = []
    if artifact.vehicle != vehicle_label:
        mismatches.append(f"vehicle {artifact.vehicle!r} != {vehicle_label!r}")
    if artifact.airframe != Path(vehicle.spawn.usd_path).name:
        mismatches.append(
            f"airframe {artifact.airframe!r} != {Path(vehicle.spawn.usd_path).name!r}"
        )
    if artifact.command_rate_hz != config.rl.policy_rate_hz:
        mismatches.append(f"cadence {artifact.command_rate_hz} != {config.rl.policy_rate_hz}")
    if action.kind not in artifact.certified_control_modes or action.action_dim != 4:
        mismatches.append(f"action contract {(action.kind, action.action_dim)!r} is incompatible")
    if mismatches:
        raise ValueError("controller artifact is incompatible: " + "; ".join(mismatches))
    if not artifact.advisory_quality_passed:
        raise ValueError(
            "controller artifact failed its quality gates: "
            + "; ".join(artifact.advisory_quality_reasons)
        )
    controller_path = resolve_controller_path(artifact_path)
    return artifact, _file_sha256(controller_path)


def _capped_phases(
    *,
    phases: tuple[RlTrainingPhaseConfig, ...],
    maximum_iterations: int | None,
) -> tuple[RlTrainingPhaseConfig, ...]:
    if maximum_iterations is None:
        return phases
    remaining = maximum_iterations
    capped = []
    for phase in phases:
        if remaining == 0:
            break
        phase_iterations = min(remaining, int(phase.max_iterations))
        capped.append(
            replace(
                phase,
                max_iterations=phase_iterations,
                evaluation_interval=min(phase.evaluation_interval, phase_iterations),
            )
        )
        remaining -= phase_iterations
    return tuple(capped)


def _evaluation_payload(evaluation: PolicyEvaluation) -> dict[str, Any]:
    return {
        "metrics": {metric.name: metric.value for metric in evaluation.metrics},
        "reward_means": {reward.name: reward.mean for reward in evaluation.reward_means},
    }


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
        raise ValueError("cached policy evaluation has an invalid metric contract")
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


def _cache_path(
    *,
    cache_dir: Path,
    policy_content_digest: str,
    comparison_fingerprint: str,
    controller_digest: str,
) -> Path:
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "evaluator_revision": _EVALUATOR_REVISION,
        "policy_content_digest": policy_content_digest,
        "comparison_fingerprint": comparison_fingerprint,
        "controller_digest": controller_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.json"


def _read_cached_evaluation(
    *,
    cache_path: Path,
    iteration: int,
    checkpoint_path: Path,
    policy_digest: str,
    policy_content_digest: str,
) -> PolicyEvaluation | None:
    if not cache_path.is_file():
        return None
    payload = json.loads(cache_path.read_text())
    if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation cache schema in {cache_path}")
    if payload.get("policy_content_digest") != policy_content_digest:
        raise ValueError("cached policy content digest does not match its cache key")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"evaluation cache entry is incomplete: {cache_path}")
    return _evaluation_from_payload(
        payload=evaluation,
        iteration=iteration,
        checkpoint_path=checkpoint_path,
        policy_digest=policy_digest,
        source="cache",
    )


def _write_cached_evaluation(
    *,
    cache_path: Path,
    evaluation: PolicyEvaluation,
    policy_content_digest: str,
    source_run_id: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path=cache_path,
        payload={
            "schema_version": _CACHE_SCHEMA_VERSION,
            "source_run_id": source_run_id,
            "policy_content_digest": policy_content_digest,
            "evaluation": _evaluation_payload(evaluation),
        },
    )


def _evaluation_worker_main(
    *,
    config_path: str,
    run_dir: str,
    task_queue,
    result_queue,
) -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        config = load_config(config_path)
        metadata = rl.load_policy_metadata(artifact_dir=Path(run_dir))
        evaluation_config = config.rl.press.evaluation
        seeds = tuple(int(seed) for seed in evaluation_config.seeds)
        env, _env_cfg = _build_env(
            task_id=config.rl.training.task_id,
            seed=seeds[0],
            num_envs=evaluation_config.episodes_per_seed,
            experiment_config=config,
            controller_settings=metadata.controller_settings,
            policy_contract=metadata.contract,
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

                rollout = evaluate_press_approach_policy(
                    env=env,
                    policy=policy,
                    seeds=seeds,
                    episodes_per_seed=evaluation_config.episodes_per_seed,
                    goal_config=config.rl.press.goal,
                    report=_report,
                    trace_worst_episode=False,
                )
                evaluation = press_policy_evaluation(
                    iteration=int(iteration),
                    checkpoint_path=Path(checkpoint_name),
                    policy_digest=str(policy_digest),
                    result=rollout,
                    evaluation_config=evaluation_config,
                    source="fresh",
                )
                result_queue.put(("evaluation", evaluation))
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
        timeout_s: float,
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
                "task_queue": self._tasks,
                "result_queue": self._results,
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
    ) -> PolicyEvaluation:
        self._tasks.put((iteration, str(checkpoint_path), str(artifact_dir), policy_digest))
        kind, payload = self._receive()
        if kind == "error":
            raise RuntimeError(f"held-out evaluator failed:\n{payload}")
        if kind != "evaluation" or not isinstance(payload, PolicyEvaluation):
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
    run_dir: Path,
    iteration: int,
    metadata: dict[str, Any],
) -> tuple[Path, str, str]:
    artifact_dir = run_dir / "evaluated" / f"model_{iteration}"
    training.export_policy(runner, artifact_dir, metadata=metadata)
    policy_path = artifact_dir / POLICY_FILENAME
    return artifact_dir, _file_sha256(policy_path), _policy_content_sha256(policy_path)


def _report_evaluation(
    *,
    evaluation: PolicyEvaluation,
    phase_index: int,
    phase_iteration: int,
    made_progress: bool,
    plateau_state: PlateauState,
    plateau_reached: bool,
    run_dir: Path,
) -> dict[str, Any]:
    metrics = {metric.name: metric.value for metric in evaluation.metrics}
    _report(
        f"evaluation phase={phase_index + 1} phase_iteration={phase_iteration} "
        f"iteration={evaluation.iteration} source={evaluation.source} "
        f"cost={metrics['held_out_task_cost']:.6f} "
        f"progress={str(made_progress).lower()} "
        f"stale={plateau_state.stale_evaluations} "
        f"success={metrics['approach_success_rate']:.3f} "
        f"collision={metrics['collision_rate']:.3f} "
        f"other_flight_failure={metrics['other_flight_failure_rate']:.3f} "
        f"timeout={metrics['timeout_rate']:.3f}"
    )
    return {
        "phase_index": phase_index,
        "phase_iteration": phase_iteration,
        "iteration": evaluation.iteration,
        "checkpoint": str(evaluation.checkpoint_path.relative_to(run_dir)),
        "policy": str(
            (run_dir / "evaluated" / f"model_{evaluation.iteration}" / POLICY_FILENAME).relative_to(
                run_dir
            )
        ),
        "policy_digest": evaluation.policy_digest,
        "source": evaluation.source,
        "metrics": metrics,
        "reward_means": {reward.name: reward.mean for reward in evaluation.reward_means},
        "plateau": {
            "made_progress": made_progress,
            "reference_value": plateau_state.reference_value,
            "stale_evaluations": plateau_state.stale_evaluations,
            "reached": plateau_reached,
        },
    }


def _training_report(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    phases: tuple[RlTrainingPhaseConfig, ...],
    comparison_fingerprint: str,
    evaluations: list[dict[str, Any]],
    phase_results: list[dict[str, Any]],
    selected: PolicyEvaluation | None,
    acceptance,
) -> dict[str, Any]:
    criteria = config.rl.training.plateau
    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "artifact_run_id": run_dir.name,
        "policy_name": config.rl.training.policy_name,
        "policy_interface": rl.PolicyInterface.PRESS_APPROACH.value,
        "comparison_fingerprint": comparison_fingerprint,
        "phases": [asdict(phase) for phase in phases],
        "plateau_criteria": asdict(criteria),
        "evaluation_workload": {
            "seeds": list(config.rl.press.evaluation.seeds),
            "episodes_per_seed": config.rl.press.evaluation.episodes_per_seed,
        },
        "evaluations": evaluations,
        "phase_results": phase_results,
        "selected_iteration": selected.iteration if selected is not None else None,
        "selected_metric": (
            {
                "name": criteria.metric_name,
                "direction": criteria.direction,
                "value": selected.metric(criteria.metric_name),
            }
            if selected is not None
            else None
        ),
        "acceptance": (
            {"passed": acceptance.passed, "reasons": list(acceptance.reasons)}
            if acceptance is not None
            else None
        ),
    }


def _publish_selected(
    *,
    runner,
    run_dir: Path,
    policy_root: Path,
    metadata: dict[str, Any],
    report: dict[str, Any],
    selected: PolicyEvaluation,
    acceptance,
    comparison_fingerprint: str,
    device: str,
    config: ExperimentConfig,
) -> bool:
    selected_metadata = metadata | {
        "selected_checkpoint_iteration": selected.iteration,
        "selection_method": config.rl.training.plateau.metric_name,
        "checkpoint_evaluation": {
            "comparison_fingerprint": comparison_fingerprint,
            "policy_digest": selected.policy_digest,
            "metrics": {metric.name: metric.value for metric in selected.metrics},
            "reward_means": {reward.name: reward.mean for reward in selected.reward_means},
            "acceptance": {
                "passed": acceptance.passed,
                "reasons": list(acceptance.reasons),
            },
        },
    }
    training.load_checkpoint(
        runner=runner,
        checkpoint_path=selected.checkpoint_path,
        device=device,
    )
    selected_policy = run_dir / "evaluated" / f"model_{selected.iteration}" / POLICY_FILENAME
    with tempfile.TemporaryDirectory(prefix=".training-publication-", dir=run_dir) as stage_name:
        stage_dir = Path(stage_name)
        shutil.copy2(selected_policy, stage_dir / POLICY_FILENAME)
        (stage_dir / METADATA_FILENAME).write_text(
            json.dumps(selected_metadata, indent=2, allow_nan=False) + "\n"
        )
        (stage_dir / _REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        for filename in (POLICY_FILENAME, METADATA_FILENAME, _REPORT_FILENAME):
            (stage_dir / filename).replace(run_dir / filename)
    criteria = config.rl.training.plateau
    return publish_policy_labels(
        policy_root=policy_root,
        run_dir=run_dir,
        comparison_fingerprint=comparison_fingerprint,
        metric_name=criteria.metric_name,
        metric_value=selected.metric(criteria.metric_name),
        direction=criteria.direction,
    )


def _run_new_training(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    config_path: Path,
    seed: int,
    policy_root: Path,
    repo_root: Path,
    device: str,
    controller_artifact: ControllerArtifact,
    controller_digest: str,
) -> None:
    training_config = config.rl.training
    phases = _capped_phases(
        phases=tuple(training_config.phases),
        maximum_iterations=args.max_iterations,
    )
    total_maximum = sum(phase.max_iterations for phase in phases)
    run_dir = _make_run_dir(out_root=policy_root, run_id=os.environ.get(_RUN_ID_ENV))
    _report(
        f"training {training_config.task_id} for at most {total_maximum} iterations "
        f"(policy={training_config.policy_name}, seed={seed}, device={device}, "
        f"start=scratch-neutral-output) -> {run_dir}"
    )
    env, env_cfg = _build_env(
        task_id=training_config.task_id,
        seed=seed,
        num_envs=args.num_envs,
        experiment_config=config,
        controller_settings=controller_artifact.selected_settings,
        use_training_reset_distribution=True,
    )
    worker: _EvaluationWorker | None = None
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(run_dir / "logs" / "evaluation"))
    try:
        metadata = _artifact_metadata(
            config=config,
            env_cfg=env_cfg,
            source_metadata=_git_source_metadata(repo_root=repo_root),
            artifact_run_id=run_dir.name,
            controller_artifact=controller_artifact,
            controller_digest=controller_digest,
        )
        if metadata["policy_interface"] != rl.PolicyInterface.PRESS_APPROACH.value:
            raise ValueError("merged training requires a registered deterministic policy evaluator")
        output_initialization = env_cfg.policy_config.actor.output_initialization
        metadata["initialization"] = {"method": output_initialization}
        metadata["training_phases"] = [asdict(phase) for phase in phases]
        metadata["plateau_criteria"] = asdict(training_config.plateau)
        _write_artifact_metadata(run_dir=run_dir, metadata=metadata)
        policy_metadata = rl.load_policy_metadata(artifact_dir=run_dir)
        comparison_fingerprint = _press_comparison_fingerprint(
            config=config,
            contract=policy_metadata.contract,
        )
        if comparison_fingerprint is None:
            raise ValueError("press training did not produce a comparison fingerprint")
        metadata["comparison_fingerprint"] = comparison_fingerprint
        _write_artifact_metadata(run_dir=run_dir, metadata=metadata)
        runner = training.build_runner(
            env=env,
            training_config=training_config,
            phase=phases[0],
            policy_config=env_cfg.policy_config,
            log_dir=str(run_dir / "logs"),
            device=device,
        )
        training.initialize_actor_output(runner=runner, method=output_initialization)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        initial_checkpoint = logs_dir / "model_0.pt"
        training.save_checkpoint(runner=runner, checkpoint_path=initial_checkpoint)
        cache_dir = policy_root / ".evaluation-cache-v2"
        evaluations: list[PolicyEvaluation] = []
        evaluation_records: list[dict[str, Any]] = []
        phase_results: list[dict[str, Any]] = []

        def evaluate_candidate(
            *,
            iteration: int,
            checkpoint_path: Path,
        ) -> PolicyEvaluation:
            nonlocal worker
            artifact_dir, policy_digest, policy_content_digest = _export_candidate(
                runner=runner,
                run_dir=run_dir,
                iteration=iteration,
                metadata=metadata,
            )
            candidate_cache = _cache_path(
                cache_dir=cache_dir,
                policy_content_digest=policy_content_digest,
                comparison_fingerprint=comparison_fingerprint,
                controller_digest=controller_digest,
            )
            cached = _read_cached_evaluation(
                cache_path=candidate_cache,
                iteration=iteration,
                checkpoint_path=checkpoint_path,
                policy_digest=policy_digest,
                policy_content_digest=policy_content_digest,
            )
            if cached is not None:
                return cached
            if worker is None:
                worker = _EvaluationWorker(
                    config_path=config_path,
                    run_dir=run_dir,
                    timeout_s=training_config.evaluation_worker_timeout_s,
                )
            fresh = worker.evaluate(
                iteration=iteration,
                checkpoint_path=checkpoint_path,
                artifact_dir=artifact_dir,
                policy_digest=policy_digest,
            )
            _write_cached_evaluation(
                cache_path=candidate_cache,
                evaluation=fresh,
                policy_content_digest=policy_content_digest,
                source_run_id=run_dir.name,
            )
            return fresh

        initial = evaluate_candidate(iteration=0, checkpoint_path=initial_checkpoint)
        evaluations.append(initial)
        global_iteration = 0
        randomize_episode_lengths = True
        for phase_index, phase in enumerate(phases):
            training.configure_phase(runner=runner, phase=phase)
            if hasattr(env, "set_training_hard_case_fraction"):
                env.set_training_hard_case_fraction(float(phase.hard_case_fraction))
            elif phase.hard_case_fraction != 0.0:
                raise ValueError("hard-case reset mixture is unsupported by the selected task")
            plateau_state = PlateauState(
                reference_value=evaluations[-1].metric(training_config.plateau.metric_name),
                stale_evaluations=0,
            )
            if phase_index == 0:
                evaluation_records.append(
                    _report_evaluation(
                        evaluation=initial,
                        phase_index=0,
                        phase_iteration=0,
                        made_progress=True,
                        plateau_state=plateau_state,
                        plateau_reached=False,
                        run_dir=run_dir,
                    )
                )
                for metric in initial.metrics:
                    if metric.value is not None and math.isfinite(metric.value):
                        writer.add_scalar(f"Evaluation/{metric.name}", metric.value, 0)
                writer.add_scalar("Evaluation/plateau_stale_evaluations", 0, 0)
                writer.flush()
                _write_json(
                    path=run_dir / _REPORT_FILENAME,
                    payload=_training_report(
                        run_dir=run_dir,
                        config=config,
                        phases=phases,
                        comparison_fingerprint=comparison_fingerprint,
                        evaluations=evaluation_records,
                        phase_results=phase_results,
                        selected=initial,
                        acceptance=None,
                    ),
                )
            _report(
                f"phase {phase_index + 1}/{len(phases)} "
                f"max_iterations={phase.max_iterations} "
                f"evaluation_interval={phase.evaluation_interval} "
                f"learning_rate={phase.learning_rate} entropy={phase.entropy_coef} "
                f"hard_case_fraction={phase.hard_case_fraction}"
            )
            phase_iteration = 0
            phase_reason = "maximum_iterations"
            while phase_iteration < phase.max_iterations:
                chunk = min(
                    phase.evaluation_interval,
                    phase.max_iterations - phase_iteration,
                )
                result = training.train_chunk(
                    runner=runner,
                    completed_iterations=global_iteration,
                    iterations=chunk,
                    randomize_initial_episode_lengths=randomize_episode_lengths,
                )
                randomize_episode_lengths = False
                global_iteration += result.iterations
                phase_iteration += result.iterations
                checkpoint_path = logs_dir / f"model_{global_iteration}.pt"
                if not checkpoint_path.is_file():
                    raise RuntimeError(f"training did not write {checkpoint_path}")
                evaluation = evaluate_candidate(
                    iteration=global_iteration,
                    checkpoint_path=checkpoint_path,
                )
                evaluations.append(evaluation)
                plateau = advance_plateau(
                    criteria=training_config.plateau,
                    state=plateau_state,
                    evaluation=evaluation,
                )
                plateau_state = plateau.state
                evaluation_records.append(
                    _report_evaluation(
                        evaluation=evaluation,
                        phase_index=phase_index,
                        phase_iteration=phase_iteration,
                        made_progress=plateau.made_progress,
                        plateau_state=plateau.state,
                        plateau_reached=plateau.plateau_reached,
                        run_dir=run_dir,
                    )
                )
                for metric in evaluation.metrics:
                    if metric.value is not None and math.isfinite(metric.value):
                        writer.add_scalar(
                            f"Evaluation/{metric.name}",
                            metric.value,
                            global_iteration,
                        )
                writer.add_scalar(
                    "Evaluation/plateau_stale_evaluations",
                    plateau.state.stale_evaluations,
                    global_iteration,
                )
                writer.flush()
                current_best = select_best_evaluation(
                    criteria=training_config.plateau,
                    evaluations=tuple(evaluations),
                )
                _write_json(
                    path=run_dir / _REPORT_FILENAME,
                    payload=_training_report(
                        run_dir=run_dir,
                        config=config,
                        phases=phases,
                        comparison_fingerprint=comparison_fingerprint,
                        evaluations=evaluation_records,
                        phase_results=phase_results,
                        selected=current_best,
                        acceptance=None,
                    ),
                )
                if plateau.plateau_reached:
                    phase_reason = "plateau"
                    break
            phase_results.append(
                {
                    "phase_index": phase_index,
                    "completed_iterations": phase_iteration,
                    "terminal_reason": phase_reason,
                }
            )
            current_best = select_best_evaluation(
                criteria=training_config.plateau,
                evaluations=tuple(evaluations),
            )
            _write_json(
                path=run_dir / _REPORT_FILENAME,
                payload=_training_report(
                    run_dir=run_dir,
                    config=config,
                    phases=phases,
                    comparison_fingerprint=comparison_fingerprint,
                    evaluations=evaluation_records,
                    phase_results=phase_results,
                    selected=current_best,
                    acceptance=None,
                ),
            )

        selected = select_best_evaluation(
            criteria=training_config.plateau,
            evaluations=tuple(evaluations),
        )
        # AppLauncher must start before the Isaac Lab workflow import.
        from robotics_experiments.workflows.isaac_lab import evaluate_press_acceptance

        acceptance = evaluate_press_acceptance(
            evaluation=selected,
            evaluation_config=config.rl.press.evaluation,
            minimum_range_m=config.rl.press.goal.range_min_m,
        )
        final_report = _training_report(
            run_dir=run_dir,
            config=config,
            phases=phases,
            comparison_fingerprint=comparison_fingerprint,
            evaluations=evaluation_records,
            phase_results=phase_results,
            selected=selected,
            acceptance=acceptance,
        )
        best_updated = _publish_selected(
            runner=runner,
            run_dir=run_dir,
            policy_root=policy_root,
            metadata=metadata,
            report=final_report,
            selected=selected,
            acceptance=acceptance,
            comparison_fingerprint=comparison_fingerprint,
            device=device,
            config=config,
        )
    finally:
        writer.close()
        if worker is not None:
            worker.close()
        env.close()

    _report(
        f"train-rl COMPLETE iterations={global_iteration} run={run_dir} "
        f"selected={selected.iteration} cost="
        f"{selected.metric(training_config.plateau.metric_name):.6f} "
        f"acceptance={str(acceptance.passed).lower()} "
        f"latest={policy_root / 'latest'} best_updated={str(best_updated).lower()}"
    )


def _run(*, args: argparse.Namespace, config: ExperimentConfig, config_path: Path) -> None:
    training_config = config.rl.training
    seed = args.seed if args.seed is not None else int(training_config.seed)
    _seed_everything(seed)
    repo_root = _repo_root()
    policy_root = (
        repo_root / ".build" / "training" / _validate_policy_name(training_config.policy_name)
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    controller_artifact, controller_digest = _load_training_controller(
        args=args,
        config=config,
        repo_root=repo_root,
    )
    _run_new_training(
        args=args,
        config=config,
        config_path=config_path,
        seed=seed,
        policy_root=policy_root,
        repo_root=repo_root,
        device=device,
        controller_artifact=controller_artifact,
        controller_digest=controller_digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="cap the configured phase sequence for procedural verification",
    )
    parser.add_argument("--seed", type=int, default=None, help="override rl.training.seed")
    parser.add_argument("--num-envs", type=int, default=None, help="override rl.num_envs")
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=None,
        help="override rl.training.controller_artifact for training",
    )
    args, _unknown = parser.parse_known_args()
    if args.max_iterations is not None and args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if args.num_envs is not None and args.num_envs < 1:
        raise SystemExit("--num-envs must be >= 1")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        config_path = Path(os.environ[_CONFIG_PATH_ENV])
        _run(args=args, config=load_config(config_path), config_path=config_path)
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

"""Train a configured RL policy and retain its complete checkpoint run.

The F2b task->policy entrypoint: composes the Hydra config, builds the
config-selected Gymnasium task (``rl.training.task_id`` -- the hover task
for F2b; F3 points it at the press task), runs the ``rl.training`` adapter,
and writes checkpoints plus the policy contract under
``.build/training/<policy-name>/<run-id>/``. The outer ``make train-rl``
workflow then evaluates the checkpoints and publishes the selected artifact.

Run inside the Isaac Lab container (make train-rl):
    /isaac-sim/python.sh /isaac-sim/proj/isaac/runtime/standalone/train_rl.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import robotics_experiments.rl as rl
import robotics_experiments.rl.training.rsl_rl as training
from robotics_experiments.alignment import ControllerArtifact, load_controller_artifact
from robotics_experiments.alignment._artifact import resolve_controller_path
from robotics_experiments.config import ExperimentConfig, load_config
from robotics_experiments.rl import METADATA_FILENAME
from robotics_experiments.workflows.isaac_lab import make_isaac_lab_env

_POLICY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CHECKPOINT_NAME_PATTERN = re.compile(r"model_(\d+)\.pt")
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"
_CONTROLLER_ARTIFACT_ENV = "CONTROLLER_ARTIFACT"


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_root() -> Path:
    # train_rl.py -> standalone -> runtime -> isaac -> repo root
    return Path(__file__).resolve().parents[3]


def _make_run_dir(out_root: Path, run_id: str | None = None) -> Path:
    if run_id is None:
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{os.getpid()}"
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "RL_RUN_ID must be a non-empty single path component using only "
            "letters, digits, '.', '_' or '-'"
        )
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _validate_policy_name(policy_name: str) -> str:
    if not isinstance(policy_name, str):
        raise ValueError(f"rl.training.policy_name must be a string, got {policy_name!r}")
    if _POLICY_NAME_PATTERN.fullmatch(policy_name) is None:
        raise ValueError(
            "rl.training.policy_name must be a non-empty single path component "
            "using only letters, digits, '_' or '-'"
        )
    return policy_name


def _policy_out_root(training_root: Path, policy_name: str) -> Path:
    return training_root / _validate_policy_name(policy_name)


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


def _checkpoint_run_dir(*, checkpoint_path: Path, out_root: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file() or checkpoint_path.parent.name != "logs":
        raise ValueError(
            "--export-checkpoint must name an existing "
            ".build/training/<policy>/<run>/logs checkpoint"
        )
    run_dir = checkpoint_path.parent.parent
    try:
        relative_run = run_dir.relative_to(out_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"checkpoint {checkpoint_path} is outside policy artifact root {out_root}"
        ) from exc
    if len(relative_run.parts) != 1:
        raise ValueError(f"checkpoint {checkpoint_path} is not inside one policy run directory")
    return run_dir


def _build_env(
    *,
    task_id: str,
    seed: int,
    num_envs: int | None,
    experiment_config: ExperimentConfig,
    controller_settings,
    policy_contract=None,
):
    return make_isaac_lab_env(
        task_id=task_id,
        experiment_config=experiment_config,
        controller_settings=controller_settings,
        seed=seed,
        num_envs=num_envs,
        policy_contract=policy_contract,
    )


def _artifact_metadata(
    *,
    config: ExperimentConfig,
    env_cfg,
    source_metadata: dict[str, str | bool],
    artifact_run_id: str,
    controller_artifact: ControllerArtifact,
    controller_digest: str,
):
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
            if str(env_cfg.policy_interface) == rl.PolicyInterface.PRESS_V1.value
            else None
        ),
    )
    return metadata | source_metadata | {"artifact_run_id": artifact_run_id}


def _write_artifact_metadata(*, run_dir: Path, metadata: dict) -> None:
    (run_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def _stored_artifact_metadata(*, run_dir: Path) -> tuple[dict, rl.PolicyMetadata]:
    metadata_path = run_dir / rl.METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(
            f"checkpoint run {run_dir} is missing its stored {rl.METADATA_FILENAME}; "
            "the original training contract cannot be recovered"
        )
    loaded = rl.load_policy_metadata(artifact_dir=run_dir)
    metadata = dict(loaded.values)
    if not isinstance(metadata.get("source_commit"), str) or not metadata["source_commit"]:
        raise ValueError(f"checkpoint run {run_dir} metadata has no source_commit provenance")
    if type(metadata.get("source_dirty")) is not bool:
        raise ValueError(f"checkpoint run {run_dir} metadata has no source_dirty provenance")
    return metadata, loaded


def _initialization_provenance(
    *,
    checkpoint_path: Path,
    training_root: Path,
    repo_root: Path,
) -> tuple[dict[str, str | int | None], object]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file() or checkpoint_path.parent.name != "logs":
        raise ValueError(
            "--initialize-checkpoint must name an existing "
            ".build/training/<source-policy>/<run>/logs checkpoint"
        )
    run_dir = checkpoint_path.parent.parent
    try:
        relative_run = run_dir.relative_to(training_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"initialization checkpoint {checkpoint_path} is outside {training_root}"
        ) from exc
    if len(relative_run.parts) != 2:
        raise ValueError(
            f"initialization checkpoint {checkpoint_path} is not inside one policy run"
        )
    source_policy, source_run = relative_run.parts
    _validate_policy_name(source_policy)
    match = _CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError("--initialize-checkpoint must name a numeric model_<iteration>.pt")
    source_metadata, source_policy_metadata = _stored_artifact_metadata(run_dir=run_dir)
    relative_checkpoint = checkpoint_path.resolve().relative_to(repo_root.resolve())
    return (
        {
            "method": "actor-only",
            "source_policy": source_policy,
            "source_stage": source_metadata.get("curriculum_stage"),
            "source_run": source_run,
            "checkpoint_iteration": int(match.group(1)),
            "checkpoint_path": str(relative_checkpoint),
            "source_commit": str(source_metadata["source_commit"]),
        },
        source_policy_metadata.contract,
    )


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
    if artifact.command_interface.kind != "body-flu-velocity-yaw-rate-v1":
        mismatches.append(f"interface {artifact.command_interface.kind!r}")
    if action.kind != "velocity" or action.action_dim != artifact.command_interface.dimension:
        mismatches.append(f"action contract {(action.kind, action.action_dim)!r} is incompatible")
    if mismatches:
        raise ValueError("controller artifact is incompatible: " + "; ".join(mismatches))
    controller_path = resolve_controller_path(artifact_path)
    digest = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    if not artifact.advisory_quality_passed:
        _report(
            "WARNING: consuming advisory-quality-failed controller "
            f"{artifact.alignment_id}/{artifact.run_id}: "
            + "; ".join(artifact.advisory_quality_reasons)
        )
    return artifact, digest


def _validate_initialization_contract(
    *,
    source_contract,
    env_cfg,
    action_dim: int,
) -> None:
    if source_contract.obs_dim != int(env_cfg.observation_space):
        raise ValueError(
            "initialization checkpoint observation dimension is incompatible: "
            f"{source_contract.obs_dim} != {int(env_cfg.observation_space)}"
        )
    if source_contract.action_dim != action_dim:
        raise ValueError(
            "initialization checkpoint action dimension is incompatible: "
            f"{source_contract.action_dim} != {action_dim}"
        )


def _validate_training_mode(
    *,
    training_mode: str,
    initialization_checkpoint: Path | None,
) -> None:
    if training_mode == "scratch":
        if initialization_checkpoint is not None:
            raise ValueError(
                "scratch training forbids --initialize-checkpoint; use the "
                "actor-only training mode"
            )
        return
    if training_mode == "actor-only":
        if initialization_checkpoint is None:
            raise ValueError("actor-only training requires --initialize-checkpoint")
        return
    raise ValueError(f"unsupported rl.training.training_mode={training_mode!r}")


def _run_new_training(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    seed: int,
    iterations: int,
    out_root: Path,
    training_root: Path,
    repo_root: Path,
    device: str,
    controller_artifact: ControllerArtifact,
    controller_digest: str,
) -> None:
    training_cfg = config.rl.training
    initialization_checkpoint = args.initialize_checkpoint
    _validate_training_mode(
        training_mode=training_cfg.training_mode,
        initialization_checkpoint=initialization_checkpoint,
    )
    initialization = None
    source_contract = None
    if initialization_checkpoint is not None:
        initialization, source_contract = _initialization_provenance(
            checkpoint_path=initialization_checkpoint,
            training_root=training_root,
            repo_root=repo_root,
        )
    run_dir = _make_run_dir(out_root, os.environ.get(_RUN_ID_ENV))
    initialization_report = (
        ", actor_init=scratch-neutral-output"
        if initialization is None
        else (
            f", actor_init={initialization['source_run']}/"
            f"model_{initialization['checkpoint_iteration']}.pt"
        )
    )
    _report(
        f"training {training_cfg.task_id} for {iterations} iterations "
        f"(policy={training_cfg.policy_name}, seed={seed}, device={device}"
        f"{initialization_report}) -> {run_dir}"
    )
    env, env_cfg = _build_env(
        task_id=training_cfg.task_id,
        seed=seed,
        num_envs=args.num_envs,
        experiment_config=config,
        controller_settings=controller_artifact.selected_settings,
    )
    try:
        if source_contract is not None:
            _validate_initialization_contract(
                source_contract=source_contract,
                env_cfg=env_cfg,
                action_dim=int(config.rl.action.action_dim),
            )
        metadata = _artifact_metadata(
            config=config,
            env_cfg=env_cfg,
            source_metadata=_git_source_metadata(repo_root=repo_root),
            artifact_run_id=run_dir.name,
            controller_artifact=controller_artifact,
            controller_digest=controller_digest,
        )
        output_initialization = env_cfg.policy_config.actor.output_initialization
        metadata["training_mode"] = training_cfg.training_mode
        metadata["initialization"] = (
            {"method": output_initialization} if initialization is None else initialization
        )
        _write_artifact_metadata(run_dir=run_dir, metadata=metadata)
        runner = training.build_runner(
            env=env,
            training_config=training_cfg,
            policy_config=env_cfg.policy_config,
            log_dir=str(run_dir / "logs"),
            device=device,
        )
        if initialization_checkpoint is None:
            training.initialize_actor_output(
                runner=runner,
                method=output_initialization,
            )
        else:
            training.initialize_actor_from_checkpoint(
                runner=runner,
                checkpoint_path=initialization_checkpoint,
                device=device,
            )
        result = training.train(runner, iterations)
    finally:
        env.close()

    _report(
        f"train-rl COMPLETE iterations={result.iterations} "
        f"run={run_dir} checkpoints=ready-for-selection"
    )


def _run_manual_export(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    seed: int,
    out_root: Path,
    device: str,
) -> None:
    training_cfg = config.rl.training
    checkpoint_path = args.export_checkpoint
    assert checkpoint_path is not None
    run_dir = _checkpoint_run_dir(checkpoint_path=checkpoint_path, out_root=out_root)
    metadata, policy_metadata = _stored_artifact_metadata(run_dir=run_dir)
    _report(
        f"exporting {training_cfg.task_id} checkpoint {checkpoint_path} "
        f"(policy={training_cfg.policy_name}, device={device}) -> {run_dir}"
    )
    env, _env_cfg = _build_env(
        task_id=training_cfg.task_id,
        seed=seed,
        num_envs=args.num_envs,
        experiment_config=config,
        controller_settings=policy_metadata.controller_settings,
        policy_contract=policy_metadata.contract,
    )
    try:
        runner = training.build_runner(
            env=env,
            training_config=training_cfg,
            policy_config=_env_cfg.policy_config,
            log_dir=str(run_dir / "logs"),
            device=device,
        )
        training.load_checkpoint(
            runner=runner,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        training.export_policy(runner, run_dir, metadata=metadata)
    finally:
        env.close()
    _report(
        f"train-rl EXPORT COMPLETE checkpoint={checkpoint_path} "
        f"artifact={run_dir} publication=manual-only"
    )


def _run(*, args: argparse.Namespace, config: ExperimentConfig) -> None:
    training_cfg = config.rl.training
    seed = args.seed if args.seed is not None else int(training_cfg.seed)
    iterations = (
        args.max_iterations if args.max_iterations is not None else int(training_cfg.max_iterations)
    )
    _seed_everything(seed)
    repo_root = _repo_root()
    training_root = repo_root / ".build" / "training"
    out_root = _policy_out_root(training_root, training_cfg.policy_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.export_checkpoint is not None:
        _run_manual_export(
            args=args,
            config=config,
            seed=seed,
            out_root=out_root,
            device=device,
        )
        return
    _validate_training_mode(
        training_mode=training_cfg.training_mode,
        initialization_checkpoint=args.initialize_checkpoint,
    )
    controller_artifact, controller_digest = _load_training_controller(
        args=args,
        config=config,
        repo_root=repo_root,
    )
    _run_new_training(
        args=args,
        config=config,
        seed=seed,
        iterations=iterations,
        out_root=out_root,
        training_root=training_root,
        repo_root=repo_root,
        device=device,
        controller_artifact=controller_artifact,
        controller_digest=controller_digest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # --scene / --strict are accepted for make-harness parity; training builds
    # the Lab task directly and has no strict mode.
    parser.add_argument("--scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="override rl.training.max_iterations (e.g. a smoke run's 5)",
    )
    parser.add_argument("--seed", type=int, default=None, help="override rl.training.seed")
    parser.add_argument("--num-envs", type=int, default=None, help="override rl.num_envs")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--export-checkpoint",
        type=Path,
        default=None,
        help="export an existing run checkpoint without additional training",
    )
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=None,
        help="override rl.training.controller_artifact for training",
    )
    checkpoint_group.add_argument(
        "--initialize-checkpoint",
        type=Path,
        default=None,
        help="start a fresh run initialized from only an existing checkpoint's actor weights",
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
        try:
            config_path = os.environ[_CONFIG_PATH_ENV]
        except KeyError as exc:
            raise RuntimeError(f"missing required environment variable {_CONFIG_PATH_ENV}") from exc
        _run(args=args, config=load_config(config_path))
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

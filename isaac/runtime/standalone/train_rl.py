"""Train a press-pipeline policy and export the validated artifact.

The F2b task->policy entrypoint: composes the Hydra config, builds the
config-selected Gymnasium task (``rl.training.task_id`` -- the hover task
for F2b; F3 points it at the press task), runs the ``rl.training`` adapter,
and writes the policy artifact (TorchScript + metadata sidecar) under
``.build/training/<policy-name>/<run-id>/``, updating the named
``.build/training/<policy-name>/latest`` symlink the deploy-side routine configs
reference.

Run inside the Isaac Lab container (make train-rl):
    /isaac-sim/python.sh /isaac-sim/proj/isaac/runtime/standalone/train_rl.py
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robotics_experiments.config import ExperimentConfig, load_config

_LATEST_LINK = "latest"
_POLICY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"


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


def _make_run_dir(out_root: Path) -> Path:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{os.getpid()}"
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


def _update_latest_symlink(out_root: Path, run_dir: Path) -> None:
    link = out_root / _LATEST_LINK
    temp = out_root / f".{_LATEST_LINK}.tmp"
    temp.unlink(missing_ok=True)
    temp.symlink_to(run_dir.name)
    temp.replace(link)


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
):
    from robotics_experiments.workflows.isaac_lab import make_isaac_lab_env

    return make_isaac_lab_env(
        task_id=task_id,
        experiment_config=experiment_config,
        seed=seed,
        num_envs=num_envs,
    )


def _artifact_metadata(
    *, config: ExperimentConfig, env_cfg, source_metadata: dict[str, str | bool]
):
    import robotics_experiments.rl.press as press

    metadata = press.policy_metadata(
        action_cfg=config.rl.action,
        obs_schema_version=int(env_cfg.observation_schema_version),
        obs_dim=int(env_cfg.observation_space),
        policy_rate_hz=config.rl.policy_rate_hz,
    )
    return metadata | source_metadata


def _run(*, args: argparse.Namespace, config: ExperimentConfig) -> None:
    import robotics_experiments.rl.training.rsl_rl as training

    training_cfg = config.rl.training
    seed = args.seed if args.seed is not None else int(training_cfg.seed)
    iterations = (
        args.max_iterations if args.max_iterations is not None else int(training_cfg.max_iterations)
    )
    _seed_everything(seed)

    out_root = _policy_out_root(_repo_root() / ".build" / "training", training_cfg.policy_name)
    checkpoint_path = args.export_checkpoint
    run_dir = (
        _checkpoint_run_dir(checkpoint_path=checkpoint_path, out_root=out_root)
        if checkpoint_path is not None
        else _make_run_dir(out_root)
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if checkpoint_path is None:
        _report(
            f"training {training_cfg.task_id} for {iterations} iterations "
            f"(policy={training_cfg.policy_name}, seed={seed}, device={device}) -> {run_dir}"
        )
    else:
        _report(
            f"exporting {training_cfg.task_id} checkpoint {checkpoint_path} "
            f"(policy={training_cfg.policy_name}, device={device}) -> {run_dir}"
        )

    env, env_cfg = _build_env(
        task_id=training_cfg.task_id,
        seed=seed,
        num_envs=args.num_envs,
        experiment_config=config,
    )
    try:
        runner = training.build_runner(
            env,
            training_cfg,
            log_dir=str(run_dir / "logs"),
            device=device,
        )
        if checkpoint_path is None:
            result = training.train(runner, iterations)
        else:
            training.load_checkpoint(
                runner=runner,
                checkpoint_path=checkpoint_path,
                device=device,
            )
        metadata = _artifact_metadata(
            config=config,
            env_cfg=env_cfg,
            source_metadata=_git_source_metadata(repo_root=_repo_root()),
        )
        training.export_policy(runner, run_dir, metadata=metadata)
    finally:
        env.close()

    _update_latest_symlink(out_root, run_dir)
    if checkpoint_path is None:
        _report(
            f"train-rl COMPLETE iterations={result.iterations} "
            f"artifact={run_dir} latest={out_root / _LATEST_LINK}"
        )
    else:
        _report(
            f"train-rl EXPORT COMPLETE checkpoint={checkpoint_path} "
            f"artifact={run_dir} latest={out_root / _LATEST_LINK}"
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
    parser.add_argument(
        "--export-checkpoint",
        type=Path,
        default=None,
        help="export an existing run checkpoint without additional training",
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

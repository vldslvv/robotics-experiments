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
import importlib
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from robotics_experiments.config import get_config

_LATEST_LINK = "latest"
_POLICY_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


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


def _update_latest_symlink(out_root: Path, run_dir: Path) -> None:
    link = out_root / _LATEST_LINK
    temp = out_root / f".{_LATEST_LINK}.tmp"
    temp.unlink(missing_ok=True)
    temp.symlink_to(run_dir.name)
    temp.replace(link)


def _build_env(task_id: str, seed: int, num_envs: int | None):
    # Isaac Lab task code; imported after AppLauncher.
    import gymnasium as gym

    from robotics_experiments.workflows.isaac_lab import register_isaac_lab_tasks

    if not register_isaac_lab_tasks():
        raise RuntimeError("Isaac Lab task registration failed (gymnasium/isaaclab missing)")
    cfg_entry = gym.spec(task_id).kwargs["env_cfg_entry_point"]
    module_name, class_name = cfg_entry.split(":")
    env_cfg = getattr(importlib.import_module(module_name), class_name)()
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    return gym.make(task_id, cfg=env_cfg).unwrapped, env_cfg


def _run(args: argparse.Namespace) -> None:
    import robotics_experiments.rl.training as training
    from robotics_experiments.rl.press.observation import HOVER_OBSERVATION_SCHEMA_VERSION
    from robotics_experiments.rl.press.policy import policy_metadata

    config = get_config()
    training_cfg = config.rl.training
    seed = args.seed if args.seed is not None else int(training_cfg.seed)
    iterations = (
        args.max_iterations if args.max_iterations is not None else int(training_cfg.max_iterations)
    )
    _seed_everything(seed)

    out_root = _policy_out_root(_repo_root() / ".build" / "training", training_cfg.policy_name)
    run_dir = _make_run_dir(out_root)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _report(
        f"training {training_cfg.task_id} for {iterations} iterations "
        f"(policy={training_cfg.policy_name}, seed={seed}, device={device}) -> {run_dir}"
    )

    env, env_cfg = _build_env(training_cfg.task_id, seed=seed, num_envs=args.num_envs)
    try:
        runner = training.build_runner(
            env, training_cfg, log_dir=str(run_dir / "logs"), device=device
        )
        result = training.train(runner, iterations)
        metadata = policy_metadata(
            config.rl.action,
            obs_schema_version=HOVER_OBSERVATION_SCHEMA_VERSION,
            obs_dim=int(env_cfg.observation_space),
        )
        training.export_policy(runner, run_dir, metadata=metadata)
    finally:
        env.close()

    _update_latest_symlink(out_root, run_dir)
    _report(
        f"train-rl COMPLETE iterations={result.iterations} "
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
    args, _unknown = parser.parse_known_args()
    if args.max_iterations is not None and args.max_iterations < 1:
        raise SystemExit("--max-iterations must be >= 1")
    if args.num_envs is not None and args.num_envs < 1:
        raise SystemExit("--num-envs must be >= 1")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        _run(args)
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

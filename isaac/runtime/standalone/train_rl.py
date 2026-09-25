"""Run a curriculum, or one isolated attempt worker inside Isaac Lab."""

import argparse
import json
import os
import signal
from pathlib import Path

import torch

from robotics_experiments.config import load_config, load_curriculum_config
from robotics_experiments.rl.workflow import _load_pinned_policy_inputs
from robotics_experiments.workflows.isaac_lab.curriculum import run_curriculum
from robotics_experiments.workflows.isaac_lab.training import LearnerCheckpoint, run_attempt

__all__ = ["main"]


def _interrupt(_signum, _frame, /) -> None:
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--procedural-smoke",
        action="store_true",
        help="run one update per phase without using reserved test seeds or publishing",
    )
    parser.add_argument("--attempt-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--monitoring-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--update-offset", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--controller-artifact", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--controller-artifact-digest", help=argparse.SUPPRESS)
    parser.add_argument("--flight-parameters-artifact", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--flight-parameters-artifact-digest", help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()
    for number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(number, _interrupt)
    config_path = Path(os.environ["ROBOTICS_EXPERIMENT_CONFIG_PATH"]).resolve(strict=True)
    repo_root = Path(__file__).resolve().parents[3]
    if not args.attempt_worker:
        result = run_curriculum(
            config=load_curriculum_config(config_path),
            config_path=config_path,
            repo_root=repo_root,
            run_id=os.environ.get("RL_RUN_ID"),
            procedural_smoke=args.procedural_smoke,
        )
        if not result.complete and not args.procedural_smoke:
            raise RuntimeError(f"curriculum stopped; see {result.report_path}")
        return
    if any(
        value is None
        for value in (
            args.attempt_index,
            args.monitoring_dir,
            args.update_offset,
            args.controller_artifact,
            args.controller_artifact_digest,
            args.flight_parameters_artifact,
            args.flight_parameters_artifact_digest,
        )
    ):
        parser.error(
            "attempt workers require their index, monitoring directory, update offset "
            "and pinned evidence paths/digests"
        )
    config = load_config(config_path)
    if (
        not 0 <= args.attempt_index < len(config.rl.training.attempt_seeds)
        or args.update_offset < 0
    ):
        parser.error("invalid attempt index or update offset")
    inputs = _load_pinned_policy_inputs(
        controller_path=args.controller_artifact,
        controller_digest=args.controller_artifact_digest,
        flight_parameters_path=args.flight_parameters_artifact,
        flight_parameters_digest=args.flight_parameters_artifact_digest,
    )
    encoded = os.environ.get("RL_INITIAL_LEARNER")
    learner = LearnerCheckpoint(**json.loads(encoded)) if encoded else None
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    try:
        run_attempt(
            config=config,
            config_path=config_path,
            inputs=inputs,
            initial_learner=learner,
            run_dir=Path(os.environ["RL_CAMPAIGN_RUN_DIR"]),
            monitoring_dir=args.monitoring_dir,
            update_offset=args.update_offset,
            attempt_index=args.attempt_index,
            procedural_smoke=args.procedural_smoke,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()

"""Prepare every named asset and every selected experiment's switch model."""

import argparse
import fcntl
from pathlib import Path

from isaac.runtime.standalone.build_copter_usd_standalone import build_all_airframe_usds
from isaac.runtime.standalone.build_switch_usd_standalone import build_switch_usd
from robotics_experiments.config import load_config, load_curriculum_config, load_switch_model

__all__ = ["build_assets"]


def build_assets(*, config_path: Path, repo_root: Path) -> tuple[Path, ...]:
    """Serialize generation; a failure propagates before a dependent launch."""
    asset_root = repo_root / ".build/assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    with (asset_root / "generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            experiments = [load_config(config_path)]
        except ValueError as exc:
            if str(exc) != "expected an experiment configuration using base_experiment":
                raise
            curriculum = load_curriculum_config(config_path)
            experiments = [
                load_config(repo_root / stage.config_path) for stage in curriculum.stages
            ]
        paths = [Path(path) for path in build_all_airframe_usds()]
        models = [
            load_switch_model(path)
            for path in sorted((repo_root / "configs/experiment/switches/model").glob("*.yaml"))
        ]
        models.extend(experiment.switches.model for experiment in experiments)
        paths.extend(build_switch_usd(model=model, repo_root=repo_root) for model in models)
        result = tuple(dict.fromkeys(paths))
        for path in result:
            print(f"Asset ready: {path}", flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    build_assets(config_path=args.config, repo_root=Path(__file__).resolve().parents[3])

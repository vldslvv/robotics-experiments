"""Host-safe policy-artifact preflight for configured routines."""

from __future__ import annotations

import argparse
from pathlib import Path

from robotics_experiments.config import load_config
from robotics_experiments.runtime._policy import (
    _preflight_policy_artifacts,
)

__all__: list[str] = []


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate configured policy artifacts")
    parser.add_argument("config", type=Path, help="Hydra experiment YAML")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    preflight = _preflight_policy_artifacts(
        config=load_config(args.config),
        repo_root=repo_root,
    )
    if not preflight.policies:
        print("Policy artifact preflight: no policy steps configured")
        return
    for policy in preflight.policies:
        values = policy.metadata.values
        print(
            "Policy artifact preflight: "
            f"vehicle={policy.vehicle_label} artifact={policy.artifact} "
            f"interface={policy.metadata.contract.policy_interface.value} "
            f"source_commit={values.get('source_commit', 'unknown')} "
            f"source_dirty={values.get('source_dirty', 'unknown')}"
        )


if __name__ == "__main__":
    _main()

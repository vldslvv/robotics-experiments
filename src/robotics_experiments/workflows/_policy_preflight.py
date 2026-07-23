"""Host-safe policy-artifact preflight for configured routines."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robotics_experiments.config import ExperimentConfig, load_config
from robotics_experiments.rl.press import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    HOVER_POLICY_INTERFACE,
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    PRESS_POLICY_INTERFACE,
    load_policy_metadata,
    validate_policy_artifact,
)

__all__: list[str] = []


def _artifact_path(*, descriptor: Mapping[str, Any], repo_root: Path) -> Path:
    artifact = Path(str(descriptor["artifact"]))
    return artifact if artifact.is_absolute() else repo_root / artifact


def _interface_contract(*, interface: str) -> tuple[int, int]:
    if interface == HOVER_POLICY_INTERFACE:
        return HOVER_OBSERVATION_SCHEMA_VERSION, HOVER_OBSERVATION_DIM
    if interface == PRESS_POLICY_INTERFACE:
        return OBSERVATION_SCHEMA_VERSION, PRESS_OBSERVATION_DIM
    raise ValueError(f"unsupported policy_interface={interface!r}")


def _preflight_policy_artifacts(
    *,
    config: ExperimentConfig,
    repo_root: Path,
) -> tuple[tuple[str, Path, dict[str, Any]], ...]:
    """Validate every policy descriptor without importing Torch or Isaac."""
    validated = []
    for label, vehicle in config.vehicles.items():
        for descriptor in vehicle.routine.steps:
            if descriptor.get("kind") != "policy":
                continue
            artifact = _artifact_path(descriptor=descriptor, repo_root=repo_root)
            interface = str(load_policy_metadata(artifact_dir=artifact)["policy_interface"])
            obs_schema_version, obs_dim = _interface_contract(interface=interface)
            metadata = validate_policy_artifact(
                artifact_dir=artifact,
                obs_schema_version=obs_schema_version,
                obs_dim=obs_dim,
                policy_interface=interface,
            )
            if metadata.policy_rate_hz > config.driver.control_rate_hz:
                raise ValueError(
                    f"policy artifact {artifact} cadence {metadata.policy_rate_hz} Hz "
                    f"exceeds driver.control_rate_hz={config.driver.control_rate_hz}"
                )
            validated.append((label, artifact, load_policy_metadata(artifact_dir=artifact)))
    press_rates = {
        float(metadata["policy_rate_hz"])
        for _label, _artifact, metadata in validated
        if metadata["policy_interface"] == PRESS_POLICY_INTERFACE
    }
    if len(press_rates) > 1:
        raise ValueError(
            "press-v1 policy artifacts in one fleet must share one policy_rate_hz; "
            f"found {sorted(press_rates)}"
        )
    return tuple(validated)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate configured policy artifacts")
    parser.add_argument("config", type=Path, help="Hydra experiment YAML")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    validated = _preflight_policy_artifacts(
        config=load_config(args.config),
        repo_root=repo_root,
    )
    if not validated:
        print("Policy artifact preflight: no policy steps configured")
        return
    for label, artifact, metadata in validated:
        print(
            "Policy artifact preflight: "
            f"vehicle={label} artifact={artifact} "
            f"interface={metadata['policy_interface']} "
            f"source_commit={metadata.get('source_commit', 'unknown')} "
            f"source_dirty={metadata.get('source_dirty', 'unknown')}"
        )


if __name__ == "__main__":
    _main()

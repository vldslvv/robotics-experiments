"""Policy artifact contracts, validation, loading, and hover composition.

A policy artifact owns the contract used to interpret its normalized output
after export. Hydra authors that contract for a new training run, but
evaluation and deployment parse the stored metadata instead of comparing it
with the active RL training config.

Torch is imported only by :func:`load_vector_policy`, keeping metadata
preflight on the lightweight core import path.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from robotics_experiments.rl.press.deploy import press_action_to_offboard
from robotics_experiments.rl.press.observation import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    hover_observation_from_state,
)

if TYPE_CHECKING:
    from robotics_experiments.commanding import AttitudeRateSetpoint, VelocitySetpoint
    from robotics_experiments.px4_bridge import VehicleState

__all__ = [
    "HOVER_POLICY_INTERFACE",
    "LoadedVectorPolicy",
    "METADATA_FILENAME",
    "POLICY_FILENAME",
    "PRESS_POLICY_INTERFACE",
    "PolicyContract",
    "build_policy_step_fn",
    "load_policy_contract",
    "load_policy_metadata",
    "load_vector_policy",
    "policy_metadata",
    "validate_policy_artifact",
]

POLICY_FILENAME = "policy.pt"
METADATA_FILENAME = "policy_meta.json"
HOVER_POLICY_INTERFACE = "hover-v1"
PRESS_POLICY_INTERFACE = "press-v1"

_SCALE_FIELDS = (
    "max_body_velocity_mps",
    "max_yaw_rate_radps",
    "max_body_rate_radps",
    "hover_thrust_fraction",
)
_INTERFACE_OBSERVATIONS = {
    HOVER_POLICY_INTERFACE: (
        HOVER_OBSERVATION_SCHEMA_VERSION,
        HOVER_OBSERVATION_DIM,
    ),
    PRESS_POLICY_INTERFACE: (
        OBSERVATION_SCHEMA_VERSION,
        PRESS_OBSERVATION_DIM,
    ),
}
_ACTION_DIMENSIONS = {"velocity": 4, "ctbr": 4}


@dataclass(frozen=True)
class PolicyContract:
    """Artifact-owned interpretation of observations and normalized actions."""

    policy_interface: str
    obs_schema_version: int
    obs_dim: int
    policy_rate_hz: float
    action_kind: str
    action_dim: int
    max_body_velocity_mps: float
    max_yaw_rate_radps: float
    max_body_rate_radps: float
    hover_thrust_fraction: float


@dataclass(frozen=True)
class LoadedVectorPolicy:
    """Validated artifact contract paired with its CPU inference callable."""

    contract: PolicyContract
    inference: Callable[[np.ndarray], np.ndarray]


def _metadata_value(
    *,
    metadata: Mapping[str, Any],
    name: str,
    artifact_dir: Path,
) -> Any:
    if name not in metadata:
        raise ValueError(f"policy artifact {artifact_dir} metadata is missing {name}")
    return metadata[name]


def _integer(
    *,
    metadata: Mapping[str, Any],
    name: str,
    artifact_dir: Path,
) -> int:
    value = _metadata_value(metadata=metadata, name=name, artifact_dir=artifact_dir)
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"policy artifact {artifact_dir} metadata {name} must be a positive integer, "
            f"got {value!r}"
        )
    return value


def _finite_positive(*, value: Any, name: str, artifact_dir: Path) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            f"policy artifact {artifact_dir} metadata {name} must be finite and positive, "
            f"got {value!r}"
        )
    return float(value)


def _infer_legacy_interface(
    *,
    metadata: dict[str, Any],
    artifact_dir: Path,
) -> None:
    if "policy_interface" in metadata:
        return
    scales = metadata.get("action_scales")
    complete = (
        all(name in metadata for name in ("policy_rate_hz", "action_kind", "action_dim"))
        and isinstance(scales, dict)
        and all(name in scales for name in _SCALE_FIELDS)
    )
    if (
        complete
        and metadata.get("obs_schema_version") == HOVER_OBSERVATION_SCHEMA_VERSION
        and metadata.get("obs_dim") == HOVER_OBSERVATION_DIM
    ):
        metadata["policy_interface"] = HOVER_POLICY_INTERFACE
        return
    if (
        complete
        and metadata.get("obs_schema_version") == OBSERVATION_SCHEMA_VERSION
        and metadata.get("obs_dim") == PRESS_OBSERVATION_DIM
        and metadata.get("policy_rate_hz") == 20.0
        and metadata.get("action_kind") == "velocity"
        and metadata.get("action_dim") == 4
    ):
        metadata["policy_interface"] = PRESS_POLICY_INTERFACE
        warnings.warn(
            f"policy artifact {artifact_dir} uses legacy press metadata without "
            "policy_interface; inferred press-v1 from its complete schema tuple. "
            "Re-export future artifacts with an explicit policy_interface.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    raise ValueError(
        f"policy artifact {artifact_dir} metadata has no policy_interface; "
        "re-export it from its checkpoint"
    )


def policy_metadata(
    *,
    action_cfg: Any,
    obs_schema_version: int,
    obs_dim: int,
    policy_rate_hz: float,
    policy_interface: str,
) -> dict[str, Any]:
    """Compose metadata for a contract authored by a training config."""
    return {
        "policy_interface": str(policy_interface),
        "obs_schema_version": int(obs_schema_version),
        "obs_dim": int(obs_dim),
        "policy_rate_hz": float(policy_rate_hz),
        "action_kind": str(action_cfg.kind),
        "action_dim": int(action_cfg.action_dim),
        "action_scales": {name: float(getattr(action_cfg, name)) for name in _SCALE_FIELDS},
    }


def load_policy_metadata(*, artifact_dir: Union[str, Path]) -> dict[str, Any]:
    """Read policy metadata and apply the narrow schema-v1 legacy inference."""
    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(f"policy artifact {artifact_dir} is missing {METADATA_FILENAME}")
    try:
        loaded = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"policy artifact {artifact_dir} has malformed {METADATA_FILENAME}: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError(
            f"policy artifact {artifact_dir} metadata must be a JSON object, "
            f"got {type(loaded).__name__}"
        )
    metadata = dict(loaded)
    _infer_legacy_interface(metadata=metadata, artifact_dir=artifact_dir)
    return metadata


def load_policy_contract(*, artifact_dir: Union[str, Path]) -> PolicyContract:
    """Parse and validate the artifact-owned policy contract."""
    artifact_dir = Path(artifact_dir)
    metadata = load_policy_metadata(artifact_dir=artifact_dir)
    interface = _metadata_value(
        metadata=metadata,
        name="policy_interface",
        artifact_dir=artifact_dir,
    )
    if not isinstance(interface, str) or interface not in _INTERFACE_OBSERVATIONS:
        raise ValueError(
            f"policy artifact {artifact_dir} has unsupported policy_interface={interface!r}"
        )
    obs_schema_version = _integer(
        metadata=metadata,
        name="obs_schema_version",
        artifact_dir=artifact_dir,
    )
    obs_dim = _integer(metadata=metadata, name="obs_dim", artifact_dir=artifact_dir)
    expected_observation = _INTERFACE_OBSERVATIONS[interface]
    if (obs_schema_version, obs_dim) != expected_observation:
        raise ValueError(
            f"policy artifact {artifact_dir} has unsupported {interface} observation "
            f"contract {(obs_schema_version, obs_dim)!r}; expected {expected_observation!r}"
        )
    policy_rate_hz = _finite_positive(
        value=_metadata_value(
            metadata=metadata,
            name="policy_rate_hz",
            artifact_dir=artifact_dir,
        ),
        name="policy_rate_hz",
        artifact_dir=artifact_dir,
    )
    action_kind = _metadata_value(
        metadata=metadata,
        name="action_kind",
        artifact_dir=artifact_dir,
    )
    if not isinstance(action_kind, str) or action_kind not in _ACTION_DIMENSIONS:
        raise ValueError(
            f"policy artifact {artifact_dir} has unsupported action_kind={action_kind!r}"
        )
    action_dim = _integer(metadata=metadata, name="action_dim", artifact_dir=artifact_dir)
    if action_dim != _ACTION_DIMENSIONS[action_kind]:
        raise ValueError(
            f"policy artifact {artifact_dir} has unsupported action_dim={action_dim} "
            f"for action_kind={action_kind!r}; expected {_ACTION_DIMENSIONS[action_kind]}"
        )
    scales = _metadata_value(
        metadata=metadata,
        name="action_scales",
        artifact_dir=artifact_dir,
    )
    if not isinstance(scales, dict):
        raise ValueError(f"policy artifact {artifact_dir} metadata action_scales must be an object")
    parsed_scales = {
        name: _finite_positive(
            value=_metadata_value(
                metadata=scales,
                name=name,
                artifact_dir=artifact_dir,
            ),
            name=f"action_scales.{name}",
            artifact_dir=artifact_dir,
        )
        for name in _SCALE_FIELDS
    }
    if parsed_scales["hover_thrust_fraction"] > 1.0:
        raise ValueError(
            f"policy artifact {artifact_dir} metadata "
            "action_scales.hover_thrust_fraction must not exceed 1"
        )
    return PolicyContract(
        policy_interface=str(interface),
        obs_schema_version=obs_schema_version,
        obs_dim=obs_dim,
        policy_rate_hz=policy_rate_hz,
        action_kind=str(action_kind),
        action_dim=action_dim,
        **parsed_scales,
    )


def _validate_runtime_observation(
    *,
    contract: PolicyContract,
    artifact_dir: Path,
    obs_schema_version: int,
    obs_dim: int,
    policy_interface: str,
) -> None:
    runtime = (policy_interface, obs_schema_version, obs_dim)
    artifact = (
        contract.policy_interface,
        contract.obs_schema_version,
        contract.obs_dim,
    )
    if runtime != artifact:
        raise ValueError(
            f"policy artifact {artifact_dir} contract {artifact!r} is incompatible "
            f"with runtime observation {runtime!r}"
        )


def validate_policy_artifact(
    *,
    artifact_dir: Union[str, Path],
    obs_schema_version: int,
    obs_dim: int,
    policy_interface: str,
) -> PolicyContract:
    """Validate an artifact and the runtime observation contract without Torch."""
    artifact_dir = Path(artifact_dir)
    module_path = artifact_dir / POLICY_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not module_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"policy artifact {artifact_dir} is missing {POLICY_FILENAME} or "
            f"{METADATA_FILENAME}; train and select one first "
            "(make train-rl, then make select-rl-checkpoint)"
        )
    contract = load_policy_contract(artifact_dir=artifact_dir)
    _validate_runtime_observation(
        contract=contract,
        artifact_dir=artifact_dir,
        obs_schema_version=obs_schema_version,
        obs_dim=obs_dim,
        policy_interface=policy_interface,
    )
    return contract


def load_vector_policy(
    *,
    artifact_dir: Union[str, Path],
    obs_schema_version: int,
    obs_dim: int,
    policy_interface: str,
) -> LoadedVectorPolicy:
    """Load a vector policy after validating artifact/runtime compatibility."""
    artifact_dir = Path(artifact_dir)
    module_path = artifact_dir / POLICY_FILENAME
    contract = validate_policy_artifact(
        artifact_dir=artifact_dir,
        obs_schema_version=obs_schema_version,
        obs_dim=obs_dim,
        policy_interface=policy_interface,
    )

    import torch

    module = torch.jit.load(str(module_path), map_location="cpu")
    module.eval()

    def run(observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        expected_tail = (contract.obs_dim,)
        if obs.ndim not in (1, 2) or obs.shape[-1:] != expected_tail:
            raise ValueError(
                f"policy artifact {artifact_dir} received observation shape "
                f"{obs.shape}, expected ({contract.obs_dim},) or "
                f"(batch, {contract.obs_dim})"
            )
        if single:
            obs = obs.reshape(1, contract.obs_dim)
        with torch.no_grad():
            action = module(torch.from_numpy(obs))
        action_np = action.detach().cpu().numpy().astype(np.float64)
        if action_np.shape != (obs.shape[0], contract.action_dim):
            raise ValueError(
                f"policy artifact {artifact_dir} produced action shape "
                f"{action_np.shape}, expected ({obs.shape[0]}, {contract.action_dim})"
            )
        return action_np[0] if single else action_np

    return LoadedVectorPolicy(contract=contract, inference=run)


def build_policy_step_fn(
    *,
    artifact_dir: Union[str, Path],
) -> Callable[["VehicleState", np.ndarray], "Union[VelocitySetpoint, AttitudeRateSetpoint]"]:
    """Compose the hover policy's state-to-OFFBOARD callable."""
    loaded = load_vector_policy(
        artifact_dir=artifact_dir,
        obs_schema_version=HOVER_OBSERVATION_SCHEMA_VERSION,
        obs_dim=HOVER_OBSERVATION_DIM,
        policy_interface=HOVER_POLICY_INTERFACE,
    )

    def step(state: "VehicleState", target_enu: np.ndarray):
        observation = hover_observation_from_state(state, target_enu)
        action = np.clip(loaded.inference(observation), -1.0, 1.0)
        return press_action_to_offboard(action, loaded.contract)

    return step

"""Policy artifact contracts, validation, and loading.

A policy artifact owns the contract used to interpret its normalized output
after export. Hydra authors that contract for a new training run, but
evaluation and deployment parse the stored metadata instead of comparing it
with the active RL training config.

Torch is an ordinary runtime dependency and is imported eagerly.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from robotics_experiments.config import PressActionConfig
from robotics_experiments.rl._interfaces import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    INTERFACE_OBSERVATIONS,
    PRESS_OBSERVATION_DIM,
    PRESS_OBSERVATION_SCHEMA_VERSION,
    PolicyInterface,
)

__all__ = [
    "LoadedPolicy",
    "METADATA_FILENAME",
    "POLICY_FILENAME",
    "PolicyContract",
    "PolicyInterface",
    "PolicyMetadata",
    "load_policy",
    "load_policy_metadata",
    "policy_metadata",
    "validate_policy_artifact",
]

POLICY_FILENAME = "policy.pt"
METADATA_FILENAME = "policy_meta.json"


_SCALE_FIELDS = (
    "max_body_velocity_mps",
    "max_yaw_rate_radps",
    "max_body_rate_radps",
    "hover_thrust_fraction",
)
_ACTION_DIMENSIONS = {"velocity": 4, "ctbr": 4}


@dataclass(frozen=True)
class PolicyContract:
    """Artifact-owned interpretation of observations and normalized actions."""

    policy_interface: PolicyInterface
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
class PolicyMetadata:
    """Validated runtime contract paired with the complete stored JSON record."""

    contract: PolicyContract
    values: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedPolicy:
    """Validated artifact contract paired with its CPU inference callable."""

    metadata: PolicyMetadata
    inference: Callable[[np.ndarray], np.ndarray]

    @property
    def contract(self) -> PolicyContract:
        """Return the already-validated artifact contract."""
        return self.metadata.contract


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
        metadata["policy_interface"] = PolicyInterface.HOVER_V1.value
        return
    if (
        complete
        and metadata.get("obs_schema_version") == PRESS_OBSERVATION_SCHEMA_VERSION
        and metadata.get("obs_dim") == PRESS_OBSERVATION_DIM
        and metadata.get("policy_rate_hz") == 20.0
        and metadata.get("action_kind") == "velocity"
        and metadata.get("action_dim") == 4
    ):
        metadata["policy_interface"] = PolicyInterface.PRESS_V1.value
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
    action_cfg: PressActionConfig,
    obs_schema_version: int,
    obs_dim: int,
    policy_rate_hz: float,
    policy_interface: str | PolicyInterface,
) -> dict[str, Any]:
    """Compose metadata for a contract authored by a training config."""
    interface = PolicyInterface(policy_interface)
    return {
        "policy_interface": interface.value,
        "obs_schema_version": int(obs_schema_version),
        "obs_dim": int(obs_dim),
        "policy_rate_hz": float(policy_rate_hz),
        "action_kind": str(action_cfg.kind),
        "action_dim": int(action_cfg.action_dim),
        "action_scales": {name: float(getattr(action_cfg, name)) for name in _SCALE_FIELDS},
    }


def _read_metadata(*, artifact_dir: Path) -> dict[str, Any]:
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


def _parse_metadata(*, artifact_dir: Path, metadata: dict[str, Any]) -> PolicyMetadata:
    interface_value = _metadata_value(
        metadata=metadata,
        name="policy_interface",
        artifact_dir=artifact_dir,
    )
    try:
        interface = PolicyInterface(interface_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"policy artifact {artifact_dir} has unsupported "
            f"policy_interface={interface_value!r}"
        ) from exc
    obs_schema_version = _integer(
        metadata=metadata,
        name="obs_schema_version",
        artifact_dir=artifact_dir,
    )
    obs_dim = _integer(metadata=metadata, name="obs_dim", artifact_dir=artifact_dir)
    expected_observation = INTERFACE_OBSERVATIONS[interface]
    if (obs_schema_version, obs_dim) != expected_observation:
        raise ValueError(
            f"policy artifact {artifact_dir} has unsupported {interface.value} observation "
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
    contract = PolicyContract(
        policy_interface=interface,
        obs_schema_version=obs_schema_version,
        obs_dim=obs_dim,
        policy_rate_hz=policy_rate_hz,
        action_kind=str(action_kind),
        action_dim=action_dim,
        **parsed_scales,
    )
    return PolicyMetadata(
        contract=contract,
        values=MappingProxyType(metadata),
    )


def _load_metadata(*, artifact_dir: Path) -> PolicyMetadata:
    return _parse_metadata(
        artifact_dir=artifact_dir,
        metadata=_read_metadata(artifact_dir=artifact_dir),
    )


def load_policy_metadata(*, artifact_dir: str | Path) -> PolicyMetadata:
    """Read and validate an artifact's complete metadata record once."""
    return _load_metadata(artifact_dir=Path(artifact_dir))


def validate_policy_artifact(
    *,
    artifact_dir: str | Path,
) -> PolicyMetadata:
    """Validate a complete artifact without importing Torch."""
    artifact_dir = Path(artifact_dir)
    module_path = artifact_dir / POLICY_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not module_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"policy artifact {artifact_dir} is missing {POLICY_FILENAME} or "
            f"{METADATA_FILENAME}; train and select one first "
            "(make train-rl, then make select-rl-checkpoint)"
        )
    return _load_metadata(artifact_dir=artifact_dir)


def load_policy(
    *,
    artifact_dir: str | Path,
) -> LoadedPolicy:
    """Load a vector policy after reading and validating its artifact once."""
    artifact_dir = Path(artifact_dir)
    module_path = artifact_dir / POLICY_FILENAME
    if not module_path.is_file():
        raise ValueError(
            f"policy artifact {artifact_dir} is missing {POLICY_FILENAME} or "
            f"{METADATA_FILENAME}; train and select one first "
            "(make train-rl, then make select-rl-checkpoint)"
        )
    metadata = _load_metadata(artifact_dir=artifact_dir)
    contract = metadata.contract

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

    return LoadedPolicy(metadata=metadata, inference=run)

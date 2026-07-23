"""Validated policy-artifact loading + the deploy-time policy step composer.

The policy artifact is a directory holding a TorchScript module
(``policy.pt``) and a JSON metadata sidecar (``policy_meta.json``) recording
the contract the policy was trained against: the observation schema version
and dimension, policy rate, action level (``kind``/``action_dim``), and action
command scales. The vector loader refuses any artifact whose metadata mismatches
the active config -- fail loudly at load, never a shape error deep in the
driver. A bare checkpoint without metadata is not a valid artifact.

Torch is imported only inside functions (like ``MavlinkCommandPort`` imports
pymavlink inside methods), so this module stays on the core import path and
the runtime never depends on the training stack.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Union

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
    "METADATA_FILENAME",
    "POLICY_FILENAME",
    "PRESS_POLICY_INTERFACE",
    "build_policy_step_fn",
    "load_policy_metadata",
    "load_vector_policy",
    "policy_metadata",
    "validate_policy_artifact",
]

POLICY_FILENAME = "policy.pt"
METADATA_FILENAME = "policy_meta.json"
HOVER_POLICY_INTERFACE = "hover-v1"
PRESS_POLICY_INTERFACE = "press-v1"

# Action command scales recorded in (and validated against) the metadata; the
# full set is recorded regardless of kind so a scale edit invalidates old
# artifacts loudly instead of silently rescaling their actions.
_SCALE_FIELDS = (
    "max_body_velocity_mps",
    "max_yaw_rate_radps",
    "max_body_rate_radps",
    "hover_thrust_fraction",
)


def policy_metadata(
    *,
    action_cfg: Any,
    obs_schema_version: int,
    obs_dim: int,
    policy_rate_hz: float,
    policy_interface: str,
) -> dict[str, Any]:
    """Compose the artifact metadata for a policy trained on this config."""
    return {
        "policy_interface": str(policy_interface),
        "obs_schema_version": int(obs_schema_version),
        "obs_dim": int(obs_dim),
        "policy_rate_hz": float(policy_rate_hz),
        "action_kind": str(action_cfg.kind),
        "action_dim": int(action_cfg.action_dim),
        "action_scales": {name: float(getattr(action_cfg, name)) for name in _SCALE_FIELDS},
    }


def _validate_metadata(found: dict[str, Any], expected: dict[str, Any], artifact_dir: Path) -> None:
    for key, expected_value in expected.items():
        if key == "action_scales":
            continue
        if found.get(key) != expected_value:
            raise ValueError(
                f"policy artifact {artifact_dir} metadata mismatch: "
                f"{key} is {found.get(key)!r}, active config expects {expected_value!r}"
            )
    found_scales = found.get("action_scales")
    if not isinstance(found_scales, dict):
        raise ValueError(f"policy artifact {artifact_dir} metadata has no action_scales")
    for name, expected_value in expected["action_scales"].items():
        if name not in found_scales or not np.isclose(found_scales[name], expected_value):
            raise ValueError(
                f"policy artifact {artifact_dir} metadata mismatch: "
                f"action_scales.{name} is {found_scales.get(name)!r}, "
                f"active config expects {expected_value!r}"
            )


def validate_policy_artifact(
    *,
    artifact_dir: Union[str, Path],
    action_cfg: Any,
    obs_schema_version: int,
    obs_dim: int,
    policy_rate_hz: float,
    policy_interface: str,
) -> dict[str, Any]:
    """Validate an artifact contract without importing or executing Torch.

    This host-safe boundary is used before launching simulator containers and
    again by the runtime loader. The TorchScript module itself is loaded only
    by :func:`load_vector_policy` inside the environment that executes it.
    """
    artifact_dir = Path(artifact_dir)
    module_path = artifact_dir / POLICY_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not module_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"policy artifact {artifact_dir} is missing {POLICY_FILENAME} or "
            f"{METADATA_FILENAME}; train one first (make train-rl)"
        )
    metadata = load_policy_metadata(artifact_dir=artifact_dir)
    expected = policy_metadata(
        action_cfg=action_cfg,
        obs_schema_version=obs_schema_version,
        obs_dim=obs_dim,
        policy_rate_hz=policy_rate_hz,
        policy_interface=policy_interface,
    )
    _validate_metadata(metadata, expected, artifact_dir)
    return metadata


def load_vector_policy(
    *,
    artifact_dir: Union[str, Path],
    action_cfg: Any,
    obs_schema_version: int,
    obs_dim: int,
    policy_rate_hz: float,
    policy_interface: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """Load a policy artifact, validating its metadata against the active config.

    Args:
        artifact_dir: Directory holding ``policy.pt`` + ``policy_meta.json``.
        action_cfg: The active ``rl.action`` config the runtime will map
            actions through.
        obs_schema_version: The observation schema version the caller will
            assemble (e.g. ``HOVER_OBSERVATION_SCHEMA_VERSION``).
        obs_dim: The observation vector length the caller will feed.
        policy_rate_hz: The inference rate the caller will use.

    Returns:
        A pure ``obs_vector -> normalized action`` callable (numpy in/out; the
        TorchScript module runs on CPU inside).

    Raises:
        ValueError: missing artifact files or any metadata/config mismatch.
    """
    artifact_dir = Path(artifact_dir)
    module_path = artifact_dir / POLICY_FILENAME
    validate_policy_artifact(
        artifact_dir=artifact_dir,
        action_cfg=action_cfg,
        obs_schema_version=obs_schema_version,
        obs_dim=obs_dim,
        policy_rate_hz=policy_rate_hz,
        policy_interface=policy_interface,
    )

    import torch

    module = torch.jit.load(str(module_path), map_location="cpu")
    module.eval()
    action_dim = int(action_cfg.action_dim)

    def run(observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        expected_tail = (obs_dim,)
        if obs.ndim not in (1, 2) or obs.shape[-1:] != expected_tail:
            raise ValueError(
                f"policy artifact {artifact_dir} received observation shape "
                f"{obs.shape}, expected ({obs_dim},) or (batch, {obs_dim})"
            )
        if single:
            obs = obs.reshape(1, obs_dim)
        with torch.no_grad():
            action = module(torch.from_numpy(obs))
        action_np = action.detach().cpu().numpy().astype(np.float64)
        if action_np.shape != (obs.shape[0], action_dim):
            raise ValueError(
                f"policy artifact {artifact_dir} produced action shape "
                f"{action_np.shape}, expected ({obs.shape[0]}, {action_dim})"
            )
        return action_np[0] if single else action_np

    return run


def load_policy_metadata(*, artifact_dir: Union[str, Path]) -> dict[str, Any]:
    """Read the sidecar that selects the runtime policy adapter.

    Legacy schema-v1 artifacts are recognized only when their complete
    metadata tuple identifies one existing interface. Legacy press inference
    warns because future exports must record the interface explicitly.
    """
    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise ValueError(f"policy artifact {artifact_dir} is missing {METADATA_FILENAME}")
    metadata = json.loads(metadata_path.read_text())
    if "policy_interface" not in metadata:
        if (
            metadata.get("obs_schema_version") == HOVER_OBSERVATION_SCHEMA_VERSION
            and metadata.get("obs_dim") == HOVER_OBSERVATION_DIM
        ):
            metadata["policy_interface"] = HOVER_POLICY_INTERFACE
        elif (
            metadata.get("obs_schema_version") == OBSERVATION_SCHEMA_VERSION
            and metadata.get("obs_dim") == PRESS_OBSERVATION_DIM
            and metadata.get("policy_rate_hz") == 20.0
            and metadata.get("action_kind") == "velocity"
            and metadata.get("action_dim") == 4
            and isinstance(metadata.get("action_scales"), dict)
            and all(name in metadata["action_scales"] for name in _SCALE_FIELDS)
        ):
            metadata["policy_interface"] = PRESS_POLICY_INTERFACE
            warnings.warn(
                f"policy artifact {artifact_dir} uses legacy press metadata without "
                "policy_interface; inferred press-v1 from its complete schema tuple. "
                "Re-export future artifacts with an explicit policy_interface.",
                RuntimeWarning,
                stacklevel=1,
            )
        else:
            raise ValueError(
                f"policy artifact {artifact_dir} metadata has no policy_interface; "
                "re-export it from its checkpoint"
            )
    return metadata


def build_policy_step_fn(
    *,
    artifact_dir: Union[str, Path],
    action_cfg: Any,
    policy_rate_hz: float,
) -> Callable[["VehicleState", np.ndarray], "Union[VelocitySetpoint, AttitudeRateSetpoint]"]:
    """Compose the ``{kind: policy}`` routine step's per-tick pure function.

    Loads the artifact against the hover observation layout and returns
    ``(vehicle_state, target_enu) -> OFFBOARD setpoint payload``: pack the
    hover observation, run the policy, clip to [-1, 1] (the env-side clamp),
    and map through
    :func:`robotics_experiments.rl.press.press_action_to_offboard`.

    Raises:
        ValueError: propagated from the vector artifact loader on a missing or
            mismatched artifact (surfaces at routine build time).
    """
    policy = load_vector_policy(
        artifact_dir=artifact_dir,
        action_cfg=action_cfg,
        obs_schema_version=HOVER_OBSERVATION_SCHEMA_VERSION,
        obs_dim=HOVER_OBSERVATION_DIM,
        policy_rate_hz=policy_rate_hz,
        policy_interface=HOVER_POLICY_INTERFACE,
    )

    def step(state: "VehicleState", target_enu: np.ndarray):
        observation = hover_observation_from_state(state, target_enu)
        action = np.clip(policy(observation), -1.0, 1.0)
        return press_action_to_offboard(action, action_cfg)

    return step

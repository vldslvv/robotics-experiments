"""Validated policy-artifact loading + the deploy-time policy step composer.

The policy artifact is a directory holding a TorchScript module
(``policy.pt``) and a JSON metadata sidecar (``policy_meta.json``) recording
the contract the policy was trained against: the observation schema version
and dimension, the action level (``kind``/``action_dim``), and the action
command scales. ``load_policy`` refuses any artifact whose metadata mismatches
the active config -- fail loudly at load, never a shape error deep in the
driver. A bare checkpoint without metadata is not a valid artifact.

Torch is imported only inside functions (like ``MavlinkCommandPort`` imports
pymavlink inside methods), so this module stays on the core import path and
the runtime never depends on the training stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np

from .deploy import press_action_to_offboard
from .observation import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    hover_observation_from_state,
)

if TYPE_CHECKING:
    from robotics_experiments.commanding import AttitudeRateSetpoint, VelocitySetpoint
    from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = [
    "METADATA_FILENAME",
    "POLICY_FILENAME",
    "build_policy_step_fn",
    "load_policy",
    "policy_metadata",
]

POLICY_FILENAME = "policy.pt"
METADATA_FILENAME = "policy_meta.json"

# Action command scales recorded in (and validated against) the metadata; the
# full set is recorded regardless of kind so a scale edit invalidates old
# artifacts loudly instead of silently rescaling their actions.
_SCALE_FIELDS = (
    "max_body_velocity_mps",
    "max_yaw_rate_radps",
    "max_body_rate_radps",
    "hover_thrust_fraction",
)


def policy_metadata(action_cfg: Any, *, obs_schema_version: int, obs_dim: int) -> dict[str, Any]:
    """Compose the artifact metadata for a policy trained on this config."""
    return {
        "obs_schema_version": int(obs_schema_version),
        "obs_dim": int(obs_dim),
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


def load_policy(
    artifact_dir: Union[str, Path],
    *,
    action_cfg: Any,
    obs_schema_version: int,
    obs_dim: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Load a policy artifact, validating its metadata against the active config.

    Args:
        artifact_dir: Directory holding ``policy.pt`` + ``policy_meta.json``.
        action_cfg: The active ``rl.action`` config the runtime will map
            actions through.
        obs_schema_version: The observation schema version the caller will
            assemble (e.g. ``HOVER_OBSERVATION_SCHEMA_VERSION``).
        obs_dim: The observation vector length the caller will feed.

    Returns:
        A pure ``obs_vector -> normalized action`` callable (numpy in/out; the
        TorchScript module runs on CPU inside).

    Raises:
        ValueError: missing artifact files or any metadata/config mismatch.
    """
    artifact_dir = Path(artifact_dir)
    module_path = artifact_dir / POLICY_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not module_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"policy artifact {artifact_dir} is missing {POLICY_FILENAME} or "
            f"{METADATA_FILENAME}; train one first (make train-rl)"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = policy_metadata(action_cfg, obs_schema_version=obs_schema_version, obs_dim=obs_dim)
    _validate_metadata(metadata, expected, artifact_dir)

    import torch

    module = torch.jit.load(str(module_path), map_location="cpu")
    module.eval()
    action_dim = int(action_cfg.action_dim)

    def run(observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, obs_dim)
        with torch.no_grad():
            action = module(torch.from_numpy(obs))
        action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if action_np.shape != (action_dim,):
            raise ValueError(
                f"policy artifact {artifact_dir} produced action shape "
                f"{action_np.shape}, expected ({action_dim},)"
            )
        return action_np

    return run


def build_policy_step_fn(
    artifact_dir: Union[str, Path], action_cfg: Any
) -> Callable[["VehicleState", np.ndarray], "Union[VelocitySetpoint, AttitudeRateSetpoint]"]:
    """Compose the ``{kind: policy}`` routine step's per-tick pure function.

    Loads the artifact against the hover observation layout and returns
    ``(vehicle_state, target_enu) -> OFFBOARD setpoint payload``: pack the
    hover observation, run the policy, clip to [-1, 1] (the env-side clamp),
    and map through :func:`press_action_to_offboard`.

    Raises:
        ValueError: propagated from :func:`load_policy` on a missing or
            mismatched artifact (surfaces at routine build time).
    """
    policy = load_policy(
        artifact_dir,
        action_cfg=action_cfg,
        obs_schema_version=HOVER_OBSERVATION_SCHEMA_VERSION,
        obs_dim=HOVER_OBSERVATION_DIM,
    )

    def step(state: "VehicleState", target_enu: np.ndarray):
        observation = hover_observation_from_state(state, target_enu)
        action = np.clip(policy(observation), -1.0, 1.0)
        return press_action_to_offboard(action, action_cfg)

    return step

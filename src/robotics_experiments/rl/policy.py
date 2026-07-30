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
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from robotics_experiments.alignment import ControllerArtifact, ControllerSettings
from robotics_experiments.config import RlActionConfig
from robotics_experiments.rl._interfaces import INTERFACE_OBSERVATIONS, PolicyInterface

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
    curriculum_stage: str | None = None


@dataclass(frozen=True)
class PolicyMetadata:
    """Validated runtime contract paired with the complete stored JSON record."""

    contract: PolicyContract
    controller_settings: ControllerSettings
    controller_run_id: str
    controller_digest: str
    controller_comparison_fingerprint: str
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


def policy_metadata(
    *,
    action_cfg: RlActionConfig,
    controller_artifact: ControllerArtifact,
    controller_digest: str,
    obs_schema_version: int,
    obs_dim: int,
    policy_rate_hz: float,
    policy_interface: str | PolicyInterface,
    curriculum_stage: str | None = None,
) -> dict[str, Any]:
    """Compose metadata for a contract authored by a training config."""
    interface = PolicyInterface(policy_interface)
    metadata = {
        "policy_interface": interface.value,
        "obs_schema_version": int(obs_schema_version),
        "obs_dim": int(obs_dim),
        "policy_rate_hz": float(policy_rate_hz),
        "action_kind": str(action_cfg.kind),
        "action_dim": int(action_cfg.action_dim),
        "action_scales": {name: float(getattr(action_cfg, name)) for name in _SCALE_FIELDS},
        "controller": {
            "schema_version": controller_artifact.schema_version,
            "alignment_id": controller_artifact.alignment_id,
            "run_id": controller_artifact.run_id,
            "controller_digest": controller_digest,
            "controller_comparison_fingerprint": controller_artifact.comparison_fingerprint,
            "target_definition_fingerprint": (controller_artifact.target_definition_fingerprint),
            "reference_definition_fingerprint": (
                controller_artifact.reference_definition_fingerprint
            ),
            "settings": asdict(controller_artifact.selected_settings),
            "advisory_quality_passed": controller_artifact.advisory_quality_passed,
        },
    }
    if interface is PolicyInterface.PRESS_V1:
        if curriculum_stage != "approach":
            raise ValueError(
                "press-v1 metadata requires curriculum_stage='approach', "
                f"got {curriculum_stage!r}"
            )
        metadata["curriculum_stage"] = curriculum_stage
    elif curriculum_stage is not None:
        raise ValueError(
            f"{interface.value} metadata must not declare " f"curriculum_stage={curriculum_stage!r}"
        )
    return metadata


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
    return dict(loaded)


def _controller_settings(
    *,
    metadata: Mapping[str, Any],
    artifact_dir: Path,
) -> tuple[ControllerSettings, str, str, str]:
    controller = _metadata_value(
        metadata=metadata,
        name="controller",
        artifact_dir=artifact_dir,
    )
    if not isinstance(controller, dict):
        raise ValueError(
            f"policy artifact {artifact_dir} metadata controller must be an object; "
            "legacy policy artifacts without controller provenance are unsupported"
        )
    required_strings = (
        "alignment_id",
        "run_id",
        "controller_digest",
        "controller_comparison_fingerprint",
        "target_definition_fingerprint",
        "reference_definition_fingerprint",
    )
    for name in required_strings:
        value = controller.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"policy artifact {artifact_dir} metadata controller.{name} "
                "must be a non-empty string"
            )
    schema_version = controller.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"policy artifact {artifact_dir} has unsupported controller schema")
    if type(controller.get("advisory_quality_passed")) is not bool:
        raise ValueError(
            f"policy artifact {artifact_dir} controller quality status must be boolean"
        )
    settings = controller.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(
            f"policy artifact {artifact_dir} metadata controller.settings must be an object"
        )
    if set(settings) != {item.name for item in fields(ControllerSettings)}:
        raise ValueError(
            f"policy artifact {artifact_dir} metadata controller.settings fields mismatch"
        )

    def vector(name: str) -> tuple[float, float, float]:
        value = settings.get(name)
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(
                f"policy artifact {artifact_dir} alignment.settings.{name} "
                "must contain three finite values"
            )
        parsed = tuple(
            _finite_nonnegative(
                value=item,
                name=f"alignment.settings.{name}",
                artifact_dir=artifact_dir,
            )
            for item in value
        )
        return parsed[0], parsed[1], parsed[2]

    def rate_output_scale() -> tuple[float, float, float]:
        if schema_version == 2 and "rate_output_scale_nm" not in settings:
            return 1.0, 1.0, 1.0
        return vector("rate_output_scale_nm")

    def hover_thrust_fraction() -> float:
        if schema_version in (2, 3) and "hover_thrust_fraction" not in settings:
            return 0.0
        return _finite_positive(
            value=settings.get("hover_thrust_fraction"),
            name="alignment.settings.hover_thrust_fraction",
            artifact_dir=artifact_dir,
        )

    def hover_thrust_normalized() -> float:
        if schema_version in (2, 3, 4):
            return 0.0
        return _finite_positive(
            value=settings.get("hover_thrust_normalized"),
            name="alignment.settings.hover_thrust_normalized",
            artifact_dir=artifact_dir,
        )

    def horizontal_schedule(name: str, legacy_value: float) -> float:
        if schema_version in (2, 3, 4, 5):
            return legacy_value
        return _finite_positive(
            value=settings.get(name),
            name=f"alignment.settings.{name}",
            artifact_dir=artifact_dir,
        )

    delay = settings.get("command_delay_s")
    if (
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(delay)
        or delay < 0.0
    ):
        raise ValueError(
            f"policy artifact {artifact_dir} alignment.settings.command_delay_s "
            "must be finite and non-negative"
        )
    controller_settings = ControllerSettings(
        command_delay_s=float(delay),
        kp_vel=vector("kp_vel"),
        horizontal_velocity_error_gain_scale=horizontal_schedule(
            "horizontal_velocity_error_gain_scale",
            1.0,
        ),
        horizontal_velocity_error_gain_start_mps=horizontal_schedule(
            "horizontal_velocity_error_gain_start_mps",
            0.15,
        ),
        horizontal_velocity_error_gain_full_mps=horizontal_schedule(
            "horizontal_velocity_error_gain_full_mps",
            0.30,
        ),
        ki_vel=vector("ki_vel"),
        kd_vel=vector("kd_vel"),
        velocity_filter_hz=_finite_nonnegative(
            value=settings.get("velocity_filter_hz"),
            name="alignment.settings.velocity_filter_hz",
            artifact_dir=artifact_dir,
        ),
        velocity_derivative_filter_hz=_finite_positive(
            value=settings.get("velocity_derivative_filter_hz"),
            name="alignment.settings.velocity_derivative_filter_hz",
            artifact_dir=artifact_dir,
        ),
        velocity_integral_limit_mps2=vector("velocity_integral_limit_mps2"),
        hover_thrust_fraction=hover_thrust_fraction(),
        hover_thrust_normalized=hover_thrust_normalized(),
        kp_att=vector("kp_att"),
        attitude_yaw_weight=_finite_nonnegative(
            value=settings.get("attitude_yaw_weight"),
            name="alignment.settings.attitude_yaw_weight",
            artifact_dir=artifact_dir,
        ),
        kp_rate=vector("kp_rate"),
        ki_rate=vector("ki_rate"),
        kd_rate=vector("kd_rate"),
        kff_rate=vector("kff_rate"),
        rate_output_scale_nm=rate_output_scale(),
        rate_derivative_filter_hz=_finite_positive(
            value=settings.get("rate_derivative_filter_hz"),
            name="alignment.settings.rate_derivative_filter_hz",
            artifact_dir=artifact_dir,
        ),
        rate_integral_limit_nm=vector("rate_integral_limit_nm"),
        max_tilt_cmd_deg=_finite_positive(
            value=settings.get("max_tilt_cmd_deg"),
            name="alignment.settings.max_tilt_cmd_deg",
            artifact_dir=artifact_dir,
        ),
        min_collective_thrust_fraction=_finite_nonnegative(
            value=settings.get("min_collective_thrust_fraction"),
            name="alignment.settings.min_collective_thrust_fraction",
            artifact_dir=artifact_dir,
        ),
        max_collective_thrust_fraction=_finite_positive(
            value=settings.get("max_collective_thrust_fraction"),
            name="alignment.settings.max_collective_thrust_fraction",
            artifact_dir=artifact_dir,
        ),
        horizontal_thrust_margin_fraction=_finite_nonnegative(
            value=settings.get("horizontal_thrust_margin_fraction"),
            name="alignment.settings.horizontal_thrust_margin_fraction",
            artifact_dir=artifact_dir,
        ),
        thruster_tau_inc_s=_finite_nonnegative(
            value=settings.get("thruster_tau_inc_s"),
            name="alignment.settings.thruster_tau_inc_s",
            artifact_dir=artifact_dir,
        ),
        thruster_tau_dec_s=_finite_nonnegative(
            value=settings.get("thruster_tau_dec_s"),
            name="alignment.settings.thruster_tau_dec_s",
            artifact_dir=artifact_dir,
        ),
        thruster_max_thrust_rate_nps=_finite_positive(
            value=settings.get("thruster_max_thrust_rate_nps"),
            name="alignment.settings.thruster_max_thrust_rate_nps",
            artifact_dir=artifact_dir,
        ),
    )
    if not 0.0 <= controller_settings.attitude_yaw_weight <= 1.0:
        raise ValueError(
            f"policy artifact {artifact_dir} alignment.settings.attitude_yaw_weight "
            "must be in [0, 1]"
        )
    if controller_settings.hover_thrust_fraction > 1.0:
        raise ValueError(
            f"policy artifact {artifact_dir} alignment.settings.hover_thrust_fraction "
            "must not exceed 1"
        )
    if controller_settings.hover_thrust_normalized > 1.0:
        raise ValueError(
            f"policy artifact {artifact_dir} alignment.settings.hover_thrust_normalized "
            "must not exceed 1"
        )
    if controller_settings.horizontal_velocity_error_gain_scale < 1.0:
        raise ValueError(
            f"policy artifact {artifact_dir} alignment.settings."
            "horizontal_velocity_error_gain_scale must be at least 1"
        )
    if (
        controller_settings.horizontal_velocity_error_gain_start_mps
        >= controller_settings.horizontal_velocity_error_gain_full_mps
    ):
        raise ValueError(
            f"policy artifact {artifact_dir} alignment horizontal velocity error "
            "gain transition must satisfy start < full"
        )
    if not (
        0.0
        <= controller_settings.min_collective_thrust_fraction
        < controller_settings.max_collective_thrust_fraction
        <= 1.0
    ):
        raise ValueError(
            f"policy artifact {artifact_dir} alignment collective thrust fractions "
            "must satisfy 0 <= min < max <= 1"
        )
    if (
        controller_settings.horizontal_thrust_margin_fraction
        >= controller_settings.max_collective_thrust_fraction
    ):
        raise ValueError(
            f"policy artifact {artifact_dir} alignment horizontal thrust margin "
            "must be below the maximum thrust fraction"
        )
    return (
        controller_settings,
        str(controller["run_id"]),
        str(controller["controller_digest"]),
        str(controller["controller_comparison_fingerprint"]),
    )


def _finite_nonnegative(*, value: Any, name: str, artifact_dir: Path) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(
            f"policy artifact {artifact_dir} metadata {name} must be finite and "
            f"non-negative, got {value!r}"
        )
    return float(value)


def _parse_metadata(*, artifact_dir: Path, metadata: dict[str, Any]) -> PolicyMetadata:
    (
        controller_settings,
        controller_run_id,
        controller_digest,
        controller_comparison_fingerprint,
    ) = _controller_settings(metadata=metadata, artifact_dir=artifact_dir)
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
    curriculum_stage = metadata.get("curriculum_stage")
    if interface is PolicyInterface.PRESS_V1:
        if curriculum_stage is None:
            raise ValueError(
                f"policy artifact {artifact_dir} press-v1 metadata is missing "
                "curriculum_stage; historical stage-less press artifacts are unsupported"
            )
        if curriculum_stage != "approach":
            raise ValueError(
                f"policy artifact {artifact_dir} has unsupported "
                f"curriculum_stage={curriculum_stage!r}"
            )
    elif curriculum_stage is not None:
        raise ValueError(
            f"policy artifact {artifact_dir} {interface.value} metadata must not "
            f"declare curriculum_stage={curriculum_stage!r}"
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
        curriculum_stage=curriculum_stage,
        **parsed_scales,
    )
    return PolicyMetadata(
        contract=contract,
        controller_settings=controller_settings,
        controller_run_id=controller_run_id,
        controller_digest=controller_digest,
        controller_comparison_fingerprint=controller_comparison_fingerprint,
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

"""Strict loading for the controller-v1 artifact."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from robotics_experiments.controller._models import (
    ControllerArtifact,
    ControllerSettings,
)

__all__: list[str] = []

_ARTIFACT_KIND = "robotics-experiments-controller"
_SCHEMA_VERSION = 1


def _fingerprint(value: object, /) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _object(value: Any, name: str, /) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_fields(value: dict[str, Any], model: type[Any], name: str, /) -> None:
    expected = {item.name for item in fields(model)}
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )


def _string(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(
    *,
    value: Any,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _vector(
    value: Any,
    name: str,
    nonnegative: bool = True,
    /,
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite values")
    parsed = tuple(_finite(value=item, name=name, nonnegative=nonnegative) for item in value)
    return parsed[0], parsed[1], parsed[2]


def _settings(value: Any, name: str, /) -> ControllerSettings:
    payload = _object(value, name)
    _exact_fields(payload, ControllerSettings, name)
    result = ControllerSettings(
        command_delay_s=_finite(
            value=payload["command_delay_s"], name=f"{name}.command_delay_s", nonnegative=True
        ),
        velocity_controller_rate_hz=_finite(
            value=payload["velocity_controller_rate_hz"],
            name=f"{name}.velocity_controller_rate_hz",
            positive=True,
        ),
        attitude_controller_rate_hz=_finite(
            value=payload["attitude_controller_rate_hz"],
            name=f"{name}.attitude_controller_rate_hz",
            positive=True,
        ),
        kp_vel=_vector(payload["kp_vel"], f"{name}.kp_vel"),
        horizontal_velocity_error_gain_scale=_finite(
            value=payload["horizontal_velocity_error_gain_scale"],
            name=f"{name}.horizontal_velocity_error_gain_scale",
            positive=True,
        ),
        horizontal_velocity_error_gain_start_mps=_finite(
            value=payload["horizontal_velocity_error_gain_start_mps"],
            name=f"{name}.horizontal_velocity_error_gain_start_mps",
            positive=True,
        ),
        horizontal_velocity_error_gain_full_mps=_finite(
            value=payload["horizontal_velocity_error_gain_full_mps"],
            name=f"{name}.horizontal_velocity_error_gain_full_mps",
            positive=True,
        ),
        horizontal_velocity_error_gain_end_mps=_finite(
            value=payload["horizontal_velocity_error_gain_end_mps"],
            name=f"{name}.horizontal_velocity_error_gain_end_mps",
            positive=True,
        ),
        ki_vel=_vector(payload["ki_vel"], f"{name}.ki_vel"),
        kd_vel=_vector(payload["kd_vel"], f"{name}.kd_vel"),
        velocity_observation_scale=_vector(
            payload["velocity_observation_scale"],
            f"{name}.velocity_observation_scale",
        ),
        velocity_observation_bias_mps=_vector(
            payload["velocity_observation_bias_mps"],
            f"{name}.velocity_observation_bias_mps",
            False,
        ),
        velocity_estimator_bias_decay_hz=_finite(
            value=payload["velocity_estimator_bias_decay_hz"],
            name=f"{name}.velocity_estimator_bias_decay_hz",
            nonnegative=True,
        ),
        velocity_filter_hz=_vector(payload["velocity_filter_hz"], f"{name}.velocity_filter_hz"),
        velocity_derivative_filter_hz=_finite(
            value=payload["velocity_derivative_filter_hz"],
            name=f"{name}.velocity_derivative_filter_hz",
            positive=True,
        ),
        velocity_integral_limit_mps2=_vector(
            payload["velocity_integral_limit_mps2"], f"{name}.velocity_integral_limit_mps2"
        ),
        hover_thrust_normalized=_finite(
            value=payload["hover_thrust_normalized"],
            name=f"{name}.hover_thrust_normalized",
            positive=True,
        ),
        ctbr_effective_collective_scale=_finite(
            value=payload["ctbr_effective_collective_scale"],
            name=f"{name}.ctbr_effective_collective_scale",
            positive=True,
        ),
        kp_att=_vector(payload["kp_att"], f"{name}.kp_att"),
        attitude_yaw_weight=_finite(
            value=payload["attitude_yaw_weight"],
            name=f"{name}.attitude_yaw_weight",
            nonnegative=True,
        ),
        kp_rate=_vector(payload["kp_rate"], f"{name}.kp_rate"),
        ki_rate=_vector(payload["ki_rate"], f"{name}.ki_rate"),
        kd_rate=_vector(payload["kd_rate"], f"{name}.kd_rate"),
        kff_rate=_vector(payload["kff_rate"], f"{name}.kff_rate"),
        angular_velocity_filter_hz=_finite(
            value=payload["angular_velocity_filter_hz"],
            name=f"{name}.angular_velocity_filter_hz",
            nonnegative=True,
        ),
        rate_derivative_filter_hz=_finite(
            value=payload["rate_derivative_filter_hz"],
            name=f"{name}.rate_derivative_filter_hz",
            positive=True,
        ),
        yaw_torque_filter_hz=_finite(
            value=payload["yaw_torque_filter_hz"],
            name=f"{name}.yaw_torque_filter_hz",
            positive=True,
        ),
        rate_integral_limit=_vector(payload["rate_integral_limit"], f"{name}.rate_integral_limit"),
        max_tilt_cmd_deg=_finite(
            value=payload["max_tilt_cmd_deg"], name=f"{name}.max_tilt_cmd_deg", positive=True
        ),
        min_collective_thrust_fraction=_finite(
            value=payload["min_collective_thrust_fraction"],
            name=f"{name}.min_collective_thrust_fraction",
            nonnegative=True,
        ),
        max_collective_thrust_fraction=_finite(
            value=payload["max_collective_thrust_fraction"],
            name=f"{name}.max_collective_thrust_fraction",
            positive=True,
        ),
        horizontal_thrust_margin_fraction=_finite(
            value=payload["horizontal_thrust_margin_fraction"],
            name=f"{name}.horizontal_thrust_margin_fraction",
            nonnegative=True,
        ),
        thruster_tau_inc_s=_finite(
            value=payload["thruster_tau_inc_s"], name=f"{name}.thruster_tau_inc_s", nonnegative=True
        ),
        thruster_tau_dec_s=_finite(
            value=payload["thruster_tau_dec_s"], name=f"{name}.thruster_tau_dec_s", nonnegative=True
        ),
        thruster_max_thrust_rate_nps=_finite(
            value=payload["thruster_max_thrust_rate_nps"],
            name=f"{name}.thruster_max_thrust_rate_nps",
            positive=True,
        ),
    )
    if not 0.0 <= result.attitude_yaw_weight <= 1.0:
        raise ValueError(f"{name}.attitude_yaw_weight must be in [0, 1]")
    if not 0.0 < result.hover_thrust_normalized <= 1.0:
        raise ValueError(f"{name}.hover_thrust_normalized must be in (0, 1]")
    if result.horizontal_velocity_error_gain_scale < 1.0:
        raise ValueError(f"{name}.horizontal_velocity_error_gain_scale must be at least 1")
    if (
        not result.horizontal_velocity_error_gain_start_mps
        < result.horizontal_velocity_error_gain_full_mps
        < result.horizontal_velocity_error_gain_end_mps
    ):
        raise ValueError(f"{name} horizontal gain transition must satisfy start < full < end")
    if (
        not 0.0
        <= result.min_collective_thrust_fraction
        < result.max_collective_thrust_fraction
        <= 1.0
    ):
        raise ValueError(f"{name} collective limits must satisfy 0 <= min < max <= 1")
    if result.horizontal_thrust_margin_fraction >= result.max_collective_thrust_fraction:
        raise ValueError(f"{name}.horizontal_thrust_margin_fraction must be below max collective")
    return result


def resolve_controller_path(path: str | Path, /) -> Path:
    """Resolve a strict v1 controller file or bundle directory."""
    artifact_path = Path(path)
    return artifact_path / "controller.json" if artifact_path.is_dir() else artifact_path


def controller_artifact_from_payload(value: Any, /) -> ControllerArtifact:
    """Validate an in-memory exact controller-v1 artifact payload."""
    payload = _object(value, "controller artifact")
    _exact_fields(payload, ControllerArtifact, "controller artifact")
    if payload["artifact_kind"] != _ARTIFACT_KIND or payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("controller artifact is not the supported controller-v1 contract")
    modes = payload["certified_control_modes"]
    if (
        not isinstance(modes, list)
        or not modes
        or any(mode not in ("velocity", "ctbr") for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError(
            "certified_control_modes must be a non-empty unique list of velocity and/or ctbr"
        )
    settings = _settings(payload["selected_settings"], "selected_settings")
    settings_fingerprint = _string(payload["settings_fingerprint"], "settings_fingerprint")
    if settings_fingerprint != _fingerprint(asdict(settings)):
        raise ValueError("controller artifact settings_fingerprint does not match settings")
    reasons = payload["advisory_quality_reasons"]
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ValueError("advisory_quality_reasons must be a string list")
    if (
        type(payload["advisory_quality_passed"]) is not bool
        or type(payload["source_dirty"]) is not bool
    ):
        raise ValueError("controller quality and dirty provenance must be booleans")
    strings = {
        name: _string(payload[name], name)
        for name in (
            "alignment_id",
            "run_id",
            "target_controller_id",
            "reference_controller_id",
            "target_backend",
            "reference_backend",
            "target_definition_fingerprint",
            "reference_definition_fingerprint",
            "comparison_fingerprint",
            "experiment_config",
            "experiment_fingerprint",
            "vehicle",
            "airframe",
            "probe_suite_fingerprint",
            "source_commit",
            "target_image",
            "reference_image",
        )
    }
    if strings["target_backend"] != "isaac-lab" or strings["reference_backend"] != "px4-sitl":
        raise ValueError("controller artifact backend contract must be isaac-lab/px4-sitl")
    return ControllerArtifact(
        artifact_kind=_ARTIFACT_KIND,
        schema_version=1,
        command_rate_hz=_finite(
            value=payload["command_rate_hz"], name="command_rate_hz", positive=True
        ),
        certified_control_modes=tuple(modes),
        selected_settings=settings,
        settings_fingerprint=settings_fingerprint,
        advisory_quality_passed=payload["advisory_quality_passed"],
        advisory_quality_reasons=tuple(reasons),
        source_dirty=payload["source_dirty"],
        **strings,
    )


def load_controller_artifact(path: str | Path, /) -> ControllerArtifact:
    """Load an exact controller-v1 artifact with no compatibility behavior."""
    artifact_path = resolve_controller_path(path)
    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load controller artifact {artifact_path}: {exc}") from exc
    return controller_artifact_from_payload(payload)

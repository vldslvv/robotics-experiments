"""Strict flight-campaign configuration loading."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from robotics_experiments.flight_parameters._models import (
    FlightConfig,
    FlightExecutionConfig,
    HoverFlightConfig,
)

__all__: list[str] = []

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _object(value: Any, name: str, /) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact(value: dict[str, Any], expected: set[str], name: str, /) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{name} fields mismatch: missing={missing}, extra={extra}")


def _string(value: Any, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _identifier(value: Any, name: str, /) -> str:
    result = _string(value, name)
    if _IDENTIFIER_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{name} must contain only letters, digits, dot, underscore, or dash")
    return result


def _finite(value: Any, name: str, /, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_integer(value: Any, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _vector3(value: Any, name: str, /) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _fingerprint(value: object, /) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_flight_config(path: str | Path, /) -> FlightConfig:
    """Load the exact supported flight-config-v1 YAML contract."""

    config_path = Path(path)
    try:
        value = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    except Exception as exc:
        raise ValueError(f"cannot load flight config {config_path}: {exc}") from exc
    payload = _object(value, "flight config")
    _exact(
        payload,
        {
            "schema_version",
            "flight_id",
            "source_backend",
            "state_source",
            "experiment_config",
            "vehicle",
            "scenario",
            "position_margin_m",
            "attitude_margin_rad",
            "execution",
        },
        "flight config",
    )
    if payload["schema_version"] != 1:
        raise ValueError("flight config is not the supported schema version 1")
    if payload["source_backend"] != "px4-sitl":
        raise ValueError("flight config currently supports only source_backend 'px4-sitl'")
    if payload["state_source"] != "simulator-ground-truth":
        raise ValueError(
            "flight config currently supports only state_source 'simulator-ground-truth'"
        )
    scenario_payload = _object(payload["scenario"], "flight config scenario")
    _exact(
        scenario_payload,
        {
            "kind",
            "repetitions",
            "mode_transition_timeout_s",
            "mode_report_max_age_s",
            "settle_s",
            "measure_s",
            "sample_rate_hz",
            "target_position_enu_m",
            "target_yaw_rad",
        },
        "flight config scenario",
    )
    if scenario_payload["kind"] != "hover":
        raise ValueError("flight config currently supports only scenario kind 'hover'")
    scenario = HoverFlightConfig(
        kind="hover",
        repetitions=_positive_integer(scenario_payload["repetitions"], "scenario.repetitions"),
        mode_transition_timeout_s=_finite(
            scenario_payload["mode_transition_timeout_s"],
            "scenario.mode_transition_timeout_s",
            positive=True,
        ),
        mode_report_max_age_s=_finite(
            scenario_payload["mode_report_max_age_s"],
            "scenario.mode_report_max_age_s",
            positive=True,
        ),
        settle_s=_finite(scenario_payload["settle_s"], "scenario.settle_s", positive=True),
        measure_s=_finite(scenario_payload["measure_s"], "scenario.measure_s", positive=True),
        sample_rate_hz=_finite(
            scenario_payload["sample_rate_hz"], "scenario.sample_rate_hz", positive=True
        ),
        target_position_enu_m=_vector3(
            scenario_payload["target_position_enu_m"], "scenario.target_position_enu_m"
        ),
        target_yaw_rad=_finite(scenario_payload["target_yaw_rad"], "scenario.target_yaw_rad"),
    )
    sample_count = scenario.measure_s * scenario.sample_rate_hz
    if not math.isclose(sample_count, round(sample_count), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("scenario measure_s * sample_rate_hz must be an integer")
    if round(sample_count) <= 1:
        raise ValueError("scenario measurement must contain at least two samples")
    execution_payload = _object(payload["execution"], "flight config execution")
    _exact(
        execution_payload,
        {"worker_count"},
        "flight config execution",
    )
    execution = FlightExecutionConfig(
        worker_count=_positive_integer(execution_payload["worker_count"], "execution.worker_count"),
    )
    without_fingerprint = FlightConfig(
        schema_version=1,
        flight_id=_identifier(payload["flight_id"], "flight_id"),
        source_backend="px4-sitl",
        state_source="simulator-ground-truth",
        experiment_config=_string(payload["experiment_config"], "experiment_config"),
        vehicle=_string(payload["vehicle"], "vehicle"),
        scenario=scenario,
        position_margin_m=_finite(payload["position_margin_m"], "position_margin_m", positive=True),
        attitude_margin_rad=_finite(
            payload["attitude_margin_rad"], "attitude_margin_rad", positive=True
        ),
        execution=execution,
        config_fingerprint="",
    )
    fingerprint_payload = asdict(without_fingerprint)
    fingerprint_payload.pop("config_fingerprint")
    return FlightConfig(
        **{**fingerprint_payload, "scenario": scenario, "execution": execution},
        config_fingerprint=_fingerprint(fingerprint_payload),
    )

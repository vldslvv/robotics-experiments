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
    FlightExecutionConfig,
    FlightParametersConfig,
    FlightSetConfig,
    HoverConfig,
    ResponseConfig,
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


def flight_trials(*, config: FlightParametersConfig, kind: str) -> tuple[tuple[str, int], ...]:
    """Assign seeds in hover-then-response, fit-validation-acceptance order."""
    if kind not in {"hover", "response"}:
        raise ValueError(f"unsupported flight kind: {kind!r}")
    offset = sum(group.hover_count for group in config.sets) if kind == "response" else 0
    membership = (
        group.name
        for group in config.sets
        for _ in range(group.hover_count if kind == "hover" else group.response_count)
    )
    return tuple(
        (name, config.starting_seed + offset + index) for index, name in enumerate(membership)
    )


def load_flight_parameters_config(path: str | Path, /) -> FlightParametersConfig:
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
            "flight_parameters_id",
            "source_backend",
            "state_source",
            "experiment_config",
            "vehicle",
            "starting_seed",
            "sets",
            "hover",
            "response",
            "policy_warmup_s",
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
    sets_payload = _object(payload["sets"], "sets")
    _exact(sets_payload, {"fit", "validation", "acceptance"}, "sets")
    flight_sets = []
    for name in ("fit", "validation", "acceptance"):
        item = _object(sets_payload[name], f"sets.{name}")
        _exact(item, {"hover_count", "response_count"}, f"sets.{name}")
        flight_sets.append(
            FlightSetConfig(
                name=name,
                hover_count=_positive_integer(item["hover_count"], f"sets.{name}.hover_count"),
                response_count=_positive_integer(
                    item["response_count"], f"sets.{name}.response_count"
                ),
            )
        )
    sets = tuple(flight_sets)
    scenario_payload = _object(payload["hover"], "flight parameters hover")
    _exact(
        scenario_payload,
        {
            "kind",
            "mode_transition_timeout_s",
            "mode_report_max_age_s",
            "settle_s",
            "measure_s",
            "sample_rate_hz",
            "target_position_enu_m",
            "target_yaw_rad",
        },
        "flight parameters hover",
    )
    if scenario_payload["kind"] != "hover":
        raise ValueError("flight parameters currently supports only hover kind 'hover'")
    hover = HoverConfig(
        kind="hover",
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
    sample_count = hover.measure_s * hover.sample_rate_hz
    if not math.isclose(sample_count, round(sample_count), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("scenario measure_s * sample_rate_hz must be an integer")
    if round(sample_count) <= 1:
        raise ValueError("scenario measurement must contain at least two samples")
    response_payload = _object(payload["response"], "flight parameters response")
    _exact(
        response_payload,
        {"kind", "experiment_config", "definition"},
        "flight parameters response",
    )
    if response_payload["kind"] != "controller-response":
        raise ValueError("flight parameters supports only response kind 'controller-response'")
    response = ResponseConfig(
        kind="controller-response",
        experiment_config=_string(
            response_payload["experiment_config"], "response.experiment_config"
        ),
        definition=_string(response_payload["definition"], "response.definition"),
    )
    execution_payload = _object(payload["execution"], "flight config execution")
    _exact(
        execution_payload,
        {"worker_count"},
        "flight config execution",
    )
    execution = FlightExecutionConfig(
        worker_count=_positive_integer(execution_payload["worker_count"], "execution.worker_count"),
    )
    policy_warmup_s = _finite(payload["policy_warmup_s"], "policy_warmup_s", positive=True)
    if policy_warmup_s >= hover.measure_s:
        raise ValueError("policy_warmup_s must be shorter than hover.measure_s")
    policy_samples = policy_warmup_s * hover.sample_rate_hz
    if not math.isclose(policy_samples, round(policy_samples), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("policy_warmup_s * hover.sample_rate_hz must be an integer")
    without_fingerprint = FlightParametersConfig(
        schema_version=1,
        flight_parameters_id=_identifier(payload["flight_parameters_id"], "flight_parameters_id"),
        source_backend="px4-sitl",
        state_source="simulator-ground-truth",
        experiment_config=_string(payload["experiment_config"], "experiment_config"),
        vehicle=_string(payload["vehicle"], "vehicle"),
        starting_seed=payload["starting_seed"],
        sets=sets,
        hover=hover,
        response=response,
        policy_warmup_s=policy_warmup_s,
        position_margin_m=_finite(payload["position_margin_m"], "position_margin_m", positive=True),
        attitude_margin_rad=_finite(
            payload["attitude_margin_rad"], "attitude_margin_rad", positive=True
        ),
        execution=execution,
        config_fingerprint="",
    )
    fingerprint_payload = asdict(without_fingerprint)
    fingerprint_payload.pop("config_fingerprint")
    return FlightParametersConfig(
        **{
            **fingerprint_payload,
            "sets": sets,
            "hover": hover,
            "response": response,
            "execution": execution,
        },
        config_fingerprint=_fingerprint(fingerprint_payload),
    )

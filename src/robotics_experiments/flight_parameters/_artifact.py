"""Strict capture trace and numeric hover-summary parsing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

from robotics_experiments.flight_parameters._models import (
    FlightParameters,
    FlightTrace,
    HoverParameters,
    HoverVerification,
    Px4CommandEvent,
    Px4ModeEvent,
    Px4PositionGoalEvent,
    RawFlightTrace,
    SignalStatistics,
    UpperBound,
    VectorBounds,
)

__all__: list[str] = []

_TRACE_KIND = "robotics-experiments-flight-parameters-trace"
_RAW_TRACE_KIND = "robotics-experiments-flight-parameters-raw-trace"
_VERIFICATION_KIND = "robotics-experiments-flight-parameters-px4-hover-verification"


def _object(value: Any, name: str, /) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_dataclass(value: dict[str, Any], cls: Any, name: str, /) -> None:
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        raise ValueError(f"{name} fields mismatch")


def _string(value: Any, name: str, /, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not value and not allow_empty):
        qualifier = "string" if allow_empty else "non-empty string"
        raise ValueError(f"{name} must be a {qualifier}")
    return value


def _finite(value: Any, name: str, /, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _integer(value: Any, name: str, /, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise ValueError(f"{name} must be an integer" + (" and positive" if positive else ""))
    return value


def _boolean(value: Any, name: str, /) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _vector(value: Any, name: str, /) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _quaternion(value: Any, name: str, /) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must contain four numbers")
    result = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    norm = math.sqrt(sum(item * item for item in result))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise ValueError(f"{name} must be normalized")
    return result  # type: ignore[return-value]


def _digest(value: Any, name: str, /) -> str:
    result = _string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _verification_fingerprint(payload: dict[str, Any], /) -> str:
    value = dict(payload)
    value.pop("verification_fingerprint", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _set_name(value: Any, /) -> str:
    if value not in ("fit", "validation", "acceptance"):
        raise ValueError("set_name must be fit, validation, or acceptance")
    return value


def _seed(value: Any, /) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return value


def flight_trace_from_payload(value: Any, /) -> FlightTrace:
    """Validate an in-memory exact trace-v1 payload."""

    payload = _object(value, "flight trace")
    _exact_dataclass(payload, FlightTrace, "flight trace")
    if payload["artifact_kind"] != _TRACE_KIND or payload["schema_version"] != 1:
        raise ValueError("flight trace is not the supported trace-v1 contract")
    time = payload["time_s"]
    positions = payload["position_enu_m"]
    attitudes = payload["attitude_flu_enu_wxyz"]
    linear = payload["linear_velocity_enu_mps"]
    angular = payload["angular_velocity_enu_radps"]
    if not isinstance(time, list) or not time:
        raise ValueError("flight trace time_s must be a non-empty list")
    time_values = tuple(_finite(item, f"time_s[{index}]") for index, item in enumerate(time))
    if time_values[0] != 0.0 or any(
        later <= earlier for earlier, later in zip(time_values, time_values[1:], strict=False)
    ):
        raise ValueError("flight trace timestamps must start at zero and increase")
    if any(
        not isinstance(samples, list) or len(samples) != len(time_values)
        for samples in (positions, attitudes, linear, angular)
    ):
        raise ValueError("flight trace sample arrays must have equal lengths")
    completed = payload["completed"]
    if type(completed) is not bool:
        raise ValueError("flight trace completed must be boolean")
    failure_reason = _string(payload["failure_reason"], "failure_reason", allow_empty=True)
    if completed == bool(failure_reason):
        raise ValueError("complete traces have no failure reason; failed traces require one")
    measurement_start_us = _integer(
        payload["measurement_start_us"], "measurement_start_us", positive=True
    )
    measurement_end_us = _integer(
        payload["measurement_end_us"], "measurement_end_us", positive=True
    )
    if measurement_end_us <= measurement_start_us:
        raise ValueError("flight trace measurement interval must increase")
    return FlightTrace(
        artifact_kind=_TRACE_KIND,
        schema_version=1,
        flight_parameters_id=_string(payload["flight_parameters_id"], "flight_parameters_id"),
        campaign_run_id=_string(payload["campaign_run_id"], "campaign_run_id"),
        trial_index=_integer(payload["trial_index"], "trial_index", positive=True),
        set_name=_set_name(payload["set_name"]),
        seed=_seed(payload["seed"]),
        source_backend=_string(payload["source_backend"], "source_backend"),
        state_source=_string(payload["state_source"], "state_source"),
        config_fingerprint=_string(payload["config_fingerprint"], "config_fingerprint"),
        experiment_fingerprint=_string(payload["experiment_fingerprint"], "experiment_fingerprint"),
        vehicle_fingerprint=_string(payload["vehicle_fingerprint"], "vehicle_fingerprint"),
        source_image=_string(payload["source_image"], "source_image"),
        verification_fingerprint=_digest(
            payload["verification_fingerprint"], "verification_fingerprint"
        ),
        measurement_start_us=measurement_start_us,
        measurement_end_us=measurement_end_us,
        target_position_enu_m=_vector(payload["target_position_enu_m"], "target_position_enu_m"),
        target_attitude_flu_enu_wxyz=_quaternion(
            payload["target_attitude_flu_enu_wxyz"], "target_attitude_flu_enu_wxyz"
        ),
        time_s=time_values,
        position_enu_m=tuple(
            _vector(item, f"position_enu_m[{index}]") for index, item in enumerate(positions)
        ),
        attitude_flu_enu_wxyz=tuple(
            _quaternion(item, f"attitude_flu_enu_wxyz[{index}]")
            for index, item in enumerate(attitudes)
        ),
        linear_velocity_enu_mps=tuple(
            _vector(item, f"linear_velocity_enu_mps[{index}]") for index, item in enumerate(linear)
        ),
        angular_velocity_enu_radps=tuple(
            _vector(item, f"angular_velocity_enu_radps[{index}]")
            for index, item in enumerate(angular)
        ),
        completed=completed,
        failure_reason=failure_reason,
    )


def raw_flight_trace_from_payload(value: Any, /) -> RawFlightTrace:
    """Validate an in-memory exact raw-trace-v1 payload."""

    payload = _object(value, "raw flight trace")
    _exact_dataclass(payload, RawFlightTrace, "raw flight trace")
    if payload["artifact_kind"] != _RAW_TRACE_KIND or payload["schema_version"] != 1:
        raise ValueError("raw flight trace is not the supported raw-trace-v1 contract")
    raw_time = payload["time_us"]
    positions = payload["position_enu_m"]
    attitudes = payload["attitude_flu_enu_wxyz"]
    linear = payload["linear_velocity_enu_mps"]
    angular = payload["angular_velocity_enu_radps"]
    if not isinstance(raw_time, list) or not raw_time:
        raise ValueError("raw flight trace time_us must be a non-empty list")
    time_us = tuple(
        _integer(item, f"time_us[{index}]", positive=True) for index, item in enumerate(raw_time)
    )
    if any(later <= earlier for earlier, later in zip(time_us, time_us[1:], strict=False)):
        raise ValueError("raw flight trace timestamps must increase")
    if any(
        not isinstance(samples, list) or len(samples) != len(time_us)
        for samples in (positions, attitudes, linear, angular)
    ):
        raise ValueError("raw flight trace sample arrays must have equal lengths")
    completed = _boolean(payload["completed"], "completed")
    failure_reason = _string(payload["failure_reason"], "failure_reason", allow_empty=True)
    if completed == bool(failure_reason):
        raise ValueError("complete raw traces have no failure reason; failed traces require one")
    return RawFlightTrace(
        artifact_kind=_RAW_TRACE_KIND,
        schema_version=1,
        flight_parameters_id=_string(payload["flight_parameters_id"], "flight_parameters_id"),
        campaign_run_id=_string(payload["campaign_run_id"], "campaign_run_id"),
        trial_index=_integer(payload["trial_index"], "trial_index", positive=True),
        set_name=_set_name(payload["set_name"]),
        seed=_seed(payload["seed"]),
        source_backend=_string(payload["source_backend"], "source_backend"),
        state_source=_string(payload["state_source"], "state_source"),
        config_fingerprint=_string(payload["config_fingerprint"], "config_fingerprint"),
        experiment_fingerprint=_string(payload["experiment_fingerprint"], "experiment_fingerprint"),
        vehicle_fingerprint=_string(payload["vehicle_fingerprint"], "vehicle_fingerprint"),
        source_image=_string(payload["source_image"], "source_image"),
        requested_target_position_enu_m=_vector(
            payload["requested_target_position_enu_m"], "requested_target_position_enu_m"
        ),
        requested_target_attitude_flu_enu_wxyz=_quaternion(
            payload["requested_target_attitude_flu_enu_wxyz"],
            "requested_target_attitude_flu_enu_wxyz",
        ),
        px4_home_lat_lon_alt=_vector(payload["px4_home_lat_lon_alt"], "px4_home_lat_lon_alt"),
        time_us=time_us,
        position_enu_m=tuple(
            _vector(item, f"position_enu_m[{index}]") for index, item in enumerate(positions)
        ),
        attitude_flu_enu_wxyz=tuple(
            _quaternion(item, f"attitude_flu_enu_wxyz[{index}]")
            for index, item in enumerate(attitudes)
        ),
        linear_velocity_enu_mps=tuple(
            _vector(item, f"linear_velocity_enu_mps[{index}]") for index, item in enumerate(linear)
        ),
        angular_velocity_enu_radps=tuple(
            _vector(item, f"angular_velocity_enu_radps[{index}]")
            for index, item in enumerate(angular)
        ),
        completed=completed,
        failure_reason=failure_reason,
    )


def load_raw_flight_trace(path: str | Path, /) -> RawFlightTrace:
    """Load an exact raw-trace-v1 JSON payload."""

    trace_path = Path(path)
    try:
        return raw_flight_trace_from_payload(json.loads(trace_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load raw flight trace {trace_path}: {exc}") from exc


def _command_event(value: Any, name: str, /) -> Px4CommandEvent:
    payload = _object(value, name)
    _exact_dataclass(payload, Px4CommandEvent, name)
    return Px4CommandEvent(
        timestamp_us=_integer(payload["timestamp_us"], f"{name}.timestamp_us", positive=True),
        command_id=_integer(payload["command_id"], f"{name}.command_id", positive=True),
        ack_timestamp_us=_integer(
            payload["ack_timestamp_us"], f"{name}.ack_timestamp_us", positive=True
        ),
        ack_result=_integer(payload["ack_result"], f"{name}.ack_result"),
    )


def _mode_event(value: Any, name: str, /) -> Px4ModeEvent:
    payload = _object(value, name)
    _exact_dataclass(payload, Px4ModeEvent, name)
    return Px4ModeEvent(
        timestamp_us=_integer(payload["timestamp_us"], f"{name}.timestamp_us", positive=True),
        nav_state=_integer(payload["nav_state"], f"{name}.nav_state"),
        mode=_string(payload["mode"], f"{name}.mode"),
    )


def _goal_event(value: Any, name: str, /) -> Px4PositionGoalEvent:
    payload = _object(value, name)
    _exact_dataclass(payload, Px4PositionGoalEvent, name)
    return Px4PositionGoalEvent(
        timestamp_us=_integer(payload["timestamp_us"], f"{name}.timestamp_us", positive=True),
        latitude_deg=_finite(payload["latitude_deg"], f"{name}.latitude_deg"),
        longitude_deg=_finite(payload["longitude_deg"], f"{name}.longitude_deg"),
        altitude_m=_finite(payload["altitude_m"], f"{name}.altitude_m"),
        yaw_valid=_boolean(payload["yaw_valid"], f"{name}.yaw_valid"),
        yaw_rad=_finite(payload["yaw_rad"], f"{name}.yaw_rad"),
    )


def hover_verification_from_payload(value: Any, /) -> HoverVerification:
    """Validate an exact PX4 hover-verification-v1 payload."""

    payload = _object(value, "hover verification")
    _exact_dataclass(payload, HoverVerification, "hover verification")
    if payload["artifact_kind"] != _VERIFICATION_KIND or payload["schema_version"] != 1:
        raise ValueError("hover verification is not the supported verification-v1 contract")
    fingerprint = _digest(payload["verification_fingerprint"], "verification_fingerprint")
    raw_commands = payload["command_events"]
    raw_modes = payload["mode_events"]
    raw_goals = payload["position_goal_events"]
    if (
        not isinstance(raw_commands, list)
        or not isinstance(raw_modes, list)
        or not isinstance(raw_goals, list)
    ):
        raise ValueError("hover verification event fields must be lists")
    commands = tuple(
        _command_event(item, f"command_events[{index}]") for index, item in enumerate(raw_commands)
    )
    modes = tuple(
        _mode_event(item, f"mode_events[{index}]") for index, item in enumerate(raw_modes)
    )
    goals = tuple(
        _goal_event(item, f"position_goal_events[{index}]") for index, item in enumerate(raw_goals)
    )
    if not commands or not modes or not goals:
        raise ValueError("hover verification requires command, mode, and position-goal evidence")
    loiter_enter_us = _integer(payload["loiter_enter_us"], "loiter_enter_us", positive=True)
    measurement_start_us = _integer(
        payload["measurement_start_us"], "measurement_start_us", positive=True
    )
    measurement_end_us = _integer(
        payload["measurement_end_us"], "measurement_end_us", positive=True
    )
    if not loiter_enter_us <= measurement_start_us < measurement_end_us:
        raise ValueError("hover verification timestamps are not ordered")
    return HoverVerification(
        artifact_kind=_VERIFICATION_KIND,
        schema_version=1,
        flight_parameters_id=_string(payload["flight_parameters_id"], "flight_parameters_id"),
        campaign_run_id=_string(payload["campaign_run_id"], "campaign_run_id"),
        trial_index=_integer(payload["trial_index"], "trial_index", positive=True),
        config_fingerprint=_string(payload["config_fingerprint"], "config_fingerprint"),
        px4_software_version=_string(payload["px4_software_version"], "px4_software_version"),
        raw_trace_sha256=_digest(payload["raw_trace_sha256"], "raw_trace_sha256"),
        ulog_sha256=_digest(payload["ulog_sha256"], "ulog_sha256"),
        command_events=commands,
        mode_events=modes,
        position_goal_events=goals,
        loiter_enter_us=loiter_enter_us,
        measurement_start_us=measurement_start_us,
        measurement_end_us=measurement_end_us,
        mode_report_max_age_us=_integer(
            payload["mode_report_max_age_us"], "mode_report_max_age_us", positive=True
        ),
        control_mode_report_count=_integer(
            payload["control_mode_report_count"], "control_mode_report_count", positive=True
        ),
        land_state_report_count=_integer(
            payload["land_state_report_count"], "land_state_report_count", positive=True
        ),
        requested_target_position_enu_m=_vector(
            payload["requested_target_position_enu_m"], "requested_target_position_enu_m"
        ),
        effective_target_position_enu_m=_vector(
            payload["effective_target_position_enu_m"], "effective_target_position_enu_m"
        ),
        effective_target_attitude_flu_enu_wxyz=_quaternion(
            payload["effective_target_attitude_flu_enu_wxyz"],
            "effective_target_attitude_flu_enu_wxyz",
        ),
        px4_yaw_target_ned_rad=_finite(payload["px4_yaw_target_ned_rad"], "px4_yaw_target_ned_rad"),
        verification_fingerprint=fingerprint,
    )


def load_hover_verification(path: str | Path, /) -> HoverVerification:
    """Load an exact PX4 hover-verification-v1 JSON payload."""

    verification_path = Path(path)
    try:
        return hover_verification_from_payload(
            json.loads(verification_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load hover verification {verification_path}: {exc}") from exc


def load_flight_trace(path: str | Path, /) -> FlightTrace:
    """Load an exact trace-v1 JSON payload."""

    trace_path = Path(path)
    try:
        return flight_trace_from_payload(json.loads(trace_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load flight trace {trace_path}: {exc}") from exc


def _vector_bounds(value: Any, name: str, /) -> VectorBounds:
    payload = _object(value, name)
    _exact_dataclass(payload, VectorBounds, name)
    result = VectorBounds(
        observed_min=_vector(payload["observed_min"], f"{name}.observed_min"),
        observed_max=_vector(payload["observed_max"], f"{name}.observed_max"),
        lower=_vector(payload["lower"], f"{name}.lower"),
        upper=_vector(payload["upper"], f"{name}.upper"),
    )
    if any(
        not lower <= observed_min <= observed_max <= upper
        for lower, observed_min, observed_max, upper in zip(
            result.lower, result.observed_min, result.observed_max, result.upper, strict=True
        )
    ):
        raise ValueError(f"{name} bounds must contain ordered observed extrema")
    return result


def _upper_bound(value: Any, name: str, /) -> UpperBound:
    payload = _object(value, name)
    _exact_dataclass(payload, UpperBound, name)
    observed = _finite(payload["observed_max"], f"{name}.observed_max")
    upper = _finite(payload["upper"], f"{name}.upper", positive=True)
    if observed < 0.0 or observed > upper:
        raise ValueError(f"{name} must satisfy 0 <= observed_max <= upper")
    return UpperBound(observed_max=observed, upper=upper)


def _signal_statistics(value: Any, name: str, /) -> SignalStatistics:
    payload = _object(value, name)
    _exact_dataclass(payload, SignalStatistics, name)
    minimum = _finite(payload["minimum"], f"{name}.minimum")
    maximum = _finite(payload["maximum"], f"{name}.maximum")
    maximum_absolute = _finite(payload["maximum_absolute"], f"{name}.maximum_absolute")
    rmse = _finite(payload["rmse"], f"{name}.rmse")
    p95_absolute = _finite(payload["p95_absolute"], f"{name}.p95_absolute")
    p99_absolute = _finite(payload["p99_absolute"], f"{name}.p99_absolute")
    if minimum > maximum:
        raise ValueError(f"{name} minimum must not exceed maximum")
    if min(maximum_absolute, rmse, p95_absolute, p99_absolute) < 0.0:
        raise ValueError(f"{name} absolute statistics must be non-negative")
    if maximum_absolute < max(abs(minimum), abs(maximum)):
        raise ValueError(f"{name} maximum_absolute must contain the signed extrema")
    if p95_absolute > p99_absolute or p99_absolute > maximum_absolute:
        raise ValueError(f"{name} percentiles must be ordered below maximum_absolute")
    return SignalStatistics(
        name=_string(payload["name"], f"{name}.name"),
        unit=_string(payload["unit"], f"{name}.unit"),
        minimum=minimum,
        maximum=maximum,
        maximum_absolute=maximum_absolute,
        rmse=rmse,
        p95_absolute=p95_absolute,
        p99_absolute=p99_absolute,
    )


def _statistics_list(value: Any, name: str, /) -> tuple[SignalStatistics, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_signal_statistics(item, f"{name}[{index}]") for index, item in enumerate(value))
    names = tuple(item.name for item in result)
    if len(set(names)) != len(names):
        raise ValueError(f"{name} signal names must be unique")
    return result


def flight_parameters_from_payload(value: Any, /) -> FlightParameters:
    """Validate the exact numeric hover summary shared by alignment and policies."""
    payload = _object(value, "flight parameters")
    _exact_dataclass(payload, FlightParameters, "flight parameters")
    hover_payload = _object(payload["hover"], "hover")
    _exact_dataclass(hover_payload, HoverParameters, "hover")
    hover = HoverParameters(
        position_error_enu_m=_vector_bounds(hover_payload["position_error_enu_m"], "position"),
        attitude_error_rpy_rad=_vector_bounds(hover_payload["attitude_error_rpy_rad"], "attitude"),
        horizontal_position_error_m=_upper_bound(
            hover_payload["horizontal_position_error_m"], "horizontal"
        ),
        position_error_m=_upper_bound(hover_payload["position_error_m"], "position norm"),
        tilt_error_rad=_upper_bound(hover_payload["tilt_error_rad"], "tilt"),
        orientation_error_rad=_upper_bound(hover_payload["orientation_error_rad"], "orientation"),
        acceptance_limits=_statistics_list(hover_payload["acceptance_limits"], "acceptance limits"),
    )
    return FlightParameters(
        run_id=_string(payload["run_id"], "run_id"),
        sample_rate_hz=_finite(payload["sample_rate_hz"], "sample_rate_hz", positive=True),
        policy_warmup_s=_finite(payload["policy_warmup_s"], "policy_warmup_s", positive=True),
        policy_window_s=_finite(payload["policy_window_s"], "policy_window_s", positive=True),
        position_margin_m=_finite(payload["position_margin_m"], "position_margin_m", positive=True),
        attitude_margin_rad=_finite(
            payload["attitude_margin_rad"], "attitude_margin_rad", positive=True
        ),
        hover=hover,
    )

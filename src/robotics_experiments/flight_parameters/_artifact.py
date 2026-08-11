"""Strict flight trace/envelope IO, bundle validation, and publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from robotics_experiments.flight_parameters._models import (
    FlightEnvelope,
    FlightEnvelopeBundle,
    FlightTrace,
    HoverEnvelope,
    HoverVerification,
    Px4CommandEvent,
    Px4ModeEvent,
    Px4PositionGoalEvent,
    RawFlightTrace,
    UpperBound,
    VectorBounds,
)

__all__: list[str] = []

_TRACE_KIND = "robotics-experiments-flight-trace"
_RAW_TRACE_KIND = "robotics-experiments-raw-flight-trace"
_VERIFICATION_KIND = "robotics-experiments-px4-hover-verification"
_ENVELOPE_KIND = "robotics-experiments-flight-envelope"


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


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, name: str, /) -> str:
    result = _string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _artifact_fingerprint(payload: dict[str, Any], /) -> str:
    """Return the canonical SHA-256 for an envelope payload without its digest."""

    value = dict(payload)
    value.pop("artifact_fingerprint", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verification_fingerprint(payload: dict[str, Any], /) -> str:
    value = dict(payload)
    value.pop("verification_fingerprint", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _evidence_fingerprint(fingerprints: tuple[str, ...], /) -> str:
    encoded = json.dumps(fingerprints, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def flight_trace_from_payload(value: Any, /) -> FlightTrace:
    """Validate an in-memory exact trace-v1 payload."""

    payload = _object(value, "flight trace")
    _exact_dataclass(payload, FlightTrace, "flight trace")
    if payload["artifact_kind"] != _TRACE_KIND or payload["schema_version"] != 1:
        raise ValueError("flight trace is not the supported trace-v1 contract")
    time = payload["time_s"]
    positions = payload["position_enu_m"]
    attitudes = payload["attitude_flu_enu_wxyz"]
    if not isinstance(time, list) or not time:
        raise ValueError("flight trace time_s must be a non-empty list")
    time_values = tuple(_finite(item, f"time_s[{index}]") for index, item in enumerate(time))
    if time_values[0] != 0.0 or any(
        later <= earlier for earlier, later in zip(time_values, time_values[1:], strict=False)
    ):
        raise ValueError("flight trace timestamps must start at zero and increase")
    if not isinstance(positions, list) or not isinstance(attitudes, list):
        raise ValueError("flight trace samples must be lists")
    if len(time_values) != len(positions) or len(time_values) != len(attitudes):
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
        flight_id=_string(payload["flight_id"], "flight_id"),
        campaign_run_id=_string(payload["campaign_run_id"], "campaign_run_id"),
        trial_index=_integer(payload["trial_index"], "trial_index", positive=True),
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
    if not isinstance(raw_time, list) or not raw_time:
        raise ValueError("raw flight trace time_us must be a non-empty list")
    time_us = tuple(
        _integer(item, f"time_us[{index}]", positive=True) for index, item in enumerate(raw_time)
    )
    if any(later <= earlier for earlier, later in zip(time_us, time_us[1:], strict=False)):
        raise ValueError("raw flight trace timestamps must increase")
    if not isinstance(positions, list) or not isinstance(attitudes, list):
        raise ValueError("raw flight trace samples must be lists")
    if len(time_us) != len(positions) or len(time_us) != len(attitudes):
        raise ValueError("raw flight trace sample arrays must have equal lengths")
    completed = _boolean(payload["completed"], "completed")
    failure_reason = _string(payload["failure_reason"], "failure_reason", allow_empty=True)
    if completed == bool(failure_reason):
        raise ValueError("complete raw traces have no failure reason; failed traces require one")
    return RawFlightTrace(
        artifact_kind=_RAW_TRACE_KIND,
        schema_version=1,
        flight_id=_string(payload["flight_id"], "flight_id"),
        campaign_run_id=_string(payload["campaign_run_id"], "campaign_run_id"),
        trial_index=_integer(payload["trial_index"], "trial_index", positive=True),
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
    if fingerprint != _verification_fingerprint(payload):
        raise ValueError("hover verification fingerprint does not match its payload")
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
        flight_id=_string(payload["flight_id"], "flight_id"),
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


def flight_envelope_from_payload(value: Any, /) -> FlightEnvelope:
    """Validate an in-memory exact flight-envelope-v1 payload."""

    payload = _object(value, "flight envelope")
    _exact_dataclass(payload, FlightEnvelope, "flight envelope")
    if payload["artifact_kind"] != _ENVELOPE_KIND or payload["schema_version"] != 1:
        raise ValueError("flight envelope is not the supported envelope-v1 contract")
    if payload["scenario_kind"] != "hover":
        raise ValueError("flight envelope currently supports only hover")
    fingerprint = _string(payload["artifact_fingerprint"], "artifact_fingerprint")
    if fingerprint != _artifact_fingerprint(payload):
        raise ValueError("flight envelope artifact_fingerprint does not match its payload")
    hover_payload = _object(payload["hover"], "hover")
    _exact_dataclass(hover_payload, HoverEnvelope, "hover")
    hover = HoverEnvelope(
        position_error_enu_m=_vector_bounds(
            hover_payload["position_error_enu_m"], "hover.position_error_enu_m"
        ),
        horizontal_position_error_m=_upper_bound(
            hover_payload["horizontal_position_error_m"],
            "hover.horizontal_position_error_m",
        ),
        position_error_m=_upper_bound(hover_payload["position_error_m"], "hover.position_error_m"),
        attitude_error_rpy_rad=_vector_bounds(
            hover_payload["attitude_error_rpy_rad"], "hover.attitude_error_rpy_rad"
        ),
        tilt_error_rad=_upper_bound(hover_payload["tilt_error_rad"], "hover.tilt_error_rad"),
        orientation_error_rad=_upper_bound(
            hover_payload["orientation_error_rad"], "hover.orientation_error_rad"
        ),
    )
    if type(payload["source_dirty"]) is not bool:
        raise ValueError("source_dirty must be boolean")
    return FlightEnvelope(
        artifact_kind=_ENVELOPE_KIND,
        schema_version=1,
        flight_id=_string(payload["flight_id"], "flight_id"),
        run_id=_string(payload["run_id"], "run_id"),
        scenario_kind="hover",
        source_backend=_string(payload["source_backend"], "source_backend"),
        state_source=_string(payload["state_source"], "state_source"),
        experiment_config=_string(payload["experiment_config"], "experiment_config"),
        vehicle=_string(payload["vehicle"], "vehicle"),
        config_fingerprint=_string(payload["config_fingerprint"], "config_fingerprint"),
        experiment_fingerprint=_string(payload["experiment_fingerprint"], "experiment_fingerprint"),
        vehicle_fingerprint=_string(payload["vehicle_fingerprint"], "vehicle_fingerprint"),
        source_image=_string(payload["source_image"], "source_image"),
        source_commit=_string(payload["source_commit"], "source_commit"),
        source_dirty=payload["source_dirty"],
        evidence_fingerprint=_string(payload["evidence_fingerprint"], "evidence_fingerprint"),
        trial_count=_integer(payload["trial_count"], "trial_count", positive=True),
        sample_count=_integer(payload["sample_count"], "sample_count", positive=True),
        settle_s=_finite(payload["settle_s"], "settle_s", positive=True),
        measure_s=_finite(payload["measure_s"], "measure_s", positive=True),
        sample_rate_hz=_finite(payload["sample_rate_hz"], "sample_rate_hz", positive=True),
        position_margin_m=_finite(payload["position_margin_m"], "position_margin_m", positive=True),
        attitude_margin_rad=_finite(
            payload["attitude_margin_rad"], "attitude_margin_rad", positive=True
        ),
        hover=hover,
        artifact_fingerprint=fingerprint,
    )


def _resolve_flight_envelope_path(path: str | Path, /) -> Path:
    """Resolve an envelope file or immutable/publication bundle directory."""

    artifact_path = Path(path)
    return artifact_path / "flight_envelope.json" if artifact_path.is_dir() else artifact_path


def load_flight_envelope(path: str | Path, /) -> FlightEnvelope:
    """Load an exact flight-envelope-v1 artifact."""

    artifact_path = _resolve_flight_envelope_path(path)
    try:
        return flight_envelope_from_payload(json.loads(artifact_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load flight envelope {artifact_path}: {exc}") from exc


def write_flight_envelope(*, envelope: FlightEnvelope, run_dir: Path) -> Path:
    """Write one new immutable envelope payload."""

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "flight_envelope.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(asdict(envelope), stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
    load_flight_envelope(path)
    return path


def finalize_flight_envelope_bundle(run_dir: Path, /) -> Path:
    """Inventory and validate every immutable payload in a completed run."""

    envelope = load_flight_envelope(run_dir)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"flight envelope manifest already exists: {manifest_path}")
    payloads = sorted(path for path in run_dir.rglob("*") if path.is_file())
    if not payloads or not (run_dir / "flight_envelope.json").is_file():
        raise ValueError("flight envelope bundle has no envelope payload")
    manifest = {
        "schema_version": 1,
        "flight_id": envelope.flight_id,
        "run_id": envelope.run_id,
        "files": [
            {
                "role": "flight-envelope" if path.name == "flight_envelope.json" else "evidence",
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in payloads
        ],
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
    validate_flight_envelope_bundle(run_dir)
    return manifest_path


def validate_flight_envelope_bundle(run_dir: str | Path, /) -> None:
    """Validate envelope identity plus exact manifest coverage and hashes."""

    bundle = Path(run_dir)
    envelope = load_flight_envelope(bundle)
    manifest_path = bundle / "manifest.json"
    try:
        manifest = _object(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load flight envelope manifest {manifest_path}: {exc}") from exc
    if set(manifest) != {"schema_version", "flight_id", "run_id", "files"}:
        raise ValueError("flight envelope manifest fields mismatch")
    if (
        manifest["schema_version"] != 1
        or manifest["flight_id"] != envelope.flight_id
        or manifest["run_id"] != envelope.run_id
    ):
        raise ValueError("flight envelope manifest identity mismatch")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("flight envelope manifest files must be a non-empty list")
    inventoried: set[Path] = set()
    for index, raw_entry in enumerate(entries):
        entry = _object(raw_entry, f"manifest.files[{index}]")
        if set(entry) != {"role", "path", "size_bytes", "sha256"}:
            raise ValueError(f"manifest.files[{index}] fields mismatch")
        relative = Path(_string(entry["path"], f"manifest.files[{index}].path"))
        if relative.is_absolute() or ".." in relative.parts or relative == Path("manifest.json"):
            raise ValueError("flight envelope manifest paths must be relative payload paths")
        if relative in inventoried:
            raise ValueError(f"flight envelope manifest duplicates {relative}")
        inventoried.add(relative)
        path = bundle / relative
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"flight envelope manifest size mismatch for {relative}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"flight envelope manifest hash mismatch for {relative}")
    payloads = {
        path.relative_to(bundle)
        for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if inventoried != payloads:
        raise ValueError("flight envelope manifest does not inventory every payload exactly once")
    verification_fingerprints = []
    for trial_index in range(1, envelope.trial_count + 1):
        trial = bundle / "evidence" / f"trial-{trial_index:03d}"
        raw_path = trial / "raw_trace.json"
        ulog_path = trial / "flight.ulg"
        bridge_path = trial / "bridge.log"
        verification_path = trial / "hover_verification.json"
        trace_path = trial / "trace.json"
        if not all(
            path.is_file()
            for path in (raw_path, ulog_path, bridge_path, verification_path, trace_path)
        ):
            raise ValueError(f"flight envelope trial {trial_index} lacks verified hover evidence")
        raw = load_raw_flight_trace(raw_path)
        verification = load_hover_verification(verification_path)
        trace = load_flight_trace(trace_path)
        if (
            raw.flight_id != envelope.flight_id
            or raw.campaign_run_id != envelope.run_id
            or raw.trial_index != trial_index
            or verification.flight_id != envelope.flight_id
            or verification.campaign_run_id != envelope.run_id
            or verification.trial_index != trial_index
            or trace.flight_id != envelope.flight_id
            or trace.campaign_run_id != envelope.run_id
            or trace.trial_index != trial_index
        ):
            raise ValueError(f"flight envelope trial {trial_index} evidence identity mismatch")
        if verification.raw_trace_sha256 != _sha256(raw_path):
            raise ValueError(f"flight envelope trial {trial_index} raw trace digest mismatch")
        if verification.ulog_sha256 != _sha256(ulog_path):
            raise ValueError(f"flight envelope trial {trial_index} ULog digest mismatch")
        if trace.verification_fingerprint != verification.verification_fingerprint:
            raise ValueError(f"flight envelope trial {trial_index} verification link mismatch")
        verification_fingerprints.append(verification.verification_fingerprint)
    if envelope.evidence_fingerprint != _evidence_fingerprint(tuple(verification_fingerprints)):
        raise ValueError("flight envelope evidence fingerprint mismatch")


def load_flight_envelope_bundle(*, path: str | Path) -> FlightEnvelopeBundle:
    """Resolve and validate one immutable measurement bundle exactly once."""

    run_dir = Path(path).resolve(strict=True)
    if not run_dir.is_dir():
        raise ValueError(f"flight envelope bundle must be a directory, got {run_dir}")
    validate_flight_envelope_bundle(run_dir)
    envelope_path = run_dir / "flight_envelope.json"
    manifest_path = run_dir / "manifest.json"
    return FlightEnvelopeBundle(
        run_dir=run_dir,
        envelope=load_flight_envelope(envelope_path),
        envelope_digest=_sha256(envelope_path),
        manifest_digest=_sha256(manifest_path),
    )


def publish_flight_envelope(run_dir: str | Path, /) -> Path:
    """Atomically advance the relative ``latest`` link after bundle validation."""

    immutable_run = Path(run_dir).resolve()
    validate_flight_envelope_bundle(immutable_run)
    root = immutable_run.parent
    link = root / "latest"
    relative_target = os.path.relpath(immutable_run, start=root)
    temporary = root / f".latest.{os.getpid()}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(relative_target, target_is_directory=True)
    os.replace(temporary, link)
    return link

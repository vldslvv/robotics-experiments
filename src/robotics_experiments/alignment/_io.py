"""YAML/JSON loading, validation, fingerprinting, and publication."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import mmap
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from robotics_experiments.alignment._metrics import metric
from robotics_experiments.alignment._models import (
    AlignmentArtifact,
    CandidateScore,
    CommandInterface,
    ControllerDefinition,
    ControllerSettings,
    ParameterSpec,
    ProbeCommand,
    ProbeProfile,
    ProbeTrace,
    SearchSpec,
    SettingValue,
)

__all__: list[str] = []

_TRACE_SCHEMA_VERSION = 3
_SUPPORTED_TRACE_SCHEMA_VERSIONS = frozenset({2, 3})
_ARTIFACT_SCHEMA_VERSION = 4
_SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = frozenset({2, 3, 4})
_BACKENDS = frozenset({"isaac-lab", "px4-sitl"})
_INTERFACE_KIND = "body-flu-velocity-yaw-rate-v1"
_SEARCH_STRATEGIES = frozenset({"staged-beam-v1", "staged-beam-v2"})
_SETTINGS_FIELDS = {item.name for item in fields(ControllerSettings)}
_CANDIDATE_ENTRY = re.compile(rb'(?m)^    "([0-9a-f]{64})": ')
_CANDIDATES_MARKER = b'  "candidates": {'
_FAILURES_MARKER = b'  "failures": {'


def _canonical_json(value: Any, /) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _fingerprint(value: Any, /) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _object(*, value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _string(*, value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(*, value: Any, name: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _vector(*, value: Any, name: str, length: int = 3) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    parsed = tuple(_finite(value=item, name=f"{name}[{index}]") for index, item in enumerate(value))
    return parsed


def _settings(
    *,
    value: Any,
    name: str,
    legacy_rate_output_scale: bool = False,
    legacy_hover_thrust: bool = False,
) -> ControllerSettings:
    settings = _object(value=value, name=name)
    if legacy_rate_output_scale and "rate_output_scale_nm" not in settings:
        settings["rate_output_scale_nm"] = [1.0, 1.0, 1.0]
    if legacy_hover_thrust and "hover_thrust_fraction" not in settings:
        # Zero is an internal legacy sentinel for the exact physical
        # mass*(acceleration+gravity) feed-forward used by schema 2 and 3.
        settings["hover_thrust_fraction"] = 0.0
    missing = _SETTINGS_FIELDS - settings.keys()
    extra = settings.keys() - _SETTINGS_FIELDS
    if missing or extra:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    parsed = ControllerSettings(
        command_delay_s=_finite(
            value=settings["command_delay_s"],
            name=f"{name}.command_delay_s",
            nonnegative=True,
        ),
        kp_vel=_vector(value=settings["kp_vel"], name=f"{name}.kp_vel"),
        ki_vel=_vector(value=settings["ki_vel"], name=f"{name}.ki_vel"),
        kd_vel=_vector(value=settings["kd_vel"], name=f"{name}.kd_vel"),
        velocity_filter_hz=_finite(
            value=settings["velocity_filter_hz"],
            name=f"{name}.velocity_filter_hz",
            nonnegative=True,
        ),
        velocity_derivative_filter_hz=_finite(
            value=settings["velocity_derivative_filter_hz"],
            name=f"{name}.velocity_derivative_filter_hz",
            positive=True,
        ),
        velocity_integral_limit_mps2=_vector(
            value=settings["velocity_integral_limit_mps2"],
            name=f"{name}.velocity_integral_limit_mps2",
        ),
        hover_thrust_fraction=_finite(
            value=settings["hover_thrust_fraction"],
            name=f"{name}.hover_thrust_fraction",
            nonnegative=legacy_hover_thrust,
            positive=not legacy_hover_thrust,
        ),
        kp_att=_vector(value=settings["kp_att"], name=f"{name}.kp_att"),
        attitude_yaw_weight=_finite(
            value=settings["attitude_yaw_weight"],
            name=f"{name}.attitude_yaw_weight",
            nonnegative=True,
        ),
        kp_rate=_vector(value=settings["kp_rate"], name=f"{name}.kp_rate"),
        ki_rate=_vector(value=settings["ki_rate"], name=f"{name}.ki_rate"),
        kd_rate=_vector(value=settings["kd_rate"], name=f"{name}.kd_rate"),
        kff_rate=_vector(value=settings["kff_rate"], name=f"{name}.kff_rate"),
        rate_output_scale_nm=_vector(
            value=settings["rate_output_scale_nm"],
            name=f"{name}.rate_output_scale_nm",
        ),
        rate_derivative_filter_hz=_finite(
            value=settings["rate_derivative_filter_hz"],
            name=f"{name}.rate_derivative_filter_hz",
            positive=True,
        ),
        rate_integral_limit_nm=_vector(
            value=settings["rate_integral_limit_nm"],
            name=f"{name}.rate_integral_limit_nm",
        ),
        max_tilt_cmd_deg=_finite(
            value=settings["max_tilt_cmd_deg"],
            name=f"{name}.max_tilt_cmd_deg",
            positive=True,
        ),
        min_collective_thrust_fraction=_finite(
            value=settings["min_collective_thrust_fraction"],
            name=f"{name}.min_collective_thrust_fraction",
            nonnegative=True,
        ),
        max_collective_thrust_fraction=_finite(
            value=settings["max_collective_thrust_fraction"],
            name=f"{name}.max_collective_thrust_fraction",
            positive=True,
        ),
        horizontal_thrust_margin_fraction=_finite(
            value=settings["horizontal_thrust_margin_fraction"],
            name=f"{name}.horizontal_thrust_margin_fraction",
            nonnegative=True,
        ),
        thruster_tau_inc_s=_finite(
            value=settings["thruster_tau_inc_s"],
            name=f"{name}.thruster_tau_inc_s",
            nonnegative=True,
        ),
        thruster_tau_dec_s=_finite(
            value=settings["thruster_tau_dec_s"],
            name=f"{name}.thruster_tau_dec_s",
            nonnegative=True,
        ),
        thruster_max_thrust_rate_nps=_finite(
            value=settings["thruster_max_thrust_rate_nps"],
            name=f"{name}.thruster_max_thrust_rate_nps",
            positive=True,
        ),
    )
    for field_name in (
        "kp_vel",
        "ki_vel",
        "kd_vel",
        "velocity_integral_limit_mps2",
        "kp_att",
        "kp_rate",
        "ki_rate",
        "kd_rate",
        "kff_rate",
        "rate_output_scale_nm",
        "rate_integral_limit_nm",
    ):
        if any(value < 0.0 for value in getattr(parsed, field_name)):
            raise ValueError(f"{name}.{field_name} values must be non-negative")
    if not 0.0 <= parsed.attitude_yaw_weight <= 1.0:
        raise ValueError(f"{name}.attitude_yaw_weight must be in [0, 1]")
    if parsed.hover_thrust_fraction > 1.0:
        raise ValueError(f"{name}.hover_thrust_fraction must not exceed 1")
    if not (
        0.0 <= parsed.min_collective_thrust_fraction < parsed.max_collective_thrust_fraction <= 1.0
    ):
        raise ValueError(f"{name} collective thrust fractions must satisfy 0 <= min < max <= 1")
    if parsed.horizontal_thrust_margin_fraction >= parsed.max_collective_thrust_fraction:
        raise ValueError(
            f"{name}.horizontal_thrust_margin_fraction must be below the maximum "
            "collective thrust fraction"
        )
    return parsed


def _interface(*, value: Any, name: str) -> CommandInterface:
    interface = _object(value=value, name=name)
    kind = _string(value=interface.get("kind"), name=f"{name}.kind")
    if kind != _INTERFACE_KIND:
        raise ValueError(f"unsupported {name}.kind={kind!r}")
    dimension = interface.get("dimension")
    if type(dimension) is not int or dimension != 4:
        raise ValueError(f"{name}.dimension must be 4")
    frame = _string(value=interface.get("frame"), name=f"{name}.frame")
    if frame != "body-flu":
        raise ValueError(f"unsupported {name}.frame={frame!r}")
    return CommandInterface(
        kind=kind,
        dimension=dimension,
        frame=frame,
        linear_scale_mps=_finite(
            value=interface.get("linear_scale_mps"),
            name=f"{name}.linear_scale_mps",
            positive=True,
        ),
        yaw_rate_scale_radps=_finite(
            value=interface.get("yaw_rate_scale_radps"),
            name=f"{name}.yaw_rate_scale_radps",
            positive=True,
        ),
    )


def _profiles(*, value: Any) -> tuple[ProbeProfile, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("probes must be a non-empty list")
    profiles: list[ProbeProfile] = []
    names: set[str] = set()
    for index, raw_profile in enumerate(value):
        name = f"probes[{index}]"
        profile = _object(value=raw_profile, name=name)
        profile_name = _string(value=profile.get("name"), name=f"{name}.name")
        if profile_name in names:
            raise ValueError(f"duplicate probe name {profile_name!r}")
        names.add(profile_name)
        split = _string(value=profile.get("split"), name=f"{name}.split")
        if split not in ("fit", "validation", "stability"):
            raise ValueError(f"{name}.split must be 'fit', 'validation', or 'stability'")
        axis = _string(value=profile.get("axis"), name=f"{name}.axis")
        if axis not in ("x", "y", "z", "yaw", "combined"):
            raise ValueError(f"{name}.axis must be x, y, z, yaw, or combined")
        raw_commands = profile.get("commands")
        if not isinstance(raw_commands, list) or not raw_commands:
            raise ValueError(f"{name}.commands must be a non-empty list")
        commands = tuple(
            ProbeCommand(
                velocity_body_flu_mps=_vector(
                    value=_object(value=command, name=f"{name}.commands[{command_index}]").get(
                        "velocity_body_flu_mps"
                    ),
                    name=f"{name}.commands[{command_index}].velocity_body_flu_mps",
                ),
                yaw_rate_radps=_finite(
                    value=_object(
                        value=command,
                        name=f"{name}.commands[{command_index}]",
                    ).get("yaw_rate_radps"),
                    name=f"{name}.commands[{command_index}].yaw_rate_radps",
                ),
                duration_s=_finite(
                    value=_object(
                        value=command,
                        name=f"{name}.commands[{command_index}]",
                    ).get("duration_s"),
                    name=f"{name}.commands[{command_index}].duration_s",
                    positive=True,
                ),
            )
            for command_index, command in enumerate(raw_commands)
        )
        profiles.append(
            ProbeProfile(
                name=profile_name,
                split=split,
                axis=axis,
                commands=commands,
            )
        )
    covered = {(profile.split, profile.axis) for profile in profiles}
    expected = {
        (split, axis)
        for split in ("fit", "validation")
        for axis in ("x", "y", "z", "yaw", "combined")
    }
    if not expected <= covered:
        raise ValueError(
            "probes must cover x, y, z, yaw, and combined behavior for fit and validation; "
            f"missing={sorted(expected - covered)!r}"
        )
    return tuple(profiles)


def _candidate(*, value: Any, template: SettingValue, name: str) -> SettingValue:
    if isinstance(template, tuple):
        return _vector(value=value, name=name)
    return _finite(value=value, name=name)


def _parameters(
    *,
    value: Any,
    base_settings: ControllerSettings,
) -> tuple[ParameterSpec, ...]:
    if not isinstance(value, list):
        raise ValueError("parameters must be a list")
    parameters: list[ParameterSpec] = []
    names: set[str] = set()
    for index, raw_parameter in enumerate(value):
        name = f"parameters[{index}]"
        parameter = _object(value=raw_parameter, name=name)
        field_name = _string(value=parameter.get("field"), name=f"{name}.field")
        if field_name not in _SETTINGS_FIELDS:
            raise ValueError(f"{name}.field names unknown ControllerSettings field {field_name!r}")
        if field_name in names:
            raise ValueError(f"duplicate fitted settings field {field_name!r}")
        names.add(field_name)
        raw_candidates = parameter.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"{name}.candidates must be a non-empty list")
        template = getattr(base_settings, field_name)
        parameters.append(
            ParameterSpec(
                field=field_name,
                stage=_positive_integer(
                    value=parameter.get("stage"),
                    name=f"{name}.stage",
                ),
                candidates=tuple(
                    _candidate(
                        value=candidate,
                        template=template,
                        name=f"{name}.candidates[{candidate_index}]",
                    )
                    for candidate_index, candidate in enumerate(raw_candidates)
                ),
            )
        )
    return tuple(parameters)


def _positive_integer(*, value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(*, value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _search(*, value: Any) -> SearchSpec:
    search = _object(value=value, name="search")
    strategy = _string(value=search.get("strategy"), name="search.strategy")
    if strategy not in _SEARCH_STRATEGIES:
        raise ValueError(f"unsupported search.strategy={strategy!r}")
    return SearchSpec(
        strategy=strategy,
        beam_width=_positive_integer(
            value=search.get("beam_width"),
            name="search.beam_width",
        ),
        passes_per_stage=_positive_integer(
            value=search.get("passes_per_stage"),
            name="search.passes_per_stage",
        ),
    )


def _trace(*, value: Any, name: str) -> ProbeTrace:
    trace = _object(value=value, name=name)
    failure_reason = trace.get("failure_reason")
    if failure_reason is not None and not isinstance(failure_reason, str):
        raise ValueError(f"{name}.failure_reason must be a string or null")
    completed = trace.get("completed")
    if type(completed) is not bool:
        raise ValueError(f"{name}.completed must be boolean")
    time = trace.get("time_s")
    commands = trace.get("command_velocity_body_flu_mps")
    yaw_commands = trace.get("command_yaw_rate_radps")
    positions = trace.get("position_enu_m")
    attitudes = trace.get("attitude_flu_enu_wxyz")
    velocities = trace.get("velocity_body_flu_mps")
    world_velocities = trace.get("velocity_enu_mps")
    angular_velocities = trace.get("angular_velocity_body_flu_radps")
    if (
        not isinstance(time, list)
        or not isinstance(commands, list)
        or not isinstance(yaw_commands, list)
        or not isinstance(positions, list)
        or not isinstance(attitudes, list)
        or not isinstance(velocities, list)
        or not isinstance(world_velocities, list)
        or not isinstance(angular_velocities, list)
    ):
        raise ValueError(f"{name} samples must be lists")
    sample_lengths = (
        len(commands),
        len(yaw_commands),
        len(positions),
        len(attitudes),
        len(velocities),
        len(world_velocities),
        len(angular_velocities),
    )
    if not time or any(length != len(time) for length in sample_lengths):
        raise ValueError(f"{name} sample lists must be non-empty and equal length")
    parsed_time = tuple(_finite(value=item, name=f"{name}.time_s") for item in time)
    if any(
        next_time <= current
        for current, next_time in zip(parsed_time, parsed_time[1:], strict=False)
    ):
        raise ValueError(f"{name}.time_s must be strictly increasing")
    return ProbeTrace(
        profile=_string(value=trace.get("profile"), name=f"{name}.profile"),
        split=_string(value=trace.get("split"), name=f"{name}.split"),
        axis=_string(value=trace.get("axis"), name=f"{name}.axis"),
        time_s=parsed_time,
        command_velocity_body_flu_mps=tuple(
            _vector(value=item, name=f"{name}.command_velocity_body_flu_mps") for item in commands
        ),
        command_yaw_rate_radps=tuple(
            _finite(value=item, name=f"{name}.command_yaw_rate_radps") for item in yaw_commands
        ),
        position_enu_m=tuple(
            _vector(value=item, name=f"{name}.position_enu_m") for item in positions
        ),
        attitude_flu_enu_wxyz=tuple(
            _vector(value=item, name=f"{name}.attitude_flu_enu_wxyz", length=4)
            for item in attitudes
        ),
        velocity_body_flu_mps=tuple(
            _vector(value=item, name=f"{name}.velocity_body_flu_mps") for item in velocities
        ),
        velocity_enu_mps=tuple(
            _vector(value=item, name=f"{name}.velocity_enu_mps") for item in world_velocities
        ),
        angular_velocity_body_flu_radps=tuple(
            _vector(value=item, name=f"{name}.angular_velocity_body_flu_radps")
            for item in angular_velocities
        ),
        initial_position_enu_m=_vector(
            value=trace.get("initial_position_enu_m"),
            name=f"{name}.initial_position_enu_m",
        ),
        initial_attitude_flu_enu_wxyz=_vector(
            value=trace.get("initial_attitude_flu_enu_wxyz"),
            name=f"{name}.initial_attitude_flu_enu_wxyz",
            length=4,
        ),
        initial_linear_velocity_enu_mps=_vector(
            value=trace.get("initial_linear_velocity_enu_mps"),
            name=f"{name}.initial_linear_velocity_enu_mps",
        ),
        initial_angular_velocity_body_flu_radps=_vector(
            value=trace.get("initial_angular_velocity_body_flu_radps"),
            name=f"{name}.initial_angular_velocity_body_flu_radps",
        ),
        completed=completed,
        start_sim_time_us=_nonnegative_integer(
            value=trace.get("start_sim_time_us", 0),
            name=f"{name}.start_sim_time_us",
        ),
        failure_reason=failure_reason,
    )


def _json_value_bounds(*, payload: mmap.mmap, start: int, end: int) -> tuple[int, int]:
    while start < end and payload[start] in b" \t\r\n":
        start += 1
    while start < end and payload[end - 1] in b" \t\r\n":
        end -= 1
    if start < end and payload[end - 1] == ord(","):
        end -= 1
        while start < end and payload[end - 1] in b" \t\r\n":
            end -= 1
    return start, end


def _index_candidate_diagnostics(
    *,
    path: Path,
) -> tuple[dict[str, Any], dict[str, tuple[int, int]], dict[str, str]]:
    try:
        with path.open("rb") as diagnostics:
            with mmap.mmap(diagnostics.fileno(), 0, access=mmap.ACCESS_READ) as payload:
                candidates_start = payload.find(_CANDIDATES_MARKER)
                failures_start = payload.find(_FAILURES_MARKER, candidates_start)
                if candidates_start < 0 or failures_start < 0:
                    raise ValueError("candidate diagnostics must use the generated JSON layout")
                candidates_end = payload.rfind(b"\n  },", candidates_start, failures_start)
                failures_content_start = failures_start + len(_FAILURES_MARKER)
                failures_end = (
                    failures_content_start
                    if payload[failures_content_start] == ord("}")
                    else payload.rfind(b"\n  }\n}", failures_start)
                )
                if candidates_end < 0 or failures_end < 0:
                    raise ValueError("candidate diagnostics must use the generated JSON layout")
                header = json.loads(
                    payload[:candidates_start] + b'  "candidates": {},\n  "failures": {}\n}'
                )
                metadata = _object(value=header, name=str(path))

                candidate_matches = tuple(
                    _CANDIDATE_ENTRY.finditer(
                        payload,
                        candidates_start + len(_CANDIDATES_MARKER),
                        candidates_end,
                    )
                )
                candidates: dict[str, tuple[int, int]] = {}
                for index, match in enumerate(candidate_matches):
                    end = (
                        candidate_matches[index + 1].start()
                        if index + 1 < len(candidate_matches)
                        else candidates_end
                    )
                    start, end = _json_value_bounds(
                        payload=payload,
                        start=match.end(),
                        end=end,
                    )
                    if start == end or payload[start] != ord("["):
                        raise ValueError("candidate trace entry must be a JSON list")
                    candidates[match.group(1).decode()] = (start, end)

                failure_matches = tuple(
                    _CANDIDATE_ENTRY.finditer(
                        payload,
                        failures_start + len(_FAILURES_MARKER),
                        failures_end,
                    )
                )
                failures: dict[str, str] = {}
                for index, match in enumerate(failure_matches):
                    end = (
                        failure_matches[index + 1].start()
                        if index + 1 < len(failure_matches)
                        else failures_end
                    )
                    start, end = _json_value_bounds(
                        payload=payload,
                        start=match.end(),
                        end=end,
                    )
                    failure = json.loads(payload[start:end])
                    if not isinstance(failure, str):
                        raise ValueError("candidate failure entry must be a JSON string")
                    failures[match.group(1).decode()] = failure
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot index isaac-lab alignment traces from {path}: {exc}") from exc
    return metadata, candidates, failures


class _TraceFileEvaluator:
    def __init__(self, *, backend: str) -> None:
        self._backend = backend
        self._retain_traces = False
        self._path: Path | None = None
        self._payload: dict[str, Any] | None = None
        self._candidate_offsets: dict[str, tuple[int, int]] = {}
        self._candidate_failures: dict[str, str] = {}

    def _load_payload(self, *, path: Path) -> dict[str, Any]:
        if self._path == path and self._payload is not None:
            return self._payload
        try:
            loaded = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot load {self._backend} alignment traces from {path}: {exc}"
            ) from exc
        payload = _object(value=loaded, name=str(path))
        self._path = path
        self._payload = payload
        return payload

    def _load_candidate_payload(self, *, path: Path) -> dict[str, Any]:
        if self._path == path and self._payload is not None:
            return self._payload
        payload, offsets, failures = _index_candidate_diagnostics(path=path)
        self._path = path
        self._payload = payload
        self._candidate_offsets = offsets
        self._candidate_failures = failures
        return payload

    def _load_candidate_traces(self, *, path: Path, key: str) -> Any:
        bounds = self._candidate_offsets.get(key)
        if bounds is None:
            failure = self._candidate_failures.get(key)
            if failure is not None:
                raise RuntimeError(failure)
            return None
        start, end = bounds
        try:
            with path.open("rb") as diagnostics:
                diagnostics.seek(start)
                return json.loads(diagnostics.read(end - start))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot load isaac-lab candidate traces from {path}: {exc}"
            ) from exc

    def evaluate(
        self,
        *,
        definition: ControllerDefinition,
        settings: ControllerSettings,
        reference_traces: tuple[ProbeTrace, ...] | None,
    ) -> tuple[ProbeTrace, ...]:
        run_dir_value = os.environ.get("ROBOTICS_ALIGNMENT_RUN_DIR")
        if not run_dir_value:
            raise RuntimeError(
                f"{self._backend} evaluation requires ROBOTICS_ALIGNMENT_RUN_DIR "
                "or an explicitly injected evaluator"
            )
        run_dir = Path(run_dir_value)
        if self._backend == "px4-sitl":
            path = run_dir / "reference.json"
            key = "traces"
            payload = self._load_payload(path=path)
        else:
            path = run_dir / "target_candidates.json"
            key = _fingerprint(asdict(settings))
            payload = self._load_candidate_payload(path=path)
        if payload.get("schema_version") not in _SUPPORTED_TRACE_SCHEMA_VERSIONS:
            raise RuntimeError(f"{path} has an unsupported trace schema")
        if payload.get("controller_id") != definition.controller_id:
            raise RuntimeError(f"{path} was captured for a different controller")
        if payload.get("definition_fingerprint") != definition.definition_fingerprint:
            raise RuntimeError(f"{path} was captured from a different definition")
        if self._backend == "px4-sitl":
            raw_traces = payload.get(key)
        else:
            raw_traces = self._load_candidate_traces(path=path, key=key)
        if not isinstance(raw_traces, list):
            raise RuntimeError(f"{path} has no trace set for settings fingerprint {key!r}")
        return tuple(
            _trace(value=raw_trace, name=f"{path}.traces[{index}]")
            for index, raw_trace in enumerate(raw_traces)
        )


def load_controller_definition(path: str | Path, /) -> ControllerDefinition:
    """Load and validate one executable controller definition."""
    definition_path = Path(path)
    try:
        loaded = yaml.safe_load(definition_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load controller definition {definition_path}: {exc}") from exc
    definition = _object(value=loaded, name=str(definition_path))
    backend = _string(value=definition.get("backend"), name="backend")
    if backend not in _BACKENDS:
        raise ValueError(f"unsupported controller backend {backend!r}")
    base_settings = _settings(value=definition.get("settings"), name="settings")
    selection_metric = _string(
        value=definition.get("selection_metric"),
        name="selection_metric",
    )
    metric(name=selection_metric)
    publication_metrics_value = definition.get("publication_metrics", [])
    if not isinstance(publication_metrics_value, list):
        raise ValueError("publication_metrics must be a list")
    publication_metrics = tuple(
        _string(value=value, name="publication_metrics") for value in publication_metrics_value
    )
    raw_for_fingerprint = dict(definition)
    fingerprint = _fingerprint(raw_for_fingerprint)
    return ControllerDefinition(
        alignment_id=_string(value=definition.get("alignment_id"), name="alignment_id"),
        controller_id=_string(value=definition.get("controller_id"), name="controller_id"),
        backend=backend,
        experiment_config=_string(
            value=definition.get("experiment_config"),
            name="experiment_config",
        ),
        vehicle=_string(value=definition.get("vehicle"), name="vehicle"),
        airframe=_string(value=definition.get("airframe"), name="airframe"),
        command_rate_hz=_finite(
            value=definition.get("command_rate_hz"),
            name="command_rate_hz",
            positive=True,
        ),
        command_interface=_interface(
            value=definition.get("command_interface"),
            name="command_interface",
        ),
        base_settings=base_settings,
        probes=_profiles(value=definition.get("probes")),
        parameters=_parameters(
            value=definition.get("parameters", []),
            base_settings=base_settings,
        ),
        search=_search(value=definition.get("search")),
        selection_metric=selection_metric,
        publication_metrics=publication_metrics,
        evaluator=_TraceFileEvaluator(backend=backend),
        definition_fingerprint=fingerprint,
        image=_string(value=definition.get("image"), name="image"),
    )


def _artifact_payload(artifact: AlignmentArtifact, /) -> dict[str, Any]:
    def plain(value: Any, /) -> Any:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: plain(getattr(value, item.name)) for item in dataclasses.fields(value)
            }
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        return value

    payload = plain(artifact)
    assert isinstance(payload, dict)
    return payload


def _candidate_score(
    *,
    value: Any,
    name: str,
    legacy_rate_output_scale: bool,
    legacy_hover_thrust: bool,
) -> CandidateScore:
    candidate = _object(value=value, name=name)
    metrics = _object(value=candidate.get("metrics"), name=f"{name}.metrics")
    ranking = candidate.get("ranking_values")
    usable = candidate.get("usable")
    if type(usable) is not bool:
        raise ValueError(f"{name}.usable must be boolean")
    if not isinstance(ranking, list) or (usable and not ranking):
        raise ValueError(
            f"{name}.ranking_values must be a list and non-empty for usable candidates"
        )
    failure_reason = candidate.get("failure_reason")
    if failure_reason is not None and not isinstance(failure_reason, str):
        raise ValueError(f"{name}.failure_reason must be string or null")
    index = candidate.get("index")
    if type(index) is not int or index < 0:
        raise ValueError(f"{name}.index must be a non-negative integer")
    return CandidateScore(
        index=index,
        settings=_settings(
            value=candidate.get("settings"),
            name=f"{name}.settings",
            legacy_rate_output_scale=legacy_rate_output_scale,
            legacy_hover_thrust=legacy_hover_thrust,
        ),
        usable=usable,
        metrics={
            key: _finite(value=item, name=f"{name}.metrics.{key}") for key, item in metrics.items()
        },
        ranking_values=tuple(
            _finite(value=item, name=f"{name}.ranking_values") for item in ranking
        ),
        failure_reason=failure_reason,
    )


def load_alignment_artifact(path: str | Path, /) -> AlignmentArtifact:
    """Load a run directory, link, or alignment JSON and reject malformed provenance."""
    artifact_path = Path(path)
    if artifact_path.is_dir():
        artifact_path = artifact_path / "alignment.json"
    try:
        loaded = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load alignment artifact {artifact_path}: {exc}") from exc
    artifact = _object(value=loaded, name=str(artifact_path))
    schema_version = artifact.get("schema_version")
    if schema_version not in _SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(
            f"alignment artifact {artifact_path} has unsupported "
            f"schema_version={schema_version!r}"
        )
    legacy_rate_output_scale = schema_version == 2
    legacy_hover_thrust = schema_version in (2, 3)
    command_interface = _interface(
        value=artifact.get("command_interface"),
        name="command_interface",
    )
    target_backend = _string(
        value=artifact.get("target_backend"),
        name="target_backend",
    )
    reference_backend = _string(
        value=artifact.get("reference_backend"),
        name="reference_backend",
    )
    if target_backend != "isaac-lab" or reference_backend != "px4-sitl":
        raise ValueError(
            "alignment artifact has unsupported target/reference backend contract "
            f"{target_backend!r}/{reference_backend!r}"
        )
    raw_candidates = artifact.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("alignment artifact candidates must be a non-empty list")
    candidates = tuple(
        _candidate_score(
            value=value,
            name=f"candidates[{index}]",
            legacy_rate_output_scale=legacy_rate_output_scale,
            legacy_hover_thrust=legacy_hover_thrust,
        )
        for index, value in enumerate(raw_candidates)
    )
    if tuple(candidate.index for candidate in candidates) != tuple(range(len(candidates))):
        raise ValueError("alignment artifact candidate indices must be ordered and contiguous")
    selected_index = artifact.get("selected_candidate_index")
    if type(selected_index) is not int or selected_index < 0:
        raise ValueError("selected_candidate_index must be a non-negative integer")
    selected_matches = [candidate for candidate in candidates if candidate.index == selected_index]
    if len(selected_matches) != 1 or not selected_matches[0].usable:
        raise ValueError("selected_candidate_index must identify one usable candidate")
    parsed_selected_settings = _settings(
        value=artifact.get("selected_settings"),
        name="selected_settings",
        legacy_rate_output_scale=legacy_rate_output_scale,
        legacy_hover_thrust=legacy_hover_thrust,
    )
    if parsed_selected_settings != selected_matches[0].settings:
        raise ValueError("selected_settings do not match selected candidate")
    declared = _parameters(
        value=artifact.get("declared_parameters"),
        base_settings=parsed_selected_settings,
    )
    metric_name = _string(value=artifact.get("selection_metric"), name="selection_metric")
    metric(name=metric_name)
    quality = artifact.get("advisory_quality_passed")
    dirty = artifact.get("source_dirty")
    if type(quality) is not bool or type(dirty) is not bool:
        raise ValueError("alignment artifact quality and dirty provenance must be booleans")
    reasons = artifact.get("advisory_quality_reasons")
    ranking = artifact.get("selected_ranking_values")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ValueError("advisory_quality_reasons must be a string list")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("selected_ranking_values must be a non-empty list")
    parsed_ranking = tuple(_finite(value=item, name="selected_ranking_values") for item in ranking)
    if parsed_ranking != selected_matches[0].ranking_values:
        raise ValueError("selected_ranking_values do not match selected candidate")
    fit_metrics = _object(value=artifact.get("fit_metrics"), name="fit_metrics")
    validation_metrics = _object(
        value=artifact.get("validation_metrics"),
        name="validation_metrics",
    )
    result = AlignmentArtifact(
        schema_version=schema_version,
        alignment_id=_string(value=artifact.get("alignment_id"), name="alignment_id"),
        run_id=_string(value=artifact.get("run_id"), name="run_id"),
        target_controller_id=_string(
            value=artifact.get("target_controller_id"),
            name="target_controller_id",
        ),
        reference_controller_id=_string(
            value=artifact.get("reference_controller_id"),
            name="reference_controller_id",
        ),
        target_backend=target_backend,
        reference_backend=reference_backend,
        target_definition_fingerprint=_string(
            value=artifact.get("target_definition_fingerprint"),
            name="target_definition_fingerprint",
        ),
        reference_definition_fingerprint=_string(
            value=artifact.get("reference_definition_fingerprint"),
            name="reference_definition_fingerprint",
        ),
        comparison_fingerprint=_string(
            value=artifact.get("comparison_fingerprint"),
            name="comparison_fingerprint",
        ),
        experiment_config=_string(
            value=artifact.get("experiment_config"),
            name="experiment_config",
        ),
        experiment_fingerprint=_string(
            value=artifact.get("experiment_fingerprint"),
            name="experiment_fingerprint",
        ),
        vehicle=_string(value=artifact.get("vehicle"), name="vehicle"),
        airframe=_string(value=artifact.get("airframe"), name="airframe"),
        command_rate_hz=_finite(
            value=artifact.get("command_rate_hz"),
            name="command_rate_hz",
            positive=True,
        ),
        command_interface=command_interface,
        probe_suite_fingerprint=_string(
            value=artifact.get("probe_suite_fingerprint"),
            name="probe_suite_fingerprint",
        ),
        selection_metric=metric_name,
        selected_settings=parsed_selected_settings,
        declared_parameters=declared,
        candidates=candidates,
        selected_candidate_index=selected_index,
        selected_ranking_values=parsed_ranking,
        fit_metrics={
            key: _finite(value=value, name=f"fit_metrics.{key}")
            for key, value in fit_metrics.items()
        },
        validation_metrics={
            key: _finite(value=value, name=f"validation_metrics.{key}")
            for key, value in validation_metrics.items()
        },
        advisory_quality_passed=quality,
        advisory_quality_reasons=tuple(reasons),
        source_commit=_string(value=artifact.get("source_commit"), name="source_commit"),
        source_dirty=dirty,
        target_image=_string(value=artifact.get("target_image"), name="target_image"),
        reference_image=_string(
            value=artifact.get("reference_image"),
            name="reference_image",
        ),
    )
    return result


def write_alignment_artifact(*, artifact: AlignmentArtifact, run_dir: Path) -> Path:
    """Write the canonical immutable run result."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "alignment.json"
    if path.exists():
        raise FileExistsError(f"alignment result already exists: {path}")
    path.write_text(json.dumps(_artifact_payload(artifact), indent=2, allow_nan=False) + "\n")
    return path


def _replace_relative_symlink(*, link: Path, target: Path) -> None:
    relative_target = os.path.relpath(target, start=link.parent)
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(relative_target, target_is_directory=True)
    os.replace(temporary, link)


def publish_alignment(*, artifact: AlignmentArtifact, run_dir: Path) -> None:
    """Advance ``latest`` and the comparable-generation ``best`` links atomically."""
    root = run_dir.parent
    _replace_relative_symlink(link=root / "latest", target=run_dir)
    best_link = root / "best"
    replace_best = True
    if best_link.exists():
        best = load_alignment_artifact(best_link)
        if best.comparison_fingerprint == artifact.comparison_fingerprint:
            replace_best = (
                artifact.selected_ranking_values,
                artifact.run_id,
            ) < (
                best.selected_ranking_values,
                best.run_id,
            )
    if replace_best:
        _replace_relative_symlink(link=best_link, target=run_dir)

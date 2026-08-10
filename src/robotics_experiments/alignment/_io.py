"""YAML/JSON loading, validation, fingerprinting, and publication."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from robotics_experiments.alignment._metrics import metric
from robotics_experiments.alignment._models import (
    ControllerDefinition,
    ControllerSettings,
    ControllerStateSourceAge,
    ParameterSpec,
    ProbeCommand,
    ProbeFamily,
    ProbeProfile,
    ProbeTrace,
    SearchSpec,
    SettingValue,
)

__all__: list[str] = []

_TRACE_SCHEMA_VERSION = 1
_SUPPORTED_TRACE_SCHEMA_VERSIONS = frozenset({_TRACE_SCHEMA_VERSION})
_BACKENDS = frozenset({"isaac-lab", "px4-sitl"})
_SEARCH_STRATEGIES = frozenset({"coarse-to-fine-beam"})
_SETTINGS_FIELDS = {item.name for item in fields(ControllerSettings)}
_CANDIDATE_ENTRY = re.compile(rb'"([0-9a-f]{64})"\s*:\s*')
_CANDIDATES_KEY = b'"candidates"'
_FAILURES_KEY = b'"failures"'


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
) -> ControllerSettings:
    settings = _object(value=value, name=name)
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
        velocity_controller_rate_hz=_finite(
            value=settings["velocity_controller_rate_hz"],
            name=f"{name}.velocity_controller_rate_hz",
            positive=True,
        ),
        attitude_controller_rate_hz=_finite(
            value=settings["attitude_controller_rate_hz"],
            name=f"{name}.attitude_controller_rate_hz",
            positive=True,
        ),
        kp_vel=_vector(value=settings["kp_vel"], name=f"{name}.kp_vel"),
        horizontal_velocity_error_gain_scale=_finite(
            value=settings["horizontal_velocity_error_gain_scale"],
            name=f"{name}.horizontal_velocity_error_gain_scale",
            positive=True,
        ),
        horizontal_velocity_error_gain_start_mps=_finite(
            value=settings["horizontal_velocity_error_gain_start_mps"],
            name=f"{name}.horizontal_velocity_error_gain_start_mps",
            positive=True,
        ),
        horizontal_velocity_error_gain_full_mps=_finite(
            value=settings["horizontal_velocity_error_gain_full_mps"],
            name=f"{name}.horizontal_velocity_error_gain_full_mps",
            positive=True,
        ),
        horizontal_velocity_error_gain_end_mps=_finite(
            value=settings["horizontal_velocity_error_gain_end_mps"],
            name=f"{name}.horizontal_velocity_error_gain_end_mps",
            positive=True,
        ),
        ki_vel=_vector(value=settings["ki_vel"], name=f"{name}.ki_vel"),
        kd_vel=_vector(value=settings["kd_vel"], name=f"{name}.kd_vel"),
        velocity_observation_scale=_vector(
            value=settings["velocity_observation_scale"],
            name=f"{name}.velocity_observation_scale",
        ),
        velocity_observation_bias_mps=_vector(
            value=settings["velocity_observation_bias_mps"],
            name=f"{name}.velocity_observation_bias_mps",
        ),
        velocity_estimator_bias_decay_hz=_finite(
            value=settings["velocity_estimator_bias_decay_hz"],
            name=f"{name}.velocity_estimator_bias_decay_hz",
            nonnegative=True,
        ),
        velocity_filter_hz=_vector(
            value=settings["velocity_filter_hz"],
            name=f"{name}.velocity_filter_hz",
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
        hover_thrust_normalized=_finite(
            value=settings["hover_thrust_normalized"],
            name=f"{name}.hover_thrust_normalized",
            positive=True,
        ),
        ctbr_effective_collective_scale=_finite(
            value=settings["ctbr_effective_collective_scale"],
            name=f"{name}.ctbr_effective_collective_scale",
            positive=True,
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
        angular_velocity_filter_hz=_finite(
            value=settings["angular_velocity_filter_hz"],
            name=f"{name}.angular_velocity_filter_hz",
            nonnegative=True,
        ),
        rate_derivative_filter_hz=_finite(
            value=settings["rate_derivative_filter_hz"],
            name=f"{name}.rate_derivative_filter_hz",
            positive=True,
        ),
        yaw_torque_filter_hz=_finite(
            value=settings["yaw_torque_filter_hz"],
            name=f"{name}.yaw_torque_filter_hz",
            positive=True,
        ),
        rate_integral_limit=_vector(
            value=settings["rate_integral_limit"],
            name=f"{name}.rate_integral_limit",
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
        "velocity_filter_hz",
        "velocity_integral_limit_mps2",
        "kp_att",
        "kp_rate",
        "ki_rate",
        "kd_rate",
        "kff_rate",
        "rate_integral_limit",
    ):
        if any(value < 0.0 for value in getattr(parsed, field_name)):
            raise ValueError(f"{name}.{field_name} values must be non-negative")
    if not 0.0 <= parsed.attitude_yaw_weight <= 1.0:
        raise ValueError(f"{name}.attitude_yaw_weight must be in [0, 1]")
    if parsed.hover_thrust_normalized > 1.0:
        raise ValueError(f"{name}.hover_thrust_normalized must not exceed 1")
    if parsed.horizontal_velocity_error_gain_scale < 1.0:
        raise ValueError(f"{name}.horizontal_velocity_error_gain_scale must be at least 1")
    if (
        not parsed.horizontal_velocity_error_gain_start_mps
        < parsed.horizontal_velocity_error_gain_full_mps
        < parsed.horizontal_velocity_error_gain_end_mps
    ):
        raise ValueError(
            f"{name} horizontal velocity error gain transition must satisfy start < full < end"
        )
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


def _profiles(*, value: Any, family_name: str, control_mode: str) -> tuple[ProbeProfile, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"probe family {family_name!r} profiles must be a non-empty list")
    profiles: list[ProbeProfile] = []
    names: set[str] = set()
    for index, raw_profile in enumerate(value):
        name = f"probe_families[{family_name!r}].profiles[{index}]"
        profile = _object(value=raw_profile, name=name)
        profile_name = _string(value=profile.get("name"), name=f"{name}.name")
        if profile_name in names:
            raise ValueError(f"duplicate probe name {profile_name!r}")
        names.add(profile_name)
        split = _string(value=profile.get("split"), name=f"{name}.split")
        if split not in ("fit", "validation", "stability"):
            raise ValueError(f"{name}.split must be 'fit', 'validation', or 'stability'")
        axis = _string(value=profile.get("axis"), name=f"{name}.axis")
        if axis not in ("x", "y", "z", "roll", "pitch", "yaw", "collective", "combined"):
            raise ValueError(
                f"{name}.axis must be x, y, z, roll, pitch, yaw, collective, or combined"
            )
        raw_commands = profile.get("commands")
        if not isinstance(raw_commands, list) or not raw_commands:
            raise ValueError(f"{name}.commands must be a non-empty list")
        commands = []
        for command_index, raw_command in enumerate(raw_commands):
            command_name = f"{name}.commands[{command_index}]"
            command = _object(value=raw_command, name=command_name)
            velocity: tuple[float, float, float]
            body_rate: tuple[float, float, float]
            if control_mode == "velocity":
                expected_fields = {"velocity_body_flu_mps", "yaw_rate_radps", "duration_s"}
                if set(command) != expected_fields:
                    raise ValueError(f"{command_name} fields mismatch for velocity control")
                velocity_values = _vector(
                    value=command["velocity_body_flu_mps"],
                    name=f"{command_name}.velocity_body_flu_mps",
                )
                velocity = velocity_values[0], velocity_values[1], velocity_values[2]
                yaw_rate = _finite(
                    value=command["yaw_rate_radps"], name=f"{command_name}.yaw_rate_radps"
                )
                collective = 0.0
                body_rate = (0.0, 0.0, 0.0)
            else:
                expected_fields = {
                    "collective_thrust_normalized",
                    "body_rate_flu_radps",
                    "duration_s",
                }
                if set(command) != expected_fields:
                    raise ValueError(f"{command_name} fields mismatch for CTBR control")
                velocity = (0.0, 0.0, 0.0)
                yaw_rate = 0.0
                collective = _finite(
                    value=command["collective_thrust_normalized"],
                    name=f"{command_name}.collective_thrust_normalized",
                    nonnegative=True,
                )
                if collective > 1.0:
                    raise ValueError(
                        f"{command_name}.collective_thrust_normalized must not exceed 1"
                    )
                body_rate_values = _vector(
                    value=command["body_rate_flu_radps"],
                    name=f"{command_name}.body_rate_flu_radps",
                )
                body_rate = body_rate_values[0], body_rate_values[1], body_rate_values[2]
            commands.append(
                ProbeCommand(
                    velocity_body_flu_mps=velocity,
                    yaw_rate_radps=yaw_rate,
                    collective_thrust_normalized=collective,
                    body_rate_flu_radps=body_rate,
                    duration_s=_finite(
                        value=command["duration_s"],
                        name=f"{command_name}.duration_s",
                        positive=True,
                    ),
                )
            )
        profiles.append(
            ProbeProfile(
                name=profile_name,
                split=split,
                axis=axis,
                commands=tuple(commands),
            )
        )
    return tuple(profiles)


def _probe_families(*, value: Any) -> tuple[ProbeFamily, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("probe_families must be a non-empty list")
    families = []
    names: set[str] = set()
    profile_names: set[str] = set()
    for index, raw_family in enumerate(value):
        name = f"probe_families[{index}]"
        family = _object(value=raw_family, name=name)
        if set(family) != {"name", "control_mode", "metric", "profiles"}:
            raise ValueError(f"{name} fields must be name, control_mode, metric, and profiles")
        family_name = _string(value=family["name"], name=f"{name}.name")
        if family_name in names:
            raise ValueError(f"duplicate probe family name {family_name!r}")
        names.add(family_name)
        control_mode = _string(value=family["control_mode"], name=f"{name}.control_mode")
        if control_mode not in ("velocity", "ctbr"):
            raise ValueError(f"{name}.control_mode must be 'velocity' or 'ctbr'")
        metric_name = _string(value=family["metric"], name=f"{name}.metric")
        metric(name=metric_name)
        profiles = _profiles(
            value=family["profiles"], family_name=family_name, control_mode=control_mode
        )
        duplicates = profile_names & {profile.name for profile in profiles}
        if duplicates:
            raise ValueError(f"duplicate probe profile names {sorted(duplicates)!r}")
        profile_names.update(profile.name for profile in profiles)
        families.append(
            ProbeFamily(
                name=family_name,
                control_mode=control_mode,
                metric=metric_name,
                profiles=profiles,
            )
        )
    return tuple(families)


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
        raw_regularization_scale = parameter.get("selection_regularization_scale")
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
                selection_regularization_scale=(
                    None
                    if raw_regularization_scale is None
                    else _finite(
                        value=raw_regularization_scale,
                        name=f"{name}.selection_regularization_scale",
                        positive=True,
                    )
                ),
            )
        )
    return tuple(parameters)


def _positive_integer(*, value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _controller_schema_version(*, value: Any) -> int:
    version = _positive_integer(value=value, name="controller_schema_version")
    if version != 1:
        raise ValueError("controller_schema_version must be 1")
    return version


def _nonnegative_integer(*, value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _boolean(*, value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _search(*, value: Any, base_settings: ControllerSettings) -> SearchSpec:
    search = _object(value=value, name="search")
    strategy = _string(value=search.get("strategy"), name="search.strategy")
    if strategy not in _SEARCH_STRATEGIES:
        raise ValueError(f"unsupported search.strategy={strategy!r}")
    refinement_levels = _nonnegative_integer(
        value=search.get("refinement_levels", 0),
        name="search.refinement_levels",
    )
    raw_initial_settings = search.get("initial_settings", [])
    if not isinstance(raw_initial_settings, list):
        raise ValueError("search.initial_settings must be a list")
    initial_settings = []
    for index, value in enumerate(raw_initial_settings):
        name = f"search.initial_settings[{index}]"
        overrides = _object(value=value, name=name)
        extra = overrides.keys() - _SETTINGS_FIELDS
        if extra:
            raise ValueError(f"{name} has unknown fields {sorted(extra)!r}")
        merged = asdict(base_settings)
        merged.update(overrides)
        initial_settings.append(_settings(value=merged, name=name))
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
        refinement_levels=refinement_levels,
        min_refinement_survivors=_positive_integer(
            value=search.get("min_refinement_survivors", 2),
            name="search.min_refinement_survivors",
        ),
        initial_settings=tuple(initial_settings),
    )


def _trace(*, value: Any, name: str) -> ProbeTrace:
    trace = _object(value=value, name=name)
    expected = {item.name for item in fields(ProbeTrace)}
    if set(trace) != expected:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(expected - trace.keys())!r}, "
            f"extra={sorted(trace.keys() - expected)!r}"
        )
    failure_reason = trace["failure_reason"]
    if failure_reason is not None and not isinstance(failure_reason, str):
        raise ValueError(f"{name}.failure_reason must be a string or null")
    completed = trace["completed"]
    if type(completed) is not bool:
        raise ValueError(f"{name}.completed must be boolean")
    time = trace["time_s"]
    commands = trace["command_velocity_body_flu_mps"]
    yaw_commands = trace["command_yaw_rate_radps"]
    collective_commands = trace["command_collective_thrust_normalized"]
    body_rate_commands = trace["command_body_rate_flu_radps"]
    positions = trace["position_enu_m"]
    attitudes = trace["attitude_flu_enu_wxyz"]
    velocities = trace["velocity_body_flu_mps"]
    world_velocities = trace["velocity_enu_mps"]
    angular_velocities = trace["angular_velocity_body_flu_radps"]
    if (
        not isinstance(time, list)
        or not isinstance(commands, list)
        or not isinstance(yaw_commands, list)
        or not isinstance(collective_commands, list)
        or not isinstance(body_rate_commands, list)
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
        len(collective_commands),
        len(body_rate_commands),
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
    control_mode = _string(value=trace["control_mode"], name=f"{name}.control_mode")
    if control_mode not in ("velocity", "ctbr"):
        raise ValueError(f"{name}.control_mode must be 'velocity' or 'ctbr'")
    parsed_velocity_commands = tuple(
        _vector(value=item, name=f"{name}.command_velocity_body_flu_mps") for item in commands
    )
    parsed_yaw_commands = tuple(
        _finite(value=item, name=f"{name}.command_yaw_rate_radps") for item in yaw_commands
    )
    parsed_collective_commands = tuple(
        _finite(
            value=item,
            name=f"{name}.command_collective_thrust_normalized",
            nonnegative=True,
        )
        for item in collective_commands
    )
    if any(value > 1.0 for value in parsed_collective_commands):
        raise ValueError(f"{name}.command_collective_thrust_normalized must not exceed 1")
    parsed_body_rate_commands = tuple(
        _vector(value=item, name=f"{name}.command_body_rate_flu_radps")
        for item in body_rate_commands
    )
    if control_mode == "velocity" and any(
        collective != 0.0 or body_rate != (0.0, 0.0, 0.0)
        for collective, body_rate in zip(
            parsed_collective_commands, parsed_body_rate_commands, strict=True
        )
    ):
        raise ValueError(f"{name} velocity trace has active CTBR commands")
    if control_mode == "ctbr" and any(
        velocity != (0.0, 0.0, 0.0) or yaw_rate != 0.0
        for velocity, yaw_rate in zip(parsed_velocity_commands, parsed_yaw_commands, strict=True)
    ):
        raise ValueError(f"{name} CTBR trace has active velocity commands")
    initial_hover_thrust_normalized = _finite(
        value=trace["initial_hover_thrust_normalized"],
        name=f"{name}.initial_hover_thrust_normalized",
        nonnegative=True,
    )
    if initial_hover_thrust_normalized > 1.0:
        raise ValueError(f"{name}.initial_hover_thrust_normalized must not exceed 1")
    initial_collectives = {
        field: _finite(
            value=trace[field],
            name=f"{name}.{field}",
            nonnegative=True,
        )
        for field in (
            "initial_desired_collective_thrust_normalized",
            "initial_collective_thrust_normalized",
        )
    }
    if any(value > 1.0 for value in initial_collectives.values()):
        raise ValueError(f"{name} initial collective thrust must not exceed 1")
    source_age = _object(
        value=trace["initial_controller_state_source_age_us"],
        name=f"{name}.initial_controller_state_source_age_us",
    )
    expected_source_age = {
        "vehicle_local_position_setpoint",
        "vehicle_local_position",
        "vehicle_attitude_setpoint",
        "vehicle_rates_setpoint",
        "vehicle_torque_setpoint",
        "vehicle_angular_velocity",
        "rate_ctrl_status",
        "hover_thrust_estimate",
    }
    if set(source_age) != expected_source_age:
        raise ValueError(f"{name}.initial_controller_state_source_age_us fields mismatch")
    return ProbeTrace(
        family=_string(value=trace["family"], name=f"{name}.family"),
        control_mode=control_mode,
        family_metric=_string(value=trace["family_metric"], name=f"{name}.family_metric"),
        profile=_string(value=trace["profile"], name=f"{name}.profile"),
        split=_string(value=trace["split"], name=f"{name}.split"),
        axis=_string(value=trace["axis"], name=f"{name}.axis"),
        time_s=parsed_time,
        command_velocity_body_flu_mps=parsed_velocity_commands,
        command_yaw_rate_radps=parsed_yaw_commands,
        command_collective_thrust_normalized=parsed_collective_commands,
        command_body_rate_flu_radps=parsed_body_rate_commands,
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
            value=trace["initial_position_enu_m"],
            name=f"{name}.initial_position_enu_m",
        ),
        initial_attitude_flu_enu_wxyz=_vector(
            value=trace["initial_attitude_flu_enu_wxyz"],
            name=f"{name}.initial_attitude_flu_enu_wxyz",
            length=4,
        ),
        initial_linear_velocity_enu_mps=_vector(
            value=trace["initial_linear_velocity_enu_mps"],
            name=f"{name}.initial_linear_velocity_enu_mps",
        ),
        initial_estimated_linear_velocity_enu_mps=_vector(
            value=trace["initial_estimated_linear_velocity_enu_mps"],
            name=f"{name}.initial_estimated_linear_velocity_enu_mps",
        ),
        initial_angular_velocity_body_flu_radps=_vector(
            value=trace["initial_angular_velocity_body_flu_radps"],
            name=f"{name}.initial_angular_velocity_body_flu_radps",
        ),
        initial_estimated_angular_velocity_body_flu_radps=_vector(
            value=trace["initial_estimated_angular_velocity_body_flu_radps"],
            name=f"{name}.initial_estimated_angular_velocity_body_flu_radps",
        ),
        completed=completed,
        initial_controller_state_valid=_boolean(
            value=trace["initial_controller_state_valid"],
            name=f"{name}.initial_controller_state_valid",
        ),
        initial_controller_state_source_age_us=ControllerStateSourceAge(
            vehicle_local_position_setpoint=_nonnegative_integer(
                value=source_age["vehicle_local_position_setpoint"],
                name=(
                    f"{name}.initial_controller_state_source_age_us."
                    "vehicle_local_position_setpoint"
                ),
            ),
            vehicle_local_position=_nonnegative_integer(
                value=source_age["vehicle_local_position"],
                name=f"{name}.initial_controller_state_source_age_us.vehicle_local_position",
            ),
            vehicle_attitude_setpoint=_nonnegative_integer(
                value=source_age["vehicle_attitude_setpoint"],
                name=(
                    f"{name}.initial_controller_state_source_age_us." "vehicle_attitude_setpoint"
                ),
            ),
            vehicle_rates_setpoint=_nonnegative_integer(
                value=source_age["vehicle_rates_setpoint"],
                name=(f"{name}.initial_controller_state_source_age_us." "vehicle_rates_setpoint"),
            ),
            vehicle_torque_setpoint=_nonnegative_integer(
                value=source_age["vehicle_torque_setpoint"],
                name=(f"{name}.initial_controller_state_source_age_us." "vehicle_torque_setpoint"),
            ),
            vehicle_angular_velocity=_nonnegative_integer(
                value=source_age["vehicle_angular_velocity"],
                name=f"{name}.initial_controller_state_source_age_us.vehicle_angular_velocity",
            ),
            rate_ctrl_status=_nonnegative_integer(
                value=source_age["rate_ctrl_status"],
                name=f"{name}.initial_controller_state_source_age_us.rate_ctrl_status",
            ),
            hover_thrust_estimate=_nonnegative_integer(
                value=source_age["hover_thrust_estimate"],
                name=f"{name}.initial_controller_state_source_age_us.hover_thrust_estimate",
            ),
        ),
        initial_velocity_setpoint_enu_mps=_vector(
            value=trace["initial_velocity_setpoint_enu_mps"],
            name=f"{name}.initial_velocity_setpoint_enu_mps",
        ),
        initial_hover_thrust_normalized=initial_hover_thrust_normalized,
        initial_acceleration_setpoint_enu_mps2=_vector(
            value=trace["initial_acceleration_setpoint_enu_mps2"],
            name=f"{name}.initial_acceleration_setpoint_enu_mps2",
        ),
        initial_linear_acceleration_enu_mps2=_vector(
            value=trace["initial_linear_acceleration_enu_mps2"],
            name=f"{name}.initial_linear_acceleration_enu_mps2",
        ),
        initial_angular_acceleration_body_flu_radps2=_vector(
            value=trace["initial_angular_acceleration_body_flu_radps2"],
            name=f"{name}.initial_angular_acceleration_body_flu_radps2",
        ),
        initial_rate_integral_body_flu=_vector(
            value=trace["initial_rate_integral_body_flu"],
            name=f"{name}.initial_rate_integral_body_flu",
        ),
        initial_desired_attitude_flu_enu_wxyz=_vector(
            value=trace["initial_desired_attitude_flu_enu_wxyz"],
            name=f"{name}.initial_desired_attitude_flu_enu_wxyz",
            length=4,
        ),
        initial_desired_yaw_rate_radps=_finite(
            value=trace["initial_desired_yaw_rate_radps"],
            name=f"{name}.initial_desired_yaw_rate_radps",
        ),
        initial_desired_collective_thrust_normalized=initial_collectives[
            "initial_desired_collective_thrust_normalized"
        ],
        initial_collective_thrust_normalized=initial_collectives[
            "initial_collective_thrust_normalized"
        ],
        initial_body_rate_setpoint_flu_radps=_vector(
            value=trace["initial_body_rate_setpoint_flu_radps"],
            name=f"{name}.initial_body_rate_setpoint_flu_radps",
        ),
        initial_filtered_yaw_torque_normalized=_finite(
            value=trace["initial_filtered_yaw_torque_normalized"],
            name=f"{name}.initial_filtered_yaw_torque_normalized",
        ),
        start_sim_time_us=_nonnegative_integer(
            value=trace["start_sim_time_us"],
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


def _mapping_content_start(*, payload: mmap.mmap, key: bytes, start: int = 0) -> tuple[int, int]:
    key_start = payload.find(key, start)
    if key_start < 0:
        raise ValueError(f"candidate diagnostics omit {key.decode()}")
    separator = payload.find(b":", key_start + len(key))
    if separator < 0:
        raise ValueError(f"candidate diagnostics have malformed {key.decode()}")
    content_start = separator + 1
    while content_start < len(payload) and payload[content_start] in b" \t\r\n":
        content_start += 1
    if content_start == len(payload) or payload[content_start] != ord("{"):
        raise ValueError(f"candidate diagnostics {key.decode()} must be a mapping")
    return key_start, content_start + 1


def _mapping_end_before_key(*, payload: mmap.mmap, key_start: int) -> int:
    cursor = key_start
    while cursor and payload[cursor - 1] in b" \t\r\n":
        cursor -= 1
    if not cursor or payload[cursor - 1] != ord(","):
        raise ValueError("candidate diagnostics mappings must be adjacent object members")
    cursor -= 1
    while cursor and payload[cursor - 1] in b" \t\r\n":
        cursor -= 1
    if not cursor or payload[cursor - 1] != ord("}"):
        raise ValueError("candidate diagnostics candidate mapping is not closed")
    return cursor - 1


def _final_mapping_end(*, payload: mmap.mmap) -> int:
    cursor = len(payload)
    while cursor and payload[cursor - 1] in b" \t\r\n":
        cursor -= 1
    if not cursor or payload[cursor - 1] != ord("}"):
        raise ValueError("candidate diagnostics root object is not closed")
    cursor -= 1
    while cursor and payload[cursor - 1] in b" \t\r\n":
        cursor -= 1
    if not cursor or payload[cursor - 1] != ord("}"):
        raise ValueError("candidate diagnostics failure mapping is not closed")
    return cursor - 1


def _index_candidate_diagnostics(
    *,
    path: Path,
) -> tuple[dict[str, Any], dict[str, tuple[int, int]], dict[str, str]]:
    try:
        with path.open("rb") as diagnostics:
            with mmap.mmap(diagnostics.fileno(), 0, access=mmap.ACCESS_READ) as payload:
                candidates_key_start, candidates_content_start = _mapping_content_start(
                    payload=payload,
                    key=_CANDIDATES_KEY,
                )
                failures_key_start, failures_content_start = _mapping_content_start(
                    payload=payload,
                    key=_FAILURES_KEY,
                    start=candidates_content_start,
                )
                candidates_end = _mapping_end_before_key(
                    payload=payload,
                    key_start=failures_key_start,
                )
                failures_end = _final_mapping_end(payload=payload)
                header = json.loads(
                    payload[:candidates_key_start] + b'"candidates":{},"failures":{}}'
                )
                metadata = _object(value=header, name=str(path))

                candidate_matches = tuple(
                    _CANDIDATE_ENTRY.finditer(
                        payload,
                        candidates_content_start,
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
                        failures_content_start,
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
        controller_schema_version=_controller_schema_version(
            value=definition.get("controller_schema_version")
        ),
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
        base_settings=base_settings,
        probe_families=_probe_families(value=definition.get("probe_families")),
        parameters=_parameters(
            value=definition.get("parameters", []),
            base_settings=base_settings,
        ),
        search=_search(
            value=definition.get("search"),
            base_settings=base_settings,
        ),
        selection_metric=selection_metric,
        publication_metrics=publication_metrics,
        evaluator=_TraceFileEvaluator(backend=backend),
        definition_fingerprint=fingerprint,
        image=_string(value=definition.get("image"), name="image"),
    )

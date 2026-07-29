"""YAML/JSON loading, validation, fingerprinting, and publication."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
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
    SettingValue,
)

__all__: list[str] = []

_SCHEMA_VERSION = 1
_BACKENDS = frozenset({"isaac-lab", "px4-sitl"})
_INTERFACE_KIND = "body-flu-velocity-yaw-rate-v1"
_SETTINGS_FIELDS = {item.name for item in fields(ControllerSettings)}


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


def _settings(*, value: Any, name: str) -> ControllerSettings:
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
        kp_vel=_vector(value=settings["kp_vel"], name=f"{name}.kp_vel"),
        kd_vel=_vector(value=settings["kd_vel"], name=f"{name}.kd_vel"),
        kp_att=_vector(value=settings["kp_att"], name=f"{name}.kp_att"),
        kp_rate=_vector(value=settings["kp_rate"], name=f"{name}.kp_rate"),
        max_tilt_cmd_deg=_finite(
            value=settings["max_tilt_cmd_deg"],
            name=f"{name}.max_tilt_cmd_deg",
            positive=True,
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
    for field_name in ("kp_vel", "kd_vel", "kp_att", "kp_rate"):
        if any(value < 0.0 for value in getattr(parsed, field_name)):
            raise ValueError(f"{name}.{field_name} values must be non-negative")
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
        if split not in ("fit", "validation"):
            raise ValueError(f"{name}.split must be 'fit' or 'validation'")
        axis = _string(value=profile.get("axis"), name=f"{name}.axis")
        if axis not in ("x", "y", "z"):
            raise ValueError(f"{name}.axis must be x, y, or z")
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
    expected = {(split, axis) for split in ("fit", "validation") for axis in ("x", "y", "z")}
    if covered != expected:
        raise ValueError(
            "probes must cover x, y, and z for fit and validation; "
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
    velocities = trace.get("velocity_body_flu_mps")
    if (
        not isinstance(time, list)
        or not isinstance(commands, list)
        or not isinstance(velocities, list)
    ):
        raise ValueError(f"{name} samples must be lists")
    if not time or len(time) != len(commands) or len(time) != len(velocities):
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
        velocity_body_flu_mps=tuple(
            _vector(value=item, name=f"{name}.velocity_body_flu_mps") for item in velocities
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
        failure_reason=failure_reason,
    )


class _TraceFileEvaluator:
    def __init__(self, *, backend: str) -> None:
        self._backend = backend

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
        else:
            path = run_dir / "target_candidates.json"
            key = _fingerprint(asdict(settings))
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot load {self._backend} alignment traces from {path}: {exc}"
            ) from exc
        payload = _object(value=payload, name=str(path))
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError(f"{path} has an unsupported trace schema")
        if payload.get("controller_id") != definition.controller_id:
            raise RuntimeError(f"{path} was captured for a different controller")
        if payload.get("definition_fingerprint") != definition.definition_fingerprint:
            raise RuntimeError(f"{path} was captured from a different definition")
        if self._backend == "px4-sitl":
            raw_traces = payload.get(key)
        else:
            raw_candidates = payload.get("candidates")
            raw_failures = payload.get("failures")
            if not isinstance(raw_candidates, dict) or not isinstance(raw_failures, dict):
                raise RuntimeError(f"{path} has malformed candidate diagnostics")
            raw_traces = raw_candidates.get(key)
            if raw_traces is None and isinstance(raw_failures.get(key), str):
                raise RuntimeError(raw_failures[key])
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


def _candidate_score(*, value: Any, name: str) -> CandidateScore:
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
        settings=_settings(value=candidate.get("settings"), name=f"{name}.settings"),
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
    if artifact.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"alignment artifact {artifact_path} has unsupported "
            f"schema_version={artifact.get('schema_version')!r}"
        )
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
        _candidate_score(value=value, name=f"candidates[{index}]")
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
        schema_version=_SCHEMA_VERSION,
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

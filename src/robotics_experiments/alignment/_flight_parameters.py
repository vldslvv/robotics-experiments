"""Lossless flight-parameters preparation for the existing alignment workflow."""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path
from typing import Any

from robotics_experiments.alignment._io import _trace
from robotics_experiments.alignment._models import ControllerDefinition, ProbeProfile, ProbeTrace
from robotics_experiments.flight_parameters import load_flight_parameters_bundle

__all__: list[str] = []


def _expected_profile(*, definition: ControllerDefinition, profile_name: str) -> ProbeProfile:
    return next(profile for profile in definition.profiles if profile.name == profile_name)


def _validate_commands(*, definition: ControllerDefinition, trace: ProbeTrace) -> None:
    profile = _expected_profile(definition=definition, profile_name=trace.profile)
    commands = profile.commands
    period = 1.0 / definition.command_rate_hz
    duration_s = sum(command.duration_s for command in commands)
    actual_count = len(trace.time_s)
    expected_count = round(duration_s * definition.command_rate_hz)
    if abs(actual_count - expected_count) > 1:
        raise ValueError(
            f"flight parameters profile {trace.profile!r} has {actual_count} samples; "
            f"alignment requires {expected_count} within one bridge-quantized sample"
        )
    if trace.time_s[-1] >= duration_s or any(
        not math.isclose(actual, index * period, rel_tol=0.0, abs_tol=period / 2.0)
        for index, actual in enumerate(trace.time_s)
    ):
        raise ValueError(
            f"flight parameters profile {trace.profile!r} does not match alignment cadence"
        )
    command_ends = tuple(accumulate(command.duration_s for command in commands))
    command_indices = tuple(bisect_right(command_ends, time_s + 1.0e-9) for time_s in trace.time_s)
    if any(index >= len(commands) for index in command_indices):
        raise ValueError(
            f"flight parameters profile {trace.profile!r} exceeds its command duration"
        )
    expected_velocity = tuple(commands[index].velocity_body_flu_mps for index in command_indices)
    expected_yaw = tuple(commands[index].yaw_rate_radps for index in command_indices)
    expected_collective = tuple(
        commands[index].collective_thrust_normalized for index in command_indices
    )
    expected_body_rate = tuple(commands[index].body_rate_flu_radps for index in command_indices)
    if (
        (not profile.hold_position and trace.command_velocity_body_flu_mps != expected_velocity)
        or trace.command_yaw_rate_radps != expected_yaw
        or trace.command_collective_thrust_normalized != expected_collective
        or trace.command_body_rate_flu_radps != expected_body_rate
    ):
        raise ValueError(
            f"flight parameters profile {trace.profile!r} does not contain the exact commands"
        )


def _validated_payload(
    *, path: Path, definition: ControllerDefinition, expected_profile_count: int
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load flight parameters response trace {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("flight parameters response is not the supported schema-1 trace")
    if value.get("controller_id") != definition.controller_id:
        raise ValueError("flight parameters response controller does not match alignment")
    if value.get("definition_fingerprint") != definition.definition_fingerprint:
        raise ValueError("flight parameters response definition does not match alignment")
    raw_traces = value.get("traces")
    if not isinstance(raw_traces, list) or len(raw_traces) != expected_profile_count:
        raise ValueError("flight parameters response does not cover every alignment profile")
    traces = tuple(
        _trace(value=raw, name=f"{path}.traces[{index}]") for index, raw in enumerate(raw_traces)
    )
    for expected, trace in zip(definition.profiles, traces, strict=True):
        family = next(family for family in definition.probe_families if expected in family.profiles)
        if (
            trace.profile != expected.name
            or trace.split != expected.split
            or trace.axis != expected.axis
            or trace.family != family.name
            or trace.control_mode != family.control_mode
            or trace.family_metric != family.metric
            or not trace.completed
            or trace.failure_reason is not None
            or not trace.initial_controller_state_valid
        ):
            raise ValueError(
                f"flight parameters response is insufficient for alignment profile "
                f"{expected.name!r}"
            )
        _validate_commands(definition=definition, trace=trace)
    return value


def prepare_alignment_reference(
    *, flight_parameters_artifact: str | Path, definition: ControllerDefinition, run_dir: Path
) -> None:
    """Validate sufficient response evidence and stage it without transforming samples."""

    bundle = load_flight_parameters_bundle(path=flight_parameters_artifact)
    evidence_by_set = {}
    for set_name in ("fit", "validation", "acceptance"):
        recordings = tuple(
            trace
            for trace in bundle.flight_parameters.response.traces
            if trace.set_name == set_name
        )
        if len(recordings) != 1:
            raise ValueError(
                f"alignment requires exactly one complete {set_name} response repetition"
            )
        evidence = recordings[0]
        _validated_payload(
            path=bundle.run_dir / evidence.reference_path,
            definition=definition,
            expected_profile_count=evidence.profile_count,
        )
        if evidence.profile_count != len(definition.profiles):
            raise ValueError(
                "flight parameters response profile count is insufficient for alignment"
            )
        evidence_by_set[set_name] = evidence
    run_dir.mkdir(parents=True, exist_ok=True)
    immutable_reference = os.path.relpath(bundle.run_dir, start=run_dir)
    (run_dir / "flight_parameters").symlink_to(immutable_reference, target_is_directory=True)
    for set_name, evidence in evidence_by_set.items():
        destination_dir = run_dir if set_name == "fit" else run_dir / set_name
        destination_dir.mkdir(exist_ok=True)
        for source_relative, name in (
            (evidence.reference_path, "reference.json"),
            (evidence.ulog_path, "reference.ulg"),
            (evidence.bridge_log_path, "bridge.log"),
        ):
            (destination_dir / name).symlink_to(
                os.path.relpath(bundle.run_dir / source_relative, start=destination_dir)
            )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recording_sets": {
                    name: evidence.reference_path for name, evidence in evidence_by_set.items()
                },
                "flight_parameters_run_id": bundle.flight_parameters.run_id,
                "flight_parameters_fingerprint": (bundle.flight_parameters.artifact_fingerprint),
                "flight_parameters_artifact": immutable_reference,
                "artifacts": {
                    "reference_trace": "reference.json",
                    "px4_ulog": "reference.ulg",
                    "bridge_log": "bridge.log",
                    "actuator_audit": "actuator_audit.json",
                    "target_candidates": "target_candidates.json",
                    "gpu_telemetry": "gpu_utilization.json",
                    "controller_artifact": "controller.json",
                    "controller_boundary_replay": "controller_boundary_replay.json",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

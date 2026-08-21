"""Lossless flight-parameters preparation for the existing alignment workflow."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from robotics_experiments.alignment._io import _trace
from robotics_experiments.alignment._models import ControllerDefinition, ProbeCommand, ProbeTrace
from robotics_experiments.flight_parameters import load_flight_parameters_bundle

__all__: list[str] = []


def _expected_commands(
    *, definition: ControllerDefinition, profile_name: str
) -> tuple[ProbeCommand, ...]:
    return next(profile.commands for profile in definition.profiles if profile.name == profile_name)


def _validate_commands(*, definition: ControllerDefinition, trace: ProbeTrace) -> None:
    commands = _expected_commands(definition=definition, profile_name=trace.profile)
    expected_velocity = tuple(
        command.velocity_body_flu_mps
        for command in commands
        for _ in range(round(command.duration_s * definition.command_rate_hz))
    )
    expected_yaw = tuple(
        command.yaw_rate_radps
        for command in commands
        for _ in range(round(command.duration_s * definition.command_rate_hz))
    )
    expected_collective = tuple(
        command.collective_thrust_normalized
        for command in commands
        for _ in range(round(command.duration_s * definition.command_rate_hz))
    )
    expected_body_rate = tuple(
        command.body_rate_flu_radps
        for command in commands
        for _ in range(round(command.duration_s * definition.command_rate_hz))
    )
    actual_count = len(trace.time_s)
    expected_count = len(expected_velocity)
    if actual_count != expected_count:
        raise ValueError(
            f"flight parameters profile {trace.profile!r} has {actual_count} samples; "
            f"alignment requires {expected_count}"
        )
    if (
        trace.command_velocity_body_flu_mps != expected_velocity
        or trace.command_yaw_rate_radps != expected_yaw
        or trace.command_collective_thrust_normalized != expected_collective
        or trace.command_body_rate_flu_radps != expected_body_rate
    ):
        raise ValueError(
            f"flight parameters profile {trace.profile!r} does not contain the exact commands"
        )
    period = 1.0 / definition.command_rate_hz
    # Runtime policy ticks are quantized to the bridge tick. Validate against
    # the ideal schedule so alternating ticks cannot accumulate cadence drift.
    if any(
        not math.isclose(actual, index * period, rel_tol=0.0, abs_tol=period / 2.0)
        for index, actual in enumerate(trace.time_s)
    ):
        raise ValueError(
            f"flight parameters profile {trace.profile!r} does not match alignment cadence"
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
    if bundle.flight_parameters.response.repetitions != 1:
        raise ValueError(
            "alignment requires exactly one complete flight parameters response repetition"
        )
    evidence = bundle.flight_parameters.response.traces[0]
    source_reference = bundle.run_dir / evidence.reference_path
    _validated_payload(
        path=source_reference,
        definition=definition,
        expected_profile_count=evidence.profile_count,
    )
    if evidence.profile_count != len(definition.profiles):
        raise ValueError("flight parameters response profile count is insufficient for alignment")
    run_dir.mkdir(parents=True, exist_ok=True)
    pinned_flight_parameters = run_dir / "flight_parameters"
    pinned_flight_parameters.symlink_to(
        os.path.relpath(bundle.run_dir, start=run_dir),
        target_is_directory=True,
    )
    for source, destination in (
        (source_reference, run_dir / "reference.json"),
        (bundle.run_dir / evidence.ulog_path, run_dir / "reference.ulg"),
        (bundle.run_dir / evidence.bridge_log_path, run_dir / "bridge.log"),
    ):
        shutil.copy2(source, destination)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "flight_parameters_fingerprint": (bundle.flight_parameters.artifact_fingerprint),
                "flight_parameters_artifact": "flight_parameters",
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

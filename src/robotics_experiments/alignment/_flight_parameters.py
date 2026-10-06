"""Lossless flight-parameters preparation for the existing alignment workflow."""

from __future__ import annotations

import json
import math
import os
import random
from bisect import bisect_right
from dataclasses import asdict, fields
from itertools import accumulate
from pathlib import Path
from typing import Any

from robotics_experiments.alignment._models import ControllerDefinition, ProbeProfile, ProbeTrace
from robotics_experiments.config import load_config
from robotics_experiments.flight_parameters import (
    FlightDataset,
    flight_stage_views,
    hover_pose_errors,
    load_flight_dataset,
)

__all__: list[str] = []


def _validate_commands(*, profile: ProbeProfile, view: dict[str, Any]) -> None:
    commands = profile.commands
    command_ends = tuple(accumulate(command.duration_s for command in commands))
    command_indices = tuple(
        bisect_right(command_ends, time_s + 1.0e-9) for time_s in view["command_time_s"]
    )
    expected_velocity = [commands[index].velocity_body_flu_mps for index in command_indices]
    expected_yaw = [commands[index].yaw_rate_radps for index in command_indices]
    expected_collective = [
        commands[index].collective_thrust_normalized for index in command_indices
    ]
    expected_body_rate = [commands[index].body_rate_flu_radps for index in command_indices]
    if (
        (not profile.hold_position and view["command_velocity_body_flu_mps"] != expected_velocity)
        or view["command_yaw_rate_radps"] != expected_yaw
        or view["command_collective_thrust_normalized"] != expected_collective
        or view["command_body_rate_flu_radps"] != expected_body_rate
    ):
        raise ValueError(
            f"flight parameters profile {profile.name!r} does not contain the exact commands"
        )


def _split_flights(
    *, flight_indices: tuple[int, ...], train_percent: float, validation_percent: float, seed: int
) -> dict[str, tuple[int, ...]]:
    """Shuffle whole flights, then allocate percentages by largest remainder."""
    percentages = (train_percent, validation_percent, 100.0 - train_percent - validation_percent)
    if len(flight_indices) < 3:
        raise ValueError("alignment needs at least three complete flights")
    if any(not math.isfinite(value) or value <= 0.0 for value in percentages):
        raise ValueError("train, validation and remaining test percentages must be positive")
    exact = [len(flight_indices) * value / 100.0 for value in percentages]
    counts = [math.floor(value) for value in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: len(flight_indices) - sum(counts)]:
        counts[i] += 1
    for i in range(3):
        if counts[i] == 0:
            donor = max(range(3), key=lambda j: counts[j])
            counts[donor] -= 1
            counts[i] = 1
    shuffled = list(flight_indices)
    random.Random(seed).shuffle(shuffled)
    a, b, _ = counts
    return {
        "train": tuple(shuffled[:a]),
        "validation": tuple(shuffled[a : a + b]),
        "test": tuple(shuffled[a + b :]),
    }


def load_alignment_dataset(path: Path, /) -> tuple[FlightDataset, dict[str, Any]]:
    """Resolve the pinned source and verify disjoint, complete membership."""
    metadata = json.loads(path.read_text())
    dataset = load_flight_dataset(path.parent / metadata["source"])
    if dataset.artifact_digest != metadata["source_digest"]:
        raise ValueError("alignment source digest changed")
    members = [index for role in ("train", "validation", "test") for index in metadata[role]]
    if any(not metadata[role] for role in ("train", "validation", "test")) or (
        sorted(members) != sorted(f.flight_index for f in dataset.flights)
    ):
        raise ValueError("alignment membership must partition every flight exactly once")
    return dataset, metadata


def _response_payload(
    *, flight_index: int, view: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    value = {
        **view,
        **profile,
        "flight_index": flight_index,
        "velocity_enu_mps": view["linear_velocity_enu_mps"],
        "completed": True,
        "failure_reason": None,
    }
    return {field.name: value[field.name] for field in fields(ProbeTrace)}


def load_alignment_reference(path: Path, /) -> dict[str, Any]:
    """Read lossless views in memory; reference files contain metadata only."""
    descriptor = json.loads(path.read_text())
    dataset, membership = load_alignment_dataset(path.parent / descriptor["dataset"])
    profiles = {profile["profile"]: profile for profile in descriptor["profiles"]}
    traces = [
        _response_payload(
            flight_index=flight.flight_index, view=view, profile=profiles[view["name"]]
        )
        for flight in dataset.flights
        if flight.flight_index in membership[descriptor["set_name"]]
        for view in flight_stage_views(flight)
        if view["kind"] == "controller-response"
    ]
    return {**descriptor, "source_digest": membership["source_digest"], "traces": traces}


def prepare_alignment_reference(
    *, flight_parameters_artifact: str | Path, definition: ControllerDefinition, run_dir: Path
) -> None:
    """Split one immutable capture, validate its stages, and pin minimal metadata."""
    dataset = load_flight_dataset(flight_parameters_artifact)
    config = load_config(definition.experiment_config)
    split = config.alignment.dataset_split
    membership = _split_flights(
        flight_indices=tuple(f.flight_index for f in dataset.flights),
        train_percent=split.train_percent,
        validation_percent=split.validation_percent,
        seed=dataset.definition["start_seed"],
    )
    if (
        dataset.definition["vehicle"] != definition.vehicle
        or dataset.definition["command_rate_hz"] != definition.command_rate_hz
    ):
        raise ValueError("capture vehicle or command rate does not match alignment")
    profiles = [
        {
            "family": family.name,
            "family_metric": family.metric,
            "control_mode": family.control_mode,
            "profile": profile.name,
            "split": profile.split,
        }
        for family in definition.probe_families
        for profile in family.profiles
    ]
    for flight in dataset.flights:
        if flight.metadata["controller_state_method"] != "px4-ulog-controller-boundary-v2":
            raise ValueError("alignment needs complete reconstructed controller boundaries")
        views = flight_stage_views(flight)[1:]
        if len(views) != len(profiles):
            raise ValueError("flight does not contain every alignment response stage")
        for view, profile, expected in zip(views, profiles, definition.profiles, strict=True):
            if (view["name"], view["axis"], view["hold_position"], view["control_mode"]) != (
                expected.name,
                expected.axis,
                expected.hold_position,
                profile["control_mode"],
            ) or not view["initial_controller_state_valid"]:
                raise ValueError("flight response identity or controller state mismatch")
            if json.dumps(view["commands"], sort_keys=True) != json.dumps(
                [asdict(command) for command in expected.commands], sort_keys=True
            ):
                raise ValueError(f"capture commands differ for {expected.name}")
            _validate_commands(profile=expected, view=view)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": os.path.relpath(dataset.run_dir, start=run_dir),
        "source_digest": dataset.artifact_digest,
        **membership,
    }
    with (run_dir / "dataset.json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    (run_dir / "flight_parameters").symlink_to(metadata["source"], target_is_directory=True)
    for role in membership:
        directory = run_dir if role == "train" else run_dir / role
        directory.mkdir(exist_ok=True)
        descriptor = {
            "schema_version": 1,
            "dataset": os.path.relpath(run_dir / "dataset.json", start=directory),
            "set_name": role,
            "controller_id": definition.controller_id,
            "definition_fingerprint": definition.definition_fingerprint,
            "image": definition.image,
            "profiles": profiles,
        }
        (directory / "reference.json").write_text(json.dumps(descriptor, indent=2) + "\n")


def training_hover_limits(
    *,
    dataset: FlightDataset,
    membership: dict[str, Any],
    position_margin_m: float,
    attitude_margin_rad: float,
) -> dict[str, float]:
    """Derive bounds only from complete native-hover windows in training flights."""
    maxima: dict[str, float] = {}
    for flight in dataset.flights:
        if flight.flight_index not in membership["train"]:
            continue
        hover = flight_stage_views(flight)[0]
        for position, attitude in zip(
            hover["position_enu_m"], hover["attitude_flu_enu_wxyz"], strict=True
        ):
            values = hover_pose_errors(
                target_position_enu_m=hover["effective_target_position_enu_m"],
                target_attitude_flu_enu_wxyz=hover["effective_target_attitude_flu_enu_wxyz"],
                position_enu_m=position,
                attitude_flu_enu_wxyz=attitude,
            )
            for name, value in values.items():
                maxima[name] = max(maxima.get(name, 0.0), value)
    return {
        name: value + (position_margin_m if "position" in name else attitude_margin_rad)
        for name, value in maxima.items()
    }

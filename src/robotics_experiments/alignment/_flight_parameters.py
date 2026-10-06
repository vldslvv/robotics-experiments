"""Lossless flight-parameters preparation for the existing alignment workflow."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from robotics_experiments.alignment._models import ControllerDefinition, ProbeTrace
from robotics_experiments.config import load_config
from robotics_experiments.flight_parameters import (
    derive_hover_parameters,
    flight_parameters_from_payload,
    flight_stage_views,
    load_flight_dataset,
    load_flight_selection,
)

__all__: list[str] = []


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
    dataset, membership = load_flight_selection(path.parent / descriptor["dataset"])
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
    """Check capture compatibility and split whole flights for alignment."""
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
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": os.path.relpath(dataset.run_dir, start=run_dir),
        "source_digest": dataset.artifact_digest,
        **membership,
        "hover": asdict(
            derive_hover_parameters(
                dataset=dataset,
                flight_indices=membership["train"],
                policy_warmup_s=config.alignment.evaluation.policy_warmup_s,
                policy_window_s=config.alignment.evaluation.policy_window_s,
                position_margin_m=config.alignment.evaluation.hover_position_margin_m,
                attitude_margin_rad=config.alignment.evaluation.hover_attitude_margin_rad,
            )
        ),
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


def training_hover_limits(*, membership: dict[str, Any]) -> dict[str, float]:
    """Use the shared full-window bounds recorded during preparation."""
    hover = flight_parameters_from_payload(membership["hover"]).hover
    limits = {}
    for prefix, axes, bounds in (
        ("position", ("x", "y", "z"), hover.position_error_enu_m),
        ("attitude", ("roll", "pitch", "yaw"), hover.attitude_error_rpy_rad),
    ):
        for axis, lower, upper in zip(axes, bounds.lower, bounds.upper, strict=True):
            limits[f"hover.{prefix}.{axis}.positive"] = upper
            limits[f"hover.{prefix}.{axis}.negative"] = -lower
    limits.update(
        {
            "hover.horizontal_position": hover.horizontal_position_error_m.upper,
            "hover.position": hover.position_error_m.upper,
            "hover.tilt": hover.tilt_error_rad.upper,
            "hover.orientation": hover.orientation_error_rad.upper,
        }
    )
    return limits

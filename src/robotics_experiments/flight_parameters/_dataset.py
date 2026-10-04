"""Immutable unsplit flight datasets and lossless stage views."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from robotics_experiments.flight_parameters._evidence import flight_evidence_sha256
from robotics_experiments.flight_parameters._models import FlightTrace

__all__: list[str] = []

_KIND = "robotics-experiments-flight-dataset"


@dataclass(frozen=True)
class FlightRecording:
    """One complete physical flight and its measured stage boundaries."""

    flight_index: int
    seed: int
    directory: Path
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FlightDataset:
    """Measured evidence, with no fitted bounds or consumer dataset split."""

    run_dir: Path
    flight_parameters_id: str
    run_id: str
    definition: Mapping[str, Any]
    artifact_digest: str
    flights: tuple[FlightRecording, ...]


def _digest(path: Path, /) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _freeze(value: Any, /) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _checked_file(*, root: Path, record: Mapping[str, Any]) -> Path:
    path = root / record["path"]
    if not path.resolve(strict=True).is_relative_to(root) or path.is_symlink():
        raise ValueError("flight dataset evidence must be producer-owned files inside its run")
    if path.stat().st_size != record["size_bytes"] or _digest(path) != record["sha256"]:
        raise ValueError(f"flight dataset evidence hash or size mismatch: {path}")
    if "content_sha256" in record and flight_evidence_sha256(path) != record["content_sha256"]:
        raise ValueError(f"flight dataset evidence content mismatch: {path}")
    return path


def _read_dataset(*, run: Path, artifact: Path) -> FlightDataset:
    payload = json.loads(artifact.read_text())
    if payload.get("artifact_kind") != _KIND or payload.get("schema_version") != 1:
        raise ValueError("expected an unsplit flight dataset, schema 1")
    if payload["run_id"] != run.name or not payload["completed"]:
        raise ValueError("flight dataset is incomplete or has the wrong run identity")
    definition = json.loads(_checked_file(root=run, record=payload["definition"]).read_text())
    if definition["flight_parameters_id"] != payload["flight_parameters_id"]:
        raise ValueError("flight dataset definition identity mismatch")
    flights = []
    for index, record in enumerate(payload["flights"]):
        directory = _checked_file(root=run, record=record).parent
        completed = json.loads((directory / "complete.json").read_text())
        expected = definition["start_seed"] + index
        if completed["flight"] != {
            "index": index,
            "seed": expected,
            "directory": directory.relative_to(run).as_posix(),
        }:
            raise ValueError("flight dataset flight identity mismatch")
        if not completed["completed"]:
            raise ValueError("flight recording processing is incomplete")
        for item in completed["files"]:
            _checked_file(root=directory, record=item)
        metadata = json.loads((directory / "flight.json").read_text())
        if (metadata["flight_index"], metadata["seed"], metadata["run_id"]) != (
            index,
            expected,
            run.name,
        ):
            raise ValueError("flight recording metadata identity mismatch")
        if not metadata["completed"] or len(metadata["stages"]) != len(definition["stages"]):
            raise ValueError("flight recording stage coverage mismatch")
        flights.append(FlightRecording(index, expected, directory, _freeze(metadata)))
    if len(flights) != definition["flight_count"] or not flights:
        raise ValueError("flight dataset is incomplete")
    manifest_path = run / "manifest.json"
    if manifest_path.exists():
        _validate_manifest(
            run=run, path=manifest_path, flight_parameters_id=payload["flight_parameters_id"]
        )
    return FlightDataset(
        run,
        payload["flight_parameters_id"],
        payload["run_id"],
        _freeze(definition),
        _digest(artifact),
        tuple(flights),
    )


def load_flight_dataset(path: str | Path, /) -> FlightDataset:
    """Resolve a selector once, validate all evidence, and return frozen metadata."""
    resolved = Path(path).resolve(strict=True)
    run = resolved if resolved.is_dir() else resolved.parent
    artifact = run / "flight_parameters.json" if resolved.is_dir() else resolved
    return _read_dataset(run=run, artifact=artifact)


def _stage_views(flight: FlightRecording, /) -> tuple[dict[str, Any], ...]:
    """Read original sample ranges without interpolation or shifted windows."""
    with (flight.directory / "motion.jsonl").open() as stream:
        rows = tuple(json.loads(line) for line in stream)
    times = np.asarray([row["sim_time_us"] for row in rows])
    if len(rows) != flight.metadata["sample_count"] or np.any(np.diff(times) <= 0):
        raise ValueError("flight motion sample count or timeline mismatch")
    views = []
    for metadata in flight.metadata["stages"]:
        start, stop = metadata["sample_start"], metadata["sample_stop"]
        stage_rows = rows[start:stop]
        if not stage_rows or any(
            row["stage_index"] != metadata["stage_index"] or row["phase"] != "measurement"
            for row in stage_rows
        ):
            raise ValueError("flight measurement range includes another stage or preparation")
        view = json.loads(
            json.dumps(
                metadata, default=lambda value: dict(value) if isinstance(value, Mapping) else value
            )
        )
        view["time_s"] = [
            (row["sim_time_us"] - metadata["start_sim_time_us"]) * 1e-6 for row in stage_rows
        ]
        for key in (
            "position_enu_m",
            "attitude_flu_enu_wxyz",
            "linear_velocity_enu_mps",
            "velocity_body_flu_mps",
            "angular_velocity_body_flu_radps",
            "angular_velocity_enu_radps",
        ):
            view[key] = [row[key] for row in stage_rows]
        if metadata["kind"] == "controller-response":
            commands = [row["requested_command"] for row in stage_rows]
            view["command_velocity_body_flu_mps"] = [
                (c.get("forward_mps", 0.0), -c.get("right_mps", 0.0), -c.get("down_mps", 0.0))
                for c in commands
            ]
            view["command_yaw_rate_radps"] = [-c.get("yaw_rate_radps", 0.0) for c in commands]
            view["command_body_rate_flu_radps"] = [
                (
                    c.get("roll_rate_radps", 0.0),
                    -c.get("pitch_rate_radps", 0.0),
                    -c.get("yaw_rate_radps", 0.0),
                )
                for c in commands
            ]
            view["command_collective_thrust_normalized"] = [c.get("thrust", 0.0) for c in commands]
        views.append(view)
    return tuple(views)


def _hover_trace(
    *, dataset: FlightDataset, flight: FlightRecording, view: dict[str, Any]
) -> FlightTrace:
    """In-memory pose-analysis input, without writing a consumer artifact."""
    return FlightTrace(
        artifact_kind="robotics-experiments-flight-parameters-trace",
        schema_version=1,
        flight_parameters_id=dataset.flight_parameters_id,
        campaign_run_id=dataset.run_id,
        trial_index=flight.flight_index,
        set_name="campaign",
        seed=flight.seed,
        source_backend="px4-sitl",
        state_source="simulator-ground-truth",
        config_fingerprint="",
        experiment_fingerprint="",
        vehicle_fingerprint="",
        source_image="",
        verification_fingerprint="",
        measurement_start_us=view["start_sim_time_us"],
        measurement_end_us=view["end_sim_time_us"],
        target_position_enu_m=view["effective_target_position_enu_m"],
        target_attitude_flu_enu_wxyz=view["effective_target_attitude_flu_enu_wxyz"],
        time_s=tuple(view["time_s"]),
        position_enu_m=tuple(tuple(v) for v in view["position_enu_m"]),
        attitude_flu_enu_wxyz=tuple(tuple(v) for v in view["attitude_flu_enu_wxyz"]),
        linear_velocity_enu_mps=tuple(tuple(v) for v in view["linear_velocity_enu_mps"]),
        angular_velocity_enu_radps=tuple(tuple(v) for v in view["angular_velocity_enu_radps"]),
        completed=True,
        failure_reason="",
    )


def write_flight_dataset(*, run_dir: Path, payload: dict[str, Any]) -> None:
    """Write complete producer evidence only after all per-flight checkpoints."""
    payload = {"artifact_kind": _KIND, "schema_version": 1, **payload}
    path = run_dir / "flight_parameters.json"
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix(".json.pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _read_dataset(run=run_dir.resolve(strict=True), artifact=pending)
    pending.replace(path)


def _validate_manifest(*, run: Path, path: Path, flight_parameters_id: str) -> None:
    value = json.loads(path.read_text())
    if set(value) != {"schema_version", "run_id", "flight_parameters_id", "files"} or (
        value["schema_version"],
        value["run_id"],
        value["flight_parameters_id"],
    ) != (1, run.name, flight_parameters_id):
        raise ValueError("flight dataset manifest identity mismatch")
    entries = value["files"]
    inventoried = {item["path"] for item in entries}
    actual = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file() and p != path}
    if len(inventoried) != len(entries) or inventoried != actual:
        raise ValueError("flight dataset manifest must inventory every file exactly once")
    for item in entries:
        if set(item) != {"path", "size_bytes", "sha256"}:
            raise ValueError("flight dataset manifest file fields mismatch")
        _checked_file(root=run, record=item)


def finalize_flight_dataset(run_dir: Path, /) -> Path:
    """Validate the final inventory and atomically publish a relative selector."""
    run_dir = run_dir.resolve(strict=True)
    dataset = load_flight_dataset(run_dir)
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    pending = manifest.with_suffix(".json.pending")
    files = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != pending
    ]
    pending.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": dataset.run_id,
                "flight_parameters_id": dataset.flight_parameters_id,
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    _validate_manifest(run=run_dir, path=pending, flight_parameters_id=dataset.flight_parameters_id)
    pending.replace(manifest)
    load_flight_dataset(run_dir)
    latest = run_dir.parent / "latest"
    temporary = latest.with_name(f".latest-{dataset.run_id}")
    temporary.symlink_to(os.path.relpath(run_dir, latest.parent), target_is_directory=True)
    temporary.replace(latest)
    return latest

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


def _read_dataset(*, run: Path, artifact: Path) -> FlightDataset:
    payload = json.loads(artifact.read_text())
    if payload.get("artifact_kind") != _KIND or payload.get("schema_version") != 1:
        raise ValueError("expected an unsplit flight dataset, schema 1")
    if payload["run_id"] != run.name or not payload["completed"]:
        raise ValueError("flight dataset is incomplete or has the wrong run identity")
    definition = json.loads((run / payload["definition"]["path"]).read_text())
    flights = []
    for record in payload["flights"]:
        directory = (run / record["path"]).parent
        completed = json.loads((directory / "complete.json").read_text())
        metadata = json.loads((directory / "flight.json").read_text())
        flights.append(
            FlightRecording(
                completed["flight"]["index"],
                completed["flight"]["seed"],
                directory,
                _freeze(metadata),
            )
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
    """Resolve a selector once, read its format, and return frozen metadata."""
    resolved = Path(path).resolve(strict=True)
    run = resolved if resolved.is_dir() else resolved.parent
    artifact = run / "flight_parameters.json" if resolved.is_dir() else resolved
    return _read_dataset(run=run, artifact=artifact)


def flight_stage_views(flight: FlightRecording, /) -> tuple[dict[str, Any], ...]:
    """Read original sample ranges without interpolation or shifted windows."""
    with (flight.directory / "motion.jsonl").open() as stream:
        rows = tuple(json.loads(line) for line in stream)
    views = []
    for metadata in flight.metadata["stages"]:
        start, stop = metadata["sample_start"], metadata["sample_stop"]
        stage_rows = rows[start:stop]
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
            view["command_time_s"] = [
                (row["requested_at_us"] - metadata["start_sim_time_us"]) * 1e-6
                for row in stage_rows
            ]
            view["command_velocity_body_flu_mps"] = [
                (c.get("forward_mps", 0.0), -c.get("right_mps", 0.0), -c.get("down_mps", 0.0))
                for c in commands
            ]
            view["command_yaw_rate_radps"] = [
                -c.get("yaw_rate_radps", 0.0) if metadata["control_mode"] == "velocity" else 0.0
                for c in commands
            ]
            view["command_body_rate_flu_radps"] = [
                (
                    c.get("roll_rate_radps", 0.0),
                    -c.get("pitch_rate_radps", 0.0),
                    -c.get("yaw_rate_radps", 0.0),
                )
                if metadata["control_mode"] == "ctbr"
                else (0.0, 0.0, 0.0)
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
    payload = {
        "artifact_kind": _KIND,
        "schema_version": 1,
        **payload,
        "flight_count": len(payload["flights"]),
    }
    path = run_dir / "flight_parameters.json"
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix(".json.pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def finalize_flight_dataset(run_dir: Path, /) -> Path:
    """Inventory the completed run and atomically publish a relative selector."""
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
    pending.replace(manifest)
    latest = run_dir.parent / "latest"
    temporary = latest.with_name(f".latest-{dataset.run_id}")
    temporary.symlink_to(os.path.relpath(run_dir, latest.parent), target_is_directory=True)
    temporary.replace(latest)
    return latest


def load_flight_selection(path: Path, /) -> tuple[FlightDataset, dict[str, Any]]:
    """Read the alignment selection and its producer dataset."""
    metadata = json.loads(path.read_text())
    dataset = load_flight_dataset(path.parent / metadata["source"])
    return dataset, metadata

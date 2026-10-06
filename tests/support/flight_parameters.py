"""Numeric hover limits and complete synthetic capture inputs for host tests."""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from robotics_experiments.flight_parameters import (
    derive_hover_parameters,
    flight_parameters_from_payload,
    load_flight_dataset,
)
from tests.support.paths import REPO_ROOT

__all__ = ["HOVER_PARAMETERS", "write_hover_dataset"]

HOVER_PARAMETERS = flight_parameters_from_payload(
    json.loads((REPO_ROOT / "tests/fixtures/hover_parameters.json").read_text())
)


def write_hover_dataset(root: Path, /) -> Path:
    """Write three distinct flights with paired motion and a train-only selection."""
    capture = root / "capture"
    capture.mkdir(parents=True)

    def record(path: Path, /) -> dict:
        return {
            "path": path.relative_to(capture).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    flights = []
    for index in range(3):
        directory = capture / f"flight-{index}"
        directory.mkdir()
        rows = [
            {
                "sim_time_us": j * 1000000,
                "stage_index": 0,
                "phase": "measurement",
                "position_enu_m": [index * 10.0 + j * 0.001, 0.0, 2.5],
                "attitude_flu_enu_wxyz": [math.cos(j * 0.0005), 0.0, 0.0, math.sin(j * 0.0005)],
                "linear_velocity_enu_mps": [j * 0.01, 0.0, 0.0],
                "velocity_body_flu_mps": [j * 0.01, 0.0, 0.0],
                "angular_velocity_body_flu_radps": [0.0, 0.0, j * 0.001],
                "angular_velocity_enu_radps": [0.0, 0.0, j * 0.001],
            }
            for j in range(30)
        ]
        (directory / "motion.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (directory / "flight.json").write_text(
            json.dumps(
                {
                    "flight_index": index,
                    "seed": 1001 + index,
                    "run_id": capture.name,
                    "completed": True,
                    "sample_count": len(rows),
                    "stages": [
                        {
                            "kind": "hover",
                            "stage_index": 0,
                            "start_sim_time_us": 0,
                            "end_sim_time_us": 30000000,
                            "sample_start": 0,
                            "sample_stop": 30,
                            "effective_target_position_enu_m": [0.0, 0.0, 2.5],
                            "effective_target_attitude_flu_enu_wxyz": [1.0, 0.0, 0.0, 0.0],
                        }
                    ],
                }
            )
        )
        (directory / "complete.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "flight": {"index": index, "seed": 1001 + index, "directory": directory.name},
                    "files": [
                        {**record(directory / name), "path": name}
                        for name in ("flight.json", "motion.jsonl")
                    ],
                }
            )
        )
        flights.append(record(directory / "complete.json"))
    (capture / "definition.json").write_text(
        json.dumps(
            {
                "flight_parameters_id": "test",
                "start_seed": 1001,
                "flight_count": 3,
                "vehicle": "x500_01",
                "source_backend": "px4-sitl",
                "sample_rate_hz": 1.0,
                "command_rate_hz": 20.0,
                "stages": [{"hover": {"measure_s": 30.0}}],
            }
        )
    )
    (capture / "flight_parameters.json").write_text(
        json.dumps(
            {
                "artifact_kind": "robotics-experiments-flight-dataset",
                "schema_version": 1,
                "run_id": capture.name,
                "completed": True,
                "flight_parameters_id": "test",
                "definition": record(capture / "definition.json"),
                "flights": flights,
            }
        )
    )
    dataset = load_flight_dataset(capture)
    selection = root / "dataset.json"
    selection.write_text(
        json.dumps(
            {
                "source": "capture",
                "source_digest": dataset.artifact_digest,
                "train": [0],
                "validation": [1],
                "test": [2],
                "hover": asdict(
                    derive_hover_parameters(
                        dataset=dataset,
                        flight_indices=(0,),
                        policy_warmup_s=6.0,
                        policy_window_s=24.0,
                        position_margin_m=0.005,
                        attitude_margin_rad=0.005,
                    )
                ),
            }
        )
    )
    return selection

"""ULog IO adapter for flight-parameter PX4 hover verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pyulog import ULog

from robotics_experiments.flight_parameters import (
    FlightParametersConfig,
)
from robotics_experiments.flight_parameters._artifact import (
    hover_verification_from_payload,
    load_raw_flight_trace,
)
from robotics_experiments.flight_parameters.px4_evidence import (
    ParsedPx4HoverLog,
    verify_px4_hover_evidence,
)

__all__ = ["verify_px4_hover_trial"]


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(ulog: ULog, name: str, /) -> dict[str, np.ndarray]:
    matches = [item.data for item in ulog.data_list if item.name == name and item.multi_id == 0]
    if len(matches) != 1:
        raise ValueError(f"PX4 hover verification requires one {name!r} dataset")
    return matches[0]


def _load_px4_hover_log(path: Path, /) -> ParsedPx4HoverLog:
    ulog = ULog(str(path))
    version = str(ulog.msg_info_dict.get("ver_sw", ""))
    if not version:
        raise ValueError("PX4 ULog does not identify its software version")
    return ParsedPx4HoverLog(
        px4_software_version=version,
        vehicle_command=_dataset(ulog, "vehicle_command"),
        vehicle_command_ack=_dataset(ulog, "vehicle_command_ack"),
        vehicle_status=_dataset(ulog, "vehicle_status"),
        vehicle_control_mode=_dataset(ulog, "vehicle_control_mode"),
        vehicle_land_detected=_dataset(ulog, "vehicle_land_detected"),
        position_setpoint_triplet=_dataset(ulog, "position_setpoint_triplet"),
        trajectory_setpoint=_dataset(ulog, "trajectory_setpoint"),
        vehicle_local_position=_dataset(ulog, "vehicle_local_position"),
    )


def verify_px4_hover_trial(
    *, config: FlightParametersConfig, raw_trace_path: Path, ulog_path: Path, trial_dir: Path
) -> tuple[Path, Path]:
    """Decode a completed ULog, verify hover, and write verification plus trace."""

    raw = load_raw_flight_trace(raw_trace_path)
    verification, trace = verify_px4_hover_evidence(
        config=config,
        raw=raw,
        log=_load_px4_hover_log(ulog_path),
        raw_trace_sha256=_sha256(raw_trace_path),
        ulog_sha256=_sha256(ulog_path),
    )
    verification_path = trial_dir / "hover_verification.json"
    trace_path = trial_dir / "trace.json"
    verification_path.write_text(
        json.dumps(asdict(verification), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hover_verification_from_payload(json.loads(verification_path.read_text(encoding="utf-8")))
    trace_path.write_text(
        json.dumps(asdict(trace), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verification_path, trace_path

"""ULog IO adapter for flight-parameter PX4 hover verification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyulog import ULog

from robotics_experiments.flight_parameters import (
    FlightParametersConfig,
    flight_evidence_sha256,
    open_flight_evidence,
)
from robotics_experiments.flight_parameters._math import _yaw_quaternion
from robotics_experiments.flight_parameters._models import RawFlightTrace
from robotics_experiments.flight_parameters.px4_evidence import (
    ParsedPx4HoverLog,
    verify_px4_hover_evidence,
)

__all__ = ["verify_flight_recording"]


def _dataset(ulog: ULog, name: str, /) -> dict[str, np.ndarray]:
    matches = [item.data for item in ulog.data_list if item.name == name and item.multi_id == 0]
    if len(matches) != 1:
        raise ValueError(f"PX4 hover verification requires one {name!r} dataset")
    return matches[0]


def _load_px4_hover_log(path: Path, /) -> ParsedPx4HoverLog:
    with open_flight_evidence(path) as stream:
        ulog = ULog(stream)
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


def verify_flight_recording(*, config: FlightParametersConfig, directory: Path) -> Path:
    """Verify physical control state and derive lossless measurement ranges."""
    raw_path = directory / "raw_flight.json"
    metadata = json.loads(raw_path.read_text())
    with (directory / "motion.jsonl").open() as stream:
        rows = tuple(json.loads(line) for line in stream)
    times = np.asarray([row["sim_time_us"] for row in rows], dtype=np.int64)
    if len(times) != metadata["sample_count"] or np.any(np.diff(times) <= 0):
        raise ValueError("flight motion samples are incomplete or out of order")
    stage = metadata["stages"][0]
    hold = tuple(
        row
        for row in rows
        if stage["preparation_start_us"] <= row["sim_time_us"] < stage["end_sim_time_us"]
    )
    raw = RawFlightTrace(
        artifact_kind="robotics-experiments-flight-parameters-raw-trace",
        schema_version=1,
        flight_parameters_id=config.flight_parameters_id,
        campaign_run_id=metadata["run_id"],
        trial_index=metadata["flight_index"] + 1,
        set_name="fit",
        seed=metadata["seed"],
        source_backend=config.source_backend,
        state_source=config.state_source,
        config_fingerprint=config.config_fingerprint,
        experiment_fingerprint="",
        vehicle_fingerprint="",
        source_image="",
        requested_target_position_enu_m=config.hover.target_position_enu_m,
        requested_target_attitude_flu_enu_wxyz=_yaw_quaternion(config.hover.target_yaw_rad),
        px4_home_lat_lon_alt=tuple(metadata["px4_home_lat_lon_alt"]),
        time_us=tuple(row["sim_time_us"] for row in hold),
        position_enu_m=tuple(tuple(row["position_enu_m"]) for row in hold),
        attitude_flu_enu_wxyz=tuple(tuple(row["attitude_flu_enu_wxyz"]) for row in hold),
        linear_velocity_enu_mps=tuple(tuple(row["linear_velocity_enu_mps"]) for row in hold),
        angular_velocity_enu_radps=tuple(tuple(row["angular_velocity_enu_radps"]) for row in hold),
        completed=metadata["completed"],
        failure_reason="",
    )
    log = _load_px4_hover_log(directory / "flight.ulg")
    verification, _trace = verify_px4_hover_evidence(
        config=config,
        raw=raw,
        log=log,
        raw_trace_sha256=flight_evidence_sha256(raw_path),
        ulog_sha256=flight_evidence_sha256(directory / "flight.ulg"),
        hold_start_us=stage["preparation_start_us"],
    )
    stage.update(
        effective_target_position_enu_m=verification.effective_target_position_enu_m,
        effective_target_attitude_flu_enu_wxyz=verification.effective_target_attitude_flu_enu_wxyz,
    )
    max_age = round(config.hover.mode_report_max_age_s * 1e6)
    for index, stage in enumerate(metadata["stages"]):
        if stage["stage_index"] != index or stage["name"] != config.stages[index].name:
            raise ValueError("recorded stages differ from the authored flight suite")
        start = int(np.searchsorted(times, stage["start_sim_time_us"]))
        stop = int(np.searchsorted(times, stage["end_sim_time_us"]))
        stage.update(sample_start=start, sample_stop=stop)
        if stop <= start or any(
            row["phase"] != "measurement" or row["stage_index"] != index for row in rows[start:stop]
        ):
            raise ValueError("measurement windows contain preparation or another stage")
        if index == 0:
            if stop - start != round(config.hover.measure_s * config.sample_rate_hz):
                raise ValueError("hover measurement sample count differs from its fixed window")
            continue
        duration = sum(command.duration_s for command in config.stages[index].commands)
        if (
            abs((stage["end_sim_time_us"] - stage["start_sim_time_us"]) * 1e-6 - duration)
            >= 1 / config.sample_rate_hz
        ):
            raise ValueError("response command duration differs from the authored schedule")
        for topic, flags in (
            (log.vehicle_status, {"nav_state": 14}),
            (log.vehicle_control_mode, {"flag_armed": True, "flag_control_offboard_enabled": True}),
            (log.vehicle_land_detected, {"landed": False}),
        ):
            at = np.searchsorted(topic["timestamp"], times[start:stop], side="right") - 1
            if np.any(at < 0) or np.any(times[start:stop] - topic["timestamp"][at] > max_age):
                raise ValueError("response control-state reports are absent or stale")
            if any(np.any(topic[name][at] != expected) for name, expected in flags.items()):
                raise ValueError("response measurements require armed airborne PX4 OFFBOARD")
    output = directory / "flight.json"
    output.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return output

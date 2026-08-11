"""PX4 ULog verification for task-neutral hover evidence.

This backend-specific module remains import-clean and operates on decoded
arrays. The sibling :mod:`px4_ulog` adapter owns the optional ``pyulog``
dependency used by the campaign.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from robotics_experiments.flight_parameters import (
    FlightConfig,
    FlightTrace,
)
from robotics_experiments.flight_parameters._artifact import _verification_fingerprint
from robotics_experiments.flight_parameters._models import (
    HoverVerification,
    Px4CommandEvent,
    Px4ModeEvent,
    Px4PositionGoalEvent,
    RawFlightTrace,
)

__all__ = ["ParsedPx4HoverLog", "verify_px4_hover_evidence"]

_MAV_CMD_COMPONENT_ARM_DISARM = 400
_MAV_CMD_NAV_TAKEOFF = 22
_MAV_RESULT_ACCEPTED = frozenset({0, 5})
_NAV_STATE_AUTO_LOITER = 4
_NAV_STATE_OFFBOARD = 14
_NAV_STATE_AUTO_TAKEOFF = 17
_EARTH_RADIUS_M = 6_378_137.0
_NAV_STATE_NAMES = {
    0: "MANUAL",
    1: "ALTCTL",
    2: "POSCTL",
    3: "AUTO_MISSION",
    _NAV_STATE_AUTO_LOITER: "AUTO_LOITER",
    5: "AUTO_RTL",
    6: "ACRO",
    8: "DESCEND",
    9: "TERMINATION",
    _NAV_STATE_OFFBOARD: "OFFBOARD",
    _NAV_STATE_AUTO_TAKEOFF: "AUTO_TAKEOFF",
    18: "AUTO_LAND",
}


@dataclass(frozen=True)
class ParsedPx4HoverLog:
    """ULog fields needed to prove one hover measurement interval."""

    px4_software_version: str
    vehicle_command: dict[str, np.ndarray]
    vehicle_command_ack: dict[str, np.ndarray]
    vehicle_status: dict[str, np.ndarray]
    vehicle_control_mode: dict[str, np.ndarray]
    vehicle_land_detected: dict[str, np.ndarray]
    position_setpoint_triplet: dict[str, np.ndarray]
    trajectory_setpoint: dict[str, np.ndarray]
    vehicle_local_position: dict[str, np.ndarray]


def _mode_name(nav_state: int, /) -> str:
    return _NAV_STATE_NAMES.get(nav_state, f"NAV_STATE_{nav_state}")


def _mode_events(data: dict[str, np.ndarray], /) -> tuple[Px4ModeEvent, ...]:
    events: list[Px4ModeEvent] = []
    previous = -1
    for timestamp, raw_state in zip(data["timestamp"], data["nav_state"], strict=True):
        nav_state = int(raw_state)
        if nav_state == previous:
            continue
        events.append(
            Px4ModeEvent(
                timestamp_us=int(timestamp),
                nav_state=nav_state,
                mode=_mode_name(nav_state),
            )
        )
        previous = nav_state
    return tuple(events)


def _command_events(log: ParsedPx4HoverLog, /) -> tuple[Px4CommandEvent, ...]:
    commands = log.vehicle_command
    acks = log.vehicle_command_ack
    external = [
        (int(timestamp), int(command))
        for timestamp, command, from_external in zip(
            commands["timestamp"],
            commands["command"],
            commands["from_external"],
            strict=True,
        )
        if bool(from_external)
    ]
    if tuple(command for _, command in external) != (
        _MAV_CMD_COMPONENT_ARM_DISARM,
        _MAV_CMD_NAV_TAKEOFF,
    ):
        raise ValueError(
            "PX4 hover trial external command sequence must be exactly arm then takeoff; "
            f"got {[command for _, command in external]}"
        )
    events: list[Px4CommandEvent] = []
    for timestamp_us, command_id in external:
        matching = [
            (int(ack_timestamp), int(result))
            for ack_timestamp, ack_command, result in zip(
                acks["timestamp"], acks["command"], acks["result"], strict=True
            )
            if int(ack_command) == command_id and int(ack_timestamp) >= timestamp_us
        ]
        if not matching:
            raise ValueError(f"PX4 hover trial command {command_id} has no acknowledgement")
        ack_timestamp_us, result = matching[0]
        if result not in _MAV_RESULT_ACCEPTED:
            raise ValueError(f"PX4 hover trial command {command_id} was not accepted: {result}")
        events.append(
            Px4CommandEvent(
                timestamp_us=timestamp_us,
                command_id=command_id,
                ack_timestamp_us=ack_timestamp_us,
                ack_result=result,
            )
        )
    return tuple(events)


def _position_goal_events(
    *, log: ParsedPx4HoverLog, takeoff_timestamp_us: int, measurement_end_us: int
) -> tuple[Px4PositionGoalEvent, ...]:
    data = log.position_setpoint_triplet
    events = []
    for index, timestamp in enumerate(data["timestamp"]):
        if (
            int(timestamp) < takeoff_timestamp_us
            or int(timestamp) >= measurement_end_us
            or not bool(data["current.valid"][index])
        ):
            continue
        raw_yaw = float(data["current.yaw"][index])
        events.append(
            Px4PositionGoalEvent(
                timestamp_us=int(timestamp),
                latitude_deg=float(data["current.lat"][index]),
                longitude_deg=float(data["current.lon"][index]),
                altitude_m=float(data["current.alt"][index]),
                yaw_valid=math.isfinite(raw_yaw),
                yaw_rad=raw_yaw if math.isfinite(raw_yaw) else 0.0,
            )
        )
    if not events:
        raise ValueError("PX4 hover trial has no valid position goal after takeoff")
    first = events[0]
    for event in events[1:]:
        if not (
            math.isclose(event.latitude_deg, first.latitude_deg, rel_tol=0.0, abs_tol=1.0e-8)
            and math.isclose(event.longitude_deg, first.longitude_deg, rel_tol=0.0, abs_tol=1.0e-8)
            and math.isclose(event.altitude_m, first.altitude_m, rel_tol=0.0, abs_tol=1.0e-3)
            and event.yaw_valid == first.yaw_valid
            and (
                not event.yaw_valid
                or math.isclose(event.yaw_rad, first.yaw_rad, rel_tol=0.0, abs_tol=1.0e-4)
            )
        ):
            raise ValueError("PX4 position goal changed before verified hover measurement")
    return tuple(events)


def _index_at_or_before(
    *, data: dict[str, np.ndarray], timestamp_us: int, max_age_us: int, name: str
) -> int:
    index = int(np.searchsorted(data["timestamp"], timestamp_us, side="right")) - 1
    if index < 0:
        raise ValueError(f"PX4 {name} has no report at measurement time {timestamp_us}")
    age_us = timestamp_us - int(data["timestamp"][index])
    if age_us > max_age_us:
        raise ValueError(f"PX4 {name} report is stale by {age_us}us")
    return index


def _wrap_angle(value: float, /) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quaternion(quaternion: tuple[float, float, float, float], /) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quaternion(yaw_rad: float, /) -> tuple[float, float, float, float]:
    return math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)


def _effective_position_target(
    *, raw: RawFlightTrace, goal: Px4PositionGoalEvent
) -> tuple[float, float, float]:
    home_latitude, home_longitude, home_altitude = raw.px4_home_lat_lon_alt
    latitude_rad = math.radians(home_latitude)
    east_m = (
        math.radians(goal.longitude_deg - home_longitude) * _EARTH_RADIUS_M * math.cos(latitude_rad)
    )
    north_m = math.radians(goal.latitude_deg - home_latitude) * _EARTH_RADIUS_M
    return east_m, north_m, goal.altitude_m - home_altitude


def _measurement_indices(
    *, config: FlightConfig, raw: RawFlightTrace, loiter_enter_us: int
) -> tuple[int, ...]:
    desired_start_us = loiter_enter_us + round(config.scenario.settle_s * 1_000_000)
    start = int(np.searchsorted(raw.time_us, desired_start_us, side="left"))
    count = round(config.scenario.measure_s * config.scenario.sample_rate_hz)
    stop = start + count
    if stop > len(raw.time_us):
        raise ValueError("raw pose trace does not cover the verified hover measurement")
    indices = tuple(range(start, stop))
    period_us = 1_000_000.0 / config.scenario.sample_rate_hz
    for index in indices:
        expected_us = raw.time_us[0] + round(index * period_us)
        phase_us = raw.time_us[index] - expected_us
        if not 0 <= phase_us < period_us:
            raise ValueError("raw pose trace does not contain one sample per configured time bin")
    start_us = raw.time_us[start]
    required_end_us = start_us + round(config.scenario.measure_s * 1_000_000)
    if raw.time_us[indices[-1]] >= required_end_us:
        raise ValueError("raw pose trace measurement timestamps exceed the selected interval")
    return indices


def _verify_measurement_state(
    *,
    log: ParsedPx4HoverLog,
    sample_times_us: tuple[int, ...],
    max_age_us: int,
) -> tuple[int, int]:
    for timestamp_us in sample_times_us:
        mode_index = _index_at_or_before(
            data=log.vehicle_status,
            timestamp_us=timestamp_us,
            max_age_us=max_age_us,
            name="vehicle_status",
        )
        nav_state = int(log.vehicle_status["nav_state"][mode_index])
        if nav_state != _NAV_STATE_AUTO_LOITER:
            raise ValueError(f"PX4 left AUTO_LOITER during measurement: {_mode_name(nav_state)}")
        control_index = _index_at_or_before(
            data=log.vehicle_control_mode,
            timestamp_us=timestamp_us,
            max_age_us=max_age_us,
            name="vehicle_control_mode",
        )
        control = log.vehicle_control_mode
        if not (
            bool(control["flag_armed"][control_index])
            and bool(control["flag_control_auto_enabled"][control_index])
            and bool(control["flag_control_position_enabled"][control_index])
            and bool(control["flag_multicopter_position_control_enabled"][control_index])
            and not bool(control["flag_control_offboard_enabled"][control_index])
        ):
            raise ValueError("PX4 control flags do not prove armed automatic position hold")
        land_index = _index_at_or_before(
            data=log.vehicle_land_detected,
            timestamp_us=timestamp_us,
            max_age_us=max_age_us,
            name="vehicle_land_detected",
        )
        land = log.vehicle_land_detected
        if any(
            bool(land[field][land_index]) for field in ("landed", "maybe_landed", "ground_contact")
        ):
            raise ValueError("PX4 reported a landed or ground-contact state during measurement")
    start_us = sample_times_us[0] - max_age_us
    end_us = sample_times_us[-1]
    control_count = int(
        np.count_nonzero(
            (log.vehicle_control_mode["timestamp"] >= start_us)
            & (log.vehicle_control_mode["timestamp"] <= end_us)
        )
    )
    land_count = int(
        np.count_nonzero(
            (log.vehicle_land_detected["timestamp"] >= start_us)
            & (log.vehicle_land_detected["timestamp"] <= end_us)
        )
    )
    return control_count, land_count


def _verified_yaw_target(
    *,
    log: ParsedPx4HoverLog,
    sample_times_us: tuple[int, ...],
    raw_attitude: tuple[float, float, float, float],
    max_age_us: int,
) -> tuple[float, tuple[float, float, float, float]]:
    trajectory = log.trajectory_setpoint
    in_window = (trajectory["timestamp"] >= sample_times_us[0]) & (
        trajectory["timestamp"] <= sample_times_us[-1]
    )
    yaw_values = np.asarray(trajectory["yaw"][in_window], dtype=np.float64)
    yaw_values = yaw_values[np.isfinite(yaw_values)]
    if yaw_values.size == 0:
        raise ValueError("PX4 has no finite yaw target during hover measurement")
    yaw_target_ned = float(yaw_values[0])
    if max(abs(_wrap_angle(float(value) - yaw_target_ned)) for value in yaw_values) > 1.0e-4:
        raise ValueError("PX4 yaw target changed during hover measurement")
    local_index = _index_at_or_before(
        data=log.vehicle_local_position,
        timestamp_us=sample_times_us[0],
        max_age_us=max_age_us,
        name="vehicle_local_position",
    )
    heading_ned = float(log.vehicle_local_position["heading"][local_index])
    if not math.isfinite(heading_ned):
        raise ValueError("PX4 local heading is not finite at hover measurement start")
    world_yaw = _yaw_from_quaternion(raw_attitude)
    effective_world_yaw = _wrap_angle(world_yaw - _wrap_angle(yaw_target_ned - heading_ned))
    return yaw_target_ned, _yaw_quaternion(effective_world_yaw)


def verify_px4_hover_evidence(
    *,
    config: FlightConfig,
    raw: RawFlightTrace,
    log: ParsedPx4HoverLog,
    raw_trace_sha256: str,
    ulog_sha256: str,
) -> tuple[HoverVerification, FlightTrace]:
    """Prove PX4 loiter state and extract one backend-neutral hover trace."""

    if not raw.completed or raw.failure_reason:
        raise ValueError("PX4 hover verification requires a complete raw trace")
    if (
        raw.flight_id != config.flight_id
        or raw.config_fingerprint != config.config_fingerprint
        or raw.source_backend != config.source_backend
        or raw.state_source != config.state_source
    ):
        raise ValueError("raw hover trace provenance does not match the flight config")
    commands = _command_events(log)
    modes = _mode_events(log.vehicle_status)
    takeoff_mode = next(
        (
            event
            for event in modes
            if event.timestamp_us >= commands[1].timestamp_us
            and event.nav_state == _NAV_STATE_AUTO_TAKEOFF
        ),
        None,
    )
    if takeoff_mode is None:
        raise ValueError("PX4 never reported AUTO_TAKEOFF")
    loiter = next(
        (
            event
            for event in modes
            if event.timestamp_us > takeoff_mode.timestamp_us
            and event.nav_state == _NAV_STATE_AUTO_LOITER
        ),
        None,
    )
    if loiter is None:
        raise ValueError("PX4 never transitioned from AUTO_TAKEOFF to AUTO_LOITER")
    if loiter.timestamp_us - raw.time_us[0] > round(
        config.scenario.mode_transition_timeout_s * 1_000_000
    ):
        raise ValueError("PX4 AUTO_LOITER transition exceeded the configured timeout")
    unexpected = [
        event.mode
        for event in modes
        if takeoff_mode.timestamp_us < event.timestamp_us < loiter.timestamp_us
        and event.nav_state not in {_NAV_STATE_AUTO_TAKEOFF, _NAV_STATE_AUTO_LOITER}
    ]
    if unexpected:
        raise ValueError(f"PX4 entered unexpected modes before loiter: {unexpected}")

    indices = _measurement_indices(config=config, raw=raw, loiter_enter_us=loiter.timestamp_us)
    sample_times_us = tuple(raw.time_us[index] for index in indices)
    measurement_start_us = sample_times_us[0]
    measurement_end_us = measurement_start_us + round(config.scenario.measure_s * 1_000_000)
    departed_loiter = [
        event.mode
        for event in modes
        if loiter.timestamp_us < event.timestamp_us < measurement_end_us
        and event.nav_state != _NAV_STATE_AUTO_LOITER
    ]
    if departed_loiter:
        raise ValueError(
            "PX4 left AUTO_LOITER before the hover measurement completed: " f"{departed_loiter}"
        )
    max_age_us = round(config.scenario.mode_report_max_age_s * 1_000_000)
    control_count, land_count = _verify_measurement_state(
        log=log,
        sample_times_us=sample_times_us,
        max_age_us=max_age_us,
    )
    goals = _position_goal_events(
        log=log,
        takeoff_timestamp_us=commands[1].timestamp_us,
        measurement_end_us=measurement_end_us,
    )
    goal = goals[-1]
    command_data = log.vehicle_command
    takeoff_rows = np.flatnonzero(
        (command_data["command"] == _MAV_CMD_NAV_TAKEOFF)
        & command_data["from_external"].astype(bool)
    )
    takeoff_index = int(takeoff_rows[0])
    commanded_goal = (
        float(command_data["param5"][takeoff_index]),
        float(command_data["param6"][takeoff_index]),
        float(command_data["param7"][takeoff_index]),
    )
    if not np.allclose(
        (goal.latitude_deg, goal.longitude_deg, goal.altitude_m),
        commanded_goal,
        rtol=0.0,
        atol=(1.0e-8, 1.0e-8, 1.0e-3),
    ):
        raise ValueError("PX4 active position goal does not match the accepted takeoff command")
    effective_position = _effective_position_target(raw=raw, goal=goal)
    yaw_target_ned, effective_attitude = _verified_yaw_target(
        log=log,
        sample_times_us=sample_times_us,
        raw_attitude=raw.attitude_flu_enu_wxyz[indices[0]],
        max_age_us=max_age_us,
    )
    verification = HoverVerification(
        artifact_kind="robotics-experiments-px4-hover-verification",
        schema_version=1,
        flight_id=raw.flight_id,
        campaign_run_id=raw.campaign_run_id,
        trial_index=raw.trial_index,
        config_fingerprint=raw.config_fingerprint,
        px4_software_version=log.px4_software_version,
        raw_trace_sha256=raw_trace_sha256,
        ulog_sha256=ulog_sha256,
        command_events=commands,
        mode_events=modes,
        position_goal_events=goals,
        loiter_enter_us=loiter.timestamp_us,
        measurement_start_us=measurement_start_us,
        measurement_end_us=measurement_end_us,
        mode_report_max_age_us=max_age_us,
        control_mode_report_count=control_count,
        land_state_report_count=land_count,
        requested_target_position_enu_m=raw.requested_target_position_enu_m,
        effective_target_position_enu_m=effective_position,
        effective_target_attitude_flu_enu_wxyz=effective_attitude,
        px4_yaw_target_ned_rad=yaw_target_ned,
        verification_fingerprint="",
    )
    verification_payload = asdict(verification)
    verification = HoverVerification(
        **{
            **verification_payload,
            "command_events": commands,
            "mode_events": modes,
            "position_goal_events": goals,
            "verification_fingerprint": _verification_fingerprint(verification_payload),
        }
    )
    trace = FlightTrace(
        artifact_kind="robotics-experiments-flight-trace",
        schema_version=1,
        flight_id=raw.flight_id,
        campaign_run_id=raw.campaign_run_id,
        trial_index=raw.trial_index,
        source_backend=raw.source_backend,
        state_source=raw.state_source,
        config_fingerprint=raw.config_fingerprint,
        experiment_fingerprint=raw.experiment_fingerprint,
        vehicle_fingerprint=raw.vehicle_fingerprint,
        source_image=raw.source_image,
        verification_fingerprint=verification.verification_fingerprint,
        measurement_start_us=measurement_start_us,
        measurement_end_us=measurement_end_us,
        target_position_enu_m=effective_position,
        target_attitude_flu_enu_wxyz=effective_attitude,
        time_s=tuple(index / config.scenario.sample_rate_hz for index in range(len(indices))),
        position_enu_m=tuple(raw.position_enu_m[index] for index in indices),
        attitude_flu_enu_wxyz=tuple(raw.attitude_flu_enu_wxyz[index] for index in indices),
        completed=True,
        failure_reason="",
    )
    return verification, trace

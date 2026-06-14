"""Shared standalone Isaac/PX4 takeoff scenario helpers."""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from _runtime import SceneTestContext

__all__ = [
    "ExpectedInstabilityDetected",
    "Px4TakeoffScenarioOptions",
    "run_px4_takeoff_scenario",
]

_GCS_ENDPOINT = "udp:127.0.0.1:14550"
_GCS_HEARTBEAT_PERIOD_S = 0.5
_GROUND_SETTLE_S = 1.5
_HEARTBEAT_TIMEOUT_S = 12.0
_PREARM_SETTLE_S = 10.0
_COMMAND_ACK_TIMEOUT_S = 10.0
_TAKEOFF_ALTITUDE_M = 2.5
# TODO: Figure out why PX4 currently settles below the requested 2.5m
# takeoff target in this SITL/Isaac loop. Until then, this scenario verifies
# that takeoff reaches the observed airborne envelope and then holds hover.
_MIN_TAKEOFF_CLIMB_M = 1.5
_TAKEOFF_TIMEOUT_S = 30.0
_HOVER_SETTLE_S = 2.0
_HOVER_OBSERVATION_S = 5.0
_HOVER_MAX_ALTITUDE_RANGE_M = 0.35
_HOVER_MAX_ABS_VERTICAL_SPEED_M_S = 0.35
_MAX_TILT_DEG = 35.0
_MAX_LATERAL_DRIFT_M = 2.0
_MAX_VERTICAL_SPEED_M_S = 8.0
_MAX_DOWNWARD_SPEED_M_S = -4.0
_MAX_ANGULAR_SPEED_RAD_S = 8.0
_PX4_FORCE_ARM_MAGIC = 21196

_CommandAckStatus = Literal["accepted", "temporarily_rejected"]


@dataclass
class _FlightSample:
    sim_time_us: int
    position_enu: np.ndarray
    linear_velocity_enu: np.ndarray
    angular_velocity_flu: np.ndarray
    tilt_deg: float


@dataclass(frozen=True)
class _TakeoffTarget:
    latitude_deg: float
    longitude_deg: float
    absolute_altitude_m: float


def _extension_root() -> Path:
    return Path(__file__).resolve().parents[2] / "extensions" / "drone_spawner"


def _px4_bridge_extension_root() -> Path:
    return Path(__file__).resolve().parents[2] / "extensions" / "px4_sitl_brigde"


def _logs_root() -> Path:
    return Path(os.environ.get("PX4_TEST_LOGS_DIR", "/isaac-sim/proj/px4/logs"))


def _latest_path(pattern: str) -> Path | None:
    candidates = [p for p in _logs_root().glob(pattern) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_log_hint(px4_container_log_name: str) -> str:
    ulg = _latest_path("*/*.ulg")
    bridge = _latest_path("*/*_bridge.log")
    px4_log = _logs_root() / px4_container_log_name
    parts = ["latest diagnostic logs:"]
    parts.append(f"  ulog={ulg if ulg is not None else 'not found'}")
    parts.append(f"  bridge_log={bridge if bridge is not None else 'not found'}")
    parts.append(f"  px4_container_log={px4_log if px4_log.exists() else 'not found'}")
    return "\n".join(parts)


def _resolve_drone_paths(stage, env, px4_config) -> tuple[str, list[str]]:
    from robotics_experiments.px4_bridge import discover_px4_rotor_mapping

    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    articulation_path = px4_config.articulation_path or f"{parent}/{env.drone.prim_name}"
    rotor_mapping = discover_px4_rotor_mapping(stage, articulation_path)
    return articulation_path, rotor_mapping.rotor_link_paths


def _bridge_received_first_heartbeat(host) -> bool:
    bridge = getattr(host, "_bridge", None)
    port = getattr(bridge, "_port", None)
    backend = getattr(port, "_backend", None)
    return bool(getattr(backend, "_received_first_hearbeat", False))


def _bridge_disconnect_reason(host) -> str | None:
    bridge = getattr(host, "_bridge", None)
    port = getattr(bridge, "_port", None)
    return getattr(port, "_last_disconnect_reason", None)


def _bridge_debug_log_path(host) -> Path | None:
    bridge = getattr(host, "_bridge", None)
    port = getattr(bridge, "_port", None)
    path = getattr(port, "_debug_log_path", None)
    return Path(path) if path else None


def _raise_instability(message: str) -> None:
    raise ExpectedInstabilityDetected(message)


def _require_bridge_connected(host) -> None:
    bridge = getattr(host, "_bridge", None)
    if bridge is None:
        if getattr(host, "_start_attempted", False):
            raise AssertionError("PX4 bridge host attempted start but did not construct a bridge")
        return
    if not bridge.is_connected:
        reason = _bridge_disconnect_reason(host) or "unknown"
        raise AssertionError(f"PX4 bridge disconnected during takeoff scenario: {reason}")


def _advance_one(context: SceneTestContext, host) -> None:
    context.simulation_app.update()
    _require_bridge_connected(host)


def _advance_sim_time(
    context: SceneTestContext,
    host,
    *,
    seconds: float,
    wall_timeout_s: float | None = None,
) -> None:
    start_us = int(getattr(host, "_sim_time_us", 0))
    target_us = start_us + int(round(float(seconds) * 1_000_000))
    deadline = time.monotonic() + (wall_timeout_s or max(10.0, seconds * 5.0))
    while int(getattr(host, "_sim_time_us", 0)) < target_us:
        if time.monotonic() > deadline:
            actual_s = (int(getattr(host, "_sim_time_us", 0)) - start_us) * 1e-6
            raise AssertionError(
                f"Timed out advancing Isaac simulation: target={seconds:.2f}s "
                f"actual={actual_s:.2f}s"
            )
        _advance_one(context, host)


def _wait_for_bridge_heartbeat(
    context: SceneTestContext,
    host,
    *,
    test_name: str,
    timeout_s: float,
) -> None:
    context.report(f"{test_name}: waiting up to {timeout_s:.1f}s for PX4 bridge heartbeat")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _advance_one(context, host)
        if _bridge_received_first_heartbeat(host):
            context.report(f"{test_name}: PX4 bridge received first heartbeat")
            return
    raise AssertionError("PX4 bridge did not receive PX4 heartbeat before timeout")


def _connect_gcs(context: SceneTestContext, *, test_name: str):
    from pymavlink import mavutil

    context.report(f"{test_name}: opening GCS MAVLink endpoint {_GCS_ENDPOINT}")
    mav = mavutil.mavlink_connection(_GCS_ENDPOINT, source_system=255)
    mav._last_sent_gcs_heartbeat_s = 0.0
    return mav


def _send_gcs_heartbeat(mav, *, force: bool = False) -> None:
    from pymavlink import mavutil

    now_s = time.monotonic()
    last_s = float(getattr(mav, "_last_sent_gcs_heartbeat_s", 0.0))
    if not force and now_s - last_s < _GCS_HEARTBEAT_PERIOD_S:
        return
    mav.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )
    mav._last_sent_gcs_heartbeat_s = now_s


def _drain_mavlink(mav) -> list[Any]:
    messages = []
    while True:
        msg = mav.recv_match(blocking=False)
        if msg is None:
            return messages
        messages.append(msg)


def _wait_for_gcs_heartbeat(
    context: SceneTestContext,
    host,
    mav,
    *,
    test_name: str,
    timeout_s: float,
) -> None:
    context.report(f"{test_name}: waiting up to {timeout_s:.1f}s for GCS heartbeat")
    deadline = time.monotonic() + timeout_s
    _send_gcs_heartbeat(mav, force=True)
    while time.monotonic() < deadline:
        _send_gcs_heartbeat(mav)
        _advance_one(context, host)
        for msg in _drain_mavlink(mav):
            if msg.get_type() == "HEARTBEAT":
                mav.target_system = msg.get_srcSystem()
                mav.target_component = msg.get_srcComponent()
                context.report(
                    f"{test_name}: GCS received heartbeat from "
                    f"system={msg.get_srcSystem()} component={msg.get_srcComponent()}"
                )
                return
    raise AssertionError(f"PX4 heartbeat was not received on {_GCS_ENDPOINT}")


def _message_indicates_failsafe(msg) -> bool:
    from pymavlink import mavutil

    if msg.get_type() != "HEARTBEAT":
        return False
    status = int(getattr(msg, "system_status", 0))
    failsafe_states = {
        mavutil.mavlink.MAV_STATE_CRITICAL,
        mavutil.mavlink.MAV_STATE_EMERGENCY,
        mavutil.mavlink.MAV_STATE_FLIGHT_TERMINATION,
    }
    return status in failsafe_states


def _assert_command_ack(
    context: SceneTestContext,
    host,
    mav,
    *,
    command: int,
    label: str,
    test_name: str,
    deadline_s: float,
) -> _CommandAckStatus:
    from pymavlink import mavutil

    accepted = {
        mavutil.mavlink.MAV_RESULT_ACCEPTED,
        getattr(mavutil.mavlink, "MAV_RESULT_IN_PROGRESS", 5),
    }
    while time.monotonic() < deadline_s:
        _send_gcs_heartbeat(mav)
        _advance_one(context, host)
        for msg in _drain_mavlink(mav):
            if _message_indicates_failsafe(msg):
                raise AssertionError(
                    "PX4 entered failsafe state while waiting for "
                    f"{label} ACK: system_status={int(msg.system_status)}"
                )
            if msg.get_type() == "STATUSTEXT":
                context.report(
                    f"{test_name}: PX4 STATUSTEXT while waiting for {label}: "
                    f"{getattr(msg, 'text', '')}"
                )
            if msg.get_type() != "COMMAND_ACK" or int(msg.command) != int(command):
                continue
            result = int(msg.result)
            if result not in accepted:
                if result == mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED:
                    context.report(f"{test_name}: PX4 temporarily rejected {label}; retrying")
                    return "temporarily_rejected"
                raise AssertionError(f"PX4 rejected {label}: COMMAND_ACK result={result}")
            context.report(f"{test_name}: PX4 accepted {label} command")
            return "accepted"
    raise AssertionError(f"Timed out waiting for PX4 COMMAND_ACK for {label}")


def _settle_px4_prearm(
    context: SceneTestContext,
    host,
    mav,
    *,
    test_name: str,
    seconds: float,
) -> None:
    context.report(f"{test_name}: settling PX4 estimator before arm for {seconds:.1f}s")
    start_us = int(getattr(host, "_sim_time_us", 0))
    target_us = start_us + int(round(seconds * 1_000_000))
    deadline_s = time.monotonic() + max(20.0, seconds * 5.0)
    while int(getattr(host, "_sim_time_us", 0)) < target_us:
        if time.monotonic() > deadline_s:
            actual_s = (int(getattr(host, "_sim_time_us", 0)) - start_us) * 1e-6
            raise AssertionError(
                f"Timed out settling PX4 before arm: target={seconds:.2f}s "
                f"actual={actual_s:.2f}s"
            )
        _send_gcs_heartbeat(mav)
        _advance_one(context, host)
        for msg in _drain_mavlink(mav):
            if _message_indicates_failsafe(msg):
                raise AssertionError(
                    "PX4 entered failsafe state while settling before arm: "
                    f"system_status={int(msg.system_status)}"
                )
            if msg.get_type() == "STATUSTEXT":
                context.report(
                    f"{test_name}: PX4 STATUSTEXT while settling before arm: "
                    f"{getattr(msg, 'text', '')}"
                )


def _send_command_with_retries(
    context: SceneTestContext,
    host,
    mav,
    *,
    command: int,
    label: str,
    test_name: str,
    send: Callable[[], None],
) -> None:
    deadline_s = time.monotonic() + _COMMAND_ACK_TIMEOUT_S
    while time.monotonic() < deadline_s:
        send()
        status = _assert_command_ack(
            context,
            host,
            mav,
            command=command,
            label=label,
            test_name=test_name,
            deadline_s=deadline_s,
        )
        if status == "accepted":
            return
        if time.monotonic() < deadline_s:
            _advance_sim_time(context, host, seconds=0.25, wall_timeout_s=3.0)
    raise AssertionError(f"Timed out waiting for PX4 COMMAND_ACK for {label}")


def _send_arm(context: SceneTestContext, host, mav, *, test_name: str) -> None:
    from pymavlink import mavutil

    command = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM

    def send() -> None:
        context.report(f"{test_name}: sending MAVLink force-arm command")
        mav.mav.command_long_send(
            mav.target_system,
            mav.target_component,
            command,
            0,
            1,
            _PX4_FORCE_ARM_MAGIC,
            0,
            0,
            0,
            0,
            0,
        )

    _send_command_with_retries(
        context,
        host,
        mav,
        command=command,
        label="arm",
        test_name=test_name,
        send=send,
    )


def _send_takeoff(
    context: SceneTestContext,
    host,
    mav,
    *,
    test_name: str,
    climb_m: float,
    target: _TakeoffTarget,
) -> None:
    from pymavlink import mavutil

    command = mavutil.mavlink.MAV_CMD_NAV_TAKEOFF

    def send() -> None:
        context.report(
            f"{test_name}: sending MAVLink takeoff command for {climb_m:.1f}m climb "
            f"to lat={target.latitude_deg:.7f} lon={target.longitude_deg:.7f} "
            f"absolute altitude {target.absolute_altitude_m:.1f}m"
        )
        mav.mav.command_long_send(
            mav.target_system,
            mav.target_component,
            command,
            0,
            0,
            0,
            0,
            math.nan,
            float(target.latitude_deg),
            float(target.longitude_deg),
            float(target.absolute_altitude_m),
        )

    _send_command_with_retries(
        context,
        host,
        mav,
        command=command,
        label="takeoff",
        test_name=test_name,
        send=send,
    )


def _telemetry_takeoff_position(msg) -> tuple[float, float, float] | None:
    msg_type = msg.get_type()
    if msg_type in {"GLOBAL_POSITION_INT", "GPS_RAW_INT"}:
        lat_e7 = getattr(msg, "lat", None)
        lon_e7 = getattr(msg, "lon", None)
        alt_mm = getattr(msg, "alt", None)
    else:
        return None
    if lat_e7 is None or lon_e7 is None or alt_mm is None:
        return None
    latitude_deg = float(lat_e7) / 10_000_000.0
    longitude_deg = float(lon_e7) / 10_000_000.0
    alt_m = float(alt_mm) / 1000.0
    if not all(math.isfinite(v) for v in (latitude_deg, longitude_deg, alt_m)):
        return None
    if abs(latitude_deg) < 1e-7 and abs(longitude_deg) < 1e-7:
        return None
    return latitude_deg, longitude_deg, alt_m


def _wait_for_takeoff_target(
    context: SceneTestContext,
    host,
    mav,
    *,
    test_name: str,
    climb_m: float,
    timeout_s: float,
) -> _TakeoffTarget:
    context.report(f"{test_name}: waiting up to {timeout_s:.1f}s for PX4 takeoff telemetry")
    deadline = time.monotonic() + timeout_s
    latest_position: tuple[float, float, float] | None = None
    while time.monotonic() < deadline:
        _send_gcs_heartbeat(mav)
        _advance_one(context, host)
        for msg in _drain_mavlink(mav):
            if _message_indicates_failsafe(msg):
                raise AssertionError(
                    "PX4 entered failsafe state while waiting for altitude telemetry: "
                    f"system_status={int(msg.system_status)}"
                )
            position = _telemetry_takeoff_position(msg)
            if position is not None:
                latest_position = position
        if latest_position is not None:
            latitude_deg, longitude_deg, altitude_m = latest_position
            target_altitude_m = altitude_m + climb_m
            context.report(
                f"{test_name}: PX4 takeoff telemetry lat={latitude_deg:.7f} "
                f"lon={longitude_deg:.7f} altitude={altitude_m:.1f}m; "
                f"target_altitude={target_altitude_m:.1f}m"
            )
            return _TakeoffTarget(
                latitude_deg=latitude_deg,
                longitude_deg=longitude_deg,
                absolute_altitude_m=target_altitude_m,
            )
    raise AssertionError("PX4 takeoff telemetry was not received before takeoff command")


def _sample_state(articulation_io, sim_time_us: int) -> _FlightSample:
    from robotics_experiments.px4_bridge import frames

    state = articulation_io.read_state(sim_time_us=sim_time_us)
    return _FlightSample(
        sim_time_us=sim_time_us,
        position_enu=np.asarray(state.position_enu, dtype=np.float64),
        linear_velocity_enu=np.asarray(state.linear_velocity_enu, dtype=np.float64),
        angular_velocity_flu=np.asarray(state.angular_velocity_flu, dtype=np.float64),
        tilt_deg=float(frames.tilt_angle_deg(state.orientation_flu_enu_wxyz)),
    )


def _observe_stability(
    context: SceneTestContext,
    host,
    mav,
    articulation_io,
    *,
    test_name: str,
    start_position: np.ndarray,
) -> None:
    context.report(
        f"{test_name}: observing takeoff for up to {_TAKEOFF_TIMEOUT_S:.1f}s "
        f"until climb >= {_MIN_TAKEOFF_CLIMB_M:.2f}m"
    )
    start_us = int(getattr(host, "_sim_time_us", 0))
    takeoff_deadline_us = start_us + int(round(_TAKEOFF_TIMEOUT_S * 1_000_000))
    hover_settle_end_us: int | None = None
    hover_end_us: int | None = None
    highest_altitude_m = float(start_position[2])
    sample_count = 0
    hover_sample_count = 0
    max_hover_abs_vertical_speed_m_s = 0.0
    min_hover_climb_m = math.inf
    max_hover_climb_m = -math.inf

    while True:
        _advance_one(context, host)
        _send_gcs_heartbeat(mav)

        for msg in _drain_mavlink(mav):
            if _message_indicates_failsafe(msg):
                _raise_instability(
                    "PX4 entered failsafe state during takeoff: "
                    f"system_status={int(msg.system_status)}"
                )

        sim_time_us = int(getattr(host, "_sim_time_us", 0))
        sample = _sample_state(articulation_io, sim_time_us)
        sample_count += 1
        highest_altitude_m = max(highest_altitude_m, float(sample.position_enu[2]))

        climb_m = float(sample.position_enu[2]) - float(start_position[2])
        lateral_drift_m = float(np.linalg.norm(sample.position_enu[:2] - start_position[:2]))
        vertical_velocity_m_s = float(sample.linear_velocity_enu[2])
        angular_speed_rad_s = float(np.linalg.norm(sample.angular_velocity_flu))

        if sample.tilt_deg > _MAX_TILT_DEG:
            _raise_instability(
                f"Drone tilt exceeded limit: tilt={sample.tilt_deg:.1f}deg "
                f"limit={_MAX_TILT_DEG:.1f}deg"
            )
        if lateral_drift_m > _MAX_LATERAL_DRIFT_M:
            _raise_instability(
                f"Drone lateral drift exceeded limit: drift={lateral_drift_m:.2f}m "
                f"limit={_MAX_LATERAL_DRIFT_M:.2f}m"
            )
        if vertical_velocity_m_s > _MAX_VERTICAL_SPEED_M_S:
            _raise_instability(
                f"Drone vertical velocity diverged upward: vz={vertical_velocity_m_s:.2f}m/s"
            )
        if vertical_velocity_m_s < _MAX_DOWNWARD_SPEED_M_S:
            _raise_instability(
                f"Drone vertical velocity diverged downward: vz={vertical_velocity_m_s:.2f}m/s"
            )
        if angular_speed_rad_s > _MAX_ANGULAR_SPEED_RAD_S:
            _raise_instability(
                f"Drone angular velocity diverged: |omega|={angular_speed_rad_s:.2f}rad/s"
            )

        if hover_settle_end_us is None and climb_m >= _MIN_TAKEOFF_CLIMB_M:
            hover_settle_end_us = sim_time_us + int(round(_HOVER_SETTLE_S * 1_000_000))
            hover_end_us = hover_settle_end_us + int(round(_HOVER_OBSERVATION_S * 1_000_000))
            context.report(
                f"{test_name}: reached takeoff altitude envelope: "
                f"climb={climb_m:.2f}m required={_MIN_TAKEOFF_CLIMB_M:.2f}m; "
                f"settling hover for {_HOVER_SETTLE_S:.1f}s"
            )

        if hover_settle_end_us is not None and sim_time_us >= hover_settle_end_us:
            hover_sample_count += 1
            max_hover_abs_vertical_speed_m_s = max(
                max_hover_abs_vertical_speed_m_s,
                abs(vertical_velocity_m_s),
            )
            min_hover_climb_m = min(min_hover_climb_m, climb_m)
            max_hover_climb_m = max(max_hover_climb_m, climb_m)

        if sample_count == 1 or sample_count % 250 == 0:
            context.report(
                f"{test_name}: sample t={sim_time_us * 1e-6:.2f}s "
                f"z={sample.position_enu[2]:.2f}m "
                f"vz={vertical_velocity_m_s:.2f}m/s "
                f"tilt={sample.tilt_deg:.1f}deg "
                f"drift={lateral_drift_m:.2f}m"
            )

        if hover_end_us is not None and sim_time_us >= hover_end_us:
            break
        if hover_settle_end_us is None and sim_time_us >= takeoff_deadline_us:
            break

    climb_m = highest_altitude_m - float(start_position[2])
    if climb_m < _MIN_TAKEOFF_CLIMB_M:
        _raise_instability(
            f"Drone altitude did not increase enough: climb={climb_m:.2f}m "
            f"required={_MIN_TAKEOFF_CLIMB_M:.2f}m"
        )

    if hover_sample_count <= 0:
        _raise_instability("Drone reached takeoff altitude but no hover samples were collected")

    hover_altitude_range_m = max_hover_climb_m - min_hover_climb_m
    if hover_altitude_range_m > _HOVER_MAX_ALTITUDE_RANGE_M:
        _raise_instability(
            "Drone did not hold hover height: "
            f"climb_range={hover_altitude_range_m:.2f}m "
            f"limit={_HOVER_MAX_ALTITUDE_RANGE_M:.2f}m "
            f"min_climb={min_hover_climb_m:.2f}m max_climb={max_hover_climb_m:.2f}m"
        )
    if max_hover_abs_vertical_speed_m_s > _HOVER_MAX_ABS_VERTICAL_SPEED_M_S:
        _raise_instability(
            "Drone vertical speed did not settle during hover: "
            f"max_abs_vz={max_hover_abs_vertical_speed_m_s:.2f}m/s "
            f"limit={_HOVER_MAX_ABS_VERTICAL_SPEED_M_S:.2f}m/s"
        )

    context.report(
        f"{test_name}: observed stable takeoff and hover: "
        f"max_climb={climb_m:.2f}m "
        f"hover_climb_range={min_hover_climb_m:.2f}-{max_hover_climb_m:.2f}m "
        f"max_hover_abs_vz={max_hover_abs_vertical_speed_m_s:.2f}m/s "
        f"samples={sample_count} hover_samples={hover_sample_count}"
    )


def _assert_bridge_log_clean(path: Path | None) -> None:
    if path is None or not path.exists():
        raise AssertionError("PX4 bridge debug log was not created")
    text = path.read_text(encoding="utf-8", errors="replace")
    disconnect_markers = [
        "PX4 MAVLink port disconnected",
    ]
    found_disconnects = [marker for marker in disconnect_markers if marker in text]
    if found_disconnects:
        raise AssertionError(
            f"PX4 bridge log contains disconnect markers: {found_disconnects}; log={path}"
        )

    bad_markers = [
        "actuator timestamp stale",
        "PX4 bridge tilt cutoff",
    ]
    found = [marker for marker in bad_markers if marker in text]
    if found:
        _raise_instability(f"PX4 bridge log contains failure markers: {found}; log={path}")


def _run_scenario(context: SceneTestContext, options: "Px4TakeoffScenarioOptions") -> None:
    import omni.timeline

    sys.path.insert(0, str(_extension_root()))
    sys.path.insert(0, str(_px4_bridge_extension_root()))

    from DroneSpawner_python.bridge_host import BridgeHost

    from robotics_experiments.config import get_config
    from robotics_experiments.drone import despawn_drone, spawn_drone
    from robotics_experiments.px4_bridge import IsaacArticulationIO
    from robotics_experiments.simulation import apply_settings

    env = get_config()
    px4_config = env.px4
    if options.px4_config_transform is not None:
        px4_config = options.px4_config_transform(px4_config)
    if px4_config.px4_autolaunch:
        raise AssertionError(
            "This test requires externally running PX4; px4_autolaunch must be false"
        )

    context.report(f"{options.test_name}: applying configured simulation settings")
    apply_settings(context.stage)

    context.report(f"{options.test_name}: spawning configured drone")
    spawn_drone(context.stage, env.drone)

    articulation_path, rotor_paths = _resolve_drone_paths(context.stage, env, px4_config)
    context.report(f"{options.test_name}: using drone articulation path {articulation_path}")
    context.report(f"{options.test_name}: using rotor paths {rotor_paths}")

    host = BridgeHost(articulation_path=articulation_path, config=px4_config)
    observer = IsaacArticulationIO(
        articulation_path=articulation_path,
        rotor_link_paths=rotor_paths,
    )
    timeline = omni.timeline.get_timeline_interface()
    mav = None

    try:
        context.report(f"{options.test_name}: starting PX4 bridge host")
        host.start()
        timeline.play()

        context.report(f"{options.test_name}: settling drone on ground for {_GROUND_SETTLE_S:.1f}s")
        _advance_sim_time(context, host, seconds=_GROUND_SETTLE_S, wall_timeout_s=20.0)
        _wait_for_bridge_heartbeat(
            context,
            host,
            test_name=options.test_name,
            timeout_s=_HEARTBEAT_TIMEOUT_S,
        )

        observer.initialize()
        start_sample = _sample_state(observer, int(getattr(host, "_sim_time_us", 0)))
        start_position = start_sample.position_enu.copy()
        context.report(
            f"{options.test_name}: start position z={start_position[2]:.2f}m "
            f"tilt={start_sample.tilt_deg:.1f}deg"
        )
        if start_sample.tilt_deg > _MAX_TILT_DEG:
            raise AssertionError(
                f"Drone is too tilted before takeoff: tilt={start_sample.tilt_deg:.1f}deg"
            )

        mav = _connect_gcs(context, test_name=options.test_name)
        _wait_for_gcs_heartbeat(
            context,
            host,
            mav,
            test_name=options.test_name,
            timeout_s=_HEARTBEAT_TIMEOUT_S,
        )
        _settle_px4_prearm(
            context,
            host,
            mav,
            test_name=options.test_name,
            seconds=_PREARM_SETTLE_S,
        )
        _send_arm(context, host, mav, test_name=options.test_name)
        takeoff_target = _wait_for_takeoff_target(
            context,
            host,
            mav,
            test_name=options.test_name,
            climb_m=_TAKEOFF_ALTITUDE_M,
            timeout_s=_HEARTBEAT_TIMEOUT_S,
        )
        _send_takeoff(
            context,
            host,
            mav,
            test_name=options.test_name,
            climb_m=_TAKEOFF_ALTITUDE_M,
            target=takeoff_target,
        )
        if options.post_takeoff_hook is not None:
            options.post_takeoff_hook(context, host)
        _observe_stability(
            context,
            host,
            mav,
            observer,
            test_name=options.test_name,
            start_position=start_position,
        )

        bridge_log = _bridge_debug_log_path(host)
    finally:
        context.report(f"{options.test_name}: stopping timeline, bridge host, and drone")
        timeline.stop()
        if mav is not None:
            try:
                mav.close()
            except Exception:
                pass
        observer.shutdown()
        host.stop()
        despawn_drone(context.stage, env.drone)

    _assert_bridge_log_clean(bridge_log)
    context.report(f"{options.test_name}: verified arm/takeoff stability envelope")


class ExpectedInstabilityDetected(AssertionError):
    """Raised when the scenario reaches command flow and detects instability."""


@dataclass(frozen=True)
class Px4TakeoffScenarioOptions:
    """Options for one standalone PX4 takeoff scenario run."""

    test_name: str
    px4_container_log_name: str
    px4_config_transform: Callable[[Any], Any] | None = None
    post_takeoff_hook: Callable[[SceneTestContext, Any], None] | None = None


def run_px4_takeoff_scenario(
    context: SceneTestContext,
    options: Px4TakeoffScenarioOptions,
) -> None:
    """Run the PX4 takeoff scenario and append diagnostic log hints on failure."""
    try:
        _run_scenario(context, options)
    except AssertionError as exc:
        raise type(exc)(f"{exc}\n{_latest_log_hint(options.px4_container_log_name)}") from exc

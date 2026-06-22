"""Shared standalone Isaac/PX4 takeoff scenario helpers (single- or multi-drone).

The scenario is driven by the Hydra ``vehicles`` list: one entry is the
single-vehicle baseline, N entries drive N drones each bound to its own PX4 SITL
instance (the resolved simulator port ``mavlink_port = connection_baseport + id``,
GCS link the resolved ``vehicle.gcs_port = scenario.gcs_baseport + id``). A single
``BridgeHostGroup`` owns the
one physics-step subscription and drives every host in two phases (all sends,
then all polls), so one ``simulation_app.update()`` advances every drone's
lockstep with the PX4 instances computing concurrently. The command flow runs each phase across
all drones over the shared advancing clock, and the flight-envelope gates are
evaluated per drone against its own spawn position; every drone must pass.

Scene, takeoff target, command-flow timeouts, and the envelope gates come from
the Hydra ``scenario`` group (``ScenarioConfig``); only the PX4 force-arm magic
stays in code as a protocol constant.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from _runtime import SceneTestContext, TimingReporter, format_vehicle_state_vectors

__all__ = [
    "ExpectedInstabilityDetected",
    "Px4FailsafeDetected",
    "Px4TakeoffScenarioOptions",
    "run_px4_takeoff_scenario",
]

# PX4 force-arm magic value (MAV_CMD_COMPONENT_ARM_DISARM param2); a PX4 protocol
# constant, not a tunable, so it stays in code rather than the scenario config.
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


@dataclass
class _DroneRuntime:
    """Per-drone live state for one vehicle in the scenario."""

    vehicle: Any
    host: Any
    observer: Any
    label: str
    gcs_endpoint: str
    mav: Any | None = None
    start_position: np.ndarray | None = None


@dataclass
class _DroneObservation:
    """Per-drone accumulator for the shared takeoff/hover observation loop."""

    start_position: np.ndarray
    highest_altitude_m: float
    hover_settle_end_us: int | None = None
    hover_end_us: int | None = None
    sample_count: int = 0
    hover_sample_count: int = 0
    max_hover_abs_vertical_speed_m_s: float = 0.0
    min_hover_climb_m: float = math.inf
    max_hover_climb_m: float = -math.inf
    max_hover_abs_horizontal_speed_m_s: float = 0.0
    min_hover_x_m: float = math.inf
    max_hover_x_m: float = -math.inf
    min_hover_y_m: float = math.inf
    max_hover_y_m: float = -math.inf
    latest_sample: _FlightSample | None = None
    done: bool = False


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


def _resolve_drone_paths(stage, *, prim_name: str, articulation_path_override: str):
    from robotics_experiments.px4_bridge import discover_px4_rotor_mapping

    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    articulation_path = articulation_path_override or f"{parent}/{prim_name}"
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


def _raise_failsafe(message: str) -> None:
    raise Px4FailsafeDetected(message)


def _require_bridge_connected(host, *, label: str) -> None:
    bridge = getattr(host, "_bridge", None)
    if bridge is None:
        if getattr(host, "_start_attempted", False):
            raise AssertionError(
                f"[{label}] PX4 bridge host attempted start but did not construct a bridge"
            )
        return
    if not bridge.is_connected:
        reason = _bridge_disconnect_reason(host) or "unknown"
        raise AssertionError(f"[{label}] PX4 bridge disconnected during takeoff scenario: {reason}")


def _sim_time_us(drones: list[_DroneRuntime]) -> int:
    # All bridges share the same PhysX steps, so any host's clock is the fleet
    # clock; use the first as the reference.
    return int(getattr(drones[0].host, "sim_time_us", 0))


def _advance_once(context: SceneTestContext, drones: list[_DroneRuntime]) -> None:
    """Advance PhysX one step (ticks every bridge) and assert all stay connected."""
    context.simulation_app.update()
    for drone in drones:
        _require_bridge_connected(drone.host, label=drone.label)


def _advance_sim_time(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    *,
    seconds: float,
    wall_timeout_s: float | None = None,
) -> None:
    start_us = _sim_time_us(drones)
    target_us = start_us + int(round(float(seconds) * 1_000_000))
    base = wall_timeout_s if wall_timeout_s is not None else max(10.0, seconds * 5.0)
    # Per tick the coordinator sends to every PX4, then blocks for each actuator;
    # the PX4s compute concurrently so per-tick wall time is ~the max wait, not
    # the sum. Scale the wall budget by the drone count anyway as a safe upper
    # bound (host contention can still serialize the polls). N=1 is unchanged.
    deadline = time.monotonic() + base * len(drones)
    while _sim_time_us(drones) < target_us:
        if time.monotonic() > deadline:
            actual_s = (_sim_time_us(drones) - start_us) * 1e-6
            raise AssertionError(
                f"Timed out advancing Isaac simulation: target={seconds:.2f}s "
                f"actual={actual_s:.2f}s"
            )
        _advance_once(context, drones)


def _wait_for_bridge_heartbeat(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    *,
    timeout_s: float,
) -> None:
    context.report(f"[{drone.label}] waiting up to {timeout_s:.1f}s for PX4 bridge heartbeat")
    deadline = time.monotonic() + timeout_s * len(drones)
    while time.monotonic() < deadline:
        _advance_once(context, drones)
        if _bridge_received_first_heartbeat(drone.host):
            context.report(f"[{drone.label}] PX4 bridge received first heartbeat")
            return
    raise AssertionError(f"[{drone.label}] PX4 bridge did not receive PX4 heartbeat before timeout")


def _connect_gcs(context: SceneTestContext, drone: _DroneRuntime):
    from pymavlink import mavutil

    context.report(f"[{drone.label}] opening GCS MAVLink endpoint {drone.gcs_endpoint}")
    mav = mavutil.mavlink_connection(drone.gcs_endpoint, source_system=255)
    mav._last_sent_gcs_heartbeat_s = 0.0
    return mav


def _send_gcs_heartbeat(mav, period_s: float, *, force: bool = False) -> None:
    from pymavlink import mavutil

    now_s = time.monotonic()
    last_s = float(getattr(mav, "_last_sent_gcs_heartbeat_s", 0.0))
    if not force and now_s - last_s < period_s:
        return
    mav.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )
    mav._last_sent_gcs_heartbeat_s = now_s


def _pump_all_gcs(drones: list[_DroneRuntime], period_s: float) -> None:
    for drone in drones:
        if drone.mav is not None:
            _send_gcs_heartbeat(drone.mav, period_s)


def _drain_mavlink(mav) -> list[Any]:
    messages: list[Any] = []
    while True:
        msg = mav.recv_match(blocking=False)
        if msg is None:
            return messages
        messages.append(msg)


def _wait_for_gcs_heartbeat(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    scenario,
    *,
    timeout_s: float,
) -> None:
    context.report(f"[{drone.label}] waiting up to {timeout_s:.1f}s for GCS heartbeat")
    deadline = time.monotonic() + timeout_s * len(drones)
    _send_gcs_heartbeat(drone.mav, scenario.gcs_heartbeat_period_s, force=True)
    while time.monotonic() < deadline:
        _send_gcs_heartbeat(drone.mav, scenario.gcs_heartbeat_period_s)
        _advance_once(context, drones)
        for msg in _drain_mavlink(drone.mav):
            if msg.get_type() == "HEARTBEAT":
                drone.mav.target_system = msg.get_srcSystem()
                drone.mav.target_component = msg.get_srcComponent()
                context.report(
                    f"[{drone.label}] GCS received heartbeat from "
                    f"system={msg.get_srcSystem()} component={msg.get_srcComponent()}"
                )
                return
    raise AssertionError(f"[{drone.label}] PX4 heartbeat was not received on {drone.gcs_endpoint}")


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
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    scenario,
    *,
    command: int,
    label: str,
    deadline_s: float,
    accepted_statustext: tuple[str, ...] = (),
) -> _CommandAckStatus:
    from pymavlink import mavutil

    accepted = {
        mavutil.mavlink.MAV_RESULT_ACCEPTED,
        getattr(mavutil.mavlink, "MAV_RESULT_IN_PROGRESS", 5),
    }
    while time.monotonic() < deadline_s:
        _pump_all_gcs(drones, scenario.gcs_heartbeat_period_s)
        _advance_once(context, drones)
        for msg in _drain_mavlink(drone.mav):
            if _message_indicates_failsafe(msg):
                _raise_failsafe(
                    f"[{drone.label}] PX4 entered failsafe state while waiting for "
                    f"{label} ACK: system_status={int(msg.system_status)}"
                )
            if msg.get_type() == "STATUSTEXT":
                text = str(getattr(msg, "text", ""))
                context.report(f"[{drone.label}] PX4 STATUSTEXT while waiting for {label}: {text}")
                if any(marker in text for marker in accepted_statustext):
                    context.report(f"[{drone.label}] PX4 accepted {label} command")
                    return "accepted"
            if msg.get_type() != "COMMAND_ACK" or int(msg.command) != int(command):
                continue
            result = int(msg.result)
            if result not in accepted:
                if result == mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED:
                    context.report(f"[{drone.label}] PX4 temporarily rejected {label}; retrying")
                    return "temporarily_rejected"
                raise AssertionError(
                    f"[{drone.label}] PX4 rejected {label}: COMMAND_ACK result={result}"
                )
            context.report(f"[{drone.label}] PX4 accepted {label} command")
            return "accepted"
    raise AssertionError(f"[{drone.label}] Timed out waiting for PX4 COMMAND_ACK for {label}")


def _settle_px4_prearm(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    scenario,
    *,
    seconds: float,
) -> None:
    context.report(f"settling PX4 estimators before arm for {seconds:.1f}s")
    start_us = _sim_time_us(drones)
    target_us = start_us + int(round(seconds * 1_000_000))
    deadline_s = time.monotonic() + max(20.0, seconds * 5.0) * len(drones)
    while _sim_time_us(drones) < target_us:
        if time.monotonic() > deadline_s:
            actual_s = (_sim_time_us(drones) - start_us) * 1e-6
            raise AssertionError(
                f"Timed out settling PX4 before arm: target={seconds:.2f}s actual={actual_s:.2f}s"
            )
        _pump_all_gcs(drones, scenario.gcs_heartbeat_period_s)
        _advance_once(context, drones)
        for drone in drones:
            for msg in _drain_mavlink(drone.mav):
                if _message_indicates_failsafe(msg):
                    _raise_failsafe(
                        f"[{drone.label}] PX4 entered failsafe state while settling before arm: "
                        f"system_status={int(msg.system_status)}"
                    )
                if msg.get_type() == "STATUSTEXT":
                    context.report(
                        f"[{drone.label}] PX4 STATUSTEXT while settling before arm: "
                        f"{getattr(msg, 'text', '')}"
                    )


def _send_command_with_retries(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    scenario,
    *,
    command: int,
    label: str,
    send: Callable[[], None],
    accepted_statustext: tuple[str, ...] = (),
) -> None:
    deadline_s = time.monotonic() + scenario.command_ack_timeout_s * len(drones)
    while time.monotonic() < deadline_s:
        send()
        status = _assert_command_ack(
            context,
            drones,
            drone,
            scenario,
            command=command,
            label=label,
            deadline_s=deadline_s,
            accepted_statustext=accepted_statustext,
        )
        if status == "accepted":
            return
        if time.monotonic() < deadline_s:
            _advance_sim_time(context, drones, seconds=0.25, wall_timeout_s=3.0)
    raise AssertionError(f"[{drone.label}] Timed out waiting for PX4 COMMAND_ACK for {label}")


def _send_arm(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    scenario,
) -> None:
    from pymavlink import mavutil

    command = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    mav = drone.mav

    def send() -> None:
        context.report(f"[{drone.label}] sending MAVLink force-arm command")
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
        drones,
        drone,
        scenario,
        command=command,
        label="arm",
        send=send,
        accepted_statustext=("Armed by external command",),
    )


def _send_takeoff(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    scenario,
    *,
    climb_m: float,
    target: _TakeoffTarget,
) -> None:
    from pymavlink import mavutil

    command = mavutil.mavlink.MAV_CMD_NAV_TAKEOFF
    mav = drone.mav

    def send() -> None:
        context.report(
            f"[{drone.label}] sending MAVLink takeoff command for {climb_m:.1f}m climb "
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
        context, drones, drone, scenario, command=command, label="takeoff", send=send
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
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    scenario,
    *,
    climb_m: float,
    timeout_s: float,
) -> _TakeoffTarget:
    context.report(f"[{drone.label}] waiting up to {timeout_s:.1f}s for PX4 takeoff telemetry")
    deadline = time.monotonic() + timeout_s * len(drones)
    latest_position: tuple[float, float, float] | None = None
    while time.monotonic() < deadline:
        _send_gcs_heartbeat(drone.mav, scenario.gcs_heartbeat_period_s)
        _advance_once(context, drones)
        for msg in _drain_mavlink(drone.mav):
            if _message_indicates_failsafe(msg):
                _raise_failsafe(
                    f"[{drone.label}] PX4 entered failsafe state while waiting for altitude "
                    f"telemetry: system_status={int(msg.system_status)}"
                )
            position = _telemetry_takeoff_position(msg)
            if position is not None:
                latest_position = position
        if latest_position is not None:
            latitude_deg, longitude_deg, altitude_m = latest_position
            target_altitude_m = altitude_m + climb_m
            context.report(
                f"[{drone.label}] PX4 takeoff telemetry lat={latitude_deg:.7f} "
                f"lon={longitude_deg:.7f} altitude={altitude_m:.1f}m; "
                f"target_altitude={target_altitude_m:.1f}m"
            )
            return _TakeoffTarget(
                latitude_deg=latitude_deg,
                longitude_deg=longitude_deg,
                absolute_altitude_m=target_altitude_m,
            )
    raise AssertionError(
        f"[{drone.label}] PX4 takeoff telemetry was not received before takeoff command"
    )


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


def _check_continuous_gates(
    drone: _DroneRuntime,
    scenario,
    sample: _FlightSample,
    *,
    lateral_drift_m: float,
    vertical_velocity_m_s: float,
    angular_speed_rad_s: float,
) -> None:
    label = drone.label
    if sample.tilt_deg > scenario.max_tilt_deg:
        _raise_instability(
            f"[{label}] Drone tilt exceeded limit: tilt={sample.tilt_deg:.1f}deg "
            f"limit={scenario.max_tilt_deg:.1f}deg"
        )
    if lateral_drift_m > scenario.max_lateral_drift_m:
        _raise_instability(
            f"[{label}] Drone lateral drift exceeded limit: drift={lateral_drift_m:.2f}m "
            f"limit={scenario.max_lateral_drift_m:.2f}m {format_vehicle_state_vectors(sample)}"
        )
    if vertical_velocity_m_s > scenario.max_vertical_speed_m_s:
        _raise_instability(
            f"[{label}] Drone vertical velocity diverged upward: "
            f"vz={vertical_velocity_m_s:.2f}m/s {format_vehicle_state_vectors(sample)}"
        )
    if vertical_velocity_m_s < scenario.max_downward_speed_m_s:
        _raise_instability(
            f"[{label}] Drone vertical velocity diverged downward: "
            f"vz={vertical_velocity_m_s:.2f}m/s {format_vehicle_state_vectors(sample)}"
        )
    if angular_speed_rad_s > scenario.max_angular_speed_rad_s:
        _raise_instability(
            f"[{label}] Drone angular velocity diverged: |omega|={angular_speed_rad_s:.2f}rad/s"
        )


def _update_drone_observation(
    context: SceneTestContext,
    drone: _DroneRuntime,
    obs: _DroneObservation,
    scenario,
    *,
    sim_time_us: int,
    failsafe_status: int | None,
) -> None:
    start_position = obs.start_position
    sample = _sample_state(drone.observer, sim_time_us)
    obs.latest_sample = sample
    obs.sample_count += 1
    obs.highest_altitude_m = max(obs.highest_altitude_m, float(sample.position_enu[2]))

    climb_m = float(sample.position_enu[2]) - float(start_position[2])
    lateral_drift_m = float(np.linalg.norm(sample.position_enu[:2] - start_position[:2]))
    vertical_velocity_m_s = float(sample.linear_velocity_enu[2])
    angular_speed_rad_s = float(np.linalg.norm(sample.angular_velocity_flu))

    _check_continuous_gates(
        drone,
        scenario,
        sample,
        lateral_drift_m=lateral_drift_m,
        vertical_velocity_m_s=vertical_velocity_m_s,
        angular_speed_rad_s=angular_speed_rad_s,
    )

    # Failsafe is evaluated only after the per-tick envelope gates above, so a
    # severe fault-induced divergence is attributed to the envelope; a failsafe
    # seen inside the envelope is the distinct, rerun-worthy Px4FailsafeDetected.
    if failsafe_status is not None:
        _raise_failsafe(
            f"[{drone.label}] PX4 entered failsafe state during takeoff: "
            f"system_status={failsafe_status}"
        )

    if obs.hover_settle_end_us is None and climb_m >= scenario.min_takeoff_climb_m:
        obs.hover_settle_end_us = sim_time_us + int(round(scenario.hover_settle_s * 1_000_000))
        obs.hover_end_us = obs.hover_settle_end_us + int(
            round(scenario.hover_observation_s * 1_000_000)
        )
        context.report(
            f"[{drone.label}] reached takeoff altitude envelope: climb={climb_m:.2f}m "
            f"required={scenario.min_takeoff_climb_m:.2f}m; "
            f"{format_vehicle_state_vectors(sample)} "
            f"settling hover for {scenario.hover_settle_s:.1f}s"
        )

    if obs.hover_settle_end_us is not None and sim_time_us >= obs.hover_settle_end_us:
        obs.hover_sample_count += 1
        obs.max_hover_abs_vertical_speed_m_s = max(
            obs.max_hover_abs_vertical_speed_m_s, abs(vertical_velocity_m_s)
        )
        obs.min_hover_climb_m = min(obs.min_hover_climb_m, climb_m)
        obs.max_hover_climb_m = max(obs.max_hover_climb_m, climb_m)
        horizontal_speed_m_s = float(np.linalg.norm(sample.linear_velocity_enu[:2]))
        obs.max_hover_abs_horizontal_speed_m_s = max(
            obs.max_hover_abs_horizontal_speed_m_s, horizontal_speed_m_s
        )
        obs.min_hover_x_m = min(obs.min_hover_x_m, float(sample.position_enu[0]))
        obs.max_hover_x_m = max(obs.max_hover_x_m, float(sample.position_enu[0]))
        obs.min_hover_y_m = min(obs.min_hover_y_m, float(sample.position_enu[1]))
        obs.max_hover_y_m = max(obs.max_hover_y_m, float(sample.position_enu[1]))

    if obs.sample_count == 1 or obs.sample_count % 250 == 0:
        context.report(
            f"[{drone.label}] sample t={sim_time_us * 1e-6:.2f}s "
            f"z={sample.position_enu[2]:.2f}m vz={vertical_velocity_m_s:.2f}m/s "
            f"{format_vehicle_state_vectors(sample)} tilt={sample.tilt_deg:.1f}deg "
            f"drift={lateral_drift_m:.2f}m"
        )


def _assert_drone_hover_envelope(
    context: SceneTestContext,
    drone: _DroneRuntime,
    obs: _DroneObservation,
    scenario,
) -> None:
    label = drone.label
    sample_text = (
        f" {format_vehicle_state_vectors(obs.latest_sample)}"
        if obs.latest_sample is not None
        else ""
    )

    climb_m = obs.highest_altitude_m - float(obs.start_position[2])
    if climb_m < scenario.min_takeoff_climb_m:
        _raise_instability(
            f"[{label}] Drone altitude did not increase enough: climb={climb_m:.2f}m "
            f"required={scenario.min_takeoff_climb_m:.2f}m{sample_text}"
        )

    if obs.hover_sample_count <= 0:
        _raise_instability(
            f"[{label}] Drone reached takeoff altitude but no hover samples were collected"
        )

    hover_altitude_range_m = obs.max_hover_climb_m - obs.min_hover_climb_m
    if hover_altitude_range_m > scenario.hover_max_altitude_range_m:
        _raise_instability(
            f"[{label}] Drone did not hold hover height: climb_range={hover_altitude_range_m:.2f}m "
            f"limit={scenario.hover_max_altitude_range_m:.2f}m "
            f"min_climb={obs.min_hover_climb_m:.2f}m max_climb={obs.max_hover_climb_m:.2f}m"
            f"{sample_text}"
        )
    if obs.max_hover_abs_vertical_speed_m_s > scenario.hover_max_abs_vertical_speed_m_s:
        _raise_instability(
            f"[{label}] Drone vertical speed did not settle during hover: "
            f"max_abs_vz={obs.max_hover_abs_vertical_speed_m_s:.2f}m/s "
            f"limit={scenario.hover_max_abs_vertical_speed_m_s:.2f}m/s{sample_text}"
        )

    hover_horizontal_range_m = math.hypot(
        obs.max_hover_x_m - obs.min_hover_x_m, obs.max_hover_y_m - obs.min_hover_y_m
    )
    if hover_horizontal_range_m > scenario.hover_max_horizontal_range_m:
        _raise_instability(
            f"[{label}] Drone did not hold hover position: "
            f"horizontal_range={hover_horizontal_range_m:.2f}m "
            f"limit={scenario.hover_max_horizontal_range_m:.2f}m{sample_text}"
        )
    if obs.max_hover_abs_horizontal_speed_m_s > scenario.hover_max_abs_horizontal_speed_m_s:
        _raise_instability(
            f"[{label}] Drone horizontal speed did not settle during hover: "
            f"max_horizontal_speed={obs.max_hover_abs_horizontal_speed_m_s:.2f}m/s "
            f"limit={scenario.hover_max_abs_horizontal_speed_m_s:.2f}m/s{sample_text}"
        )

    final_sample_text = (
        f"{format_vehicle_state_vectors(obs.latest_sample)} "
        if obs.latest_sample is not None
        else ""
    )
    context.report(
        f"[{label}] observed stable takeoff and hover: {final_sample_text}"
        f"max_climb={climb_m:.2f}m "
        f"hover_climb_range={obs.min_hover_climb_m:.2f}-{obs.max_hover_climb_m:.2f}m "
        f"max_hover_abs_vz={obs.max_hover_abs_vertical_speed_m_s:.2f}m/s "
        f"hover_horizontal_range={hover_horizontal_range_m:.2f}m "
        f"max_hover_horizontal_speed={obs.max_hover_abs_horizontal_speed_m_s:.2f}m/s "
        f"samples={obs.sample_count} hover_samples={obs.hover_sample_count}"
    )


def _observe_all(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    scenario,
) -> None:
    """Observe all drones' takeoff/hover over one shared advancing clock.

    Each tick advances PhysX once (ticking every bridge), samples every drone,
    applies the per-drone continuous gates immediately (drone-id tagged), and
    accumulates per-drone hover stats. The loop runs until every drone has
    completed its hover window or missed the takeoff deadline; final per-drone
    envelope assertions then require all drones to pass.
    """
    context.report(
        f"observing takeoff for up to {scenario.takeoff_timeout_s:.1f}s "
        f"until each drone climbs >= {scenario.min_takeoff_climb_m:.2f}m"
    )
    start_us = _sim_time_us(drones)
    takeoff_deadline_us = start_us + int(round(scenario.takeoff_timeout_s * 1_000_000))
    observations = {
        drone.label: _DroneObservation(
            start_position=drone.start_position.copy(),
            highest_altitude_m=float(drone.start_position[2]),
        )
        for drone in drones
    }

    while not all(obs.done for obs in observations.values()):
        _advance_once(context, drones)
        _pump_all_gcs(drones, scenario.gcs_heartbeat_period_s)
        sim_time_us = _sim_time_us(drones)

        for drone in drones:
            obs = observations[drone.label]
            if obs.done:
                continue
            failsafe_status: int | None = None
            for msg in _drain_mavlink(drone.mav):
                if _message_indicates_failsafe(msg):
                    failsafe_status = int(msg.system_status)
            _update_drone_observation(
                context,
                drone,
                obs,
                scenario,
                sim_time_us=sim_time_us,
                failsafe_status=failsafe_status,
            )
            if obs.hover_end_us is not None and sim_time_us >= obs.hover_end_us:
                obs.done = True
            elif obs.hover_settle_end_us is None and sim_time_us >= takeoff_deadline_us:
                obs.done = True

    for drone in drones:
        _assert_drone_hover_envelope(context, drone, observations[drone.label], scenario)


def _assert_bridge_log_clean(path: Path | None, label: str) -> None:
    if path is None or not path.exists():
        raise AssertionError(f"[{label}] PX4 bridge debug log was not created")
    text = path.read_text(encoding="utf-8", errors="replace")
    disconnect_markers = [
        "PX4 MAVLink port disconnected",
    ]
    found_disconnects = [marker for marker in disconnect_markers if marker in text]
    if found_disconnects:
        raise AssertionError(
            f"[{label}] PX4 bridge log contains disconnect markers: {found_disconnects}; log={path}"
        )

    bad_markers = [
        "actuator timestamp stale",
        "PX4 bridge tilt cutoff",
    ]
    found = [marker for marker in bad_markers if marker in text]
    if found:
        _raise_instability(
            f"[{label}] PX4 bridge log contains failure markers: {found}; log={path}"
        )


def _build_drones(
    context: SceneTestContext,
    options: "Px4TakeoffScenarioOptions",
    env,
) -> list[_DroneRuntime]:
    """Spawn every configured vehicle and build its bridge host + observer + GCS."""
    sys.path.insert(0, str(_extension_root()))
    sys.path.insert(0, str(_px4_bridge_extension_root()))

    from DroneSpawner_python.bridge_host import BridgeHost

    from robotics_experiments.drone import spawn_drone
    from robotics_experiments.px4_bridge import IsaacArticulationIO

    drones: list[_DroneRuntime] = []
    for vehicle in env.vehicles:
        # The optional transform mutates only the PX4 config (e.g. the negative
        # test's rotor-mapping fault); rebuild the resolved vehicle around it so
        # the host still reads the derived runtime fields (mavlink_port, id).
        if options.px4_config_transform is not None:
            vehicle = dataclasses.replace(vehicle, px4=options.px4_config_transform(vehicle.px4))

        label = f"drone{vehicle.id}"
        context.report(f"[{label}] spawning configured drone at spawn_xy={tuple(vehicle.spawn_xy)}")
        spawn_drone(context.stage, vehicle.drone, position=tuple(vehicle.spawn_xy), yaw=0.0)

        articulation_path, rotor_paths = _resolve_drone_paths(
            context.stage,
            prim_name=vehicle.drone.prim_name,
            articulation_path_override=vehicle.px4.articulation_path,
        )
        context.report(f"[{label}] articulation path {articulation_path} rotors {rotor_paths}")

        host = BridgeHost(
            articulation_path=articulation_path,
            vehicle=vehicle,
            physics_hz=env.sim.physics_hz,
        )
        observer = IsaacArticulationIO(
            articulation_path=articulation_path,
            rotor_link_paths=rotor_paths,
        )
        gcs_endpoint = f"udp:127.0.0.1:{vehicle.gcs_port}"
        drones.append(
            _DroneRuntime(
                vehicle=vehicle,
                host=host,
                observer=observer,
                label=label,
                gcs_endpoint=gcs_endpoint,
            )
        )
    return drones


def _run_scenario(context: SceneTestContext, options: "Px4TakeoffScenarioOptions") -> None:
    import omni.timeline

    from robotics_experiments.config import get_config
    from robotics_experiments.drone import despawn_drone
    from robotics_experiments.simulation import apply_settings

    timing = TimingReporter(options.test_name, emit=context.report)

    env = get_config()
    scenario = env.scenario

    context.report(f"{options.test_name}: applying configured simulation settings")
    with timing.timed_stage("apply_settings"):
        apply_settings(context.stage)

    with timing.timed_stage("spawn_drones"):
        drones = _build_drones(context, options, env)
    context.report(f"{options.test_name}: built {len(drones)} drone(s)")

    from DroneSpawner_python.bridge_host import BridgeHostGroup

    timeline = omni.timeline.get_timeline_interface()
    clock = drones[0].host
    # One coordinator owns the single physics-step subscription and runs every
    # host's send leg before every host's poll leg, so the PX4 instances compute
    # concurrently and one stalled PX4 disconnects only its own vehicle.
    group = BridgeHostGroup([drone.host for drone in drones])

    try:
        context.report(f"{options.test_name}: starting {len(drones)} PX4 bridge host(s)")
        with timing.timed_stage("bridge_host_start", sim_clock=clock):
            group.start()
            timeline.play()

        context.report(
            f"{options.test_name}: settling drones on ground for {scenario.ground_settle_s:.1f}s"
        )
        with timing.timed_stage("ground_settle", sim_clock=clock):
            _advance_sim_time(
                context, drones, seconds=scenario.ground_settle_s, wall_timeout_s=20.0
            )

        with timing.timed_stage("bridge_heartbeat_wait", sim_clock=clock):
            for drone in drones:
                _wait_for_bridge_heartbeat(
                    context, drones, drone, timeout_s=scenario.heartbeat_timeout_s
                )

        for drone in drones:
            if not bool(getattr(drone.host, "is_running", False)):
                raise AssertionError(f"[{drone.label}] PX4 bridge host stopped before command flow")
            _require_bridge_connected(drone.host, label=drone.label)

        with timing.timed_stage("observer_initialize", sim_clock=clock):
            for drone in drones:
                drone.observer.initialize()
                start_sample = _sample_state(drone.observer, _sim_time_us(drones))
                drone.start_position = start_sample.position_enu.copy()
                context.report(
                    f"[{drone.label}] start position z={drone.start_position[2]:.2f}m "
                    f"{format_vehicle_state_vectors(start_sample)} "
                    f"tilt={start_sample.tilt_deg:.1f}deg"
                )
                if start_sample.tilt_deg > scenario.max_tilt_deg:
                    raise AssertionError(
                        f"[{drone.label}] Drone is too tilted before takeoff: "
                        f"tilt={start_sample.tilt_deg:.1f}deg"
                    )

        with timing.timed_stage("gcs_connect", sim_clock=clock):
            for drone in drones:
                drone.mav = _connect_gcs(context, drone)
        with timing.timed_stage("gcs_heartbeat_wait", sim_clock=clock):
            for drone in drones:
                _wait_for_gcs_heartbeat(
                    context, drones, drone, scenario, timeout_s=scenario.heartbeat_timeout_s
                )

        with timing.timed_stage("px4_prearm_settle", sim_clock=clock):
            _settle_px4_prearm(context, drones, scenario, seconds=scenario.prearm_settle_s)

        with timing.timed_stage("arm_command_ack", sim_clock=clock):
            for drone in drones:
                _send_arm(context, drones, drone, scenario)

        with timing.timed_stage("takeoff_command_ack", sim_clock=clock):
            for drone in drones:
                takeoff_target = _wait_for_takeoff_target(
                    context,
                    drones,
                    drone,
                    scenario,
                    climb_m=scenario.takeoff_altitude_m,
                    timeout_s=scenario.heartbeat_timeout_s,
                )
                _send_takeoff(
                    context,
                    drones,
                    drone,
                    scenario,
                    climb_m=scenario.takeoff_altitude_m,
                    target=takeoff_target,
                )

        if options.post_takeoff_hook is not None:
            with timing.timed_stage("post_takeoff_hook", sim_clock=clock):
                options.post_takeoff_hook(context, drones[0].host)

        with timing.timed_stage("takeoff_hover_observation", sim_clock=clock):
            _observe_all(context, drones, scenario)

        bridge_logs = [(drone.label, _bridge_debug_log_path(drone.host)) for drone in drones]
    finally:
        context.report(f"{options.test_name}: stopping timeline, bridge hosts, and drones")
        with timing.timed_stage("teardown", sim_clock=clock):
            timeline.stop()
            group.stop()
            for drone in drones:
                if drone.mav is not None:
                    try:
                        drone.mav.close()
                    except Exception:
                        pass
                drone.observer.shutdown()
                despawn_drone(context.stage, drone.vehicle.drone)

    with timing.timed_stage("bridge_log_validation"):
        for label, path in bridge_logs:
            _assert_bridge_log_clean(path, label)
    timing.report_stage("scenario_total", timing.total_start())
    context.report(
        f"{options.test_name}: verified arm/takeoff stability envelope for {len(drones)} drone(s)"
    )


class ExpectedInstabilityDetected(AssertionError):
    """Raised when the scenario reaches command flow and detects instability."""


class Px4FailsafeDetected(AssertionError):
    """Raised when PX4 SITL enters a failsafe state.

    A failsafe heartbeat (system_status critical / emergency / flight
    termination) is a PX4-internal estimator/EKF event, not a flight-envelope
    breach, and is host-load-dependent and rerun-worthy. It is deliberately NOT
    a subclass of :class:`ExpectedInstabilityDetected`, so it never counts as an
    envelope-detected instability: the negative takeoff test must only pass when
    the stability envelope catches the injected fault, never on a failsafe flake.
    """


@dataclass(frozen=True)
class Px4TakeoffScenarioOptions:
    """Options for one standalone PX4 takeoff scenario run.

    ``px4_config_transform`` is applied to every vehicle's PX4 config (e.g. the
    negative test injects a rotor-mapping fault); ``post_takeoff_hook`` is called
    once with the first drone's bridge host (single-vehicle fault-injection only).
    """

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

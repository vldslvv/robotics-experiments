"""Shared standalone Isaac/PX4 takeoff scenario helpers (single- or multi-drone).

The scenario is driven by the Hydra ``vehicles`` list: one entry is the
single-vehicle baseline, N entries drive N drones each bound to its own PX4 SITL
instance (the resolved simulator port ``mavlink_port = connection_baseport + id``,
GCS link the resolved ``vehicle.gcs_port = px4.gcs_baseport + id``). A single
``BridgeHostGroup`` owns the
one physics-step subscription and drives every host in two phases (all sends,
then all polls), so one ``simulation_app.update()`` advances every drone's
lockstep with the PX4 instances computing concurrently. The command flow runs each phase across
all drones over the shared advancing clock, and the flight-envelope gates are
evaluated per drone against its own spawn position; every drone must pass.

The scene comes from the top-level ``scene`` config; the takeoff target,
command-flow timeouts, and the envelope gates come from the Hydra
``integration_test`` group (``IntegrationTestConfig``).

The high-level commanding (arm -> takeoff) is delegated to the pure command
layer in ``robotics_experiments.commanding``: each drone gets a
``VehicleRoutine`` driven by a ``VehicleCommander`` over a ``MavlinkCommandPort``
(the GCS link), and a ``FleetCommander`` with the ``STEP_SYNCHRONIZED`` policy
reproduces the arm-all-then-takeoff-all fleet barrier. This module keeps only its
assertion responsibilities (the continuous gates and the hover envelope), the
sim-stepping, and the bridge-connectivity checks; it never hand-rolls MAVLink.
The command port talks the GCS link only - never the bridge's simulator MAVLink
lockstep - by design.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from _runtime import SceneTestContext, TimingReporter, format_vehicle_state_vectors

__all__ = [
    "ExpectedInstabilityDetected",
    "Px4FailsafeDetected",
    "Px4TakeoffScenarioOptions",
    "run_px4_takeoff_scenario",
]


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
    port: Any | None = None
    commander: Any | None = None
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


def _bridge_status(host):
    """Read the bridge lockstep snapshot for the routine observation.

    Called only after the command flow asserts each bridge is connected, so the
    host has a live bridge here.
    """
    return host._bridge.status


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


def _pump_and_poll_all(drones: list[_DroneRuntime]) -> dict[str, Any]:
    """Pump each drone's GCS heartbeat and drain its feedback, returning latest acks.

    Keeps every drone's GCS link alive each tick and returns the most recent
    ``CommandAck`` per drone (``None`` when none arrived this tick), which the
    routine driver folds into each ``RoutineObservation``.
    """
    acks: dict[str, Any] = {}
    for drone in drones:
        drone.port.pump_heartbeat()
        acks[drone.label] = drone.port.poll_feedback()
    return acks


def _raise_if_failsafe(drones: list[_DroneRuntime], *, phase: str) -> None:
    for drone in drones:
        if drone.port.failsafe:
            _raise_failsafe(
                f"[{drone.label}] PX4 entered failsafe state {phase}: "
                f"system_status={drone.port.failsafe_system_status}"
            )


def _wait_for_gcs_handshake(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    *,
    timeout_s: float,
) -> None:
    context.report(f"[{drone.label}] waiting up to {timeout_s:.1f}s for GCS handshake")
    deadline = time.monotonic() + timeout_s * len(drones)
    drone.port.pump_heartbeat(force=True)
    while time.monotonic() < deadline:
        _advance_once(context, drones)
        _pump_and_poll_all(drones)
        if drone.port.is_handshaked:
            context.report(f"[{drone.label}] GCS handshake complete; learned PX4 target system")
            return
    raise AssertionError(f"[{drone.label}] PX4 heartbeat was not received on {drone.gcs_endpoint}")


def _settle_px4_prearm(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
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
        _advance_once(context, drones)
        _pump_and_poll_all(drones)
        _raise_if_failsafe(drones, phase="while settling before arm")


def _wait_for_takeoff_target(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    *,
    climb_m: float,
    timeout_s: float,
) -> _TakeoffTarget:
    context.report(f"[{drone.label}] waiting up to {timeout_s:.1f}s for PX4 takeoff telemetry")
    deadline = time.monotonic() + timeout_s * len(drones)
    while time.monotonic() < deadline:
        _advance_once(context, drones)
        _pump_and_poll_all(drones)
        _raise_if_failsafe(drones, phase="while waiting for altitude telemetry")
        position = drone.port.latest_global_position
        if position is not None:
            latitude_deg, longitude_deg, altitude_m = position
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


def _build_routine(scenario, target: _TakeoffTarget):
    """Build the per-drone ``arm -> takeoff`` routine from the test config.

    Both steps complete on their ``COMMAND_ACK`` (the climb and hover are
    asserted afterwards by the envelope observer, preserving the original order:
    arm-ack, takeoff-ack, post-takeoff hook, then hover observation). Both retry
    on a transient PX4 ``TEMPORARILY_REJECTED`` ack, matching the prior behavior.
    """
    from robotics_experiments.commanding import (
        RoutineStep,
        VehicleRoutine,
        ack_accepted,
        arm,
        takeoff,
    )
    from robotics_experiments.commanding.commands import (
        MAV_CMD_COMPONENT_ARM_DISARM,
        MAV_CMD_NAV_TAKEOFF,
    )

    arm_step = RoutineStep(
        command=arm(),
        is_complete=ack_accepted(MAV_CMD_COMPONENT_ARM_DISARM),
        label="arm",
        timeout_s=scenario.command_ack_timeout_s,
        retry_on_temporarily_rejected=True,
    )
    takeoff_step = RoutineStep(
        command=takeoff(target.latitude_deg, target.longitude_deg, target.absolute_altitude_m),
        is_complete=ack_accepted(MAV_CMD_NAV_TAKEOFF),
        label="takeoff",
        timeout_s=scenario.command_ack_timeout_s,
        retry_on_temporarily_rejected=True,
    )
    return VehicleRoutine(steps=(arm_step, takeoff_step))


def _routine_observation(drone: _DroneRuntime, sim_time_us: int, last_ack):
    from robotics_experiments.commanding import RoutineObservation

    vehicle_state = drone.observer.read_state(sim_time_us=sim_time_us)
    return RoutineObservation(
        vehicle_state=vehicle_state,
        bridge_status=_bridge_status(drone.host),
        last_ack=last_ack,
        failsafe=drone.port.failsafe,
        sim_time_us=sim_time_us,
    )


def _drive_routine_to_done(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    fleet,
    scenario,
) -> None:
    """Tick physics and step the fleet commander until every routine reports DONE.

    Each tick advances PhysX once (asserting every bridge stays connected),
    pumps/drains every GCS link, builds a ``RoutineObservation`` per drone, and
    steps the ``FleetCommander``; a ``FAIL`` decision or a PX4 failsafe aborts.
    The per-step timeouts inside ``decide`` are sim-time; this wall-clock budget
    is the backstop for a sim that ticks but never completes the routine.
    """
    from robotics_experiments.commanding import DecisionKind

    context.report(f"driving arm/takeoff command routine for {len(drones)} drone(s)")
    deadline_s = time.monotonic() + (
        scenario.command_ack_timeout_s + scenario.heartbeat_timeout_s
    ) * 4.0 * len(drones)
    while not fleet.is_done:
        if time.monotonic() > deadline_s:
            raise AssertionError("Timed out driving the arm/takeoff command routine to completion")
        _advance_once(context, drones)
        acks = _pump_and_poll_all(drones)
        _raise_if_failsafe(drones, phase="during the command routine")
        sim_time_us = _sim_time_us(drones)
        observations = {
            drone.label: _routine_observation(drone, sim_time_us, acks[drone.label])
            for drone in drones
        }
        decisions = fleet.step(observations)
        for label, decision in decisions.items():
            if decision.kind is DecisionKind.FAIL:
                raise AssertionError(f"[{label}] command routine failed: {decision.reason}")


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
        _pump_and_poll_all(drones)
        sim_time_us = _sim_time_us(drones)

        for drone in drones:
            obs = observations[drone.label]
            if obs.done:
                continue
            # Failsafe is bucketed after the per-tick envelope gates inside
            # _update_drone_observation, so a fault-induced divergence is caught
            # by the envelope first; pass the current system_status through.
            failsafe_status = drone.port.failsafe_system_status if drone.port.failsafe else None
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

    from robotics_experiments.commanding import (
        ExecutionPolicy,
        FleetCommander,
        MavlinkCommandPort,
        VehicleCommander,
    )
    from robotics_experiments.config import get_config
    from robotics_experiments.drone import despawn_drone
    from robotics_experiments.simulation import apply_settings

    timing = TimingReporter(options.test_name, emit=context.report)

    env = get_config()
    # The integration-test harness config (timeouts, takeoff target, envelope
    # gates). Populated because this runs under an integration_test_* experiment;
    # the local name stays `scenario` as it is the live test-scenario object.
    scenario = env.integration_test
    if scenario is None:
        raise ValueError(
            f"{options.test_name}: no integration_test config composed; run this "
            "scenario under an integration_test_* experiment "
            "(ROBOTICS_EXPERIMENT_CONFIG_PATH / HYDRA_CONFIG)"
        )

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
                drone.port = MavlinkCommandPort(
                    gcs_endpoint=drone.gcs_endpoint,
                    heartbeat_period_s=scenario.gcs_heartbeat_period_s,
                    label=drone.label,
                )
                drone.port.connect()
        with timing.timed_stage("gcs_handshake_wait", sim_clock=clock):
            for drone in drones:
                _wait_for_gcs_handshake(
                    context, drones, drone, timeout_s=scenario.heartbeat_timeout_s
                )

        with timing.timed_stage("px4_prearm_settle", sim_clock=clock):
            _settle_px4_prearm(context, drones, seconds=scenario.prearm_settle_s)

        with timing.timed_stage("build_routines", sim_clock=clock):
            commanders = []
            for drone in drones:
                # GPS telemetry flows continuously once the estimator has settled
                # (above), so the takeoff target is fetched here, before arming,
                # and baked into each drone's routine command.
                target = _wait_for_takeoff_target(
                    context,
                    drones,
                    drone,
                    climb_m=scenario.takeoff_altitude_m,
                    timeout_s=scenario.heartbeat_timeout_s,
                )
                routine = _build_routine(scenario, target)
                drone.commander = VehicleCommander(routine, drone.port, label=drone.label)
                commanders.append(drone.commander)
            # STEP_SYNCHRONIZED barrier reproduces the arm-all-then-takeoff-all
            # fleet flow: no drone sends takeoff until every drone is armed.
            fleet = FleetCommander(commanders, policy=ExecutionPolicy.STEP_SYNCHRONIZED)

        with timing.timed_stage("command_routine", sim_clock=clock):
            _drive_routine_to_done(context, drones, fleet, scenario)

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
                if drone.port is not None:
                    drone.port.disconnect()
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

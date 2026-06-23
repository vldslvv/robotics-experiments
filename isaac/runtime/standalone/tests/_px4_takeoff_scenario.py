"""Shared standalone Isaac/PX4 takeoff scenario driver (single- or multi-drone).

The scenario is driven by the Hydra ``vehicles`` list: one entry is the
single-vehicle baseline, N entries drive N drones each bound to its own PX4 SITL
instance (the resolved simulator port ``mavlink_port = connection_baseport + id``,
GCS link the resolved ``vehicle.gcs_port = px4.gcs_baseport + id``). A single
``BridgeHostGroup`` owns the one physics-step subscription and drives every host
in two phases (all sends, then all polls), so one ``simulation_app.update()``
advances every drone's lockstep with the PX4 instances computing concurrently.

This module is one of the two drivers over the shared pure command core
(``robotics_experiments.commanding``); the other is the live extension's
``FleetControlDriver``. Both build the *same* routine from ``routine.steps`` via
``build_vehicle_routine`` and drive it with a ``FleetCommander`` under the
``STEP_SYNCHRONIZED`` barrier. The flight verification is the routine's own job:
the takeoff step completes on a real climb, the hover step on a sustained stable
hover, and the configured flight-envelope gates (tilt / drift / vertical+angular
speed) are continuous invariants on those steps. This driver only adds the
generic standalone/infra checks (bridge connectivity, the heartbeat/handshake/
telemetry handshakes, the pre-arm start-tilt check, and the bridge-log scan) and
the strict-vs-forgiving reaction: in **strict** mode (the integration test, set
by ``--strict``) a flight-envelope breach raises ``ExpectedInstabilityDetected``
and any other routine ``FAIL`` raises ``AssertionError``; in **forgiving** mode (a
regular standalone run) a breach is logged and the routine is stopped, like the
live extension. The command port talks the GCS link only - never the bridge's
simulator MAVLink lockstep - by design.
"""

from __future__ import annotations

import dataclasses
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


class ExpectedInstabilityDetected(AssertionError):
    """Raised (strict mode) when the routine fails on a flight-envelope breach."""


class Px4FailsafeDetected(AssertionError):
    """Raised (strict mode) when PX4 SITL enters a failsafe state.

    A failsafe heartbeat (system_status critical / emergency / flight
    termination) is a PX4-internal estimator/EKF event, not a flight-envelope
    breach, and is host-load-dependent and rerun-worthy. It is deliberately NOT
    a subclass of :class:`ExpectedInstabilityDetected`, so it never counts as an
    envelope-detected instability: the negative takeoff test must only pass when
    the stability envelope catches the injected fault, never on a failsafe flake.
    """


class _ScenarioStopped(Exception):
    """Forgiving-mode stop: a routine FAIL or failsafe that must not fail a run.

    Raised in forgiving mode (a regular standalone run, ``--strict`` absent) to
    unwind to the scenario top, which logs it and tears down cleanly - the
    runtime counterpart of the live extension logging a warning and stopping the
    routine. Never raised in strict mode (the integration test asserts instead).
    """


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
    altitude_m: float


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


def _handle_failsafe(
    context: SceneTestContext, drones: list[_DroneRuntime], *, phase: str, strict: bool
) -> None:
    """React to any drone's PX4 failsafe: strict raises, forgiving stops the run."""
    for drone in drones:
        if drone.port is not None and drone.port.failsafe:
            message = (
                f"[{drone.label}] PX4 entered failsafe state {phase}: "
                f"system_status={drone.port.failsafe_system_status}"
            )
            if strict:
                _raise_failsafe(message)
            context.report(
                f"WARNING: {message}; stopping routine (manual control remains available)"
            )
            raise _ScenarioStopped(message)


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
    strict: bool,
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
        _handle_failsafe(context, drones, phase="while settling before arm", strict=strict)


def _wait_for_takeoff_target(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    drone: _DroneRuntime,
    *,
    timeout_s: float,
    strict: bool,
) -> _TakeoffTarget:
    context.report(f"[{drone.label}] waiting up to {timeout_s:.1f}s for PX4 takeoff telemetry")
    deadline = time.monotonic() + timeout_s * len(drones)
    while time.monotonic() < deadline:
        _advance_once(context, drones)
        _pump_and_poll_all(drones)
        _handle_failsafe(
            context, drones, phase="while waiting for altitude telemetry", strict=strict
        )
        position = drone.port.latest_global_position
        if position is not None:
            latitude_deg, longitude_deg, altitude_m = position
            context.report(
                f"[{drone.label}] PX4 takeoff telemetry lat={latitude_deg:.7f} "
                f"lon={longitude_deg:.7f} altitude={altitude_m:.1f}m"
            )
            return _TakeoffTarget(
                latitude_deg=latitude_deg,
                longitude_deg=longitude_deg,
                altitude_m=altitude_m,
            )
    raise AssertionError(
        f"[{drone.label}] PX4 takeoff telemetry was not received before takeoff command"
    )


def _envelope_gates(scenario):
    """Build the routine's flight-envelope gates from the routine config."""
    from robotics_experiments.commanding import EnvelopeGates

    return EnvelopeGates(
        max_tilt_deg=scenario.max_tilt_deg,
        max_lateral_drift_m=scenario.max_lateral_drift_m,
        max_vertical_speed_m_s=scenario.max_vertical_speed_m_s,
        max_downward_speed_m_s=scenario.max_downward_speed_m_s,
        max_angular_speed_rad_s=scenario.max_angular_speed_rad_s,
        hover_max_abs_vertical_speed_m_s=scenario.hover_max_abs_vertical_speed_m_s,
        hover_max_abs_horizontal_speed_m_s=scenario.hover_max_abs_horizontal_speed_m_s,
    )


def _build_routine(scenario, target: _TakeoffTarget):
    """Build the per-drone routine from ``routine.steps`` and the takeoff target.

    Identical to the live extension's path: the same ``build_vehicle_routine``
    seam wires the configured flight-envelope gates onto the takeoff/hover steps,
    so the routine verifies the flight in both drivers.
    """
    from robotics_experiments.commanding import build_vehicle_routine

    return build_vehicle_routine(
        scenario.steps,
        target_latitude_deg=target.latitude_deg,
        target_longitude_deg=target.longitude_deg,
        target_altitude_m=target.altitude_m,
        command_ack_timeout_s=scenario.command_ack_timeout_s,
        gates=_envelope_gates(scenario),
        takeoff_timeout_s=scenario.takeoff_timeout_s,
        hover_settle_s=scenario.hover_settle_s,
    )


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


def _takeoff_step_index(scenario) -> int | None:
    for index, descriptor in enumerate(scenario.steps):
        if descriptor.get("kind") == "takeoff":
            return index
    return None


def _drive_routine(
    context: SceneTestContext,
    drones: list[_DroneRuntime],
    fleet,
    scenario,
    *,
    strict: bool,
    post_takeoff_hook: "Callable[[SceneTestContext, Any], None] | None",
    takeoff_step_index: int | None,
) -> None:
    """Tick physics and step the fleet commander until every routine reports DONE.

    Each tick advances PhysX once (asserting every bridge stays connected),
    pumps/drains every GCS link, builds a ``RoutineObservation`` per drone, and
    steps the ``FleetCommander``. The routine itself verifies the flight (climb,
    sustained hover, and the continuous envelope invariants); a ``FAIL`` is
    handled here by mode: strict raises (``ExpectedInstabilityDetected`` for a
    flight-envelope breach, ``AssertionError`` otherwise), forgiving logs and
    stops. The per-step timeouts inside ``decide`` are sim-time; this wall-clock
    budget is the backstop for a sim that ticks but never completes the routine.
    The optional ``post_takeoff_hook`` fires once, the tick the takeoff step
    completes (used by the negative test to inject a post-takeoff fault that the
    hover step's envelope must then catch).
    """
    from robotics_experiments.commanding import DecisionKind, FailureKind

    context.report(f"driving the routine for {len(drones)} drone(s)")
    deadline_s = time.monotonic() + (
        scenario.takeoff_timeout_s + scenario.command_ack_timeout_s + scenario.handshake_timeout_s
    ) * 4.0 * len(drones)
    hook_fired = post_takeoff_hook is None or takeoff_step_index is None
    while not fleet.is_done:
        if time.monotonic() > deadline_s:
            raise AssertionError("Timed out driving the routine to completion")
        _advance_once(context, drones)
        acks = _pump_and_poll_all(drones)
        _handle_failsafe(context, drones, phase="during the routine", strict=strict)
        sim_time_us = _sim_time_us(drones)
        observations = {
            drone.label: _routine_observation(drone, sim_time_us, acks[drone.label])
            for drone in drones
        }
        decisions = fleet.step(observations)
        for label, decision in decisions.items():
            if decision.kind is DecisionKind.FAIL:
                message = f"[{label}] routine failed: {decision.reason}"
                if strict:
                    if decision.failure_kind is FailureKind.ENVELOPE:
                        _raise_instability(message)
                    raise AssertionError(message)
                context.report(
                    f"WARNING: {message}; stopping routine (manual control remains available)"
                )
                raise _ScenarioStopped(message)
        if not hook_fired and drones[0].commander.current_step_index > takeoff_step_index:
            assert post_takeoff_hook is not None
            post_takeoff_hook(context, drones[0].host)
            hook_fired = True


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

    strict = context.strict
    timing = TimingReporter(options.test_name, emit=context.report)

    env = get_config()
    # The routine config (control-flow timeouts, takeoff target tunables, and the
    # flight-envelope gates). The local name stays `scenario` as it parameterizes
    # this live scenario; the routine carries its own verification.
    scenario = env.routine
    if not scenario.steps:
        raise ValueError(
            f"{options.test_name}: routine.steps is empty; this scenario needs a non-empty "
            "routine (run it under a takeoff experiment, e.g. "
            "configs/integration_test_takeoff.yaml)"
        )

    context.report(f"{options.test_name}: running in {'strict' if strict else 'forgiving'} mode")
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

    bridge_logs: list[tuple[str, Path | None]] = []
    stopped: _ScenarioStopped | None = None
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
                    context, drones, drone, timeout_s=scenario.bridge_heartbeat_timeout_s
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
                    context, drones, drone, timeout_s=scenario.handshake_timeout_s
                )

        with timing.timed_stage("px4_prearm_settle", sim_clock=clock):
            _settle_px4_prearm(context, drones, seconds=scenario.prearm_settle_s, strict=strict)

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
                    timeout_s=scenario.takeoff_target_timeout_s,
                    strict=strict,
                )
                routine = _build_routine(scenario, target)
                drone.commander = VehicleCommander(routine, drone.port, label=drone.label)
                commanders.append(drone.commander)
            # STEP_SYNCHRONIZED barrier reproduces the arm-all-then-takeoff-all
            # fleet flow: no drone advances a step until every drone completes it.
            fleet = FleetCommander(commanders, policy=ExecutionPolicy.STEP_SYNCHRONIZED)

        with timing.timed_stage("drive_routine", sim_clock=clock):
            _drive_routine(
                context,
                drones,
                fleet,
                scenario,
                strict=strict,
                post_takeoff_hook=options.post_takeoff_hook,
                takeoff_step_index=_takeoff_step_index(scenario),
            )

        bridge_logs = [(drone.label, _bridge_debug_log_path(drone.host)) for drone in drones]
    except _ScenarioStopped as exc:
        stopped = exc
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

    if stopped is not None:
        timing.report_stage("scenario_total", timing.total_start())
        context.report(f"{options.test_name}: routine stopped before completion: {stopped}")
        return

    # Bridge-log scan is a strict-only assertion (the integration test's last
    # check that the bridge stayed healthy and clean across the flight).
    if strict:
        with timing.timed_stage("bridge_log_validation"):
            for label, path in bridge_logs:
                _assert_bridge_log_clean(path, label)
    timing.report_stage("scenario_total", timing.total_start())
    context.report(
        f"{options.test_name}: verified arm/takeoff/hover routine for {len(drones)} drone(s)"
    )


@dataclass(frozen=True)
class Px4TakeoffScenarioOptions:
    """Options for one standalone PX4 takeoff scenario run.

    ``px4_config_transform`` is applied to every vehicle's PX4 config (e.g. the
    negative test injects a rotor-mapping fault); ``post_takeoff_hook`` is called
    once with the first drone's bridge host the tick the takeoff step completes
    (single-vehicle fault-injection only).
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

"""Standalone Isaac/PX4 takeoff scenario runner (single- or multi-drone).

The scenario is driven by the Hydra ``vehicles`` list: one entry is the
single-vehicle baseline, N entries drive N drones each bound to its own PX4 SITL
instance (the resolved simulator port ``mavlink_port = connection_baseport + id``,
GCS link the resolved ``vehicle.gcs_port = px4.gcs_baseport + id``).

This module is the **strict/forgiving standalone shell** over the host-agnostic
:class:`~robotics_experiments.runtime.FleetOrchestrator`. The orchestrator owns
the production pipeline (open scene -> apply settings -> spawn fleet -> start the
lockstep bridge group -> build one per-drone
:class:`~robotics_experiments.commanding.RoutineDriver` per flying drone ->
teardown); this runner only owns the SimApp host loop, the **integration-test-only
checks** (bridge connectivity + first heartbeat, the pre-arm start-tilt check, and
the bridge-log scan), and the failure reaction. In **strict** mode (the
integration test, ``--strict``) a flight-envelope breach raises
``ExpectedInstabilityDetected``, a PX4 failsafe raises ``Px4FailsafeDetected``,
and any other routine failure raises ``AssertionError``; in **forgiving** mode a
breach is logged and that drone's routine stops, like the live extension. The
phase machine and the flight verification (real climb, sustained stable hover, the
continuous envelope invariants) live in the routine/driver. The command port talks
the GCS link only - never the bridge's simulator MAVLink lockstep - by design.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from _runtime import SceneTestContext, TimingReporter, format_vehicle_state_vectors

from robotics_experiments.commanding import FailureKind, MavlinkCommandPort
from robotics_experiments.config import get_config
from robotics_experiments.px4_bridge import IsaacArticulationIO, frames

if TYPE_CHECKING:
    from robotics_experiments.config.types import RoutineConfig
    from robotics_experiments.runtime import (
        FleetDriverEnv,
        FleetOrchestrator,
        SpawnedDrone,
    )
    from robotics_experiments.runtime._bridge_host import BridgeHost

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


@dataclass
class _FlightSample:
    sim_time_us: int
    position_enu: np.ndarray
    linear_velocity_enu: np.ndarray
    angular_velocity_flu: np.ndarray
    tilt_deg: float


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


def _bridge_received_first_heartbeat(host: BridgeHost) -> bool:
    bridge = getattr(host, "_bridge", None)
    port = getattr(bridge, "_port", None)
    backend = getattr(port, "_backend", None)
    return bool(getattr(backend, "_received_first_hearbeat", False))


def _bridge_disconnect_reason(host: BridgeHost) -> str | None:
    bridge = getattr(host, "_bridge", None)
    port = getattr(bridge, "_port", None)
    return getattr(port, "_last_disconnect_reason", None)


def _bridge_debug_log_path(host: BridgeHost) -> Path | None:
    bridge = getattr(host, "_bridge", None)
    port = getattr(bridge, "_port", None)
    path = getattr(port, "_debug_log_path", None)
    return Path(path) if path else None


def _raise_instability(message: str) -> None:
    raise ExpectedInstabilityDetected(message)


def _raise_failsafe(message: str) -> None:
    raise Px4FailsafeDetected(message)


def _require_bridge_connected(host: BridgeHost, label: str) -> None:
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


def _sim_time_us(orchestrator: FleetOrchestrator) -> int:
    # All bridges share the same PhysX steps, so any host's clock is the fleet
    # clock; use the first bridged host as the reference.
    return int(orchestrator.bridged[0].host.sim_time_us)


def _advance_once(context: SceneTestContext, orchestrator: FleetOrchestrator) -> None:
    """Advance PhysX one step (ticks every bridge) and assert all stay connected."""
    context.simulation_app.update()
    for handle in orchestrator.bridged:
        _require_bridge_connected(handle.host, handle.label)


def _advance_sim_time(
    context: SceneTestContext,
    orchestrator: FleetOrchestrator,
    seconds: float,
    wall_timeout_s: float | None = None,
) -> None:
    bridged = orchestrator.bridged
    start_us = _sim_time_us(orchestrator)
    target_us = start_us + int(round(float(seconds) * 1_000_000))
    base = wall_timeout_s if wall_timeout_s is not None else max(10.0, seconds * 5.0)
    deadline = time.monotonic() + base * len(bridged)
    while _sim_time_us(orchestrator) < target_us:
        if time.monotonic() > deadline:
            actual_s = (_sim_time_us(orchestrator) - start_us) * 1e-6
            raise AssertionError(
                f"Timed out advancing Isaac simulation: target={seconds:.2f}s "
                f"actual={actual_s:.2f}s"
            )
        _advance_once(context, orchestrator)


def _wait_for_bridge_heartbeat(
    context: SceneTestContext,
    orchestrator: FleetOrchestrator,
    handle: SpawnedDrone,
    timeout_s: float,
) -> None:
    """Infra check: wait until the bridge received its first PX4 heartbeat."""
    context.report(f"[{handle.label}] waiting up to {timeout_s:.1f}s for PX4 bridge heartbeat")
    deadline = time.monotonic() + timeout_s * len(orchestrator.bridged)
    while time.monotonic() < deadline:
        _advance_once(context, orchestrator)
        if _bridge_received_first_heartbeat(handle.host):
            context.report(f"[{handle.label}] PX4 bridge received first heartbeat")
            return
    raise AssertionError(
        f"[{handle.label}] PX4 bridge did not receive PX4 heartbeat before timeout"
    )


def _sample_state(observer: IsaacArticulationIO, sim_time_us: int) -> _FlightSample:
    state = observer.read_state(sim_time_us=sim_time_us)
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


def _build_make_env(
    context: SceneTestContext,
    strict: bool,
    post_takeoff_hook: Callable[[SceneTestContext, BridgeHost], None] | None,
    first_label: str,
) -> Callable[[SpawnedDrone, MavlinkCommandPort], FleetDriverEnv]:
    """Build the per-drone ``make_env`` closure with the strict/forgiving reaction.

    ``on_failure`` raises the scenario's typed exceptions in strict mode (so the
    integration test fails non-zero) and logs in forgiving mode (the driver then
    idles). ``on_step_advanced`` fires the optional post-takeoff hook on the first
    drone when its takeoff step completes (the negative test's fault injection).
    """
    # FleetDriverEnv lives in the Isaac-coupled runtime package; import it here so
    # module import stays free of the Kit app.
    from robotics_experiments.runtime import FleetDriverEnv

    def make_env(spawned: SpawnedDrone, port: MavlinkCommandPort) -> FleetDriverEnv:
        is_first = spawned.label == first_label

        def on_failure(kind: FailureKind, message: str) -> None:
            if strict:
                if kind is FailureKind.FAILSAFE:
                    _raise_failsafe(message)
                if kind is FailureKind.ENVELOPE:
                    _raise_instability(message)
                raise AssertionError(message)
            context.report(
                f"WARNING: {message} ({kind.name}); stopping routine "
                "(manual control remains available)"
            )

        def on_step_advanced(step_label: str) -> None:
            if post_takeoff_hook is not None and is_first and step_label == "takeoff":
                post_takeoff_hook(context, spawned.host)

        return FleetDriverEnv(
            spawned,
            port,
            log=context.report,
            on_failure=on_failure,
            on_step_advanced=on_step_advanced,
        )

    return make_env


def _drive_to_done(
    context: SceneTestContext,
    orchestrator: FleetOrchestrator,
    fleet_routine: RoutineConfig,
) -> None:
    """Tick physics and every per-drone driver until all routines are done.

    Each per-drone :class:`RoutineDriver` owns its phase machine, flight
    verification, and (via its env) the strict/forgiving reaction; this loop only
    advances the shared sim and ticks the drivers. Ticks are raw (unguarded) so a
    strict env's typed exception propagates out and fails the integration test.
    The per-phase sim-time and wall-clock guards live in the drivers; this coarse
    wall budget is a final backstop against a sim that ticks but never finishes.
    """
    active = orchestrator.flying
    context.report(f"driving the routine for {len(active)} drone(s)")
    deadline_s = time.monotonic() + (
        fleet_routine.takeoff_timeout_s
        + fleet_routine.command_ack_timeout_s
        + fleet_routine.handshake_timeout_s
        + fleet_routine.prearm_settle_s
    ) * 6.0 * len(active)
    while not orchestrator.all_done:
        if time.monotonic() > deadline_s:
            raise AssertionError("Timed out driving the routine to completion")
        _advance_once(context, orchestrator)
        for flying in active:
            flying.driver.tick()


def _run_scenario(context: SceneTestContext, options: Px4TakeoffScenarioOptions) -> None:
    # omni.timeline and FleetOrchestrator (Isaac-coupled runtime) require the
    # running Kit app, so they are imported after SimulationApp has started.
    import omni.timeline

    from robotics_experiments.runtime import FleetOrchestrator

    strict = context.strict
    timing = TimingReporter(options.test_name, emit=context.report)

    env = get_config()
    if not any(vehicle.routine.steps for vehicle in env.vehicles):
        raise ValueError(
            f"{options.test_name}: no vehicle has a non-empty routine.steps; run this scenario "
            "under a takeoff experiment (e.g. configs/takeoff_quad.yaml)"
        )
    # Fleet-level infra knobs come from the global routine default (ground settle
    # and the bridge-heartbeat infra wait apply uniformly before any drone flies).
    fleet_routine = env.routine

    context.report(f"{options.test_name}: running in {'strict' if strict else 'forgiving'} mode")
    orchestrator = FleetOrchestrator(env)

    context.report(f"{options.test_name}: opening scene and applying simulation settings")
    with timing.timed_stage("open_scene"):
        orchestrator.open_scene(context.scene_override)
        context.simulation_app.update()
        orchestrator.adopt_current_stage()
    with timing.timed_stage("apply_settings"):
        orchestrator.apply_settings()

    with timing.timed_stage("spawn_drones"):
        orchestrator.spawn_fleet()
    drone_count = len(orchestrator.drones)
    context.report(f"{options.test_name}: built {drone_count} drone(s)")
    if not orchestrator.bridged:
        raise AssertionError(f"{options.test_name}: no px4-enabled vehicle to drive")

    timeline = omni.timeline.get_timeline_interface()
    clock = orchestrator.bridged[0].host

    bridge_logs: list[tuple[str, Path | None]] = []
    try:
        context.report(
            f"{options.test_name}: starting {len(orchestrator.bridged)} PX4 bridge host(s)"
        )
        with timing.timed_stage("bridge_host_start", sim_clock=clock):
            orchestrator.start_bridges()
            timeline.play()

        context.report(
            f"{options.test_name}: settling drones on ground for "
            f"{fleet_routine.ground_settle_s:.1f}s"
        )
        with timing.timed_stage("ground_settle", sim_clock=clock):
            _advance_sim_time(
                context, orchestrator, seconds=fleet_routine.ground_settle_s, wall_timeout_s=20.0
            )

        with timing.timed_stage("bridge_heartbeat_wait", sim_clock=clock):
            for handle in orchestrator.bridged:
                _wait_for_bridge_heartbeat(
                    context,
                    orchestrator,
                    handle,
                    timeout_s=fleet_routine.bridge_heartbeat_timeout_s,
                )

        for handle in orchestrator.bridged:
            if not bool(getattr(handle.host, "is_running", False)):
                raise AssertionError(
                    f"[{handle.label}] PX4 bridge host stopped before command flow"
                )
            _require_bridge_connected(handle.host, label=handle.label)

        with timing.timed_stage("observer_initialize", sim_clock=clock):
            for handle in orchestrator.bridged:
                handle.observer.initialize()
                start_sample = _sample_state(handle.observer, _sim_time_us(orchestrator))
                context.report(
                    f"[{handle.label}] start position z={start_sample.position_enu[2]:.2f}m "
                    f"{format_vehicle_state_vectors(start_sample)} "
                    f"tilt={start_sample.tilt_deg:.1f}deg"
                )
                if start_sample.tilt_deg > handle.vehicle.routine.max_tilt_deg:
                    raise AssertionError(
                        f"[{handle.label}] Drone is too tilted before takeoff: "
                        f"tilt={start_sample.tilt_deg:.1f}deg"
                    )

        make_env = _build_make_env(
            context,
            strict=strict,
            post_takeoff_hook=options.post_takeoff_hook,
            first_label=orchestrator.drones[0].label,
        )
        with timing.timed_stage("build_drivers", sim_clock=clock):
            orchestrator.build_drivers(make_env)

        with timing.timed_stage("drive_routine", sim_clock=clock):
            _drive_to_done(context, orchestrator, fleet_routine)

        bridge_logs = [(h.label, _bridge_debug_log_path(h.host)) for h in orchestrator.bridged]
    finally:
        context.report(f"{options.test_name}: stopping timeline, bridge hosts, and drones")
        with timing.timed_stage("teardown", sim_clock=clock):
            timeline.stop()
            orchestrator.teardown()

    # Bridge-log scan is a strict-only assertion (the integration test's last check
    # that the bridge stayed healthy and clean across the flight).
    if strict:
        with timing.timed_stage("bridge_log_validation"):
            for label, path in bridge_logs:
                _assert_bridge_log_clean(path, label)
    timing.report_stage("scenario_total", timing.total_start())
    context.report(
        f"{options.test_name}: verified {options.success_label} for {drone_count} drone(s)"
    )


@dataclass(frozen=True)
class Px4TakeoffScenarioOptions:
    """Options for one standalone PX4 takeoff scenario run.

    ``post_takeoff_hook`` is called once with the first drone's bridge host the
    tick its takeoff step completes (single-vehicle fault-injection only).
    ``success_label`` is used only in the final diagnostic log line.
    """

    test_name: str
    px4_container_log_name: str
    post_takeoff_hook: Callable[[SceneTestContext, BridgeHost], None] | None = None
    success_label: str = "configured routine"


def run_px4_takeoff_scenario(
    context: SceneTestContext,
    options: Px4TakeoffScenarioOptions,
) -> None:
    """Run the PX4 takeoff scenario and append diagnostic log hints on failure."""
    try:
        _run_scenario(context, options)
    except AssertionError as exc:
        raise type(exc)(f"{exc}\n{_latest_log_hint(options.px4_container_log_name)}") from exc

"""Standalone Isaac/PX4 takeoff scenario runner (single- or multi-drone).

The scenario is driven by the Hydra ``vehicles`` list: one entry is the
single-vehicle baseline, N entries drive N drones each bound to its own PX4 SITL
instance (the resolved simulator port ``mavlink_port = connection_baseport + id``,
GCS link the resolved ``vehicle.gcs_port = px4.gcs_baseport + id``). A single
``BridgeHostGroup`` owns the one physics-step subscription and drives every host
in two phases (all sends, then all polls), so one ``simulation_app.update()``
advances every drone's lockstep with the PX4 instances computing concurrently.

This module is the **strict/forgiving standalone adapter** over the shared
per-drone :class:`~robotics_experiments.commanding.RoutineDriver`: each drone with
a non-empty resolved ``routine.steps`` gets its own driver, all ticked from one
blocking ``simulation_app.update()`` loop. The phase machine and the flight
verification (real climb, sustained stable hover, the continuous envelope
invariants) live in the routine/driver; this runner only adds the generic
standalone/infra checks (bridge connectivity + first heartbeat, the pre-arm
start-tilt check, and the bridge-log scan) and the reaction wiring. In **strict**
mode (the integration test, ``--strict``) a flight-envelope breach raises
``ExpectedInstabilityDetected``, a PX4 failsafe raises ``Px4FailsafeDetected``,
and any other routine failure raises ``AssertionError``; in **forgiving** mode a
breach is logged and that drone's routine stops, like the live extension. The
command port talks the GCS link only - never the bridge's simulator MAVLink
lockstep - by design.
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


@dataclass
class _FlightSample:
    sim_time_us: int
    position_enu: np.ndarray
    linear_velocity_enu: np.ndarray
    angular_velocity_flu: np.ndarray
    tilt_deg: float


@dataclass
class _DroneRuntime:
    """Per-drone live state for one vehicle in the scenario."""

    vehicle: Any
    host: Any
    observer: Any
    label: str
    gcs_endpoint: str
    port: Any | None = None
    driver: Any | None = None
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

    Called only after the runner asserts each bridge is connected, so the host
    has a live bridge here.
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
    """Infra check: wait until the bridge received its first PX4 heartbeat."""
    context.report(f"[{drone.label}] waiting up to {timeout_s:.1f}s for PX4 bridge heartbeat")
    deadline = time.monotonic() + timeout_s * len(drones)
    while time.monotonic() < deadline:
        _advance_once(context, drones)
        if _bridge_received_first_heartbeat(drone.host):
            context.report(f"[{drone.label}] PX4 bridge received first heartbeat")
            return
    raise AssertionError(f"[{drone.label}] PX4 bridge did not receive PX4 heartbeat before timeout")


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


class _StandaloneDriverEnv:
    """The driver seam over one live drone, with the strict/forgiving reaction.

    Reads connectivity / sim time / state from the drone's bridge host and
    observer. ``on_failure`` raises the scenario's typed exceptions in strict mode
    (so the integration test fails non-zero) and logs in forgiving mode (the
    driver then idles). ``on_step_advanced`` fires the optional post-takeoff hook
    when the takeoff step completes (the negative test's fault injection).
    """

    def __init__(
        self,
        context: SceneTestContext,
        drone: _DroneRuntime,
        *,
        strict: bool,
        post_takeoff_hook: "Callable[[SceneTestContext, Any], None] | None" = None,
    ) -> None:
        self._context = context
        self._drone = drone
        self._strict = strict
        self._post_takeoff_hook = post_takeoff_hook

    def is_bridge_connected(self) -> bool:
        bridge = getattr(self._drone.host, "_bridge", None)
        return bool(getattr(bridge, "is_connected", False))

    def sim_time_us(self) -> int:
        return int(self._drone.host.sim_time_us)

    def read_observation(self, *, sim_time_us: int, last_ack):
        from robotics_experiments.commanding import RoutineObservation

        return RoutineObservation(
            vehicle_state=self._drone.observer.read_state(sim_time_us=sim_time_us),
            bridge_status=_bridge_status(self._drone.host),
            last_ack=last_ack,
            failsafe=self._drone.port.failsafe,
            sim_time_us=sim_time_us,
        )

    def log(self, message: str) -> None:
        self._context.report(message)

    def on_failure(self, kind: Any, message: str) -> None:
        from robotics_experiments.commanding import FailureKind

        if self._strict:
            if kind == "failsafe":
                _raise_failsafe(message)
            if kind is FailureKind.ENVELOPE:
                _raise_instability(message)
            raise AssertionError(message)
        self._context.report(
            f"WARNING: {message} ({kind}); stopping routine (manual control remains available)"
        )

    def on_step_advanced(self, step_label: str) -> None:
        if self._post_takeoff_hook is not None and step_label == "takeoff":
            self._post_takeoff_hook(self._context, self._drone.host)


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

    from robotics_experiments.commanding import DriverParams, MavlinkCommandPort, RoutineDriver
    from robotics_experiments.config import get_config
    from robotics_experiments.drone import despawn_drone
    from robotics_experiments.simulation import apply_settings

    strict = context.strict
    timing = TimingReporter(options.test_name, emit=context.report)

    env = get_config()
    if not any(vehicle.routine.steps for vehicle in env.vehicles):
        raise ValueError(
            f"{options.test_name}: no vehicle has a non-empty routine.steps; run this scenario "
            "under a takeoff experiment (e.g. configs/integration_test_takeoff.yaml)"
        )
    # Fleet-level infra knobs come from the global routine default (ground settle
    # and the bridge-heartbeat infra wait apply uniformly before any drone flies).
    fleet_routine = env.routine

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
    group = BridgeHostGroup([drone.host for drone in drones])

    bridge_logs: list[tuple[str, Path | None]] = []
    try:
        context.report(f"{options.test_name}: starting {len(drones)} PX4 bridge host(s)")
        with timing.timed_stage("bridge_host_start", sim_clock=clock):
            group.start()
            timeline.play()

        context.report(
            f"{options.test_name}: settling drones on ground for "
            f"{fleet_routine.ground_settle_s:.1f}s"
        )
        with timing.timed_stage("ground_settle", sim_clock=clock):
            _advance_sim_time(
                context, drones, seconds=fleet_routine.ground_settle_s, wall_timeout_s=20.0
            )

        with timing.timed_stage("bridge_heartbeat_wait", sim_clock=clock):
            for drone in drones:
                _wait_for_bridge_heartbeat(
                    context, drones, drone, timeout_s=fleet_routine.bridge_heartbeat_timeout_s
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
                if start_sample.tilt_deg > drone.vehicle.routine.max_tilt_deg:
                    raise AssertionError(
                        f"[{drone.label}] Drone is too tilted before takeoff: "
                        f"tilt={start_sample.tilt_deg:.1f}deg"
                    )

        with timing.timed_stage("build_drivers", sim_clock=clock):
            for drone in drones:
                if not drone.vehicle.routine.steps:
                    context.report(f"[{drone.label}] no routine; left manual")
                    continue
                drone.port = MavlinkCommandPort(
                    gcs_endpoint=drone.gcs_endpoint,
                    heartbeat_period_s=drone.vehicle.routine.gcs_heartbeat_period_s,
                    label=drone.label,
                )
                # The single-vehicle fault-injection hook fires on the first drone.
                hook = options.post_takeoff_hook if drone is drones[0] else None
                driver_env = _StandaloneDriverEnv(
                    context, drone, strict=strict, post_takeoff_hook=hook
                )
                drone.driver = RoutineDriver(
                    DriverParams.from_routine_config(drone.vehicle.routine),
                    drone.port,
                    driver_env,
                    label=drone.label,
                )

        with timing.timed_stage("drive_routine", sim_clock=clock):
            _drive_to_done(context, drones, fleet_routine)

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

    # Bridge-log scan is a strict-only assertion (the integration test's last check
    # that the bridge stayed healthy and clean across the flight).
    if strict:
        with timing.timed_stage("bridge_log_validation"):
            for label, path in bridge_logs:
                _assert_bridge_log_clean(path, label)
    timing.report_stage("scenario_total", timing.total_start())
    context.report(
        f"{options.test_name}: verified arm/takeoff/hover routine for {len(drones)} drone(s)"
    )


def _drive_to_done(context: SceneTestContext, drones: list[_DroneRuntime], fleet_routine) -> None:
    """Tick physics and every per-drone driver until all routines are done.

    Each per-drone :class:`RoutineDriver` owns its phase machine, flight
    verification, and (via its env) the strict/forgiving reaction; this loop only
    advances the shared sim and ticks the drivers. The per-phase sim-time and
    wall-clock guards live in the drivers; this coarse wall budget is a final
    backstop against a sim that ticks but never finishes.
    """
    active = [drone for drone in drones if drone.driver is not None]
    context.report(f"driving the routine for {len(active)} drone(s)")
    deadline_s = time.monotonic() + (
        fleet_routine.takeoff_timeout_s
        + fleet_routine.command_ack_timeout_s
        + fleet_routine.handshake_timeout_s
        + fleet_routine.prearm_settle_s
    ) * 6.0 * len(active)
    while not all(drone.driver.is_done for drone in active):
        if time.monotonic() > deadline_s:
            raise AssertionError("Timed out driving the routine to completion")
        _advance_once(context, drones)
        for drone in active:
            drone.driver.tick()


@dataclass(frozen=True)
class Px4TakeoffScenarioOptions:
    """Options for one standalone PX4 takeoff scenario run.

    ``px4_config_transform`` is applied to every vehicle's PX4 config (e.g. the
    negative test injects a rotor-mapping fault); ``post_takeoff_hook`` is called
    once with the first drone's bridge host the tick its takeoff step completes
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

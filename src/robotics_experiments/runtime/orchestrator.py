"""Functional fleet lifecycle over Isaac and MAVLink resource adapters.

Each phase returns a frozen :class:`FleetState`; the live extension and
standalone runner explicitly rebind it at their imperative boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import carb

from robotics_experiments.commanding import (
    CommandAck,
    DriverEnvironment,
    DriverParams,
    EnvelopeGates,
    FailureKind,
    MavlinkCommandPort,
    RoutineDriverState,
    RoutineObservation,
    create_routine_driver,
    routine_driver_is_done,
    tick_routine_driver,
)
from robotics_experiments.config import ExperimentConfig, VehicleConfig
from robotics_experiments.runtime._scene import resolve_scene_path
from robotics_experiments.runtime.bridge_host import (
    BridgeHost,
    BridgeHostGroup,
    BridgeHostParams,
    read_bridge_host_status,
)

if TYPE_CHECKING:
    from pxr import Usd

    from robotics_experiments.commanding import PolicyStepFn
    from robotics_experiments.px4_bridge import IsaacArticulationIO

__all__ = [
    "FleetDriverEnv",
    "FleetState",
    "FlyingDrone",
    "SpawnedDrone",
    "adopt_current_stage",
    "apply_settings",
    "build_drivers",
    "create_fleet",
    "fleet_is_done",
    "open_scene",
    "spawn_fleet",
    "start_bridges",
    "teardown",
    "tick_drivers",
]

_LOG_PREFIX = "[FleetRuntime]"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SpawnedDrone:
    """One vehicle's immutable spawn-phase state."""

    vehicle: VehicleConfig
    label: str
    articulation_path: str
    host: Optional[BridgeHost]
    observer: Optional[IsaacArticulationIO]


@dataclass(frozen=True)
class FlyingDrone:
    """One routine-enabled drone's immutable driver composition."""

    spawned: SpawnedDrone
    port: MavlinkCommandPort
    driver: RoutineDriverState


@dataclass(frozen=True)
class FleetState:
    """Immutable progress through the fleet lifecycle."""

    config: ExperimentConfig
    stage: Optional[Usd.Stage]
    group: BridgeHostGroup
    drones: tuple[SpawnedDrone, ...]
    flying: tuple[FlyingDrone, ...]


class FleetDriverEnv:
    """Mutable simulator/resource seam used by a pure driver transition."""

    def __init__(
        self,
        spawned: SpawnedDrone,
        port: MavlinkCommandPort,
        *,
        log: Callable[[str], None],
        on_failure: Callable[[FailureKind, str], None],
        on_step_advanced: Callable[[str], None],
    ) -> None:
        if spawned.host is None or spawned.observer is None:
            raise RuntimeError(
                f"{_LOG_PREFIX} FleetDriverEnv requires a flying drone "
                f"(host/observer); got {spawned.label!r}"
            )
        self._host = spawned.host
        self._observer = spawned.observer
        self._port = port
        self._log_fn = log
        self._on_failure_fn = on_failure
        self._on_step_advanced_fn = on_step_advanced

    def is_bridge_connected(self) -> bool:
        status = read_bridge_host_status(host=self._host).bridge_status
        return status is not None and status.is_connected

    def sim_time_us(self) -> int:
        return read_bridge_host_status(host=self._host).sim_time_us

    def read_observation(
        self, *, sim_time_us: int, last_ack: Optional[CommandAck]
    ) -> RoutineObservation:
        host_status = read_bridge_host_status(host=self._host).bridge_status
        if host_status is None:
            raise RuntimeError(f"{_LOG_PREFIX} observation read before bridge startup")
        return RoutineObservation(
            vehicle_state=self._observer.read_state(sim_time_us=sim_time_us),
            bridge_status=host_status,
            last_ack=last_ack,
            failsafe=self._port.read_status().failsafe,
            sim_time_us=sim_time_us,
        )

    def log(self, message: str) -> None:
        self._log_fn(message)

    def on_failure(self, kind: FailureKind, message: str) -> None:
        self._on_failure_fn(kind, message)

    def on_step_advanced(self, step_label: str) -> None:
        self._on_step_advanced_fn(step_label)


def create_fleet(*, config: ExperimentConfig) -> FleetState:
    """Create an empty runtime state for a resolved experiment config."""
    return FleetState(
        config=config,
        stage=None,
        group=BridgeHostGroup([]),
        drones=(),
        flying=(),
    )


def open_scene(*, state: FleetState, scene_override: Optional[str] = None) -> FleetState:
    """Request the configured stage open; adoption occurs on a later frame."""
    import omni.usd

    scene = scene_override if scene_override else state.config.scene
    path = resolve_scene_path(scene, repo_root=_repo_root())
    if not path.exists():
        raise FileNotFoundError(f"Scene does not exist: {path}")
    carb.log_info(f"{_LOG_PREFIX} opening scene {path}")
    if not omni.usd.get_context().open_stage(str(path)):
        raise RuntimeError(f"Failed to open stage: {path}")
    return state


def adopt_current_stage(*, state: FleetState) -> FleetState:
    """Return state bound to the now-open USD stage."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"{_LOG_PREFIX} stage adopted before it finished opening")
    return replace(state, stage=stage)


def apply_settings(*, state: FleetState) -> FleetState:
    """Apply configured simulation settings to the adopted stage."""
    from robotics_experiments.simulation import apply_settings as apply_simulation_settings

    apply_simulation_settings(stage=state.stage, settings=state.config.sim)
    return state


def spawn_fleet(*, state: FleetState) -> FleetState:
    """Spawn the ordered vehicle mapping and compose bridge resources."""
    from robotics_experiments.drone import spawn_drone

    if state.stage is None:
        raise RuntimeError(f"{_LOG_PREFIX} spawn_fleet called before the stage was adopted")
    summary = ", ".join(
        f"{label}: id={vehicle.id} prim={vehicle.spawn.prim_name} "
        f"px4_enabled={vehicle.bridge.enabled} endpoint={vehicle.mavlink_port} "
        f"position_xy={tuple(vehicle.spawn.position_xy)}"
        for label, vehicle in state.config.vehicles.items()
    )
    carb.log_info(f"{_LOG_PREFIX} spawning {len(state.config.vehicles)} vehicle(s): {summary}")
    drones: list[SpawnedDrone] = []
    for label, vehicle in state.config.vehicles.items():
        path = spawn_drone(stage=state.stage, config=vehicle.spawn)
        host: Optional[BridgeHost] = None
        observer: Optional[IsaacArticulationIO] = None
        articulation_path = vehicle.bridge.articulation_path or path
        if vehicle.bridge.enabled:
            from robotics_experiments.px4_bridge import IsaacArticulationIO

            host = BridgeHost(
                BridgeHostParams(
                    bridge=vehicle.bridge,
                    vehicle_id=vehicle.id,
                    mavlink_port=vehicle.mavlink_port,
                    debug_log_path=vehicle.debug_log_path,
                    articulation_path=articulation_path,
                    physics_hz=state.config.sim.physics_hz,
                )
            )
            observer = IsaacArticulationIO(
                articulation_path=articulation_path,
                rotor_link_paths=[],
            )
        drones.append(
            SpawnedDrone(
                vehicle=vehicle,
                label=label,
                articulation_path=articulation_path,
                host=host,
                observer=observer,
            )
        )
    group = BridgeHostGroup([drone.host for drone in drones if drone.host is not None])
    return replace(state, group=group, drones=tuple(drones), flying=())


def start_bridges(*, state: FleetState) -> FleetState:
    """Start every host before installing the fleet-wide PhysX subscription."""
    state.group.start()
    return state


def _policy_fn_factory(*, config: ExperimentConfig, descriptor: Mapping[str, Any]) -> PolicyStepFn:
    from robotics_experiments.rl.press import build_policy_step_fn

    artifact = Path(str(descriptor["artifact"]))
    if not artifact.is_absolute():
        artifact = _repo_root() / artifact
    return build_policy_step_fn(artifact, config.rl.action)


def build_drivers(
    *,
    state: FleetState,
    make_env: Callable[[SpawnedDrone, MavlinkCommandPort], DriverEnvironment],
) -> FleetState:
    """Compose one frozen routine driver state per eligible drone."""
    flying: list[FlyingDrone] = []
    for spawned in state.drones:
        if not spawned.vehicle.routine.steps:
            continue
        if spawned.host is None:
            carb.log_warn(
                f"{_LOG_PREFIX} [{spawned.label}] has a routine but px4 is disabled; "
                "no bridge to fly it -- left manual"
            )
            continue
        port = MavlinkCommandPort(
            gcs_endpoint=f"udp:127.0.0.1:{spawned.vehicle.gcs_port}",
            heartbeat_period_s=spawned.vehicle.routine.gcs_heartbeat_period_s,
            label=spawned.label,
        )
        routine = spawned.vehicle.routine
        driver = create_routine_driver(
            params=DriverParams(
                steps=routine.steps,
                gates=EnvelopeGates(
                    max_tilt_deg=routine.max_tilt_deg,
                    max_lateral_drift_m=routine.max_lateral_drift_m,
                    max_vertical_speed_m_s=routine.max_vertical_speed_m_s,
                    max_downward_speed_m_s=routine.max_downward_speed_m_s,
                    max_angular_speed_rad_s=routine.max_angular_speed_rad_s,
                    hover_max_abs_vertical_speed_m_s=(routine.hover_max_abs_vertical_speed_m_s),
                    hover_max_abs_horizontal_speed_m_s=(routine.hover_max_abs_horizontal_speed_m_s),
                ),
                command_ack_timeout_s=routine.command_ack_timeout_s,
                bridge_heartbeat_timeout_s=routine.bridge_heartbeat_timeout_s,
                handshake_timeout_s=routine.handshake_timeout_s,
                prearm_settle_s=routine.prearm_settle_s,
                takeoff_target_timeout_s=routine.takeoff_target_timeout_s,
                takeoff_timeout_s=routine.takeoff_timeout_s,
                hover_settle_s=routine.hover_settle_s,
                goto_position_tol_m=routine.goto_position_tol_m,
                goto_heading_tol_deg=routine.goto_heading_tol_deg,
                goto_cross_track_m=routine.goto_cross_track_m,
                goto_offboard_prime_ticks=routine.goto_offboard_prime_ticks,
                home_lat_deg=spawned.vehicle.bridge.home_lat_deg,
                home_lon_deg=spawned.vehicle.bridge.home_lon_deg,
                home_alt_m=spawned.vehicle.bridge.home_alt_m,
                goto_min_heading_dist_m=state.config.driver.goto_min_heading_dist_m,
                prearm_backstop_floor_s=state.config.driver.prearm_backstop_floor_s,
                prearm_backstop_multiplier=state.config.driver.prearm_backstop_multiplier,
                drive_budget_multiplier=state.config.driver.drive_budget_multiplier,
                policy_fn_factory=lambda descriptor: _policy_fn_factory(
                    config=state.config,
                    descriptor=descriptor,
                ),
            ),
            port=port,
            env=make_env(spawned, port),
            label=spawned.label,
        )
        flying.append(FlyingDrone(spawned=spawned, port=port, driver=driver))
    return replace(state, flying=tuple(flying))


def tick_drivers(
    *,
    state: FleetState,
    on_error: Optional[Callable[[FlyingDrone, Exception], None]] = None,
) -> FleetState:
    """Tick every routine once and return their successor fleet state."""
    advanced: list[FlyingDrone] = []
    for flying in state.flying:
        try:
            driver = tick_routine_driver(state=flying.driver)
        except Exception as exc:
            if on_error is None:
                raise
            on_error(flying, exc)
            driver = flying.driver
        advanced.append(replace(flying, driver=driver))
    return replace(state, flying=tuple(advanced))


def fleet_is_done(*, state: FleetState) -> bool:
    """Return whether every configured routine reached DONE."""
    return all(routine_driver_is_done(state=flying.driver) for flying in state.flying)


def teardown(*, state: FleetState) -> FleetState:
    """Stop resources before despawning vehicles, preserving lifecycle order."""
    from robotics_experiments.drone import despawn_drone

    state.group.stop()
    for flying in state.flying:
        flying.port.disconnect()
    for spawned in state.drones:
        if spawned.observer is not None:
            spawned.observer.shutdown()
        if state.stage is not None:
            despawn_drone(stage=state.stage, config=spawned.vehicle.spawn)
    return replace(
        state,
        group=BridgeHostGroup([]),
        drones=(),
        flying=(),
    )

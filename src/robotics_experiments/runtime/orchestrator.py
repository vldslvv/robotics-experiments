"""Functional fleet lifecycle over Isaac and MAVLink resource adapters.

Each phase returns a frozen :class:`FleetState`; the live extension and
standalone runner explicitly rebind it at their imperative boundaries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import carb

from robotics_experiments._cadence import _advance_cadence, _CadenceState
from robotics_experiments.commanding import (
    CommandAck,
    DriverContext,
    DriverEnvironment,
    DriverParams,
    DriverTick,
    EnvelopeGates,
    FailureKind,
    MavlinkCommandPort,
    PolicyAdapter,
    RoutineDriverState,
    RoutineObservation,
    create_routine_driver,
    routine_driver_is_done,
    tick_routine_driver,
)
from robotics_experiments.config import ExperimentConfig, RoutineStepConfig, VehicleConfig
from robotics_experiments.drone import despawn_drone, spawn_drone
from robotics_experiments.environment import (
    SwitchesRuntimeState,
    adopt_switches_stage,
    create_switches_runtime,
    play_switches_runtime,
    step_switches_runtime,
    teardown_switches_runtime,
)
from robotics_experiments.px4_bridge import IsaacArticulationIO, vehicle_log_context
from robotics_experiments.rl import PolicyInterface
from robotics_experiments.runtime._policy import (
    _make_policy_adapter,
    _PolicyPreflight,
    _preflight_policy_artifacts,
)
from robotics_experiments.runtime._press_io import PressObservationIO
from robotics_experiments.runtime._scene import resolve_scene_path
from robotics_experiments.runtime.bridge_host import (
    BridgeHost,
    BridgeHostGroup,
    BridgeHostParams,
    read_bridge_host_status,
)
from robotics_experiments.simulation import apply_settings as apply_simulation_settings

if TYPE_CHECKING:
    from pxr import Usd

__all__ = [
    "FleetDriverEnv",
    "FleetState",
    "FlyingDrone",
    "SpawnedDrone",
    "SwitchToggleEvent",
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
_ROUTINE_LOGGER = logging.getLogger("robotics_experiments.px4_bridge.routine")


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
    policy_observer: Optional[PressObservationIO] = None


@dataclass(frozen=True)
class SwitchToggleEvent:
    """One ordered paired-lamp transition emitted by the switch runtime."""

    sequence: int
    switch_index: int
    state: bool


@dataclass(frozen=True)
class FlyingDrone:
    """One bridge-enabled drone's immutable driver composition."""

    spawned: SpawnedDrone
    context: DriverContext
    driver: RoutineDriverState


@dataclass(frozen=True)
class FleetState:
    """Immutable progress, with one ``flying`` composition per bridged drone."""

    config: ExperimentConfig
    stage: Optional[Usd.Stage]
    group: BridgeHostGroup
    drones: tuple[SpawnedDrone, ...]
    flying: tuple[FlyingDrone, ...]
    driver_cadence: _CadenceState
    switches: SwitchesRuntimeState
    switch_events: tuple[SwitchToggleEvent, ...]


@dataclass(frozen=True)
class _FleetDriverEnvStatus:
    """Latest immutable input already consumed by one routine driver."""

    observation: Optional[RoutineObservation]


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
        self._policy_observer = spawned.policy_observer
        self._port = port
        self._vehicle_id = spawned.vehicle.id
        self._log_fn = log
        self._on_failure_fn = on_failure
        self._on_step_advanced_fn = on_step_advanced
        self._status = _FleetDriverEnvStatus(observation=None)

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
        vehicle_state = self._observer.read_state(sim_time_us=sim_time_us)
        task_observation = (
            None
            if self._policy_observer is None
            else self._policy_observer.read(
                vehicle_state=vehicle_state,
                sim_time_s=sim_time_us * 1e-6,
            )
        )
        observation = RoutineObservation(
            vehicle_state=vehicle_state,
            bridge_status=host_status,
            last_ack=last_ack,
            failsafe=self._port.read_status().failsafe,
            sim_time_us=sim_time_us,
            task_observation=task_observation,
        )
        self._status = _FleetDriverEnvStatus(observation=observation)
        return observation

    def read_status(self) -> _FleetDriverEnvStatus:
        """Return telemetry from the existing driver read without sampling again."""
        return self._status

    def log(self, message: str) -> None:
        self._log_fn(message)
        with vehicle_log_context(self._vehicle_id):
            _ROUTINE_LOGGER.info(message)

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
        driver_cadence=_CadenceState(previous_time_ns=0),
        switches=create_switches_runtime(config=config.switches, stage=None),
        switch_events=(),
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
    return replace(
        state,
        stage=stage,
        switches=adopt_switches_stage(state=state.switches, stage=stage),
    )


def apply_settings(*, state: FleetState) -> FleetState:
    """Apply configured simulation settings to the adopted stage."""
    apply_simulation_settings(stage=state.stage, settings=state.config.sim)
    return state


def spawn_fleet(
    *,
    state: FleetState,
    preflight_policy_artifacts: bool = True,
) -> FleetState:
    """Spawn vehicles and optionally preflight production policy artifacts."""
    if state.stage is None:
        raise RuntimeError(f"{_LOG_PREFIX} spawn_fleet called before the stage was adopted")
    policy_preflight = _PolicyPreflight(policies=())
    if preflight_policy_artifacts and any(
        descriptor.get("kind") == "policy"
        for vehicle in state.config.vehicles.values()
        for descriptor in vehicle.routine.steps
    ):
        policy_preflight = _preflight_policy_artifacts(
            config=state.config,
            repo_root=_repo_root(),
        )
    summary = ", ".join(
        f"{label}: id={vehicle.id} prim={vehicle.spawn.prim_name} "
        f"px4_enabled={vehicle.bridge.enabled} endpoint={vehicle.mavlink_port} "
        f"position_xy={tuple(vehicle.spawn.position_xy)}"
        for label, vehicle in state.config.vehicles.items()
    )
    carb.log_info(f"{_LOG_PREFIX} spawning {len(state.config.vehicles)} vehicle(s): {summary}")
    switches = state.switches
    drones: list[SpawnedDrone] = []
    try:
        switches = play_switches_runtime(state=state.switches)
        for label, vehicle in state.config.vehicles.items():
            path = spawn_drone(stage=state.stage, config=vehicle.spawn)
            host: Optional[BridgeHost] = None
            observer: Optional[IsaacArticulationIO] = None
            articulation_path = vehicle.bridge.articulation_path or path
            if vehicle.bridge.enabled:
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
        for index, drone in enumerate(drones):
            if drone.host is not None:
                drones[index] = _prepare_policy_observer(
                    state=state,
                    spawned=drone,
                    preflight=policy_preflight,
                )
        prepared_drones = tuple(drones)
        group = BridgeHostGroup([drone.host for drone in prepared_drones if drone.host is not None])
        return replace(
            state,
            group=group,
            drones=prepared_drones,
            flying=(),
            switches=switches,
            switch_events=(),
        )
    except Exception:
        partial = replace(
            state,
            group=BridgeHostGroup([drone.host for drone in drones if drone.host is not None]),
            drones=tuple(drones),
            flying=(),
            switches=switches,
        )
        teardown(state=partial)
        raise


def start_bridges(*, state: FleetState) -> FleetState:
    """Start every host before installing the fleet-wide PhysX subscription."""
    state.group.start()
    return state


def _prepare_policy_observer(
    *,
    state: FleetState,
    spawned: SpawnedDrone,
    preflight: _PolicyPreflight,
) -> SpawnedDrone:
    press_descriptors = [
        (policy.switch_index, policy.metadata.contract)
        for policy in preflight.policies
        if policy.vehicle_label == spawned.label
        and policy.metadata.contract.policy_interface is PolicyInterface.PRESS
    ]
    if not press_descriptors:
        return spawned
    indices = {switch_index for switch_index, _contract in press_descriptors}
    if len(indices) != 1 or next(iter(indices)) < 0:
        raise ValueError("press policy steps require one shared non-negative switch_index")
    switch_index = next(iter(indices))
    if switch_index >= state.config.switches.num_switches:
        raise ValueError(
            f"press switch_index={switch_index} is outside "
            f"num_switches={state.config.switches.num_switches}"
        )
    reset = state.config.rl.press.reset
    lab_vehicle = state.config.rl.lab_vehicle
    observer = PressObservationIO(
        stage=state.stage,
        articulation_path=spawned.articulation_path,
        switch_index=switch_index,
        action_dim=press_descriptors[0][1].action_dim,
        contact_force_threshold_n=(state.config.rl.press.termination.contact_force_threshold_n),
        physical_contact_enabled=reset.physical_contact_enabled,
        switch_wall_offset_m=reset.switch_wall_offset_m,
        wall_half_extents_m=reset.wall_half_extents_m,
        pusher_effort_limit_n=lab_vehicle.pusher_effort_limit_n,
        pusher_velocity_limit_mps=lab_vehicle.pusher_velocity_limit_mps,
    )
    return replace(spawned, policy_observer=observer)


def build_drivers(
    *,
    state: FleetState,
    make_env: Callable[[SpawnedDrone, MavlinkCommandPort], DriverEnvironment],
    policy_factory: Callable[..., PolicyAdapter] | None = None,
    routine_steps: tuple[RoutineStepConfig, ...] | None = None,
) -> FleetState:
    """Compose drivers, optionally with an alignment-shell policy factory."""
    clock_host = next((drone.host for drone in state.drones if drone.host is not None), None)
    control_time_ns = (
        read_bridge_host_status(host=clock_host).sim_time_us * 1_000
        if clock_host is not None
        else 0
    )
    creation_tick = DriverTick(
        control_time_us=control_time_ns // 1_000,
        monotonic_s=time.monotonic(),
    )
    flying: list[FlyingDrone] = []
    for spawned in state.drones:
        if spawned.host is None:
            continue
        routine = spawned.vehicle.routine
        active_policy_factory = (
            policy_factory
            if policy_factory is not None
            else partial(
                _make_policy_adapter,
                config=state.config,
                repo_root=_repo_root(),
            )
        )
        port = MavlinkCommandPort(
            gcs_endpoint=f"udp:127.0.0.1:{spawned.vehicle.gcs_port}",
            heartbeat_period_s=spawned.vehicle.routine.gcs_heartbeat_period_s,
            label=spawned.label,
        )
        context = DriverContext(
            params=DriverParams(
                steps=routine.steps if routine_steps is None else routine_steps,
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
                failure_recovery_timeout_s=routine.failure_recovery_timeout_s,
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
                offboard_setpoint_rate_hz=(state.config.driver.offboard_setpoint_rate_hz),
                home_lat_deg=spawned.vehicle.bridge.home_lat_deg,
                home_lon_deg=spawned.vehicle.bridge.home_lon_deg,
                home_alt_m=spawned.vehicle.bridge.home_alt_m,
                goto_min_heading_dist_m=state.config.driver.goto_min_heading_dist_m,
                prearm_backstop_floor_s=state.config.driver.prearm_backstop_floor_s,
                prearm_backstop_multiplier=state.config.driver.prearm_backstop_multiplier,
                drive_budget_multiplier=state.config.driver.drive_budget_multiplier,
                policy_factory=active_policy_factory,
            ),
            port=port,
            env=make_env(spawned, port),
            label=spawned.label,
        )
        driver = create_routine_driver(context=context, tick=creation_tick)
        flying.append(FlyingDrone(spawned=spawned, context=context, driver=driver))
    return replace(
        state,
        flying=tuple(flying),
        driver_cadence=_CadenceState(previous_time_ns=control_time_ns),
    )


def tick_drivers(
    *,
    state: FleetState,
    physics_dt_s: float,
    monotonic_s: Optional[float] = None,
    on_error: Optional[Callable[[FlyingDrone, Exception], None]] = None,
) -> FleetState:
    """Advance the physics-derived fleet cadence and tick every driver when due."""
    previous_time_ns = state.driver_cadence.previous_time_ns
    assert previous_time_ns is not None
    control_time_ns = previous_time_ns + int(round(physics_dt_s * 1_000_000_000))
    toggles: list[tuple[int, bool]] = []
    switches = step_switches_runtime(
        state=state.switches,
        dt_s=physics_dt_s,
        on_toggle=lambda index, value: toggles.append((index, value)),
    )
    switch_events = state.switch_events + tuple(
        SwitchToggleEvent(
            sequence=len(state.switch_events) + offset, switch_index=index, state=value
        )
        for offset, (index, value) in enumerate(toggles, start=1)
    )
    cadence, due = _advance_cadence(
        state=state.driver_cadence,
        time_ns=control_time_ns,
        rate_hz=state.config.driver.control_rate_hz,
    )
    if not due:
        return replace(
            state,
            driver_cadence=cadence,
            switches=switches,
            switch_events=switch_events,
        )
    tick = DriverTick(
        control_time_us=control_time_ns // 1_000,
        monotonic_s=time.monotonic() if monotonic_s is None else monotonic_s,
    )
    advanced: list[FlyingDrone] = []
    for flying in state.flying:
        try:
            driver = tick_routine_driver(
                context=flying.context,
                state=flying.driver,
                tick=tick,
            )
        except Exception as exc:
            if on_error is None:
                raise
            on_error(flying, exc)
            driver = flying.driver
        advanced.append(flying if driver is flying.driver else replace(flying, driver=driver))
    return replace(
        state,
        driver_cadence=cadence,
        flying=tuple(advanced),
        switches=switches,
        switch_events=switch_events,
    )


def fleet_is_done(*, state: FleetState) -> bool:
    """Return whether every composed driver reached DONE."""
    return all(routine_driver_is_done(state=flying.driver) for flying in state.flying)


def teardown(*, state: FleetState) -> FleetState:
    """Stop resources before despawning vehicles, preserving lifecycle order."""
    state.group.stop()
    for flying in state.flying:
        flying.context.port.disconnect()
    for spawned in state.drones:
        if spawned.observer is not None:
            spawned.observer.shutdown()
        if state.stage is not None:
            despawn_drone(stage=state.stage, config=spawned.vehicle.spawn)
    switches = teardown_switches_runtime(state=state.switches)
    return replace(
        state,
        group=BridgeHostGroup([]),
        drones=(),
        flying=(),
        driver_cadence=_CadenceState(previous_time_ns=0),
        switches=switches,
        switch_events=(),
    )

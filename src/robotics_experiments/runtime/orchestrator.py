"""Functional fleet lifecycle over Isaac and MAVLink resource adapters.

Each phase returns a frozen :class:`FleetState`; the live extension and
standalone runner explicitly rebind it at their imperative boundaries.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import carb
import numpy as np

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
    PolicyStepResult,
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

    from robotics_experiments.px4_bridge import IsaacArticulationIO
    from robotics_experiments.runtime._press_io import PressObservationIO

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
    switches: Any
    switch_cadence: _CadenceState
    switch_rate_hz: float
    switch_events: tuple[SwitchToggleEvent, ...]


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
        vehicle_state = self._observer.read_state(sim_time_us=sim_time_us)
        policy_observation = (
            None
            if self._policy_observer is None
            else self._policy_observer.read(
                vehicle_state=vehicle_state,
                sim_time_s=sim_time_us * 1e-6,
            )
        )
        return RoutineObservation(
            vehicle_state=vehicle_state,
            bridge_status=host_status,
            last_ack=last_ack,
            failsafe=self._port.read_status().failsafe,
            sim_time_us=sim_time_us,
            policy_observation=policy_observation,
        )

    def log(self, message: str) -> None:
        self._log_fn(message)

    def on_failure(self, kind: FailureKind, message: str) -> None:
        self._on_failure_fn(kind, message)

    def on_step_advanced(self, step_label: str) -> None:
        self._on_step_advanced_fn(step_label)


def create_fleet(*, config: ExperimentConfig) -> FleetState:
    """Create an empty runtime state for a resolved experiment config."""
    from robotics_experiments.environment import create_switches_runtime

    return FleetState(
        config=config,
        stage=None,
        group=BridgeHostGroup([]),
        drones=(),
        flying=(),
        driver_cadence=_CadenceState(previous_time_ns=0),
        switches=create_switches_runtime(config=config.switches, stage=None),
        switch_cadence=_CadenceState(previous_time_ns=0),
        switch_rate_hz=config.rl.policy_rate_hz,
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
    from robotics_experiments.environment import adopt_switches_stage

    return replace(
        state,
        stage=stage,
        switches=adopt_switches_stage(state=state.switches, stage=stage),
    )


def apply_settings(*, state: FleetState) -> FleetState:
    """Apply configured simulation settings to the adopted stage."""
    from robotics_experiments.simulation import apply_settings as apply_simulation_settings

    apply_simulation_settings(stage=state.stage, settings=state.config.sim)
    return state


def spawn_fleet(*, state: FleetState) -> FleetState:
    """Spawn the ordered vehicle mapping and compose bridge resources."""
    from robotics_experiments.drone import spawn_drone
    from robotics_experiments.environment import play_switches_runtime

    if state.stage is None:
        raise RuntimeError(f"{_LOG_PREFIX} spawn_fleet called before the stage was adopted")
    switch_rate_hz = state.switch_rate_hz
    if any(
        descriptor.get("kind") == "policy"
        for vehicle in state.config.vehicles.values()
        for descriptor in vehicle.routine.steps
    ):
        switch_rate_hz = _preflight_policy_artifacts(config=state.config)
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
        for index, drone in enumerate(drones):
            if drone.host is not None:
                drones[index] = _prepare_policy_observer(state=state, spawned=drone)
        prepared_drones = tuple(drones)
        group = BridgeHostGroup([drone.host for drone in prepared_drones if drone.host is not None])
        return replace(
            state,
            group=group,
            drones=prepared_drones,
            flying=(),
            switches=switches,
            switch_rate_hz=switch_rate_hz,
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


@dataclass(frozen=True)
class _HoverPolicyProgress:
    started_us: Optional[int] = None


def _artifact_path(*, descriptor: Mapping[str, Any]) -> Path:
    artifact = Path(str(descriptor["artifact"]))
    return artifact if artifact.is_absolute() else _repo_root() / artifact


def _policy_contract(*, descriptor: Mapping[str, Any]):
    from robotics_experiments.rl.press import load_policy_contract

    return load_policy_contract(artifact_dir=_artifact_path(descriptor=descriptor))


def _policy_interface(*, descriptor: Mapping[str, Any]) -> str:
    return _policy_contract(descriptor=descriptor).policy_interface


def _preflight_policy_artifacts(*, config: ExperimentConfig) -> float:
    from robotics_experiments.rl.press import (
        HOVER_OBSERVATION_DIM,
        HOVER_OBSERVATION_SCHEMA_VERSION,
        HOVER_POLICY_INTERFACE,
        OBSERVATION_SCHEMA_VERSION,
        PRESS_OBSERVATION_DIM,
        PRESS_POLICY_INTERFACE,
        validate_policy_artifact,
    )

    press_rates: set[float] = set()
    for vehicle in config.vehicles.values():
        for descriptor in vehicle.routine.steps:
            if descriptor.get("kind") != "policy":
                continue
            contract = _policy_contract(descriptor=descriptor)
            interface = contract.policy_interface
            if interface == HOVER_POLICY_INTERFACE:
                obs_schema_version = HOVER_OBSERVATION_SCHEMA_VERSION
                obs_dim = HOVER_OBSERVATION_DIM
            elif interface == PRESS_POLICY_INTERFACE:
                obs_schema_version = OBSERVATION_SCHEMA_VERSION
                obs_dim = PRESS_OBSERVATION_DIM
            else:
                raise ValueError(f"unsupported policy_interface={interface!r}")
            contract = validate_policy_artifact(
                artifact_dir=_artifact_path(descriptor=descriptor),
                obs_schema_version=obs_schema_version,
                obs_dim=obs_dim,
                policy_interface=interface,
            )
            if contract.policy_rate_hz > config.driver.control_rate_hz:
                raise ValueError(
                    f"policy artifact {_artifact_path(descriptor=descriptor)} cadence "
                    f"{contract.policy_rate_hz} Hz exceeds driver.control_rate_hz="
                    f"{config.driver.control_rate_hz}"
                )
            if interface == PRESS_POLICY_INTERFACE:
                press_rates.add(contract.policy_rate_hz)
    if len(press_rates) > 1:
        raise ValueError(
            "press-v1 policy artifacts in one fleet must share one policy_rate_hz; "
            f"found {sorted(press_rates)}"
        )
    return next(iter(press_rates), config.rl.policy_rate_hz)


def _policy_fn_factory(*, config: ExperimentConfig, descriptor: Mapping[str, Any]) -> PolicyAdapter:
    from robotics_experiments.rl.press import (
        HOVER_OBSERVATION_DIM,
        HOVER_OBSERVATION_SCHEMA_VERSION,
        HOVER_POLICY_INTERFACE,
        OBSERVATION_SCHEMA_VERSION,
        PRESS_OBSERVATION_DIM,
        PRESS_POLICY_INTERFACE,
        HandoffGate,
        PressDeploymentState,
        hover_observation_from_state,
        initial_press_deployment_state,
        load_vector_policy,
        press_action_to_offboard,
        step_press_deployment,
    )

    artifact = _artifact_path(descriptor=descriptor)
    interface = _policy_interface(descriptor=descriptor)
    if interface == HOVER_POLICY_INTERFACE:
        if "hold_s" not in descriptor:
            raise ValueError("hover-v1 policy step requires 'hold_s'")
        hold_s = float(descriptor["hold_s"])
        loaded = load_vector_policy(
            artifact_dir=artifact,
            obs_schema_version=HOVER_OBSERVATION_SCHEMA_VERSION,
            obs_dim=HOVER_OBSERVATION_DIM,
            policy_interface=HOVER_POLICY_INTERFACE,
        )
        contract = loaded.contract

        def _hover_step(observation, target_enu, progress):
            assert isinstance(progress, _HoverPolicyProgress)
            if target_enu is None:
                return PolicyStepResult(
                    progress=progress,
                    setpoint=press_action_to_offboard(np.zeros(contract.action_dim), contract),
                    failure_reason="hover-v1 policy has no target or entry baseline",
                )
            started_us = (
                observation.sim_time_us if progress.started_us is None else progress.started_us
            )
            successor = _HoverPolicyProgress(started_us=started_us)
            return PolicyStepResult(
                progress=successor,
                setpoint=press_action_to_offboard(
                    np.clip(
                        loaded.inference(
                            hover_observation_from_state(
                                observation.vehicle_state,
                                target_enu,
                            )
                        ),
                        -1.0,
                        1.0,
                    ),
                    contract,
                ),
                complete=observation.sim_time_us - started_us >= hold_s * 1_000_000,
            )

        return PolicyAdapter(
            rate_hz=contract.policy_rate_hz,
            initial_progress=_HoverPolicyProgress(),
            step=_hover_step,
        )
    if interface == PRESS_POLICY_INTERFACE:
        loaded = load_vector_policy(
            artifact_dir=artifact,
            obs_schema_version=OBSERVATION_SCHEMA_VERSION,
            obs_dim=PRESS_OBSERVATION_DIM,
            policy_interface=PRESS_POLICY_INTERFACE,
        )
        contract = loaded.contract
        termination = config.rl.press.termination

        def _press_step(observation, target_enu, progress):
            del target_enu
            assert isinstance(progress, PressDeploymentState)
            task_observation = observation.policy_observation
            if task_observation is None:
                return PolicyStepResult(
                    progress=progress,
                    setpoint=press_action_to_offboard(np.zeros(contract.action_dim), contract),
                    failure_reason="press-v1 policy has no runtime task observation",
                )
            task_observation = replace(
                task_observation,
                previous_action=np.asarray(progress.previous_action, dtype=np.float64),
                handoff_mode=2 if progress.first_contact_seen else 1,
            )
            decision = step_press_deployment(
                state=progress,
                observation=task_observation,
                policy=loaded.inference,
                action_dim=contract.action_dim,
                handoff_gate=HandoffGate.from_config(config.rl.press.handoff),
                first_contact_speed_max_mps=termination.first_contact_speed_max_mps,
                first_contact_angular_speed_max_radps=(
                    termination.first_contact_angular_speed_max_radps
                ),
                retreat_distance_m=termination.retreat_distance_m,
                retreat_speed_max_mps=termination.retreat_speed_max_mps,
                retreat_alignment_tolerance_m=(termination.retreat_alignment_tolerance_m),
                retreat_angular_speed_max_radps=(termination.retreat_angular_speed_max_radps),
                stable_retreat_steps=termination.stable_retreat_steps,
            )
            return PolicyStepResult(
                progress=decision.state,
                setpoint=press_action_to_offboard(decision.action, contract),
                complete=decision.complete,
                failure_reason=decision.failure_reason,
            )

        return PolicyAdapter(
            rate_hz=contract.policy_rate_hz,
            initial_progress=initial_press_deployment_state(action_dim=contract.action_dim),
            step=_press_step,
        )
    raise ValueError(f"unsupported policy_interface={interface!r} in artifact {artifact}")


def _prepare_policy_observer(*, state: FleetState, spawned: SpawnedDrone) -> SpawnedDrone:
    from robotics_experiments.rl.press import PRESS_POLICY_INTERFACE

    press_descriptors = [
        (descriptor, _policy_contract(descriptor=descriptor))
        for descriptor in spawned.vehicle.routine.steps
        if descriptor.get("kind") == "policy"
        and _policy_interface(descriptor=descriptor) == PRESS_POLICY_INTERFACE
    ]
    if not press_descriptors:
        return spawned
    from robotics_experiments.runtime._press_io import PressObservationIO

    indices = {
        int(descriptor.get("switch_index", -1)) for descriptor, _contract in press_descriptors
    }
    if len(indices) != 1 or next(iter(indices)) < 0:
        raise ValueError("press-v1 policy steps require one shared non-negative switch_index")
    switch_index = next(iter(indices))
    if switch_index >= state.config.switches.num_switches:
        raise ValueError(
            f"press-v1 switch_index={switch_index} is outside "
            f"num_switches={state.config.switches.num_switches}"
        )
    reset = state.config.rl.press.reset
    lab_vehicle = state.config.rl.lab_vehicle
    observer = PressObservationIO(
        stage=state.stage,
        articulation_path=spawned.articulation_path,
        switch_index=switch_index,
        action_dim=press_descriptors[0][1].action_dim,
        contact_force_threshold_n=state.config.rl.press.contact_force_threshold_n,
        button_geometry_scale=reset.button_geometry_scale,
        button_stiffness_scale=reset.button_stiffness_scale,
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
) -> FleetState:
    """Compose one frozen routine driver state per bridge-enabled drone."""
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
        port = MavlinkCommandPort(
            gcs_endpoint=f"udp:127.0.0.1:{spawned.vehicle.gcs_port}",
            heartbeat_period_s=spawned.vehicle.routine.gcs_heartbeat_period_s,
            label=spawned.label,
        )
        context = DriverContext(
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
                offboard_setpoint_rate_hz=(state.config.driver.offboard_setpoint_rate_hz),
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
    from robotics_experiments.environment import step_switches_runtime

    previous_time_ns = state.driver_cadence.previous_time_ns
    assert previous_time_ns is not None
    control_time_ns = previous_time_ns + int(round(physics_dt_s * 1_000_000_000))
    switch_cadence, switch_due = _advance_cadence(
        state=state.switch_cadence,
        time_ns=control_time_ns,
        rate_hz=state.switch_rate_hz,
    )
    switches = state.switches
    switch_events = state.switch_events
    if switch_due:
        toggles: list[tuple[int, bool]] = []
        switches = step_switches_runtime(
            state=switches,
            on_toggle=lambda index, value: toggles.append((index, value)),
        )
        if toggles:
            first_sequence = len(switch_events)
            switch_events += tuple(
                SwitchToggleEvent(
                    sequence=first_sequence + offset,
                    switch_index=index,
                    state=value,
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
            switch_cadence=switch_cadence,
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
        switch_cadence=switch_cadence,
        switch_events=switch_events,
    )


def fleet_is_done(*, state: FleetState) -> bool:
    """Return whether every composed driver reached DONE."""
    return all(routine_driver_is_done(state=flying.driver) for flying in state.flying)


def teardown(*, state: FleetState) -> FleetState:
    """Stop resources before despawning vehicles, preserving lifecycle order."""
    from robotics_experiments.drone import despawn_drone
    from robotics_experiments.environment import teardown_switches_runtime

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
        switch_cadence=_CadenceState(previous_time_ns=0),
        switch_events=(),
    )

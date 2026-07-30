"""Mutable PhysX resource adapters around frozen PX4 bridge-host state.

This Isaac-only leaf keeps callback return values at the imperative boundary:
each callback runs a module-level transition and replaces the host's single
frozen state value. Its PX4 bridge dependencies are imported at module scope so
the physics callbacks do not repeat cached imports; the lazy runtime package
keeps this Isaac-coupled module off the core import path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import carb
import omni.physx

from robotics_experiments.config import Px4BridgeConfig
from robotics_experiments.px4_bridge import (
    BridgeStatus,
    DiscoveredRotorMapping,
    LockstepSettings,
    PegasusQuadraticThrustModel,
    Px4BridgeState,
    Px4MavlinkPort,
    SimClockState,
    advance_sim_clock,
    bridge_time_us,
    create_pegasus_thrust_model,
    create_px4_bridge,
    create_sim_clock,
    discover_px4_rotor_mapping,
    recv_and_apply,
    send_sensors,
    sim_clock_is_settling,
    start_px4_bridge,
    stop_px4_bridge,
    vehicle_log_context,
)

__all__ = [
    "BridgeHost",
    "BridgeHostGroup",
    "BridgeHostParams",
    "BridgeHostStatus",
    "read_bridge_host_status",
]

_LOG_PREFIX = "FleetRuntime.BridgeHost"


@dataclass(frozen=True)
class _PendingPoll:
    tick: int
    bridge_sim_time_us: int
    block_start: float


@dataclass(frozen=True)
class BridgeHostParams:
    """Immutable construction data for one vehicle's bridge host."""

    bridge: Px4BridgeConfig
    vehicle_id: int
    mavlink_port: int
    debug_log_path: str
    articulation_path: str
    physics_hz: float


@dataclass(frozen=True)
class BridgeHostStatus:
    """Immutable runtime read model for one bridge host."""

    sim_time_us: int
    bridge_status: Optional[BridgeStatus]
    start_attempted: bool
    debug_log_path: Optional[Path]


@dataclass(frozen=True)
class _BridgeHostState:
    clock: Optional[SimClockState] = None
    bridge: Optional[Px4BridgeState] = None
    start_attempted: bool = False
    pending: Optional[_PendingPoll] = None
    disconnect_warned: bool = False
    max_tick_block_s: float = 0.0


def _bridge_components(
    *, params: BridgeHostParams, rotor_mapping: DiscoveredRotorMapping
) -> tuple[LockstepSettings, Px4MavlinkPort, PegasusQuadraticThrustModel]:
    cfg = params.bridge
    lockstep_cfg = cfg.lockstep
    lockstep = LockstepSettings(
        lockstep_timeout_s=lockstep_cfg.lockstep_timeout_s,
        first_heartbeat_timeout_s=lockstep_cfg.first_heartbeat_timeout_s,
        bootstrap_prime_ticks=lockstep_cfg.bootstrap_prime_ticks,
        first_actuator_timeout_s=lockstep_cfg.first_actuator_timeout_s,
        max_sustained_actuator_lag_s=lockstep_cfg.max_sustained_actuator_lag_s,
        heartbeat_rate_hz=lockstep_cfg.heartbeat_rate_hz,
        physics_rate_hz=params.physics_hz,
    )
    rotor_count = len(rotor_mapping.rotor_link_paths)
    port = Px4MavlinkPort(
        num_rotors=rotor_count,
        connection_type=cfg.connection_type,
        connection_ip=cfg.connection_ip,
        port=params.mavlink_port,
        vehicle_id=params.vehicle_id,
        input_offset=cfg.input_offset,
        input_scaling=cfg.input_scaling,
        zero_position_armed=cfg.zero_position_armed,
        lockstep=lockstep,
        gps_enabled=cfg.gps_enabled,
        gps_send_rate_hz=cfg.gps_send_rate_hz,
        sensor_noise_enabled=cfg.sensor_noise_enabled,
        sensor_excitation_mode=cfg.sensor_excitation_mode,
        barometer_noise_std_pa=cfg.barometer_noise_std_pa,
        magnetometer_noise_scale=cfg.magnetometer_noise_scale,
        home_lat_deg=cfg.home_lat_deg,
        home_lon_deg=cfg.home_lon_deg,
        home_alt_m=cfg.home_alt_m,
        debug_log_path=params.debug_log_path,
        debug_log_level=cfg.debug_log_level,
        sensor_debug_log_interval=cfg.sensor_debug_log_interval,
    )
    thrust_model = create_pegasus_thrust_model(
        num_rotors=rotor_count,
        rotor_constant=cfg.rotor_constant,
        rolling_moment_coefficient=cfg.rolling_moment_coefficient,
        rot_dir=rotor_mapping.rot_dir,
        min_rotor_velocity=cfg.min_rotor_velocity,
        max_rotor_velocity=cfg.max_rotor_velocity,
        linear_drag=cfg.linear_drag,
    )
    return lockstep, port, thrust_model


class BridgeHost:
    """Minimal mutable adapter required because PhysX ignores callback returns."""

    def __init__(self, params: BridgeHostParams):
        self.params = params
        cfg = params.bridge
        self.endpoint = f"{cfg.connection_type}:{cfg.connection_ip}:{params.mavlink_port}"
        self.tag = f"{_LOG_PREFIX} v{params.vehicle_id} {self.endpoint}"
        self.first_connect_tick_budget = int(
            cfg.lockstep.first_heartbeat_timeout_s * params.physics_hz
        )
        self.slow_tick_warn_s = 0.5 * float(cfg.lockstep.lockstep_timeout_s)
        self._state = _BridgeHostState()

    def start(self) -> None:
        self._state = _start_host(host=self, state=self._state)

    def stop(self) -> None:
        self._state = _stop_host(host=self, state=self._state)

    def send_phase(self, dt: float) -> None:
        self._state = _send_phase(host=self, state=self._state, dt=dt)

    def recv_phase(self) -> None:
        self._state = _recv_phase(host=self, state=self._state)


def read_bridge_host_status(*, host: BridgeHost) -> BridgeHostStatus:
    """Read one immutable host status without exposing transition state."""
    state = host._state  # noqa: SLF001 - module status boundary
    sim_time_us = max(0, bridge_time_us(state=state.clock)) if state.clock is not None else 0
    return BridgeHostStatus(
        sim_time_us=sim_time_us,
        bridge_status=state.bridge.port.read_status() if state.bridge is not None else None,
        start_attempted=state.start_attempted,
        debug_log_path=(state.bridge.port.debug_log_path if state.bridge is not None else None),
    )


def _start_host(*, host: BridgeHost, state: _BridgeHostState) -> _BridgeHostState:
    if state.bridge is not None or not host.params.bridge.enabled:
        return state

    settle_time_s = max(0.0, float(host.params.bridge.startup_settle_time_s))
    started = replace(
        state,
        clock=create_sim_clock(
            physics_hz=host.params.physics_hz,
            settle_time_s=settle_time_s,
        ),
        bridge=None,
        start_attempted=False,
        pending=None,
    )
    if settle_time_s > 0.0:
        carb.log_info(
            f"[{host.tag}] delaying PX4 bridge start for "
            f"startup_settle_time_s={settle_time_s:.3f}"
        )
        return started
    return _start_bridge(host=host, state=started)


def _start_bridge(*, host: BridgeHost, state: _BridgeHostState) -> _BridgeHostState:
    if state.bridge is not None or state.start_attempted:
        return state

    attempted = replace(state, start_attempted=True)
    with vehicle_log_context(host.params.vehicle_id):
        return _build_bridge(host=host, state=attempted)


def _build_bridge(*, host: BridgeHost, state: _BridgeHostState) -> _BridgeHostState:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        carb.log_error(
            f"[{host.tag}] cannot discover PX4 rotors without an open USD stage. "
            "Bridge not started."
        )
        return state
    try:
        rotor_mapping = discover_px4_rotor_mapping(stage, host.params.articulation_path)
    except Exception as exc:
        carb.log_error(f"[{host.tag}] PX4 rotor discovery failed: {exc}")
        return state

    lockstep, port, thrust_model = _bridge_components(
        params=host.params,
        rotor_mapping=rotor_mapping,
    )
    cfg = host.params.bridge
    carb.log_info(
        f"[{host.tag}] starting PX4 bridge "
        f"articulation={host.params.articulation_path} "
        f"rotor_paths={rotor_mapping.rotor_link_paths} rot_dir={rotor_mapping.rot_dir} "
        f"lockstep_timeout_s={lockstep.lockstep_timeout_s} "
        f"first_heartbeat_timeout_s={lockstep.first_heartbeat_timeout_s} "
        f"bootstrap_prime_ticks={lockstep.bootstrap_prime_ticks} "
        f"first_actuator_timeout_s={lockstep.first_actuator_timeout_s} "
        f"max_sustained_actuator_lag_s={lockstep.max_sustained_actuator_lag_s} "
        f"physics_rate_hz={host.params.physics_hz} gps_enabled={cfg.gps_enabled} "
        f"gps_rate_hz={cfg.gps_send_rate_hz} "
        f"sensor_noise_enabled={cfg.sensor_noise_enabled} "
        f"sensor_excitation_mode={cfg.sensor_excitation_mode} "
        f"barometer_noise_std_pa={cfg.barometer_noise_std_pa} "
        f"magnetometer_noise_scale={cfg.magnetometer_noise_scale} "
        f"debug_log_path={host.params.debug_log_path}"
    )
    bridge = create_px4_bridge(
        articulation_path=host.params.articulation_path,
        rotor_link_paths=rotor_mapping.rotor_link_paths,
        thrust_model=thrust_model,
        port=port,
        max_tilt_angle_deg=cfg.max_tilt_angle_deg,
        rotor_spin_visual_scale=cfg.rotor_spin_visual_scale,
        rotor_spin_signs=rotor_mapping.rot_dir,
    )
    try:
        bridge = start_px4_bridge(state=bridge)
    except Exception as exc:
        carb.log_error(f"[{host.tag}] bridge start failed: {exc}")
        return state
    carb.log_info(f"[{host.tag}] PX4 bridge started")
    return replace(state, bridge=bridge)


def _stop_host(*, host: BridgeHost, state: _BridgeHostState) -> _BridgeHostState:
    if state.bridge is not None:
        try:
            with vehicle_log_context(host.params.vehicle_id):
                stop_px4_bridge(state=state.bridge)
        except Exception as exc:
            carb.log_warn(f"[{host.tag}] bridge stop raised: {exc}")
    return replace(state, bridge=None, pending=None)


def _send_phase(*, host: BridgeHost, state: _BridgeHostState, dt: float) -> _BridgeHostState:
    clock = state.clock
    if clock is None:
        raise RuntimeError(f"[{host.tag}] send_phase called before start")
    clock = advance_sim_clock(state=clock, dt=dt)
    current = replace(state, clock=clock, pending=None)
    if current.bridge is None:
        if not sim_clock_is_settling(state=clock):
            carb.log_info(
                f"[{host.tag}] startup settle complete at tick={clock.tick_count} "
                f"sim_time_us={clock.total_us}; starting PX4 bridge"
            )
            return _start_bridge(host=host, state=current)
        if clock.tick_count == 1 or clock.tick_count % 250 == 0:
            carb.log_info(
                f"[{host.tag}] waiting for startup settle tick={clock.tick_count} "
                f"dt={dt:.6f} sim_time_us={clock.total_us} "
                f"target_us={clock.settle_time_us}"
            )
        return current

    sim_time_us = bridge_time_us(state=clock)
    bridge_state = current.bridge
    if (
        not current.disconnect_warned
        and clock.tick_count >= host.first_connect_tick_budget
        and not bridge_state.port.is_connected()
    ):
        current = replace(current, disconnect_warned=True)
        carb.log_warn(
            f"[{host.tag}] PX4 still not connected after tick={clock.tick_count} "
            f"(>{host.first_connect_tick_budget}); this PX4 will stall in lockstep. "
            f"Check that a PX4 SITL instance is reachable on {host.endpoint}."
        )
    # Start before the send so the diagnostic spans this vehicle's send through
    # its paired poll, including any intervening fleet sends.
    block_start = time.monotonic()
    try:
        with vehicle_log_context(host.params.vehicle_id):
            bridge = send_sensors(
                state=bridge_state,
                dt=dt,
                sim_time_us=sim_time_us,
            )
    except Exception as exc:
        carb.log_warn(
            f"[{host.tag}] send_sensors error at tick={clock.tick_count} "
            f"sim_time_us={sim_time_us}: {exc}"
        )
        return replace(current, bridge=replace(bridge_state, pending=None))
    return replace(
        current,
        bridge=bridge,
        pending=_PendingPoll(
            tick=clock.tick_count,
            bridge_sim_time_us=sim_time_us,
            block_start=block_start,
        ),
    )


def _recv_phase(*, host: BridgeHost, state: _BridgeHostState) -> _BridgeHostState:
    pending = state.pending
    bridge = state.bridge
    if pending is None or bridge is None:
        return state

    current = replace(state, pending=None)
    try:
        with vehicle_log_context(host.params.vehicle_id):
            bridge = recv_and_apply(state=bridge)
        current = replace(current, bridge=bridge)
    except Exception as exc:
        carb.log_warn(
            f"[{host.tag}] recv_and_apply error at tick={pending.tick} "
            f"sim_time_us={pending.bridge_sim_time_us}: {exc}"
        )
        bridge = replace(bridge, pending=None)
        current = replace(current, bridge=bridge)
    block_s = time.monotonic() - pending.block_start
    current = replace(
        current,
        max_tick_block_s=max(current.max_tick_block_s, block_s),
    )
    if block_s >= host.slow_tick_warn_s:
        carb.log_warn(
            f"[{host.tag}] slow bridge tick blocked for {block_s:.3f}s "
            f"at tick={pending.tick} "
            f"(lockstep_timeout_s={host.params.bridge.lockstep.lockstep_timeout_s}); "
            "this bounds the fleet tick rate"
        )
    if pending.tick == 1 or pending.tick % 250 == 0:
        status = bridge.port.read_status()
        carb.log_info(
            f"[{host.tag}] physics tick={pending.tick} "
            f"sim_time_us={pending.bridge_sim_time_us} "
            f"connected={status.is_connected} tick_block_s={block_s:.4f} "
            f"max_tick_block_s={current.max_tick_block_s:.4f}"
        )
    return current


class BridgeHostGroup:
    """Mutable owner of the single fleet-wide PhysX subscription."""

    def __init__(self, hosts: list[BridgeHost]):
        self.hosts = list(hosts)
        self._physx_sub = None

    def start(self) -> None:
        for host in self.hosts:
            host.start()
        physx = omni.physx.get_physx_interface()
        self._physx_sub = physx.subscribe_physics_step_events(self._on_physics_step)

    def stop(self) -> None:
        self._physx_sub = None
        for host in self.hosts:
            host.stop()

    def _on_physics_step(self, dt: float) -> None:
        for host in self.hosts:
            host.send_phase(dt)
        for host in self.hosts:
            host.recv_phase()

"""PX4 bridge wiring owned by the fleet runtime.

The :class:`~robotics_experiments.runtime.orchestrator.FleetOrchestrator`
constructs each :class:`Px4Bridge` (through a ``BridgeHost``) and the single
fleet-wide physics-step subscription (through :class:`BridgeHostGroup`). The
bridge itself owns no scene state; keeping this wiring out of the bridge keeps
the bridge single-responsibility.

A host is configured through :class:`BridgeHostParams`, a frozen composition of
the vehicle's bridge behavior config (``px4``) with the derived per-vehicle
runtime state the host needs (identity, endpoint, resolved articulation path),
constructed by the orchestrator when it spawns the vehicle. The host never sees
the rest of the ``ResolvedVehicle`` (spawn, routine, ROS identity).

The host is opt-in: when the experiment config has ``px4.enabled = false``
(the default), no bridge is built and no callbacks are registered. This lets
the bridge layers be tested in stages (port alone, articulation_io alone,
full bridge) by flipping the flag.

Isaac-coupled domain leaf: this module imports ``carb``/``omni.physx`` at module
scope, so it stays off the core import path behind the lazy package
``__init__`` (see ``robotics_experiments.runtime``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import carb
import omni.physx

from robotics_experiments.config.types import Px4BridgeConfig

__all__ = ["BridgeHost", "BridgeHostGroup", "BridgeHostParams"]

_LOG_PREFIX = "FleetRuntime.BridgeHost"


@dataclass(frozen=True)
class _PendingPoll:
    """Send-leg state of one physics tick, consumed by the matching poll leg.

    Produced by :meth:`BridgeHost.send_phase` only when this tick's sensors went
    out, so ``BridgeHost._pending is None`` is the distinct "nothing to poll this
    tick" state (settling, or a send-leg error).
    """

    tick: int
    bridge_sim_time_us: int
    block_start: float


def _bridge_components(params: BridgeHostParams, rotor_mapping: Any) -> tuple[Any, Any, Any]:
    """Pure config -> bridge-component mapping: ``(lockstep, port, thrust_model)``.

    Everything here is derived from the frozen params plus the discovered rotor
    mapping; the effects (stage lookup, discovery, ``Px4Bridge`` construction and
    start) stay in :meth:`BridgeHost._build_bridge`. Imports lazily so a disabled
    bridge pays no ``px4_bridge`` import cost.
    """
    from robotics_experiments.px4_bridge import (
        LockstepSettings,
        PegasusQuadraticThrustModel,
        Px4MavlinkPort,
    )

    cfg = params.px4
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
    port = Px4MavlinkPort(
        num_rotors=rotor_mapping.num_rotors,
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
        home_lat_deg=cfg.home_lat_deg,
        home_lon_deg=cfg.home_lon_deg,
        home_alt_m=cfg.home_alt_m,
        debug_log_path=params.debug_log_path,
        debug_log_level=cfg.debug_log_level,
        sensor_debug_log_interval=cfg.sensor_debug_log_interval,
    )
    thrust_model = PegasusQuadraticThrustModel(
        num_rotors=rotor_mapping.num_rotors,
        rotor_constant=cfg.rotor_constant,
        rolling_moment_coefficient=cfg.rolling_moment_coefficient,
        rot_dir=rotor_mapping.rot_dir,
        min_rotor_velocity=cfg.min_rotor_velocity,
        max_rotor_velocity=cfg.max_rotor_velocity,
        linear_drag=cfg.linear_drag,
    )
    return lockstep, port, thrust_model


@dataclass(frozen=True)
class BridgeHostParams:
    """Everything one :class:`BridgeHost` needs, separated by responsibility.

    ``px4`` is the vehicle's bridge behavior config; the remaining fields are the
    derived per-vehicle runtime state, taken straight from the
    :class:`~robotics_experiments.config.ResolvedVehicle` (never recomputed here)
    plus the spawn result. ``articulation_path`` is resolved exactly once (the
    ``px4.articulation_path`` override, else the spawned prim path), so the host
    and the state observer read the same body.
    """

    px4: Px4BridgeConfig
    vehicle_id: int
    mavlink_port: int
    debug_log_path: str
    articulation_path: str
    physics_hz: float


class BridgeHost:
    """Lifecycle owner for a single ``Px4Bridge`` instance.

    Constructed once per Play session by the fleet orchestrator from a frozen
    :class:`BridgeHostParams`. Builds the bridge from those params, registers a
    physics-step callback that ticks it, and tears everything down on stop.
    """

    def __init__(self, params: BridgeHostParams):
        self._params = params
        self._cfg = params.px4
        self._bridge: Any | None = None
        self._clock: Any = None
        self._start_attempted: bool = False
        # Send-leg state consumed by the poll leg of the same tick (the
        # coordinator runs all sends before all polls); None = nothing to poll.
        self._pending: Optional[_PendingPoll] = None
        # Cached px4_bridge.vehicle_log_context, bound lazily when the bridge is
        # built (kept off the module top to preserve the lazy import contract).
        self._log_ctx: Any = None
        # Per-vehicle log tag so the two BridgeHosts in a multi-drone scene are
        # distinguishable in the shared Isaac console.
        self._endpoint = (
            f"{self._cfg.connection_type}:{self._cfg.connection_ip}:{params.mavlink_port}"
        )
        self._tag = f"{_LOG_PREFIX} v{params.vehicle_id} {self._endpoint}"
        # Tick budget after which a started-but-never-connected bridge is loud:
        # the first PX4 heartbeat must arrive within first_heartbeat_timeout_s.
        self._first_connect_tick_budget = int(
            self._cfg.lockstep.first_heartbeat_timeout_s * params.physics_hz
        )
        self._disconnect_warned = False
        # The hosts tick serially on the single PhysX thread and each bridge tick
        # blocks on its PX4 lockstep recv, so a slow tick here stalls every other
        # vehicle's lockstep too. Warn when one tick blocks the shared thread for
        # a large fraction of the lockstep timeout (derived, not a new knob).
        self._slow_tick_warn_s = 0.5 * float(self._cfg.lockstep.lockstep_timeout_s)
        self._max_tick_block_s = 0.0

    @property
    def is_running(self) -> bool:
        return self._bridge is not None

    @property
    def is_connected(self) -> bool:
        """Whether the bridge has a live PX4 lockstep connection.

        ``False`` before :meth:`start` or while the bridge is still settling.
        Read by the control driver to gate the routine on a connected bridge.
        """
        return self._bridge is not None and bool(self._bridge.is_connected)

    @property
    def status(self) -> Any:
        """The bridge lockstep status snapshot (``BridgeStatus``).

        The control driver folds this into each ``RoutineObservation`` so the
        pure command layer never reaches into the bridge. Only valid once the
        host is running.

        Raises:
            RuntimeError: if read before the bridge is constructed.
        """
        if self._bridge is None:
            raise RuntimeError(f"[{self._tag}] bridge status read before the bridge started")
        return self._bridge.status

    @property
    def sim_time_us(self) -> int:
        """Total accumulated sim time in microseconds since the clock started.

        Zero before :meth:`start`. This is the host-side counter (pre settle
        offset); the bridge-facing timestamp the host feeds the bridge is
        derived from it via the clock. Exposed for diagnostics and test
        scenarios that drive physics and wait on simulated time.
        """
        return int(self._clock.total_us) if self._clock is not None else 0

    def start(self) -> None:
        """Set up the sim clock and (when no settle delay) construct the bridge.

        The owning :class:`BridgeHostGroup` owns the single physics-step
        subscription and drives this host's per-leg :meth:`send_phase` /
        :meth:`recv_phase`. Idempotent: a second call while running is a no-op.
        """
        if self._bridge is not None:
            return
        if not self._cfg.enabled:
            return

        # SimClock is pure Python (no Isaac deps); import it here so a disabled
        # bridge does not pay the px4_bridge import cost.
        from robotics_experiments.px4_bridge.sim_clock import SimClock

        settle_time_s = max(0.0, float(self._cfg.startup_settle_time_s))
        self._clock = SimClock(physics_hz=self._params.physics_hz, settle_time_s=settle_time_s)
        self._start_attempted = False
        if settle_time_s > 0.0:
            carb.log_info(
                f"[{self._tag}] delaying PX4 bridge start for "
                f"startup_settle_time_s={settle_time_s:.3f}"
            )
            return

        self._start_bridge()

    def _start_bridge(self) -> None:
        """Construct the bridge after any configured startup settle delay."""
        if self._bridge is not None or self._start_attempted:
            return
        self._start_attempted = True

        # Late import so a disabled bridge does not pay the import cost
        # (and does not require the bridge extension to be loaded).
        from robotics_experiments.px4_bridge.logging import vehicle_log_context

        # Tag every bridge log record produced while building this vehicle's
        # bridge so it lands only in this vehicle's debug log file, not the
        # last-constructed vehicle's. See px4_bridge.logging.
        self._log_ctx = vehicle_log_context
        with vehicle_log_context(self._params.vehicle_id):
            self._build_bridge()

    def _build_bridge(self) -> None:
        """Discover rotors and construct the bridge for this vehicle.

        Owns the effects (stage lookup, rotor discovery, bridge construction and
        start); the pure params -> component mapping lives in
        :func:`_bridge_components`. Runs inside the vehicle's log context (see
        :meth:`_start_bridge`).
        """
        import omni.usd

        from robotics_experiments.px4_bridge import Px4Bridge, discover_px4_rotor_mapping

        articulation_path = self._params.articulation_path
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            carb.log_error(
                f"[{self._tag}] cannot discover PX4 rotors without an open USD stage. "
                "Bridge not started."
            )
            return
        try:
            rotor_mapping = discover_px4_rotor_mapping(stage, articulation_path)
        except Exception as exc:
            carb.log_error(f"[{self._tag}] PX4 rotor discovery failed: {exc}")
            return

        lockstep, port, thrust_model = _bridge_components(self._params, rotor_mapping)
        carb.log_info(
            f"[{self._tag}] starting PX4 bridge "
            f"articulation={articulation_path} rotor_paths={rotor_mapping.rotor_link_paths} "
            f"rot_dir={rotor_mapping.rot_dir} "
            f"lockstep_timeout_s={lockstep.lockstep_timeout_s} "
            f"first_heartbeat_timeout_s={lockstep.first_heartbeat_timeout_s} "
            f"bootstrap_prime_ticks={lockstep.bootstrap_prime_ticks} "
            f"first_actuator_timeout_s={lockstep.first_actuator_timeout_s} "
            f"max_sustained_actuator_lag_s={lockstep.max_sustained_actuator_lag_s} "
            f"physics_rate_hz={self._params.physics_hz} "
            f"gps_enabled={self._cfg.gps_enabled} gps_rate_hz={self._cfg.gps_send_rate_hz} "
            f"sensor_noise_enabled={self._cfg.sensor_noise_enabled} "
            f"debug_log_path={self._params.debug_log_path}"
        )
        self._bridge = Px4Bridge(
            articulation_path=articulation_path,
            rotor_link_paths=rotor_mapping.rotor_link_paths,
            thrust_model=thrust_model,
            port=port,
            max_tilt_angle_deg=self._cfg.max_tilt_angle_deg,
            rotor_spin_visual_scale=self._cfg.rotor_spin_visual_scale,
            rotor_spin_signs=rotor_mapping.rot_dir,
        )

        try:
            self._bridge.start()
        except Exception as exc:
            carb.log_error(f"[{self._tag}] bridge.start() failed: {exc}")
            self._bridge = None
            return

        carb.log_info(f"[{self._tag}] PX4 bridge started")

    def stop(self) -> None:
        """Tear down the bridge. The group owns and drops the subscription."""
        if self._bridge is not None:
            try:
                with self._log_ctx(self._params.vehicle_id):
                    self._bridge.stop()
            except Exception as exc:
                carb.log_warn(f"[{self._tag}] bridge.stop() raised: {exc}")
            self._bridge = None

    def send_phase(self, dt: float) -> None:
        """Send leg of one physics tick: advance the clock and publish sensors.

        Runs before any vehicle's poll leg. Advances the sim clock, handles the
        startup-settle gating (lazily constructing the bridge once settle
        completes), and, when the bridge is running, sends this tick's sensors to
        PX4 without blocking. Leaves the :class:`_PendingPoll` for
        :meth:`recv_phase` when a poll is owed this tick.
        """
        self._pending = None
        total_us = self._clock.advance(dt)
        tick = self._clock.tick_count
        if self._bridge is None:
            if not self._clock.is_settling:
                carb.log_info(
                    f"[{self._tag}] startup settle complete at "
                    f"tick={tick} sim_time_us={total_us}; "
                    f"starting PX4 bridge"
                )
                self._start_bridge()
                return
            elif tick == 1 or tick % 250 == 0:
                carb.log_info(
                    f"[{self._tag}] waiting for startup settle "
                    f"tick={tick} dt={dt:.6f} "
                    f"sim_time_us={total_us} target_us={self._clock.settle_time_us}"
                )
            return

        bridge_sim_time_us = self._clock.bridge_time_us()
        # A bridge that never connects starves its PX4 of sensor data, so that
        # PX4's lockstep stalls and the GCS hangs (e.g. on `commander arm`).
        # Make that obvious instead of leaving it to a 250-tick info heartbeat.
        if not self._bridge.is_connected and not self._disconnect_warned:
            if tick >= self._first_connect_tick_budget:
                self._disconnect_warned = True
                carb.log_warn(
                    f"[{self._tag}] PX4 still not connected after "
                    f"tick={tick} (>{self._first_connect_tick_budget}); this PX4 "
                    f"will stall in lockstep. Check that a PX4 SITL instance is "
                    f"reachable on {self._endpoint}."
                )
        # The meaningful block is the poll in recv_phase; start the timer here so
        # the diagnostic spans this vehicle's full send-then-poll contribution.
        block_start = time.monotonic()
        try:
            with self._log_ctx(self._params.vehicle_id):
                self._bridge.send_sensors(dt, bridge_sim_time_us)
            self._pending = _PendingPoll(
                tick=tick, bridge_sim_time_us=bridge_sim_time_us, block_start=block_start
            )
        except Exception as exc:
            # One bad tick must not crash the simulator. Log and continue;
            # persistent failure will be visible via is_connected.
            carb.log_warn(
                f"[{self._tag}] send_sensors error at tick={tick} "
                f"sim_time_us={bridge_sim_time_us}: {exc}"
            )

    def recv_phase(self) -> None:
        """Poll leg of one physics tick: block for the actuator and apply forces.

        Runs after every vehicle's send leg, so the PX4 instances compute
        concurrently and one stalled PX4 disconnects only its own vehicle. A
        no-op when this tick did not send (settling, or a send-leg error).
        """
        pending = self._pending
        if pending is None or self._bridge is None:
            return
        self._pending = None
        try:
            with self._log_ctx(self._params.vehicle_id):
                self._bridge.recv_and_apply()
        except Exception as exc:
            carb.log_warn(
                f"[{self._tag}] recv_and_apply error at tick={pending.tick} "
                f"sim_time_us={pending.bridge_sim_time_us}: {exc}"
            )
        block_s = time.monotonic() - pending.block_start
        self._max_tick_block_s = max(self._max_tick_block_s, block_s)
        # A tick that blocks for a large fraction of the lockstep timeout is the
        # poll waiting on this vehicle's PX4. With the two-phase pipeline that no
        # longer starves the other vehicles' sends, but a chronically slow poll
        # still bounds the fleet tick rate; surface the offending vehicle.
        if block_s >= self._slow_tick_warn_s:
            carb.log_warn(
                f"[{self._tag}] slow bridge tick blocked for {block_s:.3f}s "
                f"at tick={pending.tick} "
                f"(lockstep_timeout_s={self._cfg.lockstep.lockstep_timeout_s}); "
                f"this bounds the fleet tick rate"
            )
        if pending.tick == 1 or pending.tick % 250 == 0:
            carb.log_info(
                f"[{self._tag}] physics tick={pending.tick} "
                f"sim_time_us={pending.bridge_sim_time_us} "
                f"connected={self._bridge.is_connected} "
                f"tick_block_s={block_s:.4f} max_tick_block_s={self._max_tick_block_s:.4f}"
            )


class BridgeHostGroup:
    """Coordinator that drives a set of :class:`BridgeHost`s in two phases.

    Owns the single physics-step subscription for the whole group (one host for
    the default single-drone config). On each PhysX step it runs every host's
    send leg first -- sensors out to every PX4, no blocking -- then every host's
    poll leg -- block for each actuator. The PX4 instances therefore compute
    concurrently: a single stalled PX4 disconnects only its own vehicle instead
    of holding the shared physics thread and freezing the rest. Per-tick latency
    drops from the sum of the per-vehicle blocking waits to ~their max. A
    one-host group is exactly the old send-then-poll-per-tick behavior.

    Uses ``omni.physx`` and so must stay off the core import path (it lives
    behind the lazy ``robotics_experiments.runtime`` package ``__init__``).
    """

    def __init__(self, hosts: list[BridgeHost]):
        self._hosts = list(hosts)
        self._physx_sub = None

    @property
    def hosts(self) -> list[BridgeHost]:
        return self._hosts

    def start(self) -> None:
        """Start every host, then subscribe once to physics steps."""
        for host in self._hosts:
            host.start()
        physx = omni.physx.get_physx_interface()
        self._physx_sub = physx.subscribe_physics_step_events(self._on_physics_step)

    def stop(self) -> None:
        """Drop the subscription and stop every host."""
        self._physx_sub = None
        for host in self._hosts:
            host.stop()

    def _on_physics_step(self, dt: float) -> None:
        for host in self._hosts:
            host.send_phase(dt)
        for host in self._hosts:
            host.recv_phase()

"""Live control driver: run a high-level flight routine from the extension.

This is the runtime counterpart of the standalone takeoff scenario's blocking
driver. It reuses the same pure command core (``robotics_experiments.commanding``)
and the same ``MavlinkCommandPort`` over each drone's derived ``gcs_port``, but
re-expressed as a small **non-blocking, tick-driven phase machine** so it can be
driven from the live Isaac extension's event loop instead of a straight-line
script.

Separation of concerns (load-bearing): this is a *control* layer. It steps off
the Kit **app-update** event stream -- a tick separate from the bridge's
physics-step subscription -- and it talks only the GCS MAVLink link
(``vehicle.gcs_port``), never the bridge's simulator MAVLink port or its
sensor/actuator lockstep. The bridge contract forbids the bridge touching
``gcs_port``; symmetrically this driver never touches the bridge's send/recv legs.

The routine itself is config-driven: ``EnvConfig.routine.steps`` is the single
source of truth (non-empty -> the extension builds this driver; empty -> manual,
no driver). On a ``FAIL`` decision, a PX4 failsafe, or a phase timeout the driver
logs a warning and stops the routine -- it never raises into the sim (runtime is
recoverable, unlike the integration test which asserts). After the routine
completes the driver idles (keeps the GCS link warm) so PX4-console control
returns to the user.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import carb

from robotics_experiments.commanding import (
    DecisionKind,
    ExecutionPolicy,
    FleetCommander,
    MavlinkCommandPort,
    RoutineObservation,
    VehicleCommander,
    build_vehicle_routine,
)

from .bridge_host import BridgeHost

__all__ = ["FleetControlDriver"]

_LOG_PREFIX = "[DroneSpawner.control]"


class _Phase(Enum):
    """Stages of the tick-driven routine flow."""

    WAIT_BRIDGE = auto()
    CONNECT_GCS = auto()
    HANDSHAKE = auto()
    PREARM_SETTLE = auto()
    FETCH_TARGET = auto()
    BUILD = auto()
    DRIVE = auto()
    DONE = auto()


@dataclass
class _DroneControl:
    """Per-drone live control state (the control-layer analogue of a BridgeHost)."""

    vehicle: Any
    host: BridgeHost
    articulation_path: str
    label: str
    gcs_endpoint: str
    port: Optional[MavlinkCommandPort] = None
    observer: Any = None
    commander: Optional[VehicleCommander] = None
    target: Optional[tuple[float, float, float]] = field(default=None)


class FleetControlDriver:
    """Drives a config routine across a fleet from the Kit app-update stream.

    Construct from the px4-enabled vehicles' :class:`BridgeHost`s and the
    ``routine`` config, then :meth:`start` to subscribe to app-update events and
    :meth:`stop` on teardown. All meaningful behavior is the per-tick phase
    machine in :meth:`_on_update`.
    """

    def __init__(self, members: list[_DroneControl], *, routine_cfg: Any) -> None:
        self._members = members
        self._cfg = routine_cfg
        self._phase = _Phase.WAIT_BRIDGE
        self._update_sub: Any = None
        self._fleet: Optional[FleetCommander] = None
        # Wall-clock backstop for the active phase (sim-time gates are primary;
        # this catches a fully stalled sim). Set on each phase entry.
        self._phase_deadline_s = 0.0
        # Sim-time target for the prearm settle (set on entering PREARM_SETTLE).
        self._settle_target_us = 0

    @classmethod
    def from_vehicles(
        cls, entries: list[tuple[Any, BridgeHost, str]], *, routine_cfg: Any
    ) -> "FleetControlDriver":
        """Build a driver from ``(vehicle, host, articulation_path)`` entries.

        Only px4-enabled vehicles (those with a real bridge host) are passed in;
        each becomes one :class:`_DroneControl` talking its derived ``gcs_port``.
        """
        members = [
            _DroneControl(
                vehicle=vehicle,
                host=host,
                articulation_path=articulation_path,
                label=f"drone{vehicle.id}",
                gcs_endpoint=f"udp:127.0.0.1:{vehicle.gcs_port}",
            )
            for vehicle, host, articulation_path in entries
        ]
        return cls(members, routine_cfg=routine_cfg)

    def start(self) -> None:
        """Subscribe to the app-update stream and begin the routine flow.

        A no-op (with a warning) when there are no px4-enabled vehicles to fly.
        """
        if not self._members:
            carb.log_warn(
                f"{_LOG_PREFIX} routine.steps is non-empty but no px4-enabled vehicle exists; "
                "nothing to fly"
            )
            return
        import omni.kit.app

        labels = ", ".join(m.label for m in self._members)
        carb.log_info(
            f"{_LOG_PREFIX} starting routine for {len(self._members)} vehicle(s): {labels}"
        )
        self._enter(_Phase.WAIT_BRIDGE)
        app = omni.kit.app.get_app()
        self._update_sub = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="drone_spawner.control_driver"
        )

    def stop(self) -> None:
        """Drop the subscription, disconnect every GCS link, and shut down observers."""
        self._update_sub = None
        self._fleet = None
        for member in self._members:
            if member.port is not None:
                member.port.disconnect()
                member.port = None
            if member.observer is not None:
                try:
                    member.observer.shutdown()
                except Exception as exc:  # noqa: BLE001 - best-effort teardown
                    carb.log_warn(f"{_LOG_PREFIX} [{member.label}] observer shutdown failed: {exc}")
                member.observer = None

    # ---- phase machine -----------------------------------------------------

    def _enter(self, phase: _Phase) -> None:
        """Transition to ``phase`` and arm its wall-clock backstop."""
        self._phase = phase
        n = max(1, len(self._members))
        if phase is _Phase.WAIT_BRIDGE:
            self._phase_deadline_s = time.monotonic() + self._cfg.bridge_heartbeat_timeout_s * n
        elif phase is _Phase.HANDSHAKE:
            self._phase_deadline_s = time.monotonic() + self._cfg.handshake_timeout_s * n
        elif phase is _Phase.PREARM_SETTLE:
            self._phase_deadline_s = (
                time.monotonic() + max(20.0, self._cfg.prearm_settle_s * 5.0) * n
            )
        elif phase is _Phase.FETCH_TARGET:
            self._phase_deadline_s = time.monotonic() + self._cfg.takeoff_target_timeout_s * n
        elif phase is _Phase.DRIVE:
            self._phase_deadline_s = (
                time.monotonic()
                + (self._cfg.command_ack_timeout_s + self._cfg.handshake_timeout_s) * 4.0 * n
            )
        else:
            self._phase_deadline_s = 0.0

    def _abort(self, message: str) -> None:
        """Log a warning and stop the routine without raising into the sim."""
        carb.log_warn(
            f"{_LOG_PREFIX} {message}; stopping routine (manual control remains available)"
        )
        self._enter(_Phase.DONE)

    def _on_update(self, event: Any) -> None:
        """One control tick: advance the phase machine. Never raises into Kit."""
        try:
            self._tick()
        except Exception as exc:  # noqa: BLE001 - a control bug must not spam every frame
            carb.log_error(f"{_LOG_PREFIX} control tick failed: {exc}")
            self._enter(_Phase.DONE)

    def _tick(self) -> None:
        if self._phase is _Phase.DONE:
            # Keep the GCS link warm so PX4 still sees a GCS; do not command.
            self._pump_only()
            return
        if self._phase_deadline_s and time.monotonic() > self._phase_deadline_s:
            self._abort(f"phase {self._phase.name} timed out")
            return

        if self._phase is _Phase.WAIT_BRIDGE:
            self._tick_wait_bridge()
        elif self._phase is _Phase.CONNECT_GCS:
            self._tick_connect_gcs()
        elif self._phase is _Phase.HANDSHAKE:
            self._tick_handshake()
        elif self._phase is _Phase.PREARM_SETTLE:
            self._tick_prearm_settle()
        elif self._phase is _Phase.FETCH_TARGET:
            self._tick_fetch_target()
        elif self._phase is _Phase.BUILD:
            self._tick_build()
        elif self._phase is _Phase.DRIVE:
            self._tick_drive()

    def _tick_wait_bridge(self) -> None:
        if all(m.host.is_connected for m in self._members):
            carb.log_info(f"{_LOG_PREFIX} all bridges connected; connecting GCS links")
            self._enter(_Phase.CONNECT_GCS)

    def _tick_connect_gcs(self) -> None:
        from robotics_experiments.px4_bridge import IsaacArticulationIO

        for member in self._members:
            port = MavlinkCommandPort(
                gcs_endpoint=member.gcs_endpoint,
                heartbeat_period_s=self._cfg.gcs_heartbeat_period_s,
                label=member.label,
            )
            port.connect()
            port.pump_heartbeat(force=True)
            member.port = port
            # State-only observer (read_state samples body pose/velocity; rotor
            # paths are unused for it, so none are resolved here).
            member.observer = IsaacArticulationIO(
                articulation_path=member.articulation_path, rotor_link_paths=[]
            )
        self._enter(_Phase.HANDSHAKE)

    def _tick_handshake(self) -> None:
        self._pump_and_poll()
        if all(m.port is not None and m.port.is_handshaked for m in self._members):
            carb.log_info(f"{_LOG_PREFIX} GCS handshake complete; settling PX4 before arm")
            self._settle_target_us = self._sim_time_us() + int(
                round(self._cfg.prearm_settle_s * 1_000_000)
            )
            self._enter(_Phase.PREARM_SETTLE)

    def _tick_prearm_settle(self) -> None:
        self._pump_and_poll()
        if self._raise_or_abort_on_failsafe("while settling before arm"):
            return
        if self._sim_time_us() >= self._settle_target_us:
            carb.log_info(f"{_LOG_PREFIX} prearm settle complete; waiting for takeoff telemetry")
            self._enter(_Phase.FETCH_TARGET)

    def _tick_fetch_target(self) -> None:
        self._pump_and_poll()
        if self._raise_or_abort_on_failsafe("while waiting for altitude telemetry"):
            return
        for member in self._members:
            assert member.port is not None
            if member.target is None:
                member.target = member.port.latest_global_position
        if all(m.target is not None for m in self._members):
            self._enter(_Phase.BUILD)

    def _tick_build(self) -> None:
        commanders: list[VehicleCommander] = []
        for member in self._members:
            assert member.target is not None and member.port is not None
            latitude_deg, longitude_deg, altitude_m = member.target
            routine = build_vehicle_routine(
                self._cfg.steps,
                target_latitude_deg=latitude_deg,
                target_longitude_deg=longitude_deg,
                target_altitude_m=altitude_m,
                command_ack_timeout_s=self._cfg.command_ack_timeout_s,
            )
            member.commander = VehicleCommander(routine, member.port, label=member.label)
            commanders.append(member.commander)
        # STEP_SYNCHRONIZED reproduces arm-all-then-takeoff-all: no drone sends
        # takeoff until every drone is armed.
        self._fleet = FleetCommander(commanders, policy=ExecutionPolicy.STEP_SYNCHRONIZED)
        carb.log_info(f"{_LOG_PREFIX} driving {len(commanders)} routine(s) to completion")
        self._enter(_Phase.DRIVE)

    def _tick_drive(self) -> None:
        assert self._fleet is not None
        acks = self._pump_and_poll()
        if self._raise_or_abort_on_failsafe("during the command routine"):
            return
        sim_time_us = self._sim_time_us()
        observations = {
            m.label: RoutineObservation(
                vehicle_state=m.observer.read_state(sim_time_us=sim_time_us),
                bridge_status=m.host.status,
                last_ack=acks[m.label],
                failsafe=m.port.failsafe if m.port is not None else False,
                sim_time_us=sim_time_us,
            )
            for m in self._members
        }
        decisions = self._fleet.step(observations)
        for label, decision in decisions.items():
            if decision.kind is DecisionKind.FAIL:
                self._abort(f"[{label}] routine failed: {decision.reason}")
                return
        if self._fleet.is_done:
            carb.log_info(
                f"{_LOG_PREFIX} routine complete; holding hover, manual control available"
            )
            self._enter(_Phase.DONE)

    # ---- shared per-tick helpers ------------------------------------------

    def _pump_and_poll(self) -> dict[str, Any]:
        """Pump every GCS heartbeat and drain feedback; return latest acks per drone."""
        acks: dict[str, Any] = {}
        for member in self._members:
            assert member.port is not None
            member.port.pump_heartbeat()
            acks[member.label] = member.port.poll_feedback()
        return acks

    def _pump_only(self) -> None:
        """Keep links alive without collecting acks (used in the idle DONE phase)."""
        for member in self._members:
            if member.port is not None:
                member.port.pump_heartbeat()
                member.port.poll_feedback()

    def _raise_or_abort_on_failsafe(self, phase: str) -> bool:
        """Abort the routine on any drone's failsafe; return whether it aborted."""
        for member in self._members:
            if member.port is not None and member.port.failsafe:
                self._abort(
                    f"[{member.label}] PX4 entered failsafe {phase}: "
                    f"system_status={member.port.failsafe_system_status}"
                )
                return True
        return False

    def _sim_time_us(self) -> int:
        # All bridges share the same PhysX steps, so any host's clock is the
        # fleet clock; use the first as the reference.
        return int(self._members[0].host.sim_time_us)

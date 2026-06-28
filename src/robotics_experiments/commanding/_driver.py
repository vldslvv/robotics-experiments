"""The pure per-drone routine driver - one driver executes one drone's routine.

:class:`RoutineDriver` is a non-blocking, tick-driven phase machine that takes a
single drone's routine plus a :class:`_AbstractCommandPort` and drives it from
fresh GCS link to a completed (or stopped) routine. It knows only its own drone:
there is no fleet object and no cross-drone synchronization. Both workflows run
the *same* driver - the live extension ticks it from the Kit app-update stream,
the standalone scenario ticks it in a blocking loop - differing only in the
injected :class:`DriverEnvironment` (logging, the strict-vs-forgiving failure
reaction, and the Isaac state reads).

It is pure in the same sense as the rest of ``commanding``: it imports neither
Isaac nor pymavlink. Every side effect crosses a seam - the port
(`connect`/`pump_heartbeat`/`poll_feedback`/`send` plus the control-feedback
properties) and the :class:`DriverEnvironment` (bridge connectivity, sim time,
observation assembly, logging, and the failure/step callbacks). The wall-clock
backstop uses an injectable ``monotonic`` so the machine is unit-testable with a
fake clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Protocol

from ._commander import _VehicleCommander
from ._observation import CommandAck, RoutineObservation
from ._port import _AbstractCommandPort
from ._routine import (
    _build_vehicle_routine,
    _DecisionKind,
    _EnvelopeGates,
    _VehicleRoutine,
)

__all__ = ["DriverEnvironment", "DriverParams", "_DriverPhase", "RoutineDriver"]


class _DriverPhase(Enum):
    """Stages of the per-drone routine flow."""

    WAIT_BRIDGE = auto()
    CONNECT = auto()
    HANDSHAKE = auto()
    PREARM_SETTLE = auto()
    FETCH_TARGET = auto()
    BUILD = auto()
    DRIVE = auto()
    DONE = auto()


@dataclass(frozen=True)
class DriverParams:
    """The per-drone routine knobs the driver needs (a subset of ``RoutineConfig``).

    Built by the Isaac-side adapter from a (per-vehicle) ``RoutineConfig`` so
    ``commanding`` never imports ``config``. ``steps`` and ``gates`` plus the
    _takeoff target are mapped to a :class:`_VehicleRoutine` at the ``BUILD`` phase
    via :func:`_build_vehicle_routine`; the timeouts _arm the per-phase wall-clock
    backstops and the step budgets.
    """

    steps: Sequence[Mapping[str, Any]]
    gates: _EnvelopeGates
    command_ack_timeout_s: float
    bridge_heartbeat_timeout_s: float
    handshake_timeout_s: float
    prearm_settle_s: float
    takeoff_target_timeout_s: float
    takeoff_timeout_s: float
    hover_settle_s: float

    @classmethod
    def from_routine_config(cls, rc: Any) -> "DriverParams":
        """Build from a ``RoutineConfig``-shaped object (duck-typed, no config import).

        Reads the routine config's fields by name so the pure ``commanding`` layer
        never imports ``config``; both Isaac-side adapters call this with a
        per-vehicle ``ResolvedVehicle.routine``.
        """
        return cls(
            steps=rc.steps,
            gates=_EnvelopeGates(
                max_tilt_deg=rc.max_tilt_deg,
                max_lateral_drift_m=rc.max_lateral_drift_m,
                max_vertical_speed_m_s=rc.max_vertical_speed_m_s,
                max_downward_speed_m_s=rc.max_downward_speed_m_s,
                max_angular_speed_rad_s=rc.max_angular_speed_rad_s,
                hover_max_abs_vertical_speed_m_s=rc.hover_max_abs_vertical_speed_m_s,
                hover_max_abs_horizontal_speed_m_s=rc.hover_max_abs_horizontal_speed_m_s,
            ),
            command_ack_timeout_s=rc.command_ack_timeout_s,
            bridge_heartbeat_timeout_s=rc.bridge_heartbeat_timeout_s,
            handshake_timeout_s=rc.handshake_timeout_s,
            prearm_settle_s=rc.prearm_settle_s,
            takeoff_target_timeout_s=rc.takeoff_target_timeout_s,
            takeoff_timeout_s=rc.takeoff_timeout_s,
            hover_settle_s=rc.hover_settle_s,
        )


class DriverEnvironment(Protocol):
    """The injected seam for everything the pure driver cannot do itself.

    The Isaac-side adapter implements this over the live drone (a
    ``BridgeHost`` for connectivity, an ``IsaacArticulationIO`` for state, the
    bridge clock for sim time) and chooses the failure reaction:
    ``on_failure`` *raises* in the strict integration test and *logs* (then the
    driver goes to ``DONE``) in a forgiving live/regular run.
    """

    def is_bridge_connected(self) -> bool: ...

    def sim_time_us(self) -> int: ...

    def read_observation(
        self, *, sim_time_us: int, last_ack: Optional[CommandAck]
    ) -> RoutineObservation: ...

    def log(self, message: str) -> None: ...

    def on_failure(self, kind: Any, message: str) -> None:
        """React to a terminal failure. ``kind`` is a :class:`FailureKind` or one
        of the strings ``"failsafe"`` / ``"timeout"``. Strict implementations
        raise; forgiving ones log and return (the driver then idles)."""
        ...

    def on_step_advanced(self, step_label: str) -> None:
        """Called once per completed routine step (the label of the step that just
        finished). Used by the negative test to inject a post-_takeoff fault; a
        no-op for normal runs."""
        ...


class RoutineDriver:
    """Drives one drone's routine across one port via a tick-driven phase machine.

    Construct with the per-drone :class:`DriverParams`, an
    :class:`_AbstractCommandPort` (not yet connected; the driver connects it), and
    a :class:`DriverEnvironment`. Call :meth:`tick` once per control frame; the
    adapter owns the shared sim/tick cadence (a blocking ``simulation_app.update()``
    loop or the Kit app-update stream) and never the bridge's physics legs.
    """

    def __init__(
        self,
        params: DriverParams,
        port: _AbstractCommandPort,
        env: DriverEnvironment,
        *,
        label: str = "",
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._params = params
        self._port = port
        self._env = env
        self._label = label
        self._monotonic = monotonic
        self._phase = _DriverPhase.WAIT_BRIDGE
        self._deadline_s = 0.0
        self._settle_target_us = 0
        self._target: Optional[tuple[float, float, float]] = None
        self._routine: Optional[_VehicleRoutine] = None
        self._commander: Optional[_VehicleCommander] = None
        self._prev_cursor = 0
        self._enter(_DriverPhase.WAIT_BRIDGE)

    @property
    def phase(self) -> _DriverPhase:
        return self._phase

    @property
    def is_done(self) -> bool:
        return self._phase is _DriverPhase.DONE

    def tick(self) -> None:
        """Advance the phase machine one control frame. Never advances physics."""
        if self._phase is _DriverPhase.DONE:
            self._pump_only()
            return
        if self._deadline_s and self._monotonic() > self._deadline_s:
            self._fail("timeout", f"phase {self._phase.name} timed out")
            return
        if self._phase is _DriverPhase.WAIT_BRIDGE:
            self._tick_wait_bridge()
        elif self._phase is _DriverPhase.CONNECT:
            self._tick_connect()
        elif self._phase is _DriverPhase.HANDSHAKE:
            self._tick_handshake()
        elif self._phase is _DriverPhase.PREARM_SETTLE:
            self._tick_prearm_settle()
        elif self._phase is _DriverPhase.FETCH_TARGET:
            self._tick_fetch_target()
        elif self._phase is _DriverPhase.BUILD:
            self._tick_build()
        elif self._phase is _DriverPhase.DRIVE:
            self._tick_drive()

    # ---- phases ------------------------------------------------------------

    def _tick_wait_bridge(self) -> None:
        if self._env.is_bridge_connected():
            self._env.log(f"[{self._label}] bridge connected; connecting GCS link")
            self._enter(_DriverPhase.CONNECT)

    def _tick_connect(self) -> None:
        self._port.connect()
        self._port.pump_heartbeat(force=True)
        self._enter(_DriverPhase.HANDSHAKE)

    def _tick_handshake(self) -> None:
        self._pump_and_poll()
        if self._port.is_handshaked:
            self._env.log(f"[{self._label}] GCS handshake complete; settling PX4 before arm")
            self._settle_target_us = self._env.sim_time_us() + int(
                round(self._params.prearm_settle_s * 1_000_000)
            )
            self._enter(_DriverPhase.PREARM_SETTLE)

    def _tick_prearm_settle(self) -> None:
        self._pump_and_poll()
        if self._abort_on_failsafe("while settling before arm"):
            return
        if self._env.sim_time_us() >= self._settle_target_us:
            self._env.log(f"[{self._label}] prearm settle complete; waiting for takeoff telemetry")
            self._enter(_DriverPhase.FETCH_TARGET)

    def _tick_fetch_target(self) -> None:
        self._pump_and_poll()
        if self._abort_on_failsafe("while waiting for altitude telemetry"):
            return
        self._target = self._port.latest_global_position
        if self._target is not None:
            self._enter(_DriverPhase.BUILD)

    def _tick_build(self) -> None:
        assert self._target is not None
        latitude_deg, longitude_deg, altitude_m = self._target
        self._routine = _build_vehicle_routine(
            self._params.steps,
            target_latitude_deg=latitude_deg,
            target_longitude_deg=longitude_deg,
            target_altitude_m=altitude_m,
            command_ack_timeout_s=self._params.command_ack_timeout_s,
            gates=self._params.gates,
            takeoff_timeout_s=self._params.takeoff_timeout_s,
            hover_settle_s=self._params.hover_settle_s,
        )
        self._commander = _VehicleCommander(self._routine, self._port, label=self._label)
        self._prev_cursor = 0
        self._env.log(f"[{self._label}] driving routine ({len(self._routine.steps)} step(s))")
        self._enter(_DriverPhase.DRIVE)

    def _tick_drive(self) -> None:
        assert self._commander is not None and self._routine is not None
        last_ack = self._pump_and_poll()
        if self._abort_on_failsafe("during the routine"):
            return
        sim_time_us = self._env.sim_time_us()
        observation = self._env.read_observation(sim_time_us=sim_time_us, last_ack=last_ack)
        decision = self._commander.step(observation)
        if decision.kind is _DecisionKind.FAIL:
            self._fail(decision.failure_kind, f"[{self._label}] routine failed: {decision.reason}")
            return
        cursor = self._commander.current_step_index
        while self._prev_cursor < cursor:
            self._env.on_step_advanced(self._routine.steps[self._prev_cursor].label)
            self._prev_cursor += 1
        if self._commander.is_done:
            self._env.log(f"[{self._label}] routine complete; holding, manual control available")
            self._enter(_DriverPhase.DONE)

    # ---- helpers -----------------------------------------------------------

    def _enter(self, phase: _DriverPhase) -> None:
        """Transition to ``phase`` and _arm its wall-clock backstop."""
        self._phase = phase
        params = self._params
        if phase is _DriverPhase.WAIT_BRIDGE:
            self._deadline_s = self._monotonic() + params.bridge_heartbeat_timeout_s
        elif phase is _DriverPhase.HANDSHAKE:
            self._deadline_s = self._monotonic() + params.handshake_timeout_s
        elif phase is _DriverPhase.PREARM_SETTLE:
            self._deadline_s = self._monotonic() + max(20.0, params.prearm_settle_s * 5.0)
        elif phase is _DriverPhase.FETCH_TARGET:
            self._deadline_s = self._monotonic() + params.takeoff_target_timeout_s
        elif phase is _DriverPhase.DRIVE:
            self._deadline_s = (
                self._monotonic()
                + (
                    params.takeoff_timeout_s
                    + params.command_ack_timeout_s
                    + params.handshake_timeout_s
                )
                * 4.0
            )
        else:
            self._deadline_s = 0.0

    def _fail(self, kind: Any, message: str) -> None:
        """Report a terminal failure (strict raises, forgiving logs) then idle."""
        self._env.on_failure(kind, message)
        self._enter(_DriverPhase.DONE)

    def _abort_on_failsafe(self, phase: str) -> bool:
        """Fail the routine on a PX4 failsafe; return whether it did."""
        if self._port.failsafe:
            self._fail(
                "failsafe",
                f"[{self._label}] PX4 entered failsafe {phase}: "
                f"system_status={self._port.failsafe_system_status}",
            )
            return True
        return False

    def _pump_and_poll(self) -> Optional[CommandAck]:
        self._port.pump_heartbeat()
        return self._port.poll_feedback()

    def _pump_only(self) -> None:
        """Keep the GCS link warm in DONE without collecting acks."""
        self._port.pump_heartbeat()
        self._port.poll_feedback()

"""The pure per-drone routine driver - one driver executes one drone's routine.

:class:`RoutineDriver` is a non-blocking, tick-driven phase machine that takes a
single drone's routine plus a :class:`MavlinkCommandPort` and drives it from
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

from robotics_experiments.commanding._commander import _VehicleCommander
from robotics_experiments.commanding._routine_builder import build_vehicle_routine
from robotics_experiments.commanding._routine_model import (
    _DEFAULT_MIN_HEADING_DIST_M,
    DecisionKind,
    FailureKind,
    PolicyStepFn,
    VehicleRoutine,
)
from robotics_experiments.commanding._routine_predicates import EnvelopeGates
from robotics_experiments.commanding.observation import CommandAck, RoutineObservation
from robotics_experiments.commanding.port import MavlinkCommandPort
from robotics_experiments.px4_bridge import (
    DEFAULT_HOME_ALT_M,
    DEFAULT_HOME_LAT_DEG,
    DEFAULT_HOME_LON_DEG,
)

__all__ = ["DriverEnvironment", "DriverParams", "DriverPhase", "RoutineDriver"]


class DriverPhase(Enum):
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

    Built by the Isaac-side adapter from a (per-vehicle) ``RoutineConfig`` plus
    the global ``DriverConfig``-shaped execution heuristics, so ``commanding``
    never imports ``config``. ``steps`` and ``gates`` plus the takeoff target
    are mapped to a :class:`VehicleRoutine` at the ``BUILD`` phase via
    :func:`build_vehicle_routine`; the timeouts arm the per-phase wall-clock
    backstops and the step budgets.
    """

    steps: Sequence[Mapping[str, Any]]
    gates: EnvelopeGates
    command_ack_timeout_s: float
    bridge_heartbeat_timeout_s: float
    handshake_timeout_s: float
    prearm_settle_s: float
    takeoff_target_timeout_s: float
    takeoff_timeout_s: float
    hover_settle_s: float
    goto_position_tol_m: float
    goto_heading_tol_deg: float
    goto_cross_track_m: float
    goto_offboard_prime_ticks: int
    # The home geographic anchor (from the ``px4`` group) the ``goto`` builder
    # converts world-ENU waypoints against; defaulted to the bridge's frame
    # default so direct constructions and non-navigating routines need not set it.
    home_lat_deg: float = DEFAULT_HOME_LAT_DEG
    home_lon_deg: float = DEFAULT_HOME_LON_DEG
    home_alt_m: float = DEFAULT_HOME_ALT_M
    # Execution heuristics (from the Hydra ``driver`` group): the goto heading
    # epsilon and the wall-clock backstops; library-neutral fallbacks so direct
    # constructions need not set them.
    goto_min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M
    prearm_backstop_floor_s: float = 20.0
    prearm_backstop_multiplier: float = 5.0
    drive_budget_multiplier: float = 4.0
    # The injected artifact-loading seam for ``policy`` steps (built by the
    # runtime from ``robotics_experiments.rl.build_policy_step_fn``); ``None`` means a
    # routine with a policy step fails loudly at BUILD.
    policy_fn_factory: Optional[Callable[[Mapping[str, Any]], PolicyStepFn]] = None

    @classmethod
    def from_routine_config(
        cls,
        rc: Any,
        *,
        driver: Any,
        home_lat_deg: float = DEFAULT_HOME_LAT_DEG,
        home_lon_deg: float = DEFAULT_HOME_LON_DEG,
        home_alt_m: float = DEFAULT_HOME_ALT_M,
        policy_fn_factory: Optional[Callable[[Mapping[str, Any]], PolicyStepFn]] = None,
    ) -> "DriverParams":
        """Build from ``RoutineConfig``- and ``DriverConfig``-shaped objects.

        Both are duck-typed (fields read by name) so the pure ``commanding``
        layer never imports ``config``; both Isaac-side adapters call this with
        a per-vehicle ``ResolvedVehicle.routine`` and the global ``driver``
        execution-heuristics group. The home anchor comes from the per-vehicle
        ``px4`` group (passed by the orchestrator), not the routine.
        """
        return cls(
            steps=rc.steps,
            gates=EnvelopeGates(
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
            goto_position_tol_m=rc.goto_position_tol_m,
            goto_heading_tol_deg=rc.goto_heading_tol_deg,
            goto_cross_track_m=rc.goto_cross_track_m,
            goto_offboard_prime_ticks=rc.goto_offboard_prime_ticks,
            home_lat_deg=home_lat_deg,
            home_lon_deg=home_lon_deg,
            home_alt_m=home_alt_m,
            goto_min_heading_dist_m=driver.goto_min_heading_dist_m,
            prearm_backstop_floor_s=driver.prearm_backstop_floor_s,
            prearm_backstop_multiplier=driver.prearm_backstop_multiplier,
            drive_budget_multiplier=driver.drive_budget_multiplier,
            policy_fn_factory=policy_fn_factory,
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

    def on_failure(self, kind: FailureKind, message: str) -> None:
        """React to a terminal failure of the given :class:`FailureKind`. Strict
        implementations raise; forgiving ones log and return (the driver then
        idles)."""
        ...

    def on_step_advanced(self, step_label: str) -> None:
        """Called once per completed routine step (the label of the step that just
        finished). Used by the negative test to inject a post-takeoff fault; a
        no-op for normal runs."""
        ...


class RoutineDriver:
    """Drives one drone's routine across one port via a tick-driven phase machine.

    Construct with the per-drone :class:`DriverParams`, an
    :class:`MavlinkCommandPort` (not yet connected; the driver connects it), and
    a :class:`DriverEnvironment`. Call :meth:`tick` once per control frame; the
    adapter owns the shared sim/tick cadence (a blocking ``simulation_app.update()``
    loop or the Kit app-update stream) and never the bridge's physics legs.
    """

    def __init__(
        self,
        params: DriverParams,
        port: MavlinkCommandPort,
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
        self._phase = DriverPhase.WAIT_BRIDGE
        self._deadline_s = 0.0
        self._settle_target_us = 0
        self._target: Optional[tuple[float, float, float]] = None
        self._routine: Optional[VehicleRoutine] = None
        self._commander: Optional[_VehicleCommander] = None
        self._prev_cursor = 0
        self._enter(DriverPhase.WAIT_BRIDGE)

    @property
    def phase(self) -> DriverPhase:
        return self._phase

    @property
    def is_done(self) -> bool:
        return self._phase is DriverPhase.DONE

    def tick(self) -> None:
        """Advance the phase machine one control frame. Never advances physics."""
        if self._phase is DriverPhase.DONE:
            # Keep the GCS link warm in DONE (heartbeat + drain) without acting on acks.
            self._pump_and_poll()
            return
        if self._deadline_s and self._monotonic() > self._deadline_s:
            self._fail(FailureKind.TIMEOUT, f"phase {self._phase.name} timed out")
            return
        if self._phase is DriverPhase.WAIT_BRIDGE:
            self._tick_wait_bridge()
        elif self._phase is DriverPhase.CONNECT:
            self._tick_connect()
        elif self._phase is DriverPhase.HANDSHAKE:
            self._tick_handshake()
        elif self._phase is DriverPhase.PREARM_SETTLE:
            self._tick_prearm_settle()
        elif self._phase is DriverPhase.FETCH_TARGET:
            self._tick_fetch_target()
        elif self._phase is DriverPhase.BUILD:
            self._tick_build()
        elif self._phase is DriverPhase.DRIVE:
            self._tick_drive()

    # ---- phases ------------------------------------------------------------

    def _tick_wait_bridge(self) -> None:
        if self._env.is_bridge_connected():
            self._env.log(f"[{self._label}] bridge connected; connecting GCS link")
            self._enter(DriverPhase.CONNECT)

    def _tick_connect(self) -> None:
        self._port.connect()
        self._port.pump_heartbeat(force=True)
        self._enter(DriverPhase.HANDSHAKE)

    def _tick_handshake(self) -> None:
        self._pump_and_poll()
        if self._port.is_handshaked:
            self._env.log(f"[{self._label}] GCS handshake complete; settling PX4 before arm")
            self._settle_target_us = self._env.sim_time_us() + int(
                round(self._params.prearm_settle_s * 1_000_000)
            )
            self._enter(DriverPhase.PREARM_SETTLE)

    def _tick_prearm_settle(self) -> None:
        self._pump_and_poll()
        if self._abort_on_failsafe("while settling before arm"):
            return
        if self._env.sim_time_us() >= self._settle_target_us:
            self._env.log(f"[{self._label}] prearm settle complete; waiting for takeoff telemetry")
            self._enter(DriverPhase.FETCH_TARGET)

    def _tick_fetch_target(self) -> None:
        self._pump_and_poll()
        if self._abort_on_failsafe("while waiting for altitude telemetry"):
            return
        self._target = self._port.latest_global_position
        if self._target is not None:
            self._enter(DriverPhase.BUILD)

    def _tick_build(self) -> None:
        assert self._target is not None
        latitude_deg, longitude_deg, altitude_m = self._target
        try:
            self._routine = self._build_routine(latitude_deg, longitude_deg, altitude_m)
        except ValueError as exc:
            # A bad routine config (unknown kind, missing field, unloadable
            # policy artifact) is a terminal command failure, routed through
            # on_failure so the strict/forgiving reaction applies.
            self._fail(FailureKind.COMMAND, f"[{self._label}] routine build failed: {exc}")
            return
        self._commander = _VehicleCommander(self._routine, self._port)
        self._prev_cursor = 0
        self._env.log(f"[{self._label}] driving routine ({len(self._routine.steps)} step(s))")
        self._enter(DriverPhase.DRIVE)

    def _build_routine(
        self, latitude_deg: float, longitude_deg: float, altitude_m: float
    ) -> VehicleRoutine:
        return build_vehicle_routine(
            self._params.steps,
            target_latitude_deg=latitude_deg,
            target_longitude_deg=longitude_deg,
            target_altitude_m=altitude_m,
            command_ack_timeout_s=self._params.command_ack_timeout_s,
            gates=self._params.gates,
            takeoff_timeout_s=self._params.takeoff_timeout_s,
            hover_settle_s=self._params.hover_settle_s,
            home_lat_deg=self._params.home_lat_deg,
            home_lon_deg=self._params.home_lon_deg,
            home_alt_m=self._params.home_alt_m,
            goto_position_tol_m=self._params.goto_position_tol_m,
            goto_heading_tol_deg=self._params.goto_heading_tol_deg,
            goto_cross_track_m=self._params.goto_cross_track_m,
            goto_offboard_prime_ticks=self._params.goto_offboard_prime_ticks,
            goto_min_heading_dist_m=self._params.goto_min_heading_dist_m,
            policy_fn_factory=self._params.policy_fn_factory,
        )

    def _tick_drive(self) -> None:
        assert self._commander is not None and self._routine is not None
        last_ack = self._pump_and_poll()
        if self._abort_on_failsafe("during the routine"):
            return
        sim_time_us = self._env.sim_time_us()
        observation = self._env.read_observation(sim_time_us=sim_time_us, last_ack=last_ack)
        decision = self._commander.step(observation)
        if decision.kind is DecisionKind.FAIL:
            assert decision.failure_kind is not None  # always set on a FAIL decision
            self._fail(decision.failure_kind, f"[{self._label}] routine failed: {decision.reason}")
            return
        cursor = self._commander.current_step_index
        while self._prev_cursor < cursor:
            self._env.on_step_advanced(self._routine.steps[self._prev_cursor].label)
            self._prev_cursor += 1
        if self._commander.is_done:
            self._env.log(f"[{self._label}] routine complete; holding, manual control available")
            self._enter(DriverPhase.DONE)

    # ---- helpers -----------------------------------------------------------

    def _enter(self, phase: DriverPhase) -> None:
        """Transition to ``phase`` and arm its wall-clock backstop."""
        self._phase = phase
        params = self._params
        if phase is DriverPhase.WAIT_BRIDGE:
            self._deadline_s = self._monotonic() + params.bridge_heartbeat_timeout_s
        elif phase is DriverPhase.HANDSHAKE:
            self._deadline_s = self._monotonic() + params.handshake_timeout_s
        elif phase is DriverPhase.PREARM_SETTLE:
            self._deadline_s = self._monotonic() + max(
                params.prearm_backstop_floor_s,
                params.prearm_settle_s * params.prearm_backstop_multiplier,
            )
        elif phase is DriverPhase.FETCH_TARGET:
            self._deadline_s = self._monotonic() + params.takeoff_target_timeout_s
        elif phase is DriverPhase.DRIVE:
            # Wall-clock backstop generous enough for the whole routine: the sum
            # of every step's sim-time budget (goto/forward legs can be long) plus
            # the handshake/ack slack, with a safety multiplier.
            routine_budget_s = (
                sum(step.timeout_s for step in self._routine.steps) if self._routine else 0.0
            )
            self._deadline_s = (
                self._monotonic()
                + (
                    routine_budget_s
                    + params.takeoff_timeout_s
                    + params.command_ack_timeout_s
                    + params.handshake_timeout_s
                )
                * params.drive_budget_multiplier
            )
        else:
            self._deadline_s = 0.0

    def _fail(self, kind: FailureKind, message: str) -> None:
        """Report a terminal failure (strict raises, forgiving logs) then idle."""
        self._env.on_failure(kind, message)
        self._enter(DriverPhase.DONE)

    def _abort_on_failsafe(self, phase: str) -> bool:
        """Fail the routine on a PX4 failsafe; return whether it did."""
        if self._port.failsafe:
            self._fail(
                FailureKind.FAILSAFE,
                f"[{self._label}] PX4 entered failsafe {phase}: "
                f"system_status={self._port.failsafe_system_status}",
            )
            return True
        return False

    def _pump_and_poll(self) -> Optional[CommandAck]:
        self._port.pump_heartbeat()
        return self._port.poll_feedback()

"""The pure command plan and its stepper - the command-layer contract centerpiece.

A :class:`VehicleRoutine` is an ordered tuple of :class:`RoutineStep`s; each step
carries one :class:`MavlinkCommand`, a pure completion predicate over a
:class:`RoutineObservation`, and a sim-time timeout. :func:`decide` is the pure
transform at the heart of the layer - the command-layer analogue of the thrust
model's ``commands_to_dynamics``: given the routine, the current
:class:`RoutineState`, and an observation, it returns the next state and a
:class:`RoutineDecision` (``SEND`` / ``WAIT`` / ``ADVANCE`` / ``DONE`` /
``FAIL``). No sockets, no pymavlink, no Isaac - it is unit-testable with plain
dataclasses, and the side-effecting :class:`VehicleCommander` (see
``commander.py``) is the only thing that turns a ``SEND`` decision into MAVLink.

Timeouts are measured in sim time (``observation.sim_time_us``), not wall clock,
so the stepper stays pure; the driver pairs this with its own wall-clock guard
to catch a fully stalled simulation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Optional

import numpy as np

from .commands import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_NAV_TAKEOFF,
    MavlinkCommand,
    arm,
    takeoff,
)
from .observation import MAV_RESULT_TEMPORARILY_REJECTED, RoutineObservation

__all__ = [
    "DecisionKind",
    "RoutineDecision",
    "RoutineState",
    "RoutineStep",
    "VehicleRoutine",
    "ack_accepted",
    "armed",
    "build_vehicle_routine",
    "climbed_at_least",
    "decide",
]


class DecisionKind(Enum):
    """The kind of a :class:`RoutineDecision`."""

    SEND = auto()
    WAIT = auto()
    ADVANCE = auto()
    DONE = auto()
    FAIL = auto()


@dataclass(frozen=True)
class RoutineDecision:
    """The pure output of :func:`decide` for one tick.

    Exactly one ``kind``; ``command`` is set only for ``SEND`` and ``reason``
    only for ``FAIL``. Build instances through the classmethods.
    """

    kind: DecisionKind
    command: Optional[MavlinkCommand] = None
    reason: Optional[str] = None

    @classmethod
    def send(cls, command: MavlinkCommand) -> "RoutineDecision":
        return cls(DecisionKind.SEND, command=command)

    @classmethod
    def wait(cls) -> "RoutineDecision":
        return cls(DecisionKind.WAIT)

    @classmethod
    def advance(cls) -> "RoutineDecision":
        return cls(DecisionKind.ADVANCE)

    @classmethod
    def done(cls) -> "RoutineDecision":
        return cls(DecisionKind.DONE)

    @classmethod
    def fail(cls, reason: str) -> "RoutineDecision":
        return cls(DecisionKind.FAIL, reason=reason)


@dataclass(frozen=True)
class RoutineStep:
    """One step of a routine: a command, its completion predicate, and a timeout.

    Attributes:
        command: The :class:`MavlinkCommand` sent when the step begins.
        is_complete: Pure predicate over a :class:`RoutineObservation`; the step
            advances when it returns ``True``.
        label: Short human-readable name used in logs and failure messages.
        timeout_s: Sim-time budget for the step; on exceed the routine fails.
        retry_on_temporarily_rejected: When ``True``, a
            ``MAV_RESULT_TEMPORARILY_REJECTED`` ack for this command re-sends it
            instead of failing (PX4 returns this transiently before it is ready
            to arm). When ``False``, any non-accepted ack fails the routine.
    """

    command: MavlinkCommand
    is_complete: Callable[[RoutineObservation], bool]
    label: str
    timeout_s: float
    retry_on_temporarily_rejected: bool = False


@dataclass(frozen=True)
class VehicleRoutine:
    """An ordered, immutable plan of :class:`RoutineStep`s for one vehicle."""

    steps: tuple[RoutineStep, ...]


@dataclass(frozen=True)
class RoutineState:
    """Immutable progress marker threaded through :func:`decide`.

    Attributes:
        cursor: Index of the active step; ``>= len(steps)`` means the routine is
            done.
        sent: Whether the active step's command has been emitted (a ``SEND``
            decision returned) and the step is now awaiting completion.
        baseline_position_enu: The vehicle position captured when the routine
            sent its first command, for climb/displacement predicates that need
            a reference. ``None`` until the first ``SEND``.
        step_started_us: Sim time the active step began (its ``SEND`` tick), used
            to enforce the step timeout. ``None`` before the step is sent.
    """

    cursor: int = 0
    sent: bool = False
    baseline_position_enu: Optional[np.ndarray] = None
    step_started_us: Optional[int] = None

    @classmethod
    def initial(cls) -> "RoutineState":
        return cls()


def decide(
    routine: VehicleRoutine,
    state: RoutineState,
    observation: RoutineObservation,
) -> tuple[RoutineState, RoutineDecision]:
    """Pure routine stepper: map (routine, state, observation) to (state, decision).

    Returns the next :class:`RoutineState` together with the
    :class:`RoutineDecision` for this tick. The decision is purely a function of
    the inputs; the only state mutation is the returned new state. Semantics:

    - cursor past the last step -> ``DONE``.
    - step not yet sent -> ``SEND`` (capture the baseline and start the timer).
    - step sent and ``is_complete`` true -> ``ADVANCE`` (cursor moves on).
    - step sent, not complete, a temporarily-rejected ack arrived, and the step
      opts into retry -> ``SEND`` again (re-send, same step).
    - step sent, not complete, a non-accepted ack arrived (and no retry) ->
      ``FAIL``.
    - step sent, not complete, past ``timeout_s`` of sim time -> ``FAIL``.
    - otherwise -> ``WAIT``.
    """
    steps = routine.steps
    if state.cursor >= len(steps):
        return state, RoutineDecision.done()

    step = steps[state.cursor]

    if not state.sent:
        baseline = state.baseline_position_enu
        if baseline is None:
            baseline = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        new_state = replace(
            state,
            sent=True,
            baseline_position_enu=baseline,
            step_started_us=int(observation.sim_time_us),
        )
        return new_state, RoutineDecision.send(step.command)

    if step.is_complete(observation):
        new_state = replace(state, cursor=state.cursor + 1, sent=False, step_started_us=None)
        return new_state, RoutineDecision.advance()

    ack = observation.last_ack
    if ack is not None and ack.command_id == step.command.command_id and not ack.accepted:
        if step.retry_on_temporarily_rejected and ack.result == MAV_RESULT_TEMPORARILY_REJECTED:
            # Re-send the same command; PX4 returned a transient rejection. Keep
            # the step timer running so a vehicle that never becomes armable
            # still fails on the step timeout below.
            return state, RoutineDecision.send(step.command)
        return state, RoutineDecision.fail(
            f"step {step.label!r} rejected: command={step.command.command_id} result={ack.result}"
        )

    step_started_us = state.step_started_us
    assert step_started_us is not None  # set on the SEND tick, before sent=True
    elapsed_us = int(observation.sim_time_us) - step_started_us
    if elapsed_us > step.timeout_s * 1_000_000:
        return state, RoutineDecision.fail(
            f"step {step.label!r} timed out after {step.timeout_s:.1f}s "
            f"(sim elapsed {elapsed_us * 1e-6:.1f}s)"
        )

    return state, RoutineDecision.wait()


# ---- Pure completion predicates over RoutineObservation --------------------


def armed() -> Callable[[RoutineObservation], bool]:
    """Predicate: PX4 currently commands nonzero rotor references (is armed)."""

    def predicate(observation: RoutineObservation) -> bool:
        return bool(observation.bridge_status.is_armed)

    return predicate


def ack_accepted(command_id: int) -> Callable[[RoutineObservation], bool]:
    """Predicate: the last ack this tick accepted ``command_id``."""

    def predicate(observation: RoutineObservation) -> bool:
        ack = observation.last_ack
        return ack is not None and ack.command_id == command_id and ack.accepted

    return predicate


def climbed_at_least(
    min_climb_m: float, *, baseline_altitude_m: float
) -> Callable[[RoutineObservation], bool]:
    """Predicate: the vehicle climbed at least ``min_climb_m`` above a baseline.

    ``baseline_altitude_m`` is the ENU-up reference (typically the vehicle's
    pre-takeoff ground altitude), captured by the caller before the step.
    """

    def predicate(observation: RoutineObservation) -> bool:
        altitude_m = float(observation.vehicle_state.position_enu[2])
        return altitude_m - baseline_altitude_m >= min_climb_m

    return predicate


# ---- Config-driven routine assembly ----------------------------------------


def build_vehicle_routine(
    steps: Sequence[Mapping[str, Any]],
    *,
    target_latitude_deg: float,
    target_longitude_deg: float,
    target_altitude_m: float,
    command_ack_timeout_s: float,
) -> VehicleRoutine:
    """Build a :class:`VehicleRoutine` from authored step descriptors and a target.

    The single pure mapping from the ``routine.steps`` config (a list of
    ``{kind: ...}`` mappings) to the immutable routine the commander executes,
    shared by every driver (the live extension today; the standalone scenario
    once the two paths merge). Each step completes on its ``COMMAND_ACK`` and
    retries on a transient PX4 ``TEMPORARILY_REJECTED``, matching the takeoff
    scenario. ``target_*`` is the per-drone takeoff target: the drone's current
    reported geographic position and altitude, to which a ``takeoff`` step adds
    its ``climb_m``.

    An empty ``steps`` yields an empty routine (the manual case never reaches
    here, but the mapping is total). Pure: no Isaac, no pymavlink, no sockets.

    Raises:
        ValueError: on an unknown ``kind`` or a ``takeoff`` step missing
            ``climb_m`` (invalid configuration must not pass silently).
    """
    built: list[RoutineStep] = []
    for index, descriptor in enumerate(steps):
        kind = descriptor.get("kind")
        if kind == "arm":
            built.append(
                RoutineStep(
                    command=arm(),
                    is_complete=ack_accepted(MAV_CMD_COMPONENT_ARM_DISARM),
                    label="arm",
                    timeout_s=command_ack_timeout_s,
                    retry_on_temporarily_rejected=True,
                )
            )
        elif kind == "takeoff":
            if "climb_m" not in descriptor:
                raise ValueError(f"routine step {index} (takeoff) is missing required 'climb_m'")
            absolute_altitude_m = float(target_altitude_m) + float(descriptor["climb_m"])
            built.append(
                RoutineStep(
                    command=takeoff(target_latitude_deg, target_longitude_deg, absolute_altitude_m),
                    is_complete=ack_accepted(MAV_CMD_NAV_TAKEOFF),
                    label="takeoff",
                    timeout_s=command_ack_timeout_s,
                    retry_on_temporarily_rejected=True,
                )
            )
        else:
            raise ValueError(f"routine step {index} has unknown kind {kind!r}")
    return VehicleRoutine(steps=tuple(built))

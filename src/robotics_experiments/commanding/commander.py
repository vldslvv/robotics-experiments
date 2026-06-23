"""Command orchestrators (mirror ``BridgeHost`` / ``BridgeHostGroup``).

:class:`VehicleCommander` wires one :class:`VehicleRoutine` to one
:class:`AbstractCommandPort`, holding the :class:`RoutineState`: it runs the pure
:func:`decide` and performs the single side effect (``port.send`` on a ``SEND``
decision), keeping the pure decision and the impure I/O cleanly separated.

:class:`FleetCommander` drives a set of per-vehicle commanders under an execution
policy: ``INDEPENDENT`` lets each routine advance freely, while
``STEP_SYNCHRONIZED`` is a barrier - no vehicle advances to step ``N+1`` until
every vehicle has completed step ``N`` - which reproduces the takeoff scenario's
arm-all-then-takeoff-all fleet behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import Enum, auto

from .observation import CommandAck, RoutineObservation
from .port import AbstractCommandPort
from .routine import DecisionKind, RoutineDecision, RoutineState, VehicleRoutine, decide

__all__ = ["ExecutionPolicy", "FleetCommander", "VehicleCommander"]


class ExecutionPolicy(Enum):
    """How a :class:`FleetCommander` advances its vehicles' routines."""

    INDEPENDENT = auto()
    STEP_SYNCHRONIZED = auto()


class VehicleCommander:
    """Drives one routine over one port; pure decision, isolated side effect."""

    def __init__(
        self, routine: VehicleRoutine, port: AbstractCommandPort, *, label: str = ""
    ) -> None:
        self._routine = routine
        self._port = port
        self._label = label
        self._state = RoutineState.initial()
        # Latched accepted ack for the current step (see _effective_observation).
        self._accepted_ack: CommandAck | None = None

    @property
    def label(self) -> str:
        return self._label

    @property
    def is_done(self) -> bool:
        return self._state.cursor >= len(self._routine.steps)

    @property
    def current_step_index(self) -> int:
        return self._state.cursor

    def _effective_observation(self, observation: RoutineObservation) -> RoutineObservation:
        """Apply the accepted-ack latch so completion is level-triggered.

        A ``COMMAND_ACK`` is present in ``observation.last_ack`` only on the
        single tick it is drained, but a step's completion (and the fleet
        barrier) must stay satisfied across later ticks - otherwise a vehicle
        held at a barrier would revert to not-complete once its ack tick passes,
        deadlocking the barrier. So latch an *accepted* ack for the current
        step's command and re-present it on subsequent ack-less ticks.
        Rejections are deliberately not latched, keeping the retry path
        edge-triggered (one re-send per received rejection).
        """
        if not self.is_done and self._state.sent:
            step = self._routine.steps[self._state.cursor]
            ack = observation.last_ack
            if (
                ack is not None
                and ack.accepted
                and step.command is not None
                and ack.command_id == step.command.command_id
            ):
                self._accepted_ack = ack
        if observation.last_ack is None and self._accepted_ack is not None:
            return replace(observation, last_ack=self._accepted_ack)
        return observation

    def is_current_step_complete(self, observation: RoutineObservation) -> bool:
        """Whether the active step is ready to advance (barrier readiness).

        Used by :class:`FleetCommander` to evaluate the barrier without
        advancing (and without re-sending). A not-yet-sent step is never
        "complete"; a step with a ``hold_s`` window is ready only once its
        completion predicate has held for that window (read from the latched
        :class:`RoutineState`). Consults the accepted-ack latch so the answer is
        stable across ticks.
        """
        if self.is_done:
            return True
        if not self._state.sent:
            return False
        observation = self._effective_observation(observation)
        step = self._routine.steps[self._state.cursor]
        if not step.is_complete(observation, self._state.baseline_position_enu):
            return False
        if step.hold_s <= 0.0:
            return True
        hold_since_us = self._state.hold_since_us
        if hold_since_us is None:
            return False
        return int(observation.sim_time_us) - hold_since_us >= step.hold_s * 1_000_000

    def step(
        self, observation: RoutineObservation, *, allow_advance: bool = True
    ) -> RoutineDecision:
        """Run one tick: decide, optionally commit, and perform the SEND side effect.

        When ``allow_advance`` is ``False`` and the decision would ``ADVANCE``,
        the commander holds at the completed step (returns ``WAIT`` without
        committing), so a :class:`FleetCommander` barrier can keep a fast vehicle
        waiting for a slow one. Initial sends are never gated.
        """
        observation = self._effective_observation(observation)
        new_state, decision = decide(self._routine, self._state, observation)
        if decision.kind is DecisionKind.ADVANCE and not allow_advance:
            return RoutineDecision.wait()
        if decision.kind is DecisionKind.ADVANCE:
            # Entering the next step: drop the latch so its completion is judged
            # against its own command's ack.
            self._accepted_ack = None
        self._state = new_state
        if decision.kind is DecisionKind.SEND:
            assert decision.command is not None
            self._port.send(decision.command)
        return decision


class FleetCommander:
    """Drives per-vehicle commanders under an :class:`ExecutionPolicy`."""

    def __init__(self, commanders: Sequence[VehicleCommander], *, policy: ExecutionPolicy) -> None:
        labels = [c.label for c in commanders]
        if len(set(labels)) != len(labels):
            raise ValueError(f"FleetCommander requires unique commander labels, got {labels}")
        self._commanders = list(commanders)
        self._policy = policy

    @property
    def is_done(self) -> bool:
        return all(c.is_done for c in self._commanders)

    def step(self, observations: Mapping[str, RoutineObservation]) -> dict[str, RoutineDecision]:
        """Step every commander once against its observation; return per-vehicle decisions.

        Under ``STEP_SYNCHRONIZED`` no commander is allowed to advance unless
        every commander has either finished its routine or completed its current
        step, enforcing the fleet barrier.
        """
        if self._policy is ExecutionPolicy.INDEPENDENT:
            return {c.label: c.step(observations[c.label]) for c in self._commanders}

        all_ready = all(
            c.is_done or c.is_current_step_complete(observations[c.label]) for c in self._commanders
        )
        return {
            c.label: c.step(observations[c.label], allow_advance=all_ready)
            for c in self._commanders
        }

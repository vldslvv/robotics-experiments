"""The command orchestrator (mirrors ``BridgeHost``).

:class:`VehicleCommander` wires one :class:`VehicleRoutine` to one
:class:`AbstractCommandPort`, holding the :class:`RoutineState`: it runs the pure
:func:`decide` and performs the single side effect (``port.send`` on a ``SEND``
decision), keeping the pure decision and the impure I/O cleanly separated. One
commander drives one drone; the per-drone :class:`~._driver.RoutineDriver` owns the
lifecycle around it. There is no fleet object - drones are independent.
"""

from __future__ import annotations

from ._observation import RoutineObservation
from ._port import AbstractCommandPort
from ._routine import DecisionKind, RoutineDecision, RoutineState, VehicleRoutine, decide

__all__ = ["VehicleCommander"]


class VehicleCommander:
    """Drives one routine over one port; pure decision, isolated side effect."""

    def __init__(
        self, routine: VehicleRoutine, port: AbstractCommandPort, *, label: str = ""
    ) -> None:
        self._routine = routine
        self._port = port
        self._label = label
        self._state = RoutineState.initial()

    @property
    def label(self) -> str:
        return self._label

    @property
    def is_done(self) -> bool:
        return self._state.cursor >= len(self._routine.steps)

    @property
    def current_step_index(self) -> int:
        return self._state.cursor

    def step(self, observation: RoutineObservation) -> RoutineDecision:
        """Run one tick: decide, commit the new state, perform the side effect.

        The only side effects are ``SEND`` (a ``command_long``) and ``STREAM`` (an
        OFFBOARD position setpoint); every other decision is pure progress.
        """
        new_state, decision = decide(self._routine, self._state, observation)
        self._state = new_state
        if decision.kind is DecisionKind.SEND:
            assert decision.command is not None
            self._port.send(decision.command)
        elif decision.kind is DecisionKind.STREAM:
            assert decision.setpoint is not None
            self._port.send_setpoint(decision.setpoint)
        return decision

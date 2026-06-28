"""The command orchestrator (mirrors ``BridgeHost``).

:class:`_VehicleCommander` wires one :class:`_VehicleRoutine` to one
:class:`_AbstractCommandPort`, holding the :class:`_RoutineState`: it runs the pure
:func:`_decide` and performs the single side effect (``port.send`` on a ``SEND``
decision), keeping the pure decision and the impure I/O cleanly separated. One
commander drives one drone; the per-drone :class:`~.driver.RoutineDriver` owns the
lifecycle around it. There is no fleet object - drones are independent.
"""

from __future__ import annotations

from ._observation import RoutineObservation
from ._port import _AbstractCommandPort
from ._routine import _decide, _DecisionKind, _RoutineDecision, _RoutineState, _VehicleRoutine

__all__ = ["_VehicleCommander"]


class _VehicleCommander:
    """Drives one routine over one port; pure decision, isolated side effect."""

    def __init__(
        self, routine: _VehicleRoutine, port: _AbstractCommandPort, *, label: str = ""
    ) -> None:
        self._routine = routine
        self._port = port
        self._label = label
        self._state = _RoutineState.initial()

    @property
    def label(self) -> str:
        return self._label

    @property
    def is_done(self) -> bool:
        return self._state.cursor >= len(self._routine.steps)

    @property
    def current_step_index(self) -> int:
        return self._state.cursor

    def step(self, observation: RoutineObservation) -> _RoutineDecision:
        """Run one tick: _decide, commit the new state, and perform the SEND side effect."""
        new_state, decision = _decide(self._routine, self._state, observation)
        self._state = new_state
        if decision.kind is _DecisionKind.SEND:
            assert decision.command is not None
            self._port.send(decision.command)
        return decision

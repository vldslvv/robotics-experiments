"""The command orchestrator (mirrors ``BridgeHost``).

:class:`_VehicleCommander` wires one :class:`VehicleRoutine` to one
:class:`AbstractCommandPort`, holding the :class:`RoutineState`: it runs the pure
:func:`decide` and performs the single side effect (``port.send`` on a ``SEND``
decision), keeping the pure decision and the impure I/O cleanly separated. One
commander drives one drone; the per-drone :class:`~.driver.RoutineDriver` owns the
lifecycle around it. There is no fleet object - drones are independent.
"""

from __future__ import annotations

from robotics_experiments.commanding._routine_model import (
    DecisionKind,
    RoutineDecision,
    RoutineState,
    VehicleRoutine,
)
from robotics_experiments.commanding._routine_stepper import decide
from robotics_experiments.commanding.commands import (
    AttitudeRateSetpoint,
    PositionSetpoint,
    VelocitySetpoint,
)
from robotics_experiments.commanding.observation import RoutineObservation
from robotics_experiments.commanding.port import AbstractCommandPort

__all__: list[str] = []


class _VehicleCommander:
    """Drives one routine over one port; pure decision, isolated side effect."""

    def __init__(self, routine: VehicleRoutine, port: AbstractCommandPort) -> None:
        self._routine = routine
        self._port = port
        self._state = RoutineState()

    @property
    def is_done(self) -> bool:
        return self._state.cursor >= len(self._routine.steps)

    @property
    def current_step_index(self) -> int:
        return self._state.cursor

    def step(self, observation: RoutineObservation) -> RoutineDecision:
        """Run one tick: decide, commit the new state, perform the side effect.

        The only side effects are ``SEND`` (a ``command_long``) and ``STREAM``
        (an OFFBOARD setpoint, dispatched by payload type to the matching port
        method); every other decision is pure progress.
        """
        new_state, decision = decide(self._routine, self._state, observation)
        self._state = new_state
        if decision.kind is DecisionKind.SEND:
            assert decision.command is not None
            self._port.send(decision.command)
        elif decision.kind is DecisionKind.STREAM:
            setpoint = decision.setpoint
            if isinstance(setpoint, PositionSetpoint):
                self._port.send_setpoint(setpoint)
            elif isinstance(setpoint, VelocitySetpoint):
                self._port.send_velocity_setpoint(setpoint)
            elif isinstance(setpoint, AttitudeRateSetpoint):
                self._port.send_attitude_target(setpoint)
            else:
                raise TypeError(f"STREAM decision carried unsupported payload {setpoint!r}")
        return decision

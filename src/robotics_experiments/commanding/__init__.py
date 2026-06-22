"""High-level command / routine layer for driving a vehicle over its GCS link.

A domain leaf parallel to the PX4 bridge contract: a vehicle is handed a pure
:class:`VehicleRoutine` (ordered commands plus completion predicates) that the
pure :func:`decide` stepper turns into :class:`RoutineDecision`s, and a thin
side-effecting :class:`AbstractCommandPort` turns ``SEND`` decisions into MAVLink
over the resolved ``vehicle.gcs_port`` (a *control* layer, separate from the
bridge's simulator-MAVLink lockstep by design).

This package stays on the core import path: every module here imports neither
Isaac nor pymavlink at module scope (the concrete :class:`MavlinkCommandPort`
imports ``pymavlink`` only inside its methods), so ``import robotics_experiments``
succeeds with Isaac and pymavlink absent.
"""

from .commander import ExecutionPolicy, FleetCommander, VehicleCommander
from .commands import (
    MavlinkCommand,
    arm,
    land,
    reposition,
    takeoff,
)
from .observation import CommandAck, RoutineObservation
from .port import AbstractCommandPort, MavlinkCommandPort
from .routine import (
    DecisionKind,
    RoutineDecision,
    RoutineState,
    RoutineStep,
    VehicleRoutine,
    ack_accepted,
    armed,
    build_vehicle_routine,
    climbed_at_least,
    decide,
)

__all__ = [
    "AbstractCommandPort",
    "CommandAck",
    "DecisionKind",
    "ExecutionPolicy",
    "FleetCommander",
    "MavlinkCommand",
    "MavlinkCommandPort",
    "RoutineDecision",
    "RoutineObservation",
    "RoutineState",
    "RoutineStep",
    "VehicleCommander",
    "VehicleRoutine",
    "ack_accepted",
    "arm",
    "armed",
    "build_vehicle_routine",
    "climbed_at_least",
    "decide",
    "land",
    "reposition",
    "takeoff",
]

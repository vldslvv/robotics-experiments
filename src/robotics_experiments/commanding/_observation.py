"""Pure command-layer observation: the input to the routine stepper.

A :class:`RoutineObservation` is the command-layer analogue of the bridge's
:class:`VehicleState`: a frozen, dependency-free snapshot the pure ``_decide``
function reads to make a :class:`_RoutineDecision`. It reuses the bridge's
existing pure :class:`VehicleState` and :class:`BridgeStatus` and adds the
command-feedback fields (the last MAVLink ``COMMAND_ACK`` and the failsafe flag)
parsed off the GCS link by the side-effecting port.

This module imports neither pymavlink nor Isaac, so it stays on the core import
path; the ``MAV_RESULT_*`` ids are named int constants here for the same reason
the ``MAV_CMD_*`` ids are in ``_commands.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from robotics_experiments.px4_bridge.contract import BridgeStatus
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = [
    "_MAV_RESULT_ACCEPTED",
    "_MAV_RESULT_IN_PROGRESS",
    "_MAV_RESULT_TEMPORARILY_REJECTED",
    "CommandAck",
    "RoutineObservation",
]

# MAVLink command-result ids (MAV_RESULT_*). Named here as plain ints so the
# pure observation layer does not import pymavlink. Values match the MAVLink
# common message set.
_MAV_RESULT_ACCEPTED = 0
_MAV_RESULT_TEMPORARILY_REJECTED = 1
_MAV_RESULT_IN_PROGRESS = 5

# Results that mean "the vehicle took the command" - acceptance plus the
# in-progress reply PX4 may return for long-running navigation commands.
_ACCEPTED_RESULTS = frozenset({_MAV_RESULT_ACCEPTED, _MAV_RESULT_IN_PROGRESS})


@dataclass(frozen=True)
class CommandAck:
    """A parsed MAVLink ``COMMAND_ACK`` for one command id.

    Attributes:
        command_id: The ``MAV_CMD_*`` id the ack is for.
        result: The raw ``MAV_RESULT_*`` id.
        accepted: Whether ``result`` counts as acceptance (accepted/in-progress).
    """

    command_id: int
    result: int
    accepted: bool

    @classmethod
    def from_result(cls, command_id: int, result: int) -> "CommandAck":
        """Build an ack, deriving ``accepted`` from ``result``."""
        return cls(
            command_id=int(command_id),
            result=int(result),
            accepted=int(result) in _ACCEPTED_RESULTS,
        )


@dataclass(frozen=True)
class RoutineObservation:
    """Frozen snapshot the pure ``_decide`` stepper reads each tick.

    Attributes:
        vehicle_state: The bridge's pure kinematic state for this tick.
        bridge_status: The bridge lockstep snapshot (connectivity, _arm state).
        last_ack: The most recent ``COMMAND_ACK`` parsed this tick, or ``None``.
        failsafe: Whether PX4 reported a failsafe system status this tick. Read
            by the driver to raise its domain-specific failsafe error; the pure
            stepper itself progresses purely on commands, acks, and timeouts.
        sim_time_us: Simulation time in microseconds (the stepper's clock for
            step timeouts).
    """

    vehicle_state: VehicleState
    bridge_status: BridgeStatus
    last_ack: Optional[CommandAck]
    failsafe: bool
    sim_time_us: int

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from robotics_experiments.px4_bridge.contract import BridgeStatus
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = ["AbstractFlightStackBridge"]


class AbstractFlightStackBridge(ABC):
    """Stage-able lockstep seam between sim orchestration and a flight stack.

    A tick is split into two ordered halves so the data flow is explicit:
    :meth:`publish_sensors` sends state out to the flight stack, then
    :meth:`poll_commands` blocks for the paired actuator response. Lockstep is
    causal, so the two must be called publish-then-poll within the same tick;
    calling them out of order breaks lockstep.
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def publish_sensors(self, state: VehicleState) -> None:
        """Send sensors derived from ``state`` to the flight stack (out)."""

    @abstractmethod
    def poll_commands(self, state: VehicleState) -> np.ndarray:
        """Block (with timeout) for the paired actuator commands (in).

        Returns rotor references as a length-``num_rotors`` array; zeros when
        disconnected or on timeout. Must follow a :meth:`publish_sensors` call
        for the same tick.
        """

    @property
    @abstractmethod
    def status(self) -> BridgeStatus: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @property
    @abstractmethod
    def is_armed(self) -> bool: ...

    @property
    @abstractmethod
    def num_rotors(self) -> int: ...

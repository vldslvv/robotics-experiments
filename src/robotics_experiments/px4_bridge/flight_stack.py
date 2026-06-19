from __future__ import annotations

from abc import ABC, abstractmethod

from robotics_experiments.px4_bridge.contract import ActuatorCommands, BridgeStatus
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = ["AbstractFlightStackBridge"]


class AbstractFlightStackBridge(ABC):
    """Stage-able lockstep seam between sim orchestration and a flight stack.

    A tick is a single indivisible :meth:`exchange`: it sends state out to the
    flight stack and blocks for the paired actuator response. Keeping the two
    halves inside one call makes the causal lockstep order impossible to break
    from the caller side - there is no way to poll before publishing, or twice.
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def exchange(self, state: VehicleState) -> ActuatorCommands:
        """Send sensors derived from ``state``, then block for the actuator.

        One causal tick: the sensor send and the blocking actuator receive are
        a single operation. Returns the per-motor :class:`ActuatorCommands`;
        zeros when disconnected or on timeout.
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

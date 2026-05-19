from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = ["AbstractFlightStackBridge"]


class AbstractFlightStackBridge(ABC):
    """Stage-able interface between sim orchestration and a flight stack."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def step(self, state: VehicleState) -> np.ndarray: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @property
    @abstractmethod
    def is_armed(self) -> bool: ...

    @property
    @abstractmethod
    def num_rotors(self) -> int: ...

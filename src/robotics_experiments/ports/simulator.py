from __future__ import annotations

from typing import Protocol

import numpy as np

from robotics_experiments.domain.vehicle_state import VehicleState


class ArticulationIO(Protocol):
    @property
    def is_initialized(self) -> bool: ...

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def read_state(self, sim_time_us: int) -> VehicleState: ...

    def apply_forces(self, forces_n: np.ndarray) -> None: ...

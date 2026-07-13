from __future__ import annotations

from typing import Protocol

import numpy as np

from robotics_experiments.px4_bridge.thrust_model import RotorDynamics
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__: list[str] = []


class _ArticulationIO(Protocol):
    initialized: bool

    def initialize(self) -> None: ...

    def shutdown(self) -> None: ...

    def read_state(self, sim_time_us: int) -> VehicleState: ...

    def apply_dynamics(self, dynamics: RotorDynamics) -> None: ...

    def update_rotor_spin(self, rotor_omega_rad_s: np.ndarray, dt: float) -> None: ...

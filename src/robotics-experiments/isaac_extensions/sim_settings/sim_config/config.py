from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SimSettings:
    physics_hz: float = 500.0
    rendering_hz: float = 60.0
    solver_type: str = "TGS"
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

SIM_SETTINGS = SimSettings()

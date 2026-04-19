from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SimSettings:
    """Immutable simulation settings applied by the sim_settings extension.

    Attributes:
        physics_hz: PhysX steps per second.
        rendering_hz: Kit main-loop target render frequency in Hz.
        solver_type: PhysX solver identifier (``"TGS"`` or ``"PGS"``).
        gravity: World gravity vector in m/s^2 as ``(gx, gy, gz)``.
    """

    physics_hz: float = 500.0
    rendering_hz: float = 60.0
    solver_type: str = "TGS"
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

SIM_SETTINGS = SimSettings()

from dataclasses import dataclass
from typing import Tuple

from robotics_experiments.config._immutability import _FreezableConfig

__all__ = ["SimSettings"]


@dataclass
class SimSettings(_FreezableConfig):
    """Simulation settings applied by the fleet orchestrator's ``apply_settings``.

    Attributes:
        physics_hz: PhysX steps per second.
        rendering_hz: Kit main-loop target render frequency in Hz.
        solver_type: PhysX solver identifier (``"TGS"`` or ``"PGS"``).
        gravity: World gravity vector in m/s^2 as ``(gx, gy, gz)``.
    """

    physics_hz: float
    rendering_hz: float
    solver_type: str
    gravity: Tuple[float, float, float]

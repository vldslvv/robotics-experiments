from dataclasses import dataclass, field
from typing import List

from pxr import Gf


def _default_bounds() -> List[Gf.Range3d]:
    return [
        Gf.Range3d(
            Gf.Vec3d(-3.0, 17.99, 0.5),
            Gf.Vec3d(8.0, 17.99, 2.7),
        ),
    ]


@dataclass(frozen=True)
class LightSwitchConfig:
    switch_usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/LightSwitch.usd"
    lamp_usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/Lamp.usd"
    num_switches: int = 5
    press_threshold: float = 0.01
    min_spawn_dist: float = 0.5
    max_spawn_retries: int = 100
    lamp_on_intensity: float = 5000.0
    spawn_bounds: List[Gf.Range3d] = field(default_factory=_default_bounds)


LIGHTSWITCH_CONFIG = LightSwitchConfig()

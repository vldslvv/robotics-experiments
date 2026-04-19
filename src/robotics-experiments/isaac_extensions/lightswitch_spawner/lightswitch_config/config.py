from dataclasses import dataclass, field
from typing import List

from pxr import Gf


def _default_bounds() -> List[Gf.Range3d]:
    """Return the default list of spawn volumes for switches.

    Returns:
        A single axis-aligned region along the warehouse wall at y=17.99 m.
    """
    return [
        Gf.Range3d(
            Gf.Vec3d(-3.0, 17.99, 0.5),
            Gf.Vec3d(8.0, 17.99, 2.7),
        ),
    ]


@dataclass(frozen=True)
class LightSwitchConfig:
    """Immutable configuration for the light-switch spawner.

    Attributes:
        switch_usd_path: Absolute path to the switch USD asset.
        lamp_usd_path: Absolute path to the lamp USD asset.
        num_switches: Number of switch/lamp pairs to spawn.
        press_threshold: Joint position above which the button is treated as pressed.
        min_spawn_dist: Minimum pairwise distance between spawned switches, in meters.
        max_spawn_retries: Retry budget per position before accepting overlap.
        lamp_on_intensity: ``SphereLight`` intensity applied when the lamp is on.
        spawn_bounds: Candidate axis-aligned regions used to sample positions.
    """

    switch_usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/LightSwitch.usd"
    lamp_usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/Lamp.usd"
    num_switches: int = 5
    press_threshold: float = 0.01
    min_spawn_dist: float = 0.5
    max_spawn_retries: int = 100
    lamp_on_intensity: float = 5000.0
    spawn_bounds: List[Gf.Range3d] = field(default_factory=_default_bounds)


LIGHTSWITCH_CONFIG = LightSwitchConfig()

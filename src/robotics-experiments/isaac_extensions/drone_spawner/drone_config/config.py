from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DroneConfig:
    """Immutable configuration for the drone spawner.

    Attributes:
        usd_path: Absolute path to the drone USD asset referenced on spawn.
        prim_name: Child name used under the stage default prim for the drone.
        spawn_volume_prim_name: Child name used for the guide cube that
            visualizes the spawn volume.
        spawn_volume_center: World-space center of the spawn volume.
        spawn_volume_half_extents: Per-axis half-extents of the spawn volume.
    """

    usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/hexacopter_s550.usda"
    prim_name: str = "DroneSpawned"
    spawn_volume_prim_name: str = "DroneSpawnVolume"
    spawn_volume_center: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    spawn_volume_half_extents: Tuple[float, float, float] = (2.0, 2.0, 1.0)


DRONE_CONFIG = DroneConfig()

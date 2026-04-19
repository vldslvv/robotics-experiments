from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DroneConfig:
    usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/hexacopter_s550.usda"
    prim_name: str = "DroneSpawned"
    spawn_volume_prim_name: str = "DroneSpawnVolume"
    spawn_volume_center: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    spawn_volume_half_extents: Tuple[float, float, float] = (2.0, 2.0, 1.0)


DRONE_CONFIG = DroneConfig()

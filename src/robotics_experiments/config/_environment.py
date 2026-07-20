from dataclasses import dataclass, field
from typing import Dict

from robotics_experiments.config._commanding import DriverConfig, RoutineConfig
from robotics_experiments.config._immutability import _FreezableConfig
from robotics_experiments.config._px4 import Px4BridgeConfig
from robotics_experiments.config._rl import RlConfig
from robotics_experiments.config._scene import (
    DroneSpawnConfig,
    LightSwitchConfig,
    RosVehicleConfig,
)
from robotics_experiments.config._simulation import SimSettings

__all__ = ["ExperimentConfig", "VehicleConfig"]


@dataclass
class _FlightBenchmarkConfig(_FreezableConfig):
    """Opt-in full-GUI benchmark methodology parameters."""

    airborne_duration_s: float


@dataclass
class VehicleConfig(_FreezableConfig):
    """Complete authored and derived configuration for one vehicle."""

    id: int
    ros: RosVehicleConfig
    spawn: DroneSpawnConfig
    bridge: Px4BridgeConfig
    routine: RoutineConfig
    mavlink_port: int = field(init=False, default=0)
    gcs_port: int = field(init=False, default=0)
    debug_log_path: str = field(init=False, default="")
    px4_sim_model: str = field(init=False, default="")
    px4_sys_autostart: int = field(init=False, default=0)


@dataclass
class ExperimentConfig(_FreezableConfig):
    """Complete configuration for one orchestrator execution."""

    sim: SimSettings
    switches: LightSwitchConfig
    rl: RlConfig
    scene: str
    vehicles: Dict[str, VehicleConfig]
    driver: DriverConfig
    flight_benchmark: _FlightBenchmarkConfig

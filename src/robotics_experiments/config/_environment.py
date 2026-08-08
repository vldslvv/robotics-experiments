from dataclasses import dataclass, field
from typing import Dict

from robotics_experiments.config._commanding import DriverConfig, RoutineConfig
from robotics_experiments.config._immutability import _FreezableConfig
from robotics_experiments.config._px4 import Px4BridgeConfig, _Px4RuntimeConfig
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
class _AlignmentCaptureConfig(_FreezableConfig):
    """Behavior-affecting PX4 alignment capture preconditioning."""

    sensor_seed: int
    pusher_deployed: bool
    minimum_neutral_s: float
    stable_hold_s: float
    timeout_s: float
    max_linear_speed_mps: float
    max_angular_speed_radps: float
    max_tilt_deg: float


@dataclass
class _AlignmentExecutionConfig(_FreezableConfig):
    """Alignment throughput settings that do not affect comparison behavior."""

    worker_count: int
    candidate_capacity: int
    candidate_capacity_benchmark: list[int]
    gpu_target_percent: float
    gpu_sample_interval_s: float


@dataclass
class _AlignmentConfig(_FreezableConfig):
    """Controller-alignment capture and execution settings."""

    capture: _AlignmentCaptureConfig
    execution: _AlignmentExecutionConfig


@dataclass
class VehicleConfig(_FreezableConfig):
    """Complete authored and derived configuration for one vehicle."""

    id: int
    ros: RosVehicleConfig
    spawn: DroneSpawnConfig
    bridge: Px4BridgeConfig
    px4: _Px4RuntimeConfig
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
    alignment: _AlignmentConfig

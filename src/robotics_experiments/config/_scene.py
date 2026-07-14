from dataclasses import dataclass
from typing import List, Tuple

from robotics_experiments.config._immutability import _FreezableConfig

__all__ = [
    "BoundingBox",
    "DroneSpawnConfig",
    "LightSwitchConfig",
    "LightSwitchStatePublisherConfig",
    "RosQosConfig",
    "RosVehicleConfig",
]


@dataclass
class DroneSpawnConfig(_FreezableConfig):
    """Asset and world pose for one spawned vehicle."""

    usd_path: str
    prim_name: str
    position_xy: Tuple[float, float]
    yaw_rad: float
    spawn_stationary: bool
    stationary_ground_z: float
    stationary_ground_clearance_m: float
    stationary_spawn_z_fallback: float


@dataclass
class BoundingBox(_FreezableConfig):
    """Axis-aligned world-space region, authored as corner tuples.

    Used instead of ``pxr.Gf.Range3d`` so the value is serializable by
    OmegaConf / Hydra. Consumers convert to ``Gf.Range3d`` at the call site.

    Attributes:
        min: Lower corner ``(x, y, z)`` in world units.
        max: Upper corner ``(x, y, z)`` in world units.
    """

    min: Tuple[float, float, float]
    max: Tuple[float, float, float]


@dataclass
class RosQosConfig(_FreezableConfig):
    """ROS 2 QoS settings for a publisher or subscription.

    Attributes:
        depth: Queue depth for ``KEEP_LAST`` history.
        durability: Durability policy name, such as ``"transient_local"``.
        reliability: Reliability policy name, such as ``"reliable"``.
        history: History policy name, such as ``"keep_last"``.
    """

    depth: int
    durability: str
    reliability: str
    history: str


@dataclass
class RosVehicleConfig(_FreezableConfig):
    """Per-vehicle ROS graph identity.

    Attributes:
        namespace: Absolute ROS namespace for this vehicle, such as
            ``"/x500_01"``. Authored per vehicle so real fleet names can stay
            stable independently of the numeric runtime ``id`` used by PX4.
    """

    namespace: str


@dataclass
class LightSwitchStatePublisherConfig(_FreezableConfig):
    """ROS publisher configuration for light-switch state events.

    Attributes:
        topic: ROS topic used for switch toggle events.
        qos: ROS 2 QoS settings for the publisher.
    """

    topic: str
    qos: RosQosConfig


@dataclass
class LightSwitchConfig(_FreezableConfig):
    """Configuration for the light-switch spawner extension.

    Attributes:
        switch_usd_path: Absolute path to the switch USD asset.
        lamp_usd_path: Absolute path to the lamp USD asset.
        num_switches: Number of switch/lamp pairs to spawn.
        spawn_seed: Seed for deterministic switch positions and lamp heights.
        press_threshold: Joint position above which the button is treated as pressed.
        press_confirm_steps: Consecutive steps the button must stay pressed before
            a toggle is confirmed (the light-state latch / confirmation window).
        min_spawn_dist: Minimum pairwise distance between spawned switches, in meters.
        max_spawn_retries: Retry budget per position before accepting overlap.
        lamp_on_intensity: ``SphereLight`` intensity applied when the lamp is on.
        spawn_bounds: Candidate axis-aligned regions used to sample positions.
        state_publisher: ROS publisher settings for switch toggle events.
    """

    switch_usd_path: str
    lamp_usd_path: str
    num_switches: int
    spawn_seed: int
    press_threshold: float
    press_confirm_steps: int
    min_spawn_dist: float
    max_spawn_retries: int
    lamp_on_intensity: float
    spawn_bounds: List[BoundingBox]
    state_publisher: LightSwitchStatePublisherConfig

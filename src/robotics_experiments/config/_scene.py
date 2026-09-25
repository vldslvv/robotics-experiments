import math
from dataclasses import dataclass, fields
from typing import List, Tuple

from robotics_experiments.config._immutability import _FreezableConfig

__all__ = [
    "BoundingBox",
    "DroneSpawnConfig",
    "LightSwitchConfig",
    "SwitchModelConfig",
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
class SwitchModelConfig(_FreezableConfig):
    """Geometry and linear spring model, in metres, kilograms, seconds and newtons."""

    button_width_m: float
    button_height_m: float
    button_thickness_m: float
    button_rest_y_m: float
    housing_width_m: float
    housing_height_m: float
    clearance_m: float
    back_plate_thickness_m: float
    contact_offset_m: float
    travel_m: float
    activation_displacement_m: float
    activation_force_n: float
    button_mass_kg: float
    damping_ratio: float
    confirmation_s: float

    @property
    def stiffness_n_m(self) -> float:
        return self.activation_force_n / self.activation_displacement_m

    @property
    def damping_n_s_m(self) -> float:
        return 2 * self.damping_ratio * math.sqrt(self.button_mass_kg * self.stiffness_n_m)

    @property
    def back_extent_m(self) -> float:
        return (
            self.button_rest_y_m
            + self.travel_m
            + self.button_thickness_m / 2
            + self.clearance_m
            + self.back_plate_thickness_m
        )

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or (field.name != "button_rest_y_m" and value <= 0):
                raise ValueError(f"switches.model.{field.name} must be finite and positive")
        if not self.activation_displacement_m < self.travel_m:
            raise ValueError("switch activation must be inside usable travel")
        if self.clearance_m <= 2 * self.contact_offset_m:
            raise ValueError("switch clearance must exceed both contact offsets")
        if (
            self.button_width_m + 2 * self.clearance_m >= self.housing_width_m
            or self.button_height_m + 2 * self.clearance_m >= self.housing_height_m
        ):
            raise ValueError("switch housing must surround the full button swept volume")


@dataclass
class _SwitchVerificationConfig(_FreezableConfig):
    pusher_mass_kg: float
    pusher_stiffness_n_m: float
    pusher_damping_n_s_m: float
    pusher_speed_mps: float
    force_tolerance_n: float
    displacement_tolerance_m: float
    near_stop_margin_m: float
    hold_s: float
    overtravel_m: float


@dataclass
class LightSwitchConfig(_FreezableConfig):
    """Shared switch model, deterministic layout and lamp publication settings."""

    model: SwitchModelConfig
    verification: _SwitchVerificationConfig
    lamp_usd_path: str
    num_switches: int
    spawn_seed: int
    min_spawn_dist: float
    max_spawn_retries: int
    lamp_on_intensity: float
    spawn_bounds: List[BoundingBox]
    state_publisher: LightSwitchStatePublisherConfig

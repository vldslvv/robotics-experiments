from dataclasses import dataclass
from typing import Any, List

from robotics_experiments.config._commanding import DriverConfig, RoutineConfig
from robotics_experiments.config._px4 import Px4BridgeConfig
from robotics_experiments.config._rl import RlConfig
from robotics_experiments.config._scene import DroneConfig, LightSwitchConfig
from robotics_experiments.config._simulation import SimSettings

__all__ = ["EnvConfig"]


@dataclass
class EnvConfig:
    """Composite experiment config consumed by the Isaac Sim extensions.

    Attributes:
        sim: Simulation-level settings (physics, rendering, gravity).
        drone: Drone spawner configuration.
        switches: Light-switch spawner configuration.
        px4: PX4 SITL bridge configuration.
        rl: Shared RL training/scale/action config plus nested task config.
        scene: USD scene to load for this experiment. ``@``-prefixed paths are
            repository-relative. Every experiment (test or not) sets a scene;
            the standalone runner uses it when not given an explicit ``--scene``.
        vehicles: Multi-vehicle scaffolding. Each entry merges its ``drone`` /
            ``px4`` partial over the global ``drone`` / ``px4`` defaults above.
            The list is the only spawn path (one vehicle = a one-element list;
            an empty list is a valid empty fleet).

            Typing note: the annotation is intentionally ``List[Any]`` so Hydra's
            structured-config validation accepts the *partial* YAML entries
            during ``compose()``. The loader replaces the raw dicts with a
            fully enriched, validated ``list[ResolvedVehicle]``, which is the
            public/runtime type.
        routine: High-level flight routine plus its verification gates (see
            :class:`RoutineConfig`). Always present; an empty ``routine.steps``
            means manual operation, a non-empty list makes every workflow
            arm/take-off/verify the fleet. The integration test is just this
            routine driven in strict mode.
        driver: Routine-execution driver heuristics (see :class:`DriverConfig`):
            the ``goto`` heading epsilon and the driver's wall-clock backstops.
            Global (not merged per vehicle).
    """

    sim: SimSettings
    drone: DroneConfig
    switches: LightSwitchConfig
    px4: Px4BridgeConfig
    rl: RlConfig
    scene: str
    vehicles: List[Any]
    routine: RoutineConfig
    driver: DriverConfig

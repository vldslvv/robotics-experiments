from __future__ import annotations

from robotics_experiments.config import DroneConfig


def spawn_drone(stage, config: DroneConfig | None = None) -> None:
    from robotics_experiments.adapters.isaac.stage_drone import spawn_drone as _spawn

    _spawn(stage, config)


def despawn_drone(stage, config: DroneConfig | None = None) -> None:
    from robotics_experiments.adapters.isaac.stage_drone import (
        despawn_drone as _despawn,
    )

    _despawn(stage, config)

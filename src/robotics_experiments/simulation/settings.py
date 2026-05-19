from __future__ import annotations

from robotics_experiments.config import SimSettings

__all__ = ["apply_settings"]


def apply_settings(stage, settings: SimSettings | None = None) -> None:
    from robotics_experiments.simulation.isaac_settings import (
        apply_settings as _apply,
    )

    _apply(stage, settings)

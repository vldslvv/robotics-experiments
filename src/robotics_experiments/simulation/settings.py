"""Public simulation settings entrypoint.

This module intentionally does not import Isaac Sim modules at import time.
Importing ``carb`` or ``pxr`` outside the Isaac runtime can fail, so the public
``apply_settings`` function performs a lazy import of the Isaac backend only
when a caller actually applies settings to a USD stage. Keep workflow-specific
setup, such as standalone ``SimulationApp`` startup or Kit extension event
subscription, outside this module; callers share this stage-level API.
"""

from __future__ import annotations

from robotics_experiments.config import SimSettings

__all__ = ["apply_settings"]


def apply_settings(stage, settings: SimSettings | None = None) -> None:
    """Apply simulation settings to an already available USD stage.

    The caller is responsible for owning the Isaac workflow that produced the
    stage. Standalone scripts may create a ``SimulationApp`` and open a scene;
    Kit extensions may receive the stage from ``omni.usd`` events.
    """
    from robotics_experiments.simulation._isaac_settings import (
        apply_isaac_settings,
    )

    apply_isaac_settings(stage, settings)

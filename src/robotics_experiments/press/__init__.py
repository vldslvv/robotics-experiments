"""High-level RL press layer: pure actuation/observation/reward building blocks.

This package is on the core import path and must import cleanly with Isaac and
torch absent (stdlib + numpy only at module scope). The torch/isaaclab env code
lives in ``robotics_experiments.workflows.isaac_lab.tasks``; the pure reference
implementation (frozen params + pure functions) and pure deploy-mapping live
here. ``__init__`` stays re-export-only.
"""

from __future__ import annotations

from .actuation import (
    ActuationParams,
    ActuationState,
    MixerGeometry,
    action_to_rotor_omega,
    build_actuation_params,
    hover_action,
    reduce_to_body_wrench,
)
from .deploy import press_action_to_offboard

__all__ = [
    "ActuationParams",
    "ActuationState",
    "MixerGeometry",
    "action_to_rotor_omega",
    "build_actuation_params",
    "hover_action",
    "press_action_to_offboard",
    "reduce_to_body_wrench",
]

"""High-level RL press layer: pure actuation/observation/reward building blocks.

This package is on the core import path and must import cleanly with Isaac and
torch absent (stdlib + numpy only at module scope). The torch/isaaclab env code
lives in ``robotics_experiments.workflows.isaac_lab.tasks``; the pure reference
models and pure deploy-mapping live here. ``__init__`` stays re-export-only.
"""

from __future__ import annotations

from ._actuation import (
    ActuationModel,
    ActuationState,
    BodyRateActuation,
    MixerGeometry,
    VelocitySetpointActuation,
    make_actuation_model,
    reduce_to_body_wrench,
)
from ._deploy import press_action_to_offboard

__all__ = [
    "ActuationModel",
    "ActuationState",
    "BodyRateActuation",
    "MixerGeometry",
    "VelocitySetpointActuation",
    "make_actuation_model",
    "press_action_to_offboard",
    "reduce_to_body_wrench",
]

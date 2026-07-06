"""High-level RL press layer: pure actuation/observation/reward building blocks.

This package is on the core import path and must import cleanly with Isaac and
torch absent (stdlib + numpy only at module scope). The torch/isaaclab env code
lives in ``robotics_experiments.workflows.isaac_lab.tasks``; the pure reference
implementation (frozen params + pure functions) and pure deploy-mapping live
here. ``__init__`` stays re-export-only.
"""

from __future__ import annotations

from robotics_experiments.rl.press._handoff import (
    HandoffGate,
    HandoffReason,
    HandoffResult,
    handoff_ready,
)
from robotics_experiments.rl.press._provider import AbstractPoseProvider, PoseSample
from robotics_experiments.rl.press.actuation import (
    ActuationParams,
    ActuationState,
    MixerGeometry,
    action_to_rotor_omega,
    build_actuation_params,
    hover_action,
    reduce_to_body_wrench,
)
from robotics_experiments.rl.press.deploy import press_action_to_offboard
from robotics_experiments.rl.press.observation import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PressObservation,
    hover_observation_from_state,
    pack_hover_observation,
    to_vector,
    world_switch_to_contact_frame,
)

__all__ = [
    "HOVER_OBSERVATION_DIM",
    "HOVER_OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "AbstractPoseProvider",
    "ActuationParams",
    "ActuationState",
    "GroundTruthPoseProvider",
    "HandoffGate",
    "HandoffReason",
    "HandoffResult",
    "MixerGeometry",
    "PoseSample",
    "PressObservation",
    "action_to_rotor_omega",
    "build_actuation_params",
    "handoff_ready",
    "hover_action",
    "hover_observation_from_state",
    "pack_hover_observation",
    "press_action_to_offboard",
    "reduce_to_body_wrench",
    "to_vector",
    "world_switch_to_contact_frame",
]


def __getattr__(name: str):
    # GroundTruthPoseProvider lives in an Isaac-only leaf (imports pxr at
    # module scope); defer it so `import robotics_experiments.rl.press` stays
    # Isaac-free. Same pattern as environment/__init__.py.
    if name == "GroundTruthPoseProvider":
        import robotics_experiments.rl.press._isaac_provider as provider_module

        return provider_module.GroundTruthPoseProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

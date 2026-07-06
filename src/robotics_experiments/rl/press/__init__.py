"""High-level RL press public API.

This package is on the core import path and must import cleanly with Isaac and
torch absent. ``__init__`` is the curated public facade; implementation modules
remain private unless a production consumer needs a narrower module import.
"""

from __future__ import annotations

from robotics_experiments.rl.press._handoff import (
    HandoffGate,
    HandoffReason,
    HandoffResult,
    handoff_ready,
)
from robotics_experiments.rl.press._provider import AbstractPoseProvider, PoseSample
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
from robotics_experiments.rl.press.policy import (
    METADATA_FILENAME,
    POLICY_FILENAME,
    build_policy_step_fn,
    policy_metadata,
)

__all__ = [
    "HOVER_OBSERVATION_DIM",
    "HOVER_OBSERVATION_SCHEMA_VERSION",
    "METADATA_FILENAME",
    "OBSERVATION_SCHEMA_VERSION",
    "POLICY_FILENAME",
    "AbstractPoseProvider",
    "GroundTruthPoseProvider",
    "HandoffGate",
    "HandoffReason",
    "HandoffResult",
    "PoseSample",
    "PressObservation",
    "build_policy_step_fn",
    "handoff_ready",
    "hover_observation_from_state",
    "pack_hover_observation",
    "policy_metadata",
    "press_action_to_offboard",
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

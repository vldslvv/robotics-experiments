"""Public press-task reinforcement-learning contract.

The Isaac-only provider stays lazy so importing this package does not pull USD
dependencies onto the core path.
"""

from __future__ import annotations

from robotics_experiments.rl.press.deployment import (
    PressPolicyDecision,
    PressPolicyParams,
    PressPolicyProgress,
    initial_press_policy_progress,
    step_press_policy,
)
from robotics_experiments.rl.press.handoff import (
    HandoffGate,
    HandoffReason,
    HandoffResult,
    handoff_ready,
)
from robotics_experiments.rl.press.observation import (
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    PressObservation,
    PressObservationMask,
    pack_press_observation,
    to_vector,
    world_switch_to_contact_frame,
)
from robotics_experiments.rl.press.provider import AbstractPoseProvider, PoseSample
from robotics_experiments.rl.press.task import (
    PressRewardTerms,
    PressStepMetrics,
    PressTaskInput,
    PressTaskResult,
    PressTaskState,
    PressTerminationFlags,
    initial_press_task_state,
    reset_press_task_state,
    step_press_task,
)

__all__ = [
    "AbstractPoseProvider",
    "GroundTruthPoseProvider",
    "HandoffGate",
    "HandoffReason",
    "HandoffResult",
    "OBSERVATION_SCHEMA_VERSION",
    "PRESS_OBSERVATION_DIM",
    "PoseSample",
    "PressObservation",
    "PressObservationMask",
    "PressPolicyDecision",
    "PressPolicyParams",
    "PressPolicyProgress",
    "PressRewardTerms",
    "PressStepMetrics",
    "PressTaskInput",
    "PressTaskResult",
    "PressTaskState",
    "PressTerminationFlags",
    "handoff_ready",
    "initial_press_policy_progress",
    "initial_press_task_state",
    "pack_press_observation",
    "reset_press_task_state",
    "step_press_policy",
    "step_press_task",
    "to_vector",
    "world_switch_to_contact_frame",
]

_ISAAC_PROVIDER_ATTRS = {"GroundTruthPoseProvider"}


def __getattr__(name: str):
    if name in _ISAAC_PROVIDER_ATTRS:
        from robotics_experiments.rl.press.isaac_provider import GroundTruthPoseProvider

        return GroundTruthPoseProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Public press-task reinforcement-learning contract.

The Isaac-only provider stays lazy so importing this package does not pull USD
dependencies onto the core path.
"""

from __future__ import annotations

from robotics_experiments.rl.press.deployment import (
    PressApproachExecutionParams,
    PressApproachExecutionResult,
    PressApproachExecutionState,
    advance_press_approach_execution,
    initial_press_approach_execution_state,
)
from robotics_experiments.rl.press.evaluation import (
    AcceptanceCheck,
    AcceptanceResult,
    PressAcceptanceCriteria,
    create_press_acceptance_criteria,
    evaluate_press_acceptance,
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
from robotics_experiments.rl.press.policy import press_policy_contract
from robotics_experiments.rl.press.provider import AbstractPoseProvider, PoseSample
from robotics_experiments.rl.press.task import (
    PressApproachInput,
    PressApproachResult,
    PressApproachRewardTerms,
    PressApproachState,
    PressApproachTerminationFlags,
    initial_press_approach_state,
    press_approach_target_velocity,
    quaternion_geodesic_error_rad,
    reset_press_approach_state,
    step_press_approach,
)

__all__ = [
    "AbstractPoseProvider",
    "AcceptanceCheck",
    "AcceptanceResult",
    "GroundTruthPoseProvider",
    "HandoffGate",
    "HandoffReason",
    "HandoffResult",
    "OBSERVATION_SCHEMA_VERSION",
    "PRESS_OBSERVATION_DIM",
    "PoseSample",
    "PressAcceptanceCriteria",
    "PressApproachExecutionParams",
    "PressApproachExecutionResult",
    "PressApproachExecutionState",
    "PressApproachInput",
    "PressApproachResult",
    "PressApproachRewardTerms",
    "PressApproachState",
    "PressApproachTerminationFlags",
    "PressObservation",
    "PressObservationMask",
    "advance_press_approach_execution",
    "create_press_acceptance_criteria",
    "evaluate_press_acceptance",
    "handoff_ready",
    "initial_press_approach_execution_state",
    "initial_press_approach_state",
    "pack_press_observation",
    "press_approach_target_velocity",
    "press_policy_contract",
    "quaternion_geodesic_error_rad",
    "reset_press_approach_state",
    "step_press_approach",
    "to_vector",
    "world_switch_to_contact_frame",
]

_ISAAC_PROVIDER_ATTRS = {"GroundTruthPoseProvider"}


def __getattr__(name: str):
    if name in _ISAAC_PROVIDER_ATTRS:
        from robotics_experiments.rl.press.isaac_provider import GroundTruthPoseProvider

        return GroundTruthPoseProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

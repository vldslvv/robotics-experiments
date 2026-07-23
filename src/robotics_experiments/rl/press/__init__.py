"""Public press reinforcement-learning contract.

Consumers outside this package must import press concepts from this package
root rather than from implementation submodules. The Isaac-only provider leaf
stays lazy, so importing this package does not pull USD dependencies onto the
core path.
"""

from __future__ import annotations

from robotics_experiments.rl.press.actuation import (
    ActuationParams,
    ActuationState,
    MixerGeometry,
    action_to_rotor_omega,
    build_actuation_params,
    hover_action,
    rotor_forces_to_body_wrench,
)
from robotics_experiments.rl.press.deploy import press_action_to_offboard
from robotics_experiments.rl.press.deployment import (
    PressDeploymentDecision,
    PressDeploymentState,
    initial_press_deployment_state,
    step_press_deployment,
)
from robotics_experiments.rl.press.handoff import (
    HandoffGate,
    HandoffReason,
    HandoffResult,
    handoff_ready,
)
from robotics_experiments.rl.press.observation import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    PressObservation,
    PressObservationMask,
    hover_observation_from_state,
    pack_hover_observation,
    pack_press_observation,
    to_vector,
    world_switch_to_contact_frame,
)
from robotics_experiments.rl.press.policy import (
    HOVER_POLICY_INTERFACE,
    METADATA_FILENAME,
    POLICY_FILENAME,
    PRESS_POLICY_INTERFACE,
    LoadedVectorPolicy,
    PolicyContract,
    build_policy_step_fn,
    load_policy_contract,
    load_policy_metadata,
    load_vector_policy,
    policy_metadata,
    validate_policy_artifact,
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
    "ActuationParams",
    "ActuationState",
    "GroundTruthPoseProvider",
    "HOVER_OBSERVATION_DIM",
    "HOVER_OBSERVATION_SCHEMA_VERSION",
    "HOVER_POLICY_INTERFACE",
    "HandoffGate",
    "HandoffReason",
    "HandoffResult",
    "LoadedVectorPolicy",
    "METADATA_FILENAME",
    "MixerGeometry",
    "OBSERVATION_SCHEMA_VERSION",
    "PRESS_OBSERVATION_DIM",
    "POLICY_FILENAME",
    "PRESS_POLICY_INTERFACE",
    "PolicyContract",
    "PoseSample",
    "PressDeploymentDecision",
    "PressDeploymentState",
    "PressObservation",
    "PressObservationMask",
    "PressRewardTerms",
    "PressStepMetrics",
    "PressTaskInput",
    "PressTaskResult",
    "PressTaskState",
    "PressTerminationFlags",
    "action_to_rotor_omega",
    "build_actuation_params",
    "build_policy_step_fn",
    "handoff_ready",
    "hover_action",
    "hover_observation_from_state",
    "initial_press_task_state",
    "initial_press_deployment_state",
    "load_policy_contract",
    "load_policy_metadata",
    "load_vector_policy",
    "pack_hover_observation",
    "pack_press_observation",
    "policy_metadata",
    "press_action_to_offboard",
    "reset_press_task_state",
    "rotor_forces_to_body_wrench",
    "step_press_task",
    "step_press_deployment",
    "to_vector",
    "validate_policy_artifact",
    "world_switch_to_contact_frame",
]

_ISAAC_PROVIDER_ATTRS = {"GroundTruthPoseProvider"}


def __getattr__(name: str):
    if name in _ISAAC_PROVIDER_ATTRS:
        from robotics_experiments.rl.press.isaac_provider import GroundTruthPoseProvider

        return GroundTruthPoseProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

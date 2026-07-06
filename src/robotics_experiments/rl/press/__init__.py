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
from robotics_experiments.rl.press.provider import AbstractPoseProvider, PoseSample

__all__ = [
    "AbstractPoseProvider",
    "ActuationParams",
    "ActuationState",
    "GroundTruthPoseProvider",
    "HOVER_OBSERVATION_DIM",
    "HOVER_OBSERVATION_SCHEMA_VERSION",
    "HandoffGate",
    "HandoffReason",
    "HandoffResult",
    "METADATA_FILENAME",
    "MixerGeometry",
    "OBSERVATION_SCHEMA_VERSION",
    "POLICY_FILENAME",
    "PoseSample",
    "PressObservation",
    "action_to_rotor_omega",
    "build_actuation_params",
    "build_policy_step_fn",
    "handoff_ready",
    "hover_action",
    "hover_observation_from_state",
    "pack_hover_observation",
    "policy_metadata",
    "press_action_to_offboard",
    "rotor_forces_to_body_wrench",
    "to_vector",
    "world_switch_to_contact_frame",
]

_ISAAC_PROVIDER_ATTRS = {"GroundTruthPoseProvider"}


def __getattr__(name: str):
    if name in _ISAAC_PROVIDER_ATTRS:
        from robotics_experiments.rl.press.isaac_provider import GroundTruthPoseProvider

        return GroundTruthPoseProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

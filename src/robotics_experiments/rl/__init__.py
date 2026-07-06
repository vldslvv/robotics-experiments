"""Public reinforcement-learning contract.

This package root is the only supported import surface for RL consumers outside
``robotics_experiments.rl``. The Isaac-only provider leaf stays lazy, so
importing this package does not pull USD dependencies onto the core path.
Training entrypoints are also lazy, so importing this package does not pull
torch, rsl_rl, or Isaac Lab onto the core path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robotics_experiments.rl.press import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    METADATA_FILENAME,
    OBSERVATION_SCHEMA_VERSION,
    POLICY_FILENAME,
    AbstractPoseProvider,
    ActuationParams,
    ActuationState,
    HandoffGate,
    HandoffReason,
    HandoffResult,
    MixerGeometry,
    PoseSample,
    PressObservation,
    action_to_rotor_omega,
    build_actuation_params,
    build_policy_step_fn,
    handoff_ready,
    hover_action,
    hover_observation_from_state,
    pack_hover_observation,
    policy_metadata,
    press_action_to_offboard,
    rotor_forces_to_body_wrench,
    to_vector,
    world_switch_to_contact_frame,
)

if TYPE_CHECKING:
    from robotics_experiments.rl.press import GroundTruthPoseProvider
    from robotics_experiments.rl.training import (
        TrainingResult,
        build_runner,
        export_policy,
        train,
    )

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
    "TrainingResult",
    "action_to_rotor_omega",
    "build_actuation_params",
    "build_policy_step_fn",
    "build_runner",
    "export_policy",
    "handoff_ready",
    "hover_action",
    "hover_observation_from_state",
    "pack_hover_observation",
    "policy_metadata",
    "press_action_to_offboard",
    "rotor_forces_to_body_wrench",
    "train",
    "to_vector",
    "world_switch_to_contact_frame",
]

_ISAAC_PROVIDER_ATTRS = {"GroundTruthPoseProvider"}
_TRAINING_ATTRS = {"TrainingResult", "build_runner", "export_policy", "train"}


def __getattr__(name: str):
    if name in _ISAAC_PROVIDER_ATTRS:
        from robotics_experiments.rl.press import GroundTruthPoseProvider

        return GroundTruthPoseProvider
    if name in _TRAINING_ATTRS:
        import robotics_experiments.rl.training as training_module

        return getattr(training_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

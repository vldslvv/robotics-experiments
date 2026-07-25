"""Pure deploy-time transition for a trained press policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from robotics_experiments.rl.press.handoff import HandoffGate, handoff_ready
from robotics_experiments.rl.press.observation import (
    PRESS_OBSERVATION_DIM,
    PressObservation,
    PressObservationMask,
    to_vector,
)

__all__ = [
    "PressPolicyExecutionParams",
    "PressPolicyExecutionResult",
    "PressPolicyExecutionState",
    "advance_press_policy_execution",
    "initial_press_policy_execution_state",
]


@dataclass(frozen=True)
class PressPolicyExecutionParams:
    """Immutable inference, handoff, and completion parameters."""

    inference: Callable[[np.ndarray], np.ndarray]
    action_dim: int
    handoff_gate: HandoffGate
    first_contact_speed_max_mps: float
    first_contact_angular_speed_max_radps: float
    retreat_distance_m: float
    retreat_speed_max_mps: float
    retreat_alignment_tolerance_m: float
    retreat_angular_speed_max_radps: float
    stable_retreat_steps: int


@dataclass(frozen=True)
class PressPolicyExecutionState:
    """Opaque immutable state stored by the generic policy routine step."""

    previous_action: tuple[float, ...]
    initial_lamp_validated: bool = False
    handoff_latched: bool = False
    first_contact_seen: bool = False
    first_contact_stable: bool = False
    previous_lamp_state: int = -1
    toggle_count: int = 0
    stable_retreat_steps: int = 0
    initial_switch_range_m: float = math.nan
    closest_switch_range_m: float = math.nan
    peak_pusher_compression_m: float = 0.0
    peak_abs_action: float = 0.0


@dataclass(frozen=True)
class PressPolicyExecutionResult:
    """Successor state, normalized action, and generic routine outcome."""

    state: PressPolicyExecutionState
    action: np.ndarray
    complete: bool
    failure_reason: Optional[str]


def initial_press_policy_execution_state(*, action_dim: int) -> PressPolicyExecutionState:
    """Create zero-action state for one policy invocation."""
    return PressPolicyExecutionState(previous_action=(0.0,) * int(action_dim))


def _invalid_reason(*, observation: PressObservation, action_dim: int) -> Optional[str]:
    required = int(
        PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
    )
    if int(observation.obs_mask) & required != required:
        return "press observation is missing pose, contact, or lamp availability"
    if not observation.detection_valid or not observation.contact_valid:
        return "press observation contains an invalid pose or contact sample"
    if observation.lamp_state not in (0, 1) or observation.lamp_confidence <= 0.0:
        return "press observation contains an unknown lamp state"
    if np.asarray(observation.previous_action).shape != (action_dim,):
        return f"press previous_action must have shape ({action_dim},)"
    vector = to_vector(observation)
    if vector.shape != (PRESS_OBSERVATION_DIM,) or not np.all(np.isfinite(vector)):
        return "press observation is non-finite or has the wrong schema dimension"
    return None


def advance_press_policy_execution(
    *,
    params: PressPolicyExecutionParams,
    state: PressPolicyExecutionState,
    observation: PressObservation,
) -> PressPolicyExecutionResult:
    """Advance handoff, policy action, lamp-toggle, and retreat completion once."""
    observation = replace(
        observation,
        previous_action=np.asarray(state.previous_action, dtype=np.float64),
        handoff_mode=2 if state.first_contact_seen else 1,
    )
    zero = np.zeros(params.action_dim, dtype=np.float64)
    invalid = _invalid_reason(observation=observation, action_dim=params.action_dim)
    if invalid is not None:
        return PressPolicyExecutionResult(state, zero, False, invalid)

    switch_range_m = float(observation.relative_switch_pos_m[0])
    initial_switch_range_m = (
        switch_range_m if not state.initial_lamp_validated else state.initial_switch_range_m
    )
    closest_switch_range_m = (
        switch_range_m
        if not state.initial_lamp_validated
        else min(state.closest_switch_range_m, switch_range_m)
    )
    state = replace(
        state,
        initial_switch_range_m=initial_switch_range_m,
        closest_switch_range_m=closest_switch_range_m,
        peak_pusher_compression_m=max(
            state.peak_pusher_compression_m,
            float(observation.pusher_compression_m),
        ),
    )

    if not state.initial_lamp_validated:
        if observation.lamp_state != 0:
            return PressPolicyExecutionResult(
                state,
                zero,
                False,
                "press policy requires the paired lamp to start off",
            )
        state = replace(
            state,
            initial_lamp_validated=True,
            previous_lamp_state=observation.lamp_state,
        )

    toggle_count = state.toggle_count
    if observation.lamp_state != state.previous_lamp_state:
        toggle_count += 1
    if toggle_count > 1:
        return PressPolicyExecutionResult(
            replace(
                state,
                previous_lamp_state=observation.lamp_state,
                toggle_count=toggle_count,
            ),
            zero,
            False,
            "paired lamp toggled more than once during one press step",
        )

    linear_speed = float(np.linalg.norm(observation.body_lin_vel_mps))
    angular_speed = float(np.linalg.norm(observation.body_ang_vel_radps))
    first_contact = bool(observation.contact_flag) and not state.first_contact_seen
    first_contact_seen = state.first_contact_seen or bool(observation.contact_flag)
    first_contact_stable = state.first_contact_stable or (
        first_contact
        and linear_speed <= params.first_contact_speed_max_mps
        and angular_speed <= params.first_contact_angular_speed_max_radps
    )

    handoff_latched = state.handoff_latched or handoff_ready(observation, params.handoff_gate).ready
    action = zero
    if handoff_latched:
        action = np.asarray(params.inference(to_vector(observation)), dtype=np.float64)
        if action.shape != (params.action_dim,) or not np.all(np.isfinite(action)):
            return PressPolicyExecutionResult(
                state,
                zero,
                False,
                f"press policy returned invalid action shape {action.shape}",
            )
        action = np.clip(action, -1.0, 1.0)
    peak_abs_action = max(state.peak_abs_action, float(np.max(np.abs(action))))

    retreat_target_m = max(initial_switch_range_m, params.retreat_distance_m)
    lateral_error_m = float(np.linalg.norm(observation.relative_switch_pos_m[1:3]))
    stable_retreat = (
        toggle_count == 1
        and not observation.contact_flag
        and float(observation.relative_switch_pos_m[0]) >= retreat_target_m
        and linear_speed <= params.retreat_speed_max_mps
        and lateral_error_m <= params.retreat_alignment_tolerance_m
        and angular_speed <= params.retreat_angular_speed_max_radps
    )
    retreat_steps = state.stable_retreat_steps + 1 if stable_retreat else 0
    successor = PressPolicyExecutionState(
        previous_action=tuple(float(value) for value in action),
        initial_lamp_validated=True,
        handoff_latched=handoff_latched,
        first_contact_seen=first_contact_seen,
        first_contact_stable=first_contact_stable,
        previous_lamp_state=observation.lamp_state,
        toggle_count=toggle_count,
        stable_retreat_steps=retreat_steps,
        initial_switch_range_m=initial_switch_range_m,
        closest_switch_range_m=closest_switch_range_m,
        peak_pusher_compression_m=state.peak_pusher_compression_m,
        peak_abs_action=peak_abs_action,
    )
    complete = retreat_steps >= params.stable_retreat_steps
    return PressPolicyExecutionResult(successor, zero if complete else action, complete, None)

"""Pure deploy-time transition for the press approach curriculum stage."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from robotics_experiments.config import PressApproachGoalConfig, PressHandoffConfig
from robotics_experiments.flight_parameters import FlightParameters
from robotics_experiments.px4_bridge import (
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    rotate_vector_wxyz,
)
from robotics_experiments.rl import evaluate_hover_envelope
from robotics_experiments.rl.press.handoff import HandoffGate, HandoffReason, handoff_ready
from robotics_experiments.rl.press.observation import (
    PRESS_OBSERVATION_DIM,
    PressObservation,
    PressObservationMask,
    to_vector,
)

__all__ = [
    "PressApproachExecutionParams",
    "PressApproachExecutionResult",
    "PressApproachExecutionState",
    "advance_press_approach_execution",
    "initial_press_approach_execution_state",
]


@dataclass(frozen=True)
class PressApproachExecutionParams:
    """Immutable inference, handoff, and stable-goal parameters."""

    inference: Callable[[np.ndarray], np.ndarray]
    action_dim: int
    handoff_gate: HandoffGate
    goal: PressApproachGoalConfig
    flight_parameters: FlightParameters
    handoff_feedback: PressHandoffConfig
    max_body_velocity_mps: float
    max_yaw_rate_radps: float


@dataclass(frozen=True)
class PressApproachExecutionState:
    """Opaque immutable approach state stored by the generic policy step."""

    previous_action: tuple[float, ...]
    handoff_latched: bool = False
    entry_observation: tuple[float, ...] = ()
    actor_evaluations: int = 0
    stable_goal_steps: int = 0
    initial_switch_range_m: float = math.nan
    closest_switch_range_m: float = math.nan
    complete: bool = False
    contact_failure: bool = False


@dataclass(frozen=True)
class PressApproachExecutionResult:
    """Successor state, normalized action, and generic routine outcome."""

    state: PressApproachExecutionState
    action: np.ndarray
    complete: bool
    failure_reason: Optional[str]
    handoff_reason: Optional[str] = None
    actor_observation: tuple[float, ...] = ()
    raw_action: tuple[float, ...] = ()


def initial_press_approach_execution_state(
    *,
    action_dim: int,
) -> PressApproachExecutionState:
    """Create zero-action state for one approach invocation."""
    return PressApproachExecutionState(previous_action=(0.0,) * int(action_dim))


def _invalid_reason(
    *, observation: PressObservation, observation_vector: np.ndarray, action_dim: int
) -> Optional[str]:
    required = int(
        PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
    )
    if int(observation.obs_mask) & required != required:
        return "press approach observation is missing pose, contact, or lamp availability"
    if not observation.detection_valid or not observation.contact_valid:
        return "press approach observation contains an invalid pose or contact sample"
    if observation.lamp_state != 0 or observation.lamp_confidence <= 0.0:
        return "press approach requires the paired lamp to remain off"
    if np.asarray(observation.previous_action).shape != (action_dim,):
        return f"press approach previous_action must have shape ({action_dim},)"
    if observation_vector.shape != (PRESS_OBSERVATION_DIM,) or not np.all(
        np.isfinite(observation_vector)
    ):
        return "press approach observation is non-finite or has the wrong schema dimension"
    return None


def advance_press_approach_execution(
    *,
    params: PressApproachExecutionParams,
    state: PressApproachExecutionState,
    observation: PressObservation,
) -> PressApproachExecutionResult:
    """Advance handoff, inference, and stable approach completion once."""
    observation = replace(
        observation,
        previous_action=np.asarray(state.previous_action, dtype=np.float64),
        handoff_mode=1,
    )
    zero = np.zeros(params.action_dim, dtype=np.float64)
    body_to_world_wxyz = np.asarray(observation.attitude_quat_xyzw)[[3, 0, 1, 2]]
    body_to_contact_wxyz = np.asarray(observation.relative_switch_quat_xyzw)[[3, 0, 1, 2]]
    contact_to_world_wxyz = quat_multiply_wxyz(
        body_to_world_wxyz,
        quat_conjugate_wxyz(body_to_contact_wxyz),
    )
    position_error_contact_m = np.array(
        [params.goal.velocity_reference.range_center_m, 0.0, 0.0],
        dtype=np.float64,
    ) - np.asarray(observation.relative_switch_pos_m, dtype=np.float64)
    position_error_enu_m = rotate_vector_wxyz(
        contact_to_world_wxyz,
        position_error_contact_m,
    )
    observation_vector = to_vector(
        obs=observation,
        position_error_enu_m=position_error_enu_m,
    )
    invalid = _invalid_reason(
        observation=observation,
        observation_vector=observation_vector,
        action_dim=params.action_dim,
    )
    if invalid is not None:
        return PressApproachExecutionResult(state, zero, False, invalid)
    if observation.contact_flag:
        successor = replace(state, contact_failure=True)
        switch_range_m = float(observation.relative_switch_pos_m[0])
        linear_speed_mps = float(np.linalg.norm(observation.body_lin_vel_mps))
        return PressApproachExecutionResult(
            successor,
            zero,
            False,
            "press approach failed because pusher contact was detected "
            f"(range={switch_range_m:.4f} m, speed={linear_speed_mps:.4f} m/s)",
        )

    entry_observation = (
        state.entry_observation
        if state.entry_observation
        else tuple(float(value) for value in observation_vector)
    )
    switch_range_m = float(observation.relative_switch_pos_m[0])
    first_sample = math.isnan(state.initial_switch_range_m)
    initial_switch_range_m = switch_range_m if first_sample else state.initial_switch_range_m
    closest_switch_range_m = (
        switch_range_m if first_sample else min(state.closest_switch_range_m, switch_range_m)
    )
    handoff = handoff_ready(observation, params.handoff_gate, position_error_enu_m)
    handoff_latched = state.handoff_latched or handoff.ready
    action = zero
    actor_observation: tuple[float, ...] = ()
    raw_action: tuple[float, ...] = ()
    if handoff_latched:
        actor_observation = tuple(float(value) for value in observation_vector)
        action = np.asarray(params.inference(observation_vector), dtype=np.float64)
        raw_action = tuple(float(value) for value in action.reshape(-1))
        if action.shape != (params.action_dim,) or not np.all(np.isfinite(action)):
            return PressApproachExecutionResult(
                state,
                zero,
                False,
                f"press approach policy returned invalid action shape {action.shape}",
                actor_observation=actor_observation,
                raw_action=raw_action,
            )
        action = np.clip(action, -1.0, 1.0)
    elif handoff.reason in {
        HandoffReason.POSITION,
        HandoffReason.ATTITUDE,
        HandoffReason.LINEAR_SPEED,
        HandoffReason.ANGULAR_SPEED,
        HandoffReason.TOO_CLOSE,
        HandoffReason.TOO_FAR,
        HandoffReason.LATERAL,
        HandoffReason.WALL_NORMAL_SPEED,
        HandoffReason.LATERAL_SPEED,
    }:
        feedback = params.handoff_feedback
        world_velocity = -feedback.feedback_position_gain_per_s * position_error_enu_m
        body_velocity = rotate_vector_wxyz(quat_conjugate_wxyz(body_to_world_wxyz), world_velocity)
        body_velocity -= feedback.feedback_velocity_damping * observation.body_lin_vel_mps
        body_velocity = np.clip(
            body_velocity, -feedback.feedback_max_velocity_mps, feedback.feedback_max_velocity_mps
        )
        w, x, y, z = body_to_contact_wxyz
        yaw_error = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        action = np.clip(
            np.r_[
                body_velocity / params.max_body_velocity_mps,
                -feedback.feedback_yaw_gain_per_s * yaw_error / params.max_yaw_rate_radps,
            ],
            -1.0,
            1.0,
        )

    envelope = evaluate_hover_envelope(
        flight_parameters=params.flight_parameters,
        position_error_enu_m=position_error_enu_m[None, :],
        attitude_error_flu_enu_wxyz=body_to_contact_wxyz[None, :],
    )
    stable_goal = handoff_latched and bool(envelope.inside[0])
    stable_goal_steps = state.stable_goal_steps + 1 if stable_goal else 0
    complete = stable_goal_steps >= params.goal.stable_steps
    successor = PressApproachExecutionState(
        previous_action=tuple(float(value) for value in (action if handoff_latched else zero)),
        handoff_latched=handoff_latched,
        entry_observation=entry_observation,
        actor_evaluations=state.actor_evaluations + int(bool(actor_observation)),
        stable_goal_steps=stable_goal_steps,
        initial_switch_range_m=initial_switch_range_m,
        closest_switch_range_m=closest_switch_range_m,
        complete=complete,
        contact_failure=False,
    )
    return PressApproachExecutionResult(
        successor,
        zero if complete else action,
        complete,
        None,
        handoff_reason=None if handoff_latched else handoff.reason.value,
        actor_observation=actor_observation,
        raw_action=raw_action,
    )

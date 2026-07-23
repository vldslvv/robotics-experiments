"""Pure deploy-time transition for the press approach curriculum stage."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from robotics_experiments.config import PressApproachGoalConfig
from robotics_experiments.px4_bridge import rotate_vector_wxyz
from robotics_experiments.rl.press.handoff import HandoffGate, handoff_ready
from robotics_experiments.rl.press.observation import (
    PRESS_OBSERVATION_DIM,
    PressObservation,
    PressObservationMask,
    to_vector,
)
from robotics_experiments.rl.press.task import press_approach_goal_satisfied

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


@dataclass(frozen=True)
class PressApproachExecutionState:
    """Opaque immutable approach state stored by the generic policy step."""

    previous_action: tuple[float, ...]
    handoff_latched: bool = False
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


def _invalid_reason(*, observation: PressObservation, action_dim: int) -> Optional[str]:
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
    vector = to_vector(observation)
    if vector.shape != (PRESS_OBSERVATION_DIM,) or not np.all(np.isfinite(vector)):
        return "press approach observation is non-finite or has the wrong schema dimension"
    return None


def _body_z_world_z(observation: PressObservation) -> float:
    x, y = np.asarray(observation.attitude_quat_xyzw, dtype=np.float64)[:2]
    return float(1.0 - 2.0 * (x * x + y * y))


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
    invalid = _invalid_reason(observation=observation, action_dim=params.action_dim)
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

    switch_range_m = float(observation.relative_switch_pos_m[0])
    first_sample = math.isnan(state.initial_switch_range_m)
    initial_switch_range_m = switch_range_m if first_sample else state.initial_switch_range_m
    closest_switch_range_m = (
        switch_range_m if first_sample else min(state.closest_switch_range_m, switch_range_m)
    )
    handoff = handoff_ready(observation, params.handoff_gate)
    handoff_latched = state.handoff_latched or handoff.ready
    action = zero
    actor_observation: tuple[float, ...] = ()
    raw_action: tuple[float, ...] = ()
    if handoff_latched:
        observation_vector = to_vector(observation)
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

    stable_goal = handoff_latched and bool(
        press_approach_goal_satisfied(
            relative_switch_pos_m=np.asarray(observation.relative_switch_pos_m)[None, :],
            linear_velocity_mps=rotate_vector_wxyz(
                np.asarray(observation.relative_switch_quat_xyzw)[[3, 0, 1, 2]],
                np.asarray(observation.body_lin_vel_mps),
            )[None, :],
            angular_velocity_radps=np.asarray(observation.body_ang_vel_radps)[None, :],
            body_z_world_z=np.asarray([_body_z_world_z(observation)]),
            contact_flag=np.asarray([observation.contact_flag]),
            goal_cfg=params.goal,
        )[0]
    )
    stable_goal_steps = state.stable_goal_steps + 1 if stable_goal else 0
    complete = stable_goal_steps >= params.goal.stable_steps
    successor = PressApproachExecutionState(
        previous_action=tuple(float(value) for value in action),
        handoff_latched=handoff_latched,
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

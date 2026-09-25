"""Backend-neutral frozen transition for the press approach curriculum stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import torch

from robotics_experiments.config import (
    PressApproachGoalConfig,
    PressApproachRewardConfig,
    PressApproachTerminationConfig,
    PressApproachVelocityReferenceConfig,
)
from robotics_experiments.flight_parameters import FlightParameters
from robotics_experiments.rl.hover import (
    HoverEnvelopeEvaluation,
    evaluate_hover_envelope,
)

__all__ = [
    "PressApproachInput",
    "PressApproachResult",
    "PressApproachRewardTerms",
    "PressApproachState",
    "PressApproachTerminationFlags",
    "initial_press_approach_state",
    "press_approach_target_velocity",
    "quaternion_geodesic_error_rad",
    "reset_press_approach_state",
    "step_press_approach",
]

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)
Scalar: TypeAlias = bool | int | float
EnvironmentIndices: TypeAlias = Sequence[int] | np.ndarray | torch.Tensor


@dataclass(frozen=True)
class PressApproachState(Generic[ArrayT]):
    """Batched approach memory.

    Every field has shape ``[N]``.
    """

    normalized_position_error: ArrayT
    normalized_pose_error: ArrayT
    stable_goal_steps: ArrayT
    longest_stable_goal_steps: ArrayT
    closest_range_m: ArrayT


@dataclass(frozen=True)
class PressApproachInput(Generic[ArrayT]):
    """One batched physical sample consumed by :func:`step_press_approach`.

    Collision flags are privileged training facts latched across the preceding
    policy interval. They are not part of :class:`PressApproachState` or the
    policy observation.
    """

    relative_switch_pos_m: ArrayT
    relative_switch_quat_xyzw: ArrayT
    position_error_enu_m: ArrayT
    attitude_error_flu_enu_wxyz: ArrayT
    linear_velocity_mps: ArrayT
    angular_velocity_radps: ArrayT
    normalized_action: ArrayT
    body_z_world_z: ArrayT
    altitude_m: ArrayT
    body_collision_flag: ArrayT
    pusher_collision_flag: ArrayT
    time_out: ArrayT


@dataclass(frozen=True)
class PressApproachRewardTerms(Generic[ArrayT]):
    """The signed per-step Stage 1 reward contributions."""

    position_progress: ArrayT
    stabilization: ArrayT
    predictive_barrier: ArrayT
    action_effort: ArrayT
    hold_progress: ArrayT
    terminal_outcome: ArrayT

    @property
    def total(self) -> ArrayT:
        """Sum every named reward contribution."""
        return (
            self.position_progress
            + self.stabilization
            + self.predictive_barrier
            + self.action_effort
            + self.hold_progress
            + self.terminal_outcome
        )


@dataclass(frozen=True)
class PressApproachTerminationFlags(Generic[ArrayT]):
    """Per-step success, failure, and truncation facts."""

    success: ArrayT
    body_collision: ArrayT
    pusher_collision: ArrayT
    crash: ArrayT
    altitude: ArrayT
    range_undershoot: ArrayT
    relative_envelope: ArrayT
    time_out: ArrayT
    terminated: ArrayT

    @property
    def flight_failure(self) -> ArrayT:
        """Return raw flight-parameters failures, including collision-coincident ones."""
        return self.crash | self.altitude | self.range_undershoot | self.relative_envelope

    @property
    def collision(self) -> ArrayT:
        """Return the combined privileged collision fact."""
        return self.body_collision | self.pusher_collision


@dataclass(frozen=True)
class PressApproachResult(Generic[ArrayT]):
    """Successor state plus reward and termination records."""

    state: PressApproachState[ArrayT]
    rewards: PressApproachRewardTerms[ArrayT]
    termination: PressApproachTerminationFlags[ArrayT]
    range_m: ArrayT
    range_error_m: ArrayT
    alignment_error_m: ArrayT
    attitude_error_rad: ArrayT
    linear_speed_mps: ArrayT
    angular_speed_radps: ArrayT
    contact_clear: ArrayT
    strict_goal: ArrayT
    hover_evaluation: HoverEnvelopeEvaluation[ArrayT]


def _where(condition: ArrayT, yes: ArrayT | Scalar, no: ArrayT | Scalar) -> ArrayT:
    if isinstance(condition, torch.Tensor):
        return torch.where(condition, yes, no)
    return np.where(condition, yes, no)


def _zeros_like(*, value: ArrayT, integer: bool = False) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value, dtype=torch.long if integer else value.dtype)
    return np.zeros_like(value, dtype=np.int64 if integer else None)


def _norm(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.linalg.norm(value, dim=-1)
    return np.linalg.norm(value, axis=-1)


def _square(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.square()
    return np.square(value)


def _mean_last(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.mean(dim=-1)
    return value.mean(axis=-1)


def _sqrt(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.sqrt(value)
    return np.sqrt(value)


def _minimum(left: ArrayT, right: ArrayT) -> ArrayT:
    if isinstance(left, torch.Tensor):
        return torch.minimum(left, right)
    return np.minimum(left, right)


def _maximum(left: ArrayT, right: Scalar) -> ArrayT:
    if isinstance(left, torch.Tensor):
        return left.clamp_min(right)
    return np.maximum(left, right)


def _pairwise_maximum(left: ArrayT, right: ArrayT) -> ArrayT:
    if isinstance(left, torch.Tensor):
        return torch.maximum(left, right)
    return np.maximum(left, right)


def quaternion_geodesic_error_rad(*, quaternion_xyzw: ArrayT) -> ArrayT:
    """Return the full rotation error to identity for normalized quaternion batches."""
    norm = _norm(quaternion_xyzw)
    scalar = abs(quaternion_xyzw[..., 3] / norm)
    if isinstance(quaternion_xyzw, torch.Tensor):
        return 2.0 * torch.acos(scalar.clamp(0.0, 1.0))
    return 2.0 * np.arccos(np.clip(scalar, 0.0, 1.0))


def initial_press_approach_state(
    *,
    relative_switch_pos_m: ArrayT,
    relative_switch_quat_xyzw: ArrayT,
    goal_cfg: PressApproachGoalConfig,
    flight_parameters: FlightParameters,
    position_error_enu_m: ArrayT,
    attitude_error_flu_enu_wxyz: ArrayT,
) -> PressApproachState[ArrayT]:
    """Create zero progress matching a NumPy or torch batch."""
    normalized_position_error = _press_approach_position_reward_error(
        relative_switch_pos_m=relative_switch_pos_m,
        goal_cfg=goal_cfg,
    )
    hover_evaluation = evaluate_hover_envelope(
        flight_parameters=flight_parameters,
        position_error_enu_m=position_error_enu_m,
        attitude_error_flu_enu_wxyz=attitude_error_flu_enu_wxyz,
    )
    normalized_pose_error = _pairwise_maximum(
        hover_evaluation.position_utilization, hover_evaluation.attitude_utilization
    )
    return PressApproachState(
        normalized_position_error=normalized_position_error,
        normalized_pose_error=normalized_pose_error,
        stable_goal_steps=_zeros_like(value=normalized_pose_error, integer=True),
        longest_stable_goal_steps=_zeros_like(value=normalized_pose_error, integer=True),
        closest_range_m=relative_switch_pos_m[..., 0],
    )


def reset_press_approach_state(
    *,
    state: PressApproachState[ArrayT],
    env_ids: EnvironmentIndices,
    relative_switch_pos_m: ArrayT,
    relative_switch_quat_xyzw: ArrayT,
    goal_cfg: PressApproachGoalConfig,
    flight_parameters: FlightParameters,
    position_error_enu_m: ArrayT,
    attitude_error_flu_enu_wxyz: ArrayT,
) -> PressApproachState[ArrayT]:
    """Return ``state`` with only the selected environments reset."""
    fresh = initial_press_approach_state(
        relative_switch_pos_m=relative_switch_pos_m,
        relative_switch_quat_xyzw=relative_switch_quat_xyzw,
        goal_cfg=goal_cfg,
        flight_parameters=flight_parameters,
        position_error_enu_m=position_error_enu_m,
        attitude_error_flu_enu_wxyz=attitude_error_flu_enu_wxyz,
    )

    reset_fields = {}
    for field in PressApproachState.__dataclass_fields__:
        current = getattr(state, field)
        successor = current.clone() if isinstance(current, torch.Tensor) else current.copy()
        successor[env_ids] = getattr(fresh, field)
        reset_fields[field] = successor
    return PressApproachState(**reset_fields)


def press_approach_target_velocity(
    *,
    relative_switch_pos_m: ArrayT,
    linear_velocity_mps: ArrayT,
    reference_cfg: PressApproachVelocityReferenceConfig,
) -> ArrayT:
    """Return the predictive clipped contact-frame approach velocity."""
    if isinstance(relative_switch_pos_m, torch.Tensor):
        projected_range = relative_switch_pos_m[..., 0] - (
            reference_cfg.wall_normal_prediction_horizon_s
            * linear_velocity_mps[..., 0].clamp_min(0.0)
        )
        wall_normal = (
            reference_cfg.wall_normal_gain_per_s * (projected_range - reference_cfg.range_center_m)
        ).clamp(
            reference_cfg.wall_normal_min_mps,
            reference_cfg.wall_normal_max_mps,
        )
        alignment_error = torch.linalg.vector_norm(relative_switch_pos_m[..., 1:3], dim=-1)
        wall_normal = torch.where(
            (alignment_error <= reference_cfg.wall_normal_alignment_gate_m) | (wall_normal <= 0.0),
            wall_normal,
            torch.zeros_like(wall_normal),
        )
        tangential = reference_cfg.tangential_gain_per_s * relative_switch_pos_m[..., 1:3]
        tangential_norm = torch.linalg.norm(tangential, dim=-1)
        denominator = tangential_norm.clamp_min(reference_cfg.tangential_speed_max_mps)
        scale = reference_cfg.tangential_speed_max_mps / denominator
        return torch.cat((wall_normal[..., None], tangential * scale[..., None]), dim=-1)
    projected_range = relative_switch_pos_m[..., 0] - (
        reference_cfg.wall_normal_prediction_horizon_s
        * np.maximum(linear_velocity_mps[..., 0], 0.0)
    )
    wall_normal = np.clip(
        reference_cfg.wall_normal_gain_per_s * (projected_range - reference_cfg.range_center_m),
        reference_cfg.wall_normal_min_mps,
        reference_cfg.wall_normal_max_mps,
    )
    alignment_error = np.linalg.norm(relative_switch_pos_m[..., 1:3], axis=-1)
    wall_normal = np.where(
        (alignment_error <= reference_cfg.wall_normal_alignment_gate_m) | (wall_normal <= 0.0),
        wall_normal,
        np.zeros_like(wall_normal),
    )
    tangential = reference_cfg.tangential_gain_per_s * relative_switch_pos_m[..., 1:3]
    tangential_norm = np.linalg.norm(tangential, axis=-1)
    denominator = np.maximum(
        tangential_norm,
        reference_cfg.tangential_speed_max_mps,
    )
    scale = reference_cfg.tangential_speed_max_mps / denominator
    return np.concatenate(
        (wall_normal[..., None], tangential * scale[..., None]),
        axis=-1,
    )


def _press_approach_position_components(
    *,
    relative_switch_pos_m: ArrayT,
    goal_cfg: PressApproachGoalConfig,
) -> tuple[ArrayT, ArrayT]:
    range_center = 0.5 * (goal_cfg.range_min_m + goal_cfg.range_max_m)
    range_tolerance = 0.5 * (goal_cfg.range_max_m - goal_cfg.range_min_m)
    range_error = abs(relative_switch_pos_m[..., 0] - range_center) / range_tolerance
    alignment_error = _norm(relative_switch_pos_m[..., 1:3]) / goal_cfg.alignment_error_max_m
    return range_error, alignment_error


def _press_approach_position_reward_error(
    *,
    relative_switch_pos_m: ArrayT,
    goal_cfg: PressApproachGoalConfig,
) -> ArrayT:
    range_error, alignment_error = _press_approach_position_components(
        relative_switch_pos_m=relative_switch_pos_m,
        goal_cfg=goal_cfg,
    )
    return _sqrt(0.5 * (_square(range_error) + _square(alignment_error)))


def step_press_approach(
    *,
    state: PressApproachState[ArrayT],
    sample: PressApproachInput[ArrayT],
    goal_cfg: PressApproachGoalConfig,
    reward_cfg: PressApproachRewardConfig,
    termination_cfg: PressApproachTerminationConfig,
    minimum_range_m: float,
    flight_parameters: FlightParameters,
    hold_steps: int,
    approach_enabled: bool,
    training: bool,
) -> PressApproachResult[ArrayT]:
    """Advance pose costs and hold qualification once.

    Hover training continues after hold completion; evaluation and approach
    terminate on success. Physical failures always terminate.
    """
    current_range = sample.relative_switch_pos_m[..., 0]
    alignment_error = _norm(sample.relative_switch_pos_m[..., 1:3])
    relative_attitude_error = quaternion_geodesic_error_rad(
        quaternion_xyzw=sample.relative_switch_quat_xyzw
    )
    linear_speed = _norm(sample.linear_velocity_mps)
    angular_speed = _norm(sample.angular_velocity_radps)
    envelope_evaluation = evaluate_hover_envelope(
        flight_parameters=flight_parameters,
        position_error_enu_m=sample.position_error_enu_m,
        attitude_error_flu_enu_wxyz=sample.attitude_error_flu_enu_wxyz,
    )
    normalized_pose_error = _pairwise_maximum(
        envelope_evaluation.position_utilization,
        envelope_evaluation.attitude_utilization,
    )
    normalized_position_error = _press_approach_position_reward_error(
        relative_switch_pos_m=sample.relative_switch_pos_m,
        goal_cfg=goal_cfg,
    )
    position_ready = envelope_evaluation.position_utilization <= 1.0

    stable_goal = envelope_evaluation.inside & ~(
        sample.pusher_collision_flag | sample.body_collision_flag
    )
    stable_goal_steps = _where(
        stable_goal,
        state.stable_goal_steps + 1,
        _zeros_like(value=state.stable_goal_steps),
    )
    longest_stable_goal_steps = _maximum(
        state.longest_stable_goal_steps,
        stable_goal_steps,
    )
    hold_progress_steps = longest_stable_goal_steps - state.longest_stable_goal_steps

    crash = sample.body_z_world_z < termination_cfg.crash_tilt_cos
    altitude = sample.altitude_m < termination_cfg.min_altitude_m
    range_undershoot = current_range < minimum_range_m
    relative_envelope = (
        (abs(current_range) > termination_cfg.max_wall_normal_distance_m)
        | (abs(sample.relative_switch_pos_m[..., 1]) > termination_cfg.max_lateral_error_m)
        | (abs(sample.relative_switch_pos_m[..., 2]) > termination_cfg.max_vertical_error_m)
    )
    flight_failure = crash | altitude | range_undershoot | relative_envelope
    collision = sample.body_collision_flag | sample.pusher_collision_flag
    ordinary_flight_failure = flight_failure & ~collision
    failure = collision | ordinary_flight_failure
    success = (stable_goal_steps >= hold_steps) & ~failure & (approach_enabled or not training)
    terminated = success | failure
    time_out = sample.time_out & ~terminated
    termination = PressApproachTerminationFlags(
        success=success,
        body_collision=sample.body_collision_flag,
        pusher_collision=sample.pusher_collision_flag,
        crash=crash,
        altitude=altitude,
        range_undershoot=range_undershoot,
        relative_envelope=relative_envelope,
        time_out=time_out,
        terminated=terminated,
    )

    unsuccessful_termination = failure | time_out
    predicted_range = current_range - (
        goal_cfg.velocity_reference.wall_normal_prediction_horizon_s
        * _maximum(sample.linear_velocity_mps[..., 0], 0.0)
    )
    normalized_predicted_undershoot = _maximum(
        goal_cfg.range_min_m - predicted_range,
        0.0,
    ) / (goal_cfg.range_max_m - goal_cfg.range_min_m)
    position_progress = _where(
        position_ready,
        _zeros_like(value=normalized_position_error),
        reward_cfg.position_progress_weight
        * (state.normalized_position_error - normalized_position_error),
    )
    if not approach_enabled:
        position_progress = _zeros_like(value=normalized_pose_error)
    stabilization = _where(
        position_ready,
        reward_cfg.stabilization_weight / (1.0 + normalized_pose_error),
        _zeros_like(value=normalized_position_error),
    )
    predictive_barrier = -reward_cfg.predictive_barrier_weight * _square(
        normalized_predicted_undershoot
    )
    if not approach_enabled:
        pose_utilization = (
            envelope_evaluation.position_utilization + envelope_evaluation.attitude_utilization
        )
        if reward_cfg.hover_stabilization == "outside_only":
            stabilization_weight: ArrayT | Scalar = _where(
                envelope_evaluation.inside,
                0.0,
                reward_cfg.outside_envelope_weight,
            )
        elif reward_cfg.hover_stabilization == "uniform":
            stabilization_weight = reward_cfg.stabilization_weight
        elif reward_cfg.hover_stabilization == "boundary_weighted":
            stabilization_weight = _where(
                envelope_evaluation.inside,
                reward_cfg.stabilization_weight,
                reward_cfg.outside_envelope_weight,
            )
        else:
            raise ValueError(
                "unsupported reward_cfg.hover_stabilization: " f"{reward_cfg.hover_stabilization!r}"
            )
        stabilization = -stabilization_weight * pose_utilization
        predictive_barrier = _zeros_like(value=normalized_position_error)
    action_effort = _zeros_like(value=normalized_position_error)
    if not approach_enabled:
        action_effort = -reward_cfg.action_effort_weight * _mean_last(
            _square(sample.normalized_action)
        )
    rewards = PressApproachRewardTerms(
        position_progress=position_progress,
        stabilization=stabilization,
        predictive_barrier=predictive_barrier,
        action_effort=action_effort,
        hold_progress=(
            reward_cfg.hold_progress_weight * hold_progress_steps
            if approach_enabled
            else _zeros_like(value=normalized_pose_error)
        ),
        terminal_outcome=(
            reward_cfg.terminal_success_weight * success
            - reward_cfg.terminal_failure_weight * unsuccessful_termination
            if approach_enabled
            else -reward_cfg.terminal_failure_weight * failure
        ),
    )
    successor = PressApproachState(
        normalized_position_error=normalized_position_error,
        normalized_pose_error=normalized_pose_error,
        stable_goal_steps=stable_goal_steps,
        longest_stable_goal_steps=longest_stable_goal_steps,
        closest_range_m=_minimum(state.closest_range_m, current_range),
    )
    return PressApproachResult(
        state=successor,
        rewards=rewards,
        termination=termination,
        range_m=current_range,
        range_error_m=abs(current_range - 0.5 * (goal_cfg.range_min_m + goal_cfg.range_max_m)),
        alignment_error_m=alignment_error,
        attitude_error_rad=relative_attitude_error,
        linear_speed_mps=linear_speed,
        angular_speed_radps=angular_speed,
        contact_clear=~collision,
        strict_goal=stable_goal,
        hover_evaluation=envelope_evaluation,
    )

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

__all__ = [
    "PressApproachInput",
    "PressApproachResult",
    "PressApproachRewardTerms",
    "PressApproachState",
    "PressApproachTerminationFlags",
    "initial_press_approach_state",
    "press_approach_goal_satisfied",
    "press_approach_target_velocity",
    "reset_press_approach_state",
    "step_press_approach",
]

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)
Scalar: TypeAlias = bool | int | float
EnvironmentIndices: TypeAlias = Sequence[int] | np.ndarray | torch.Tensor


@dataclass(frozen=True)
class PressApproachState(Generic[ArrayT]):
    """Batched approach memory.

    Scalar fields have shape ``[N]`` and ``previous_action`` has shape
    ``[N, action_dim]``.
    """

    previous_range_m: ArrayT
    previous_action: ArrayT
    stable_goal_steps: ArrayT
    closest_range_m: ArrayT


@dataclass(frozen=True)
class PressApproachInput(Generic[ArrayT]):
    """One batched physical sample consumed by :func:`step_press_approach`.

    Collision flags are privileged training facts latched across the preceding
    policy interval. They are not part of :class:`PressApproachState` or the
    policy observation.
    """

    relative_switch_pos_m: ArrayT
    linear_velocity_mps: ArrayT
    wall_normal_velocity_command_mps: ArrayT
    angular_velocity_radps: ArrayT
    body_z_world_z: ArrayT
    altitude_m: ArrayT
    body_collision_flag: ArrayT
    pusher_collision_flag: ArrayT
    action: ArrayT
    time_out: ArrayT


@dataclass(frozen=True)
class PressApproachRewardTerms(Generic[ArrayT]):
    """The eleven signed per-step Stage 1 reward contributions."""

    range_progress: ArrayT
    range_barrier: ArrayT
    alignment_error: ArrayT
    velocity_tracking: ArrayT
    wall_normal_command_tracking: ArrayT
    upright_error: ArrayT
    action_delta: ArrayT
    success: ArrayT
    collision: ArrayT
    flight_failure: ArrayT
    timeout: ArrayT

    @property
    def total(self) -> ArrayT:
        """Sum every named reward contribution."""
        return (
            self.range_progress
            + self.range_barrier
            + self.alignment_error
            + self.velocity_tracking
            + self.wall_normal_command_tracking
            + self.upright_error
            + self.action_delta
            + self.success
            + self.collision
            + self.flight_failure
            + self.timeout
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
        """Return raw flight-envelope failures, including collision-coincident ones."""
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


def _mean_square(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.square().mean(dim=-1)
    return np.square(value).mean(axis=-1)


def _sum_square(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.square().sum(dim=-1)
    return np.square(value).sum(axis=-1)


def _square(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.square()
    return np.square(value)


def _clip(value: ArrayT, minimum: float, maximum: float) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.clamp(minimum, maximum)
    return np.clip(value, minimum, maximum)


def _minimum(left: ArrayT, right: ArrayT) -> ArrayT:
    if isinstance(left, torch.Tensor):
        return torch.minimum(left, right)
    return np.minimum(left, right)


def initial_press_approach_state(
    *,
    relative_switch_range_m: ArrayT,
    action: ArrayT,
) -> PressApproachState[ArrayT]:
    """Create zero progress matching a NumPy or torch batch."""
    return PressApproachState(
        previous_range_m=relative_switch_range_m,
        previous_action=_zeros_like(value=action),
        stable_goal_steps=_zeros_like(value=relative_switch_range_m, integer=True),
        closest_range_m=relative_switch_range_m,
    )


def reset_press_approach_state(
    *,
    state: PressApproachState[ArrayT],
    env_ids: EnvironmentIndices,
    relative_switch_range_m: ArrayT,
    action: ArrayT,
) -> PressApproachState[ArrayT]:
    """Return ``state`` with only the selected environments reset."""
    fresh = initial_press_approach_state(
        relative_switch_range_m=relative_switch_range_m,
        action=action,
    )

    def replaced(current: ArrayT, replacement: ArrayT) -> ArrayT:
        if isinstance(current, torch.Tensor):
            successor = current.clone()
        else:
            successor = current.copy()
        successor[env_ids] = replacement
        return successor

    return PressApproachState(
        **{
            field: replaced(getattr(state, field), getattr(fresh, field))
            for field in PressApproachState.__dataclass_fields__
        }
    )


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


def press_approach_goal_satisfied(
    *,
    relative_switch_pos_m: ArrayT,
    linear_velocity_mps: ArrayT,
    angular_velocity_radps: ArrayT,
    body_z_world_z: ArrayT,
    contact_flag: ArrayT,
    goal_cfg: PressApproachGoalConfig,
) -> ArrayT:
    """Return the shared per-sample stable-approach predicate."""
    current_range = relative_switch_pos_m[..., 0]
    return (
        (current_range >= goal_cfg.range_min_m)
        & (current_range <= goal_cfg.range_max_m)
        & (_norm(relative_switch_pos_m[..., 1:3]) <= goal_cfg.alignment_error_max_m)
        & (_norm(linear_velocity_mps) <= goal_cfg.linear_speed_max_mps)
        & (_norm(angular_velocity_radps) <= goal_cfg.angular_speed_max_radps)
        & (_clip(body_z_world_z, 0.0, 1.0) >= goal_cfg.body_z_alignment_min)
        & ~contact_flag
    )


def step_press_approach(
    *,
    state: PressApproachState[ArrayT],
    sample: PressApproachInput[ArrayT],
    goal_cfg: PressApproachGoalConfig,
    reward_cfg: PressApproachRewardConfig,
    termination_cfg: PressApproachTerminationConfig,
) -> PressApproachResult[ArrayT]:
    """Advance the stable, contact-free approach objective exactly once."""
    current_range = sample.relative_switch_pos_m[..., 0]
    alignment_error = _norm(sample.relative_switch_pos_m[..., 1:3])
    upright_alignment = _clip(sample.body_z_world_z, 0.0, 1.0)
    range_barrier_start = goal_cfg.range_min_m + reward_cfg.range_barrier_margin_m
    range_barrier_deficit = _clip(
        (range_barrier_start - current_range) / reward_cfg.range_barrier_margin_m,
        0.0,
        float("inf"),
    )
    target_velocity = press_approach_target_velocity(
        relative_switch_pos_m=sample.relative_switch_pos_m,
        linear_velocity_mps=sample.linear_velocity_mps,
        reference_cfg=goal_cfg.velocity_reference,
    )

    stable_goal = press_approach_goal_satisfied(
        relative_switch_pos_m=sample.relative_switch_pos_m,
        linear_velocity_mps=sample.linear_velocity_mps,
        angular_velocity_radps=sample.angular_velocity_radps,
        body_z_world_z=sample.body_z_world_z,
        contact_flag=sample.pusher_collision_flag,
        goal_cfg=goal_cfg,
    )
    stable_goal_steps = _where(
        stable_goal,
        state.stable_goal_steps + 1,
        _zeros_like(value=state.stable_goal_steps),
    )

    crash = sample.body_z_world_z < termination_cfg.crash_tilt_cos
    altitude = sample.altitude_m < termination_cfg.min_altitude_m
    range_undershoot = current_range < goal_cfg.range_min_m
    relative_envelope = (
        (abs(current_range) > termination_cfg.max_wall_normal_distance_m)
        | (abs(sample.relative_switch_pos_m[..., 1]) > termination_cfg.max_lateral_error_m)
        | (abs(sample.relative_switch_pos_m[..., 2]) > termination_cfg.max_vertical_error_m)
    )
    flight_failure = crash | altitude | range_undershoot | relative_envelope
    collision = sample.body_collision_flag | sample.pusher_collision_flag
    ordinary_flight_failure = flight_failure & ~collision
    failure = collision | ordinary_flight_failure
    success = (stable_goal_steps >= goal_cfg.stable_steps) & ~failure
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

    rewards = PressApproachRewardTerms(
        range_progress=(
            reward_cfg.range_progress_weight * (state.previous_range_m - current_range)
        ),
        range_barrier=(-reward_cfg.range_barrier_weight * _square(range_barrier_deficit)),
        alignment_error=(-reward_cfg.alignment_error_weight * _square(alignment_error)),
        velocity_tracking=(
            -reward_cfg.velocity_tracking_weight
            * _sum_square(sample.linear_velocity_mps - target_velocity)
        ),
        wall_normal_command_tracking=(
            -reward_cfg.wall_normal_command_tracking_weight
            * _square(sample.wall_normal_velocity_command_mps - target_velocity[..., 0])
        ),
        upright_error=(-reward_cfg.upright_error_weight * _square(1.0 - upright_alignment)),
        action_delta=(
            -reward_cfg.action_delta_weight * _mean_square(sample.action - state.previous_action)
        ),
        success=reward_cfg.success_weight * success,
        collision=-reward_cfg.collision_weight * collision,
        flight_failure=-reward_cfg.flight_failure_weight * ordinary_flight_failure,
        timeout=-reward_cfg.timeout_weight * time_out,
    )
    successor = PressApproachState(
        previous_range_m=current_range,
        previous_action=sample.action,
        stable_goal_steps=stable_goal_steps,
        closest_range_m=_minimum(state.closest_range_m, current_range),
    )
    return PressApproachResult(
        state=successor,
        rewards=rewards,
        termination=termination,
    )

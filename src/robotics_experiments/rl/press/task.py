"""Backend-neutral frozen transition for the ground-truth press task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import torch

from robotics_experiments.config import (
    PressEvaluationConfig,
    PressRewardConfig,
    PressTerminationConfig,
)

__all__ = [
    "PressRewardTerms",
    "PressStepMetrics",
    "PressTaskInput",
    "PressTaskPhase",
    "PressTaskResult",
    "PressTaskState",
    "PressTerminationFlags",
    "initial_press_task_state",
    "reset_press_task_state",
    "step_press_task",
]

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)
Scalar: TypeAlias = bool | int | float
EnvironmentIndices: TypeAlias = Sequence[int] | np.ndarray | torch.Tensor


class PressTaskPhase(IntEnum):
    """Press-task progression values stored in batched integer arrays."""

    APPROACH = 0
    CONTACT = 1
    RETREAT = 2


@dataclass(frozen=True)
class PressTaskState(Generic[ArrayT]):
    """Batched task memory.

    Scalar fields have shape ``[N]`` and ``previous_action`` has shape
    ``[N, action_dim]``. Ranges, positions, and compression are meters; time is
    seconds. Contact/toggle fields and peak compression are cumulative within
    an episode, while ``previous_*`` fields retain the immediately preceding
    policy step.
    """

    press_steps: ArrayT
    toggle_latched: ArrayT
    lamp_state: ArrayT
    phase: ArrayT
    initial_range_m: ArrayT
    previous_range_m: ArrayT
    previous_button_position_m: ArrayT
    previous_compression_m: ArrayT
    previous_contact_flag: ArrayT
    previous_action: ArrayT
    first_contact_seen: ArrayT
    first_contact_stable: ArrayT
    repeat_toggle_seen: ArrayT
    toggle_time_s: ArrayT
    peak_compression_m: ArrayT
    retreat_stable_steps: ArrayT
    misalignment_steps: ArrayT


@dataclass(frozen=True)
class PressTaskInput(Generic[ArrayT]):
    """One batched physical sample consumed by :func:`step_press_task`.

    Scalar fields have shape ``[N]``; position and velocity vectors have shape
    ``[N, 3]``; normalized actions have shape ``[N, action_dim]``. Position,
    altitude, button travel, and compression are meters; velocities are m/s or
    rad/s; elapsed time is seconds. Every field describes the current policy
    step, not cumulative episode state.
    """

    relative_switch_pos_m: ArrayT
    body_linear_velocity_mps: ArrayT
    body_angular_velocity_radps: ArrayT
    body_z_world_z: ArrayT
    altitude_m: ArrayT
    pusher_compression_m: ArrayT
    button_position_m: ArrayT
    contact_flag: ArrayT
    action: ArrayT
    elapsed_s: ArrayT
    time_out: ArrayT


@dataclass(frozen=True)
class PressRewardTerms(Generic[ArrayT]):
    """Batched signed per-step reward contributions, each with shape ``[N]``."""

    alive: ArrayT
    stability: ArrayT
    approach: ArrayT
    near_contact_linear_overspeed: ArrayT
    near_contact_angular_overspeed: ArrayT
    gentle_contact: ArrayT
    compression_progress: ArrayT
    button_progress: ArrayT
    latch_progress: ArrayT
    toggle: ArrayT
    repeat_toggle: ArrayT
    retreat: ArrayT
    retreat_alignment: ArrayT
    retreat_overspeed: ArrayT
    post_toggle_contact: ArrayT
    post_toggle_delay: ArrayT
    success: ArrayT
    action_energy: ArrayT
    action_delta: ArrayT
    contact_oscillation: ArrayT
    scraping: ArrayT
    misalignment: ArrayT
    failure: ArrayT
    timeout: ArrayT

    @property
    def total(self) -> ArrayT:
        """Sum every named reward component."""
        return (
            self.alive
            + self.stability
            + self.approach
            + self.near_contact_linear_overspeed
            + self.near_contact_angular_overspeed
            + self.gentle_contact
            + self.compression_progress
            + self.button_progress
            + self.latch_progress
            + self.toggle
            + self.repeat_toggle
            + self.retreat
            + self.retreat_alignment
            + self.retreat_overspeed
            + self.post_toggle_contact
            + self.post_toggle_delay
            + self.success
            + self.action_energy
            + self.action_delta
            + self.contact_oscillation
            + self.scraping
            + self.misalignment
            + self.failure
            + self.timeout
        )


@dataclass(frozen=True)
class PressTerminationFlags(Generic[ArrayT]):
    """Batched per-step outcome flags, each with shape ``[N]``.

    ``terminated`` contains success and non-timeout failures. ``time_out`` is
    kept distinct for the environment's truncation path.
    """

    success: ArrayT
    crash: ArrayT
    altitude: ArrayT
    relative_envelope: ArrayT
    time_out: ArrayT
    terminated: ArrayT


@dataclass(frozen=True)
class PressStepMetrics(Generic[ArrayT]):
    """Batched evaluator facts.

    ``toggled_this_step`` is a per-step event. The remaining boolean values are
    cumulative episode facts. Toggle time is seconds and peak compression
    overshoot is meters.
    """

    toggled_this_step: ArrayT
    first_contact_stable: ArrayT
    toggle_time_s: ArrayT
    peak_compression_overshoot_m: ArrayT
    contacted: ArrayT
    toggled: ArrayT
    repeat_toggled: ArrayT
    retreated: ArrayT


@dataclass(frozen=True)
class PressTaskResult(Generic[ArrayT]):
    """Successor state plus batched reward, termination, and metric records."""

    state: PressTaskState[ArrayT]
    rewards: PressRewardTerms[ArrayT]
    termination: PressTerminationFlags[ArrayT]
    metrics: PressStepMetrics[ArrayT]


def _where(condition: ArrayT, yes: ArrayT | Scalar, no: ArrayT | Scalar) -> ArrayT:
    if isinstance(condition, torch.Tensor):
        return torch.where(condition, yes, no)
    return np.where(condition, yes, no)


def _zeros_like(*, value: ArrayT, boolean: bool = False, integer: bool = False) -> ArrayT:
    if isinstance(value, torch.Tensor):
        dtype = torch.bool if boolean else torch.long if integer else value.dtype
        return torch.zeros_like(value, dtype=dtype)
    dtype = bool if boolean else np.int64 if integer else None
    return np.zeros_like(value, dtype=dtype)


def _full_like(*, value: ArrayT, fill: float) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.full_like(value, fill, dtype=value.dtype)
    return np.full_like(value, fill)


def _norm(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.linalg.norm(value, dim=-1)
    return np.linalg.norm(value, axis=-1)


def _mean_square(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.square().mean(dim=-1)
    return np.square(value).mean(axis=-1)


def _square(value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.square()
    return np.square(value)


def _maximum(value: ArrayT, minimum: float) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.clamp_min(minimum)
    return np.maximum(value, minimum)


def _minimum(value: ArrayT, maximum: float) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.clamp_max(maximum)
    return np.minimum(value, maximum)


def _elementwise_minimum(*, left: ArrayT, right: ArrayT) -> ArrayT:
    if isinstance(left, torch.Tensor):
        return torch.minimum(left, right)
    return np.minimum(left, right)


def initial_press_task_state(
    *, relative_switch_range_m: ArrayT, action: ArrayT
) -> PressTaskState[ArrayT]:
    """Create zero progress matching a NumPy or torch batch."""
    zeros = _zeros_like(value=relative_switch_range_m)
    false = _zeros_like(value=relative_switch_range_m, boolean=True)
    integers = _zeros_like(value=relative_switch_range_m, integer=True)
    return PressTaskState(
        press_steps=integers,
        toggle_latched=false,
        lamp_state=false,
        phase=integers,
        initial_range_m=relative_switch_range_m,
        previous_range_m=relative_switch_range_m,
        previous_button_position_m=zeros,
        previous_compression_m=zeros,
        previous_contact_flag=false,
        previous_action=_zeros_like(value=action),
        first_contact_seen=false,
        first_contact_stable=false,
        repeat_toggle_seen=false,
        toggle_time_s=_full_like(value=relative_switch_range_m, fill=-1.0),
        peak_compression_m=zeros,
        retreat_stable_steps=integers,
        misalignment_steps=integers,
    )


def reset_press_task_state(
    *,
    state: PressTaskState[ArrayT],
    env_ids: EnvironmentIndices,
    relative_switch_range_m: ArrayT,
    action: ArrayT,
) -> PressTaskState[ArrayT]:
    """Return ``state`` with only the selected environments reset."""
    fresh = initial_press_task_state(
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

    return PressTaskState(
        **{
            field: replaced(getattr(state, field), getattr(fresh, field))
            for field in PressTaskState.__dataclass_fields__
        }
    )


def _advance_latch(
    *,
    state: PressTaskState[ArrayT],
    button_position_m: ArrayT,
    press_threshold_m: float,
    confirm_steps: int,
) -> tuple[ArrayT, ArrayT, ArrayT, ArrayT]:
    """Vectorized twin of ``environment.apply_button_sample``."""
    pressed = button_position_m >= press_threshold_m
    press_steps = _where(
        pressed,
        state.press_steps + 1,
        _zeros_like(value=state.press_steps),
    )
    toggled = pressed & (press_steps >= confirm_steps) & ~state.toggle_latched
    toggle_latched = _where(pressed, state.toggle_latched | toggled, False)
    lamp_state = state.lamp_state ^ toggled
    return press_steps, toggle_latched, lamp_state, toggled


def _termination(
    *,
    sample: PressTaskInput[ArrayT],
    success: ArrayT,
    termination_cfg: PressTerminationConfig,
) -> PressTerminationFlags[ArrayT]:
    relative = sample.relative_switch_pos_m
    crash = sample.body_z_world_z < termination_cfg.crash_tilt_cos
    altitude = sample.altitude_m < termination_cfg.min_altitude_m
    relative_envelope = (
        (abs(relative[..., 0]) > termination_cfg.max_wall_normal_distance_m)
        | (abs(relative[..., 1]) > termination_cfg.max_lateral_error_m)
        | (abs(relative[..., 2]) > termination_cfg.max_vertical_error_m)
    )
    success = success & ~crash & ~altitude & ~relative_envelope
    terminated = success | crash | altitude | relative_envelope
    return PressTerminationFlags(
        success=success,
        crash=crash,
        altitude=altitude,
        relative_envelope=relative_envelope,
        time_out=sample.time_out,
        terminated=terminated,
    )


def _compute_reward_terms(
    *,
    state: PressTaskState[ArrayT],
    sample: PressTaskInput[ArrayT],
    reward_cfg: PressRewardConfig,
    termination_cfg: PressTerminationConfig,
    press_steps: ArrayT,
    phase: ArrayT,
    first_contact: ArrayT,
    first_contact_stable_now: ArrayT,
    first_toggle: ArrayT,
    repeat_toggle: ArrayT,
    retreat_target: ArrayT,
    misalignment_steps: ArrayT,
    success: ArrayT,
    termination: PressTerminationFlags[ArrayT],
    confirm_steps: int,
) -> PressRewardTerms[ArrayT]:
    """Calculate every named reward term without advancing task state."""
    common = reward_cfg.common
    approach_cfg = reward_cfg.approach
    contact_cfg = reward_cfg.contact
    retreat_cfg = reward_cfg.retreat
    control_cfg = reward_cfg.control
    terminal_cfg = reward_cfg.terminal

    current_range = sample.relative_switch_pos_m[..., 0]
    lateral_error = _norm(sample.relative_switch_pos_m[..., 1:3])
    linear_speed = _norm(sample.body_linear_velocity_mps)
    angular_speed = _norm(sample.body_angular_velocity_radps)
    precontact = state.phase == PressTaskPhase.APPROACH
    contact_phase = (phase == PressTaskPhase.CONTACT) & ~first_toggle
    retreat_phase = state.phase == PressTaskPhase.RETREAT
    approach_progress = state.previous_range_m - current_range
    compression_progress = sample.pusher_compression_m - state.previous_compression_m
    button_progress = sample.button_position_m - state.previous_button_position_m
    latch_progress = (press_steps - state.press_steps) / confirm_steps
    capped_range = _elementwise_minimum(left=current_range, right=retreat_target)
    previous_capped_range = _elementwise_minimum(
        left=state.previous_range_m,
        right=retreat_target,
    )
    retreat_progress = capped_range - previous_capped_range
    contact_changed = sample.contact_flag != state.previous_contact_flag
    scraping_speed = _norm(sample.body_linear_velocity_mps[..., 1:3])

    alive = _full_like(value=current_range, fill=common.alive_weight)
    upright_alignment = _minimum(_maximum(sample.body_z_world_z, 0.0), 1.0)
    stability = -common.stability_weight * _square(1.0 - upright_alignment)
    approach = approach_cfg.approach_progress_weight * approach_progress * precontact
    near_contact = precontact & (current_range <= approach_cfg.near_contact_range_m)
    near_contact_linear_overspeed = (
        -approach_cfg.near_contact_linear_overspeed_weight
        * _square(
            _maximum(
                linear_speed - approach_cfg.near_contact_linear_speed_max_mps,
                0.0,
            )
        )
        * near_contact
    )
    near_contact_angular_overspeed = (
        -approach_cfg.near_contact_angular_overspeed_weight
        * _square(
            _maximum(
                angular_speed - approach_cfg.near_contact_angular_speed_max_radps,
                0.0,
            )
        )
        * near_contact
    )
    gentle_contact_stable = first_contact_stable_now & (
        linear_speed <= contact_cfg.gentle_contact_speed_max_mps
    )
    gentle_contact = (
        contact_cfg.gentle_contact_weight * first_contact * _where(gentle_contact_stable, 1.0, -1.0)
    )
    compression_term = (
        contact_cfg.compression_progress_weight * compression_progress * contact_phase
    )
    button_term = contact_cfg.button_progress_weight * button_progress * contact_phase
    latch_term = contact_cfg.latch_progress_weight * latch_progress * (state.toggle_time_s < 0.0)
    # Generic contact never earns this reward: only the N-step button latch can.
    toggle = contact_cfg.toggle_weight * first_toggle
    repeat_toggle_term = -contact_cfg.repeat_toggle_weight * repeat_toggle
    retreat = retreat_cfg.retreat_progress_weight * retreat_progress * retreat_phase
    retreat_alignment = (
        -retreat_cfg.retreat_alignment_weight
        * _square(
            _maximum(
                lateral_error - termination_cfg.retreat_alignment_tolerance_m,
                0.0,
            )
        )
        * retreat_phase
    )
    retreat_overspeed = (
        -retreat_cfg.retreat_overspeed_weight
        * _maximum(linear_speed - termination_cfg.retreat_speed_max_mps, 0.0) ** 2
        * retreat_phase
    )
    post_toggle_contact = (
        -retreat_cfg.post_toggle_contact_weight * sample.contact_flag * retreat_phase
    )
    post_toggle_delay = -retreat_cfg.post_toggle_delay_weight * ~success * retreat_phase
    success_term = terminal_cfg.success_weight * success * state.first_contact_stable
    action_energy = -control_cfg.action_energy_weight * _mean_square(sample.action)
    action_delta = -control_cfg.action_delta_weight * _mean_square(
        sample.action - state.previous_action
    )
    contact_oscillation = (
        -contact_cfg.contact_oscillation_weight * contact_changed * state.first_contact_seen
    )
    scraping = -contact_cfg.scraping_weight * scraping_speed * sample.contact_flag
    misalignment = (
        -contact_cfg.misalignment_weight
        * _maximum(lateral_error - contact_cfg.misalignment_tolerance_m, 0.0)
        * (misalignment_steps >= confirm_steps)
    )
    failure = -terminal_cfg.failure_weight * (
        termination.crash | termination.altitude | termination.relative_envelope
    )
    timeout = -terminal_cfg.timeout_weight * sample.time_out
    return PressRewardTerms(
        alive=alive,
        stability=stability,
        approach=approach,
        near_contact_linear_overspeed=near_contact_linear_overspeed,
        near_contact_angular_overspeed=near_contact_angular_overspeed,
        gentle_contact=gentle_contact,
        compression_progress=compression_term,
        button_progress=button_term,
        latch_progress=latch_term,
        toggle=toggle,
        repeat_toggle=repeat_toggle_term,
        retreat=retreat,
        retreat_alignment=retreat_alignment,
        retreat_overspeed=retreat_overspeed,
        post_toggle_contact=post_toggle_contact,
        post_toggle_delay=post_toggle_delay,
        success=success_term,
        action_energy=action_energy,
        action_delta=action_delta,
        contact_oscillation=contact_oscillation,
        scraping=scraping,
        misalignment=misalignment,
        failure=failure,
        timeout=timeout,
    )


def step_press_task(
    *,
    state: PressTaskState[ArrayT],
    sample: PressTaskInput[ArrayT],
    reward_cfg: PressRewardConfig,
    termination_cfg: PressTerminationConfig,
    evaluation_cfg: PressEvaluationConfig,
    press_threshold_m: float,
    confirm_steps: int,
) -> PressTaskResult[ArrayT]:
    """Advance the latch, phase, reward, termination, and metrics exactly once."""
    # 1. Derive current physical measurements and events.
    current_range = sample.relative_switch_pos_m[..., 0]
    lateral_error = _norm(sample.relative_switch_pos_m[..., 1:3])
    linear_speed = _norm(sample.body_linear_velocity_mps)
    angular_speed = _norm(sample.body_angular_velocity_radps)
    first_contact = sample.contact_flag & ~state.first_contact_seen
    first_contact_stable_now = (linear_speed <= termination_cfg.first_contact_speed_max_mps) & (
        angular_speed <= termination_cfg.first_contact_angular_speed_max_radps
    )

    # 2. Advance the latch and explicit task phase.
    press_steps, toggle_latched, lamp_state, toggled = _advance_latch(
        state=state,
        button_position_m=sample.button_position_m,
        press_threshold_m=press_threshold_m,
        confirm_steps=confirm_steps,
    )
    first_contact_seen = state.first_contact_seen | sample.contact_flag
    first_contact_stable = state.first_contact_stable | (first_contact & first_contact_stable_now)
    phase = _where(
        first_contact_seen,
        PressTaskPhase.CONTACT,
        PressTaskPhase.APPROACH,
    )
    first_toggle = toggled & (state.toggle_time_s < 0.0)
    repeat_toggle = toggled & (state.toggle_time_s >= 0.0)
    repeat_toggle_seen = state.repeat_toggle_seen | repeat_toggle
    toggle_time = _where(first_toggle, sample.elapsed_s, state.toggle_time_s)
    phase = _where(toggle_time >= 0.0, PressTaskPhase.RETREAT, phase)
    peak_compression = _where(
        sample.pusher_compression_m > state.peak_compression_m,
        sample.pusher_compression_m,
        state.peak_compression_m,
    )
    misaligned = sample.contact_flag & (lateral_error > reward_cfg.contact.misalignment_tolerance_m)
    misalignment_steps = _where(
        misaligned,
        state.misalignment_steps + 1,
        _zeros_like(value=state.misalignment_steps),
    )

    # 3. Evaluate stable retreat and terminal outcomes. Deliberately gate on
    # the prior phase so retreat stability starts one tick after first toggle.
    retreat_target = _maximum(state.initial_range_m, termination_cfg.retreat_distance_m)
    retreat_candidate = (
        (state.phase == PressTaskPhase.RETREAT)
        & lamp_state
        & ~sample.contact_flag
        & (current_range >= retreat_target)
        & (linear_speed <= termination_cfg.retreat_speed_max_mps)
        & (lateral_error <= termination_cfg.retreat_alignment_tolerance_m)
        & (angular_speed <= termination_cfg.retreat_angular_speed_max_radps)
    )
    retreat_stable_steps = _where(
        retreat_candidate,
        state.retreat_stable_steps + 1,
        _zeros_like(value=state.retreat_stable_steps),
    )
    stable_retreat_success = retreat_stable_steps >= termination_cfg.stable_retreat_steps
    termination = _termination(
        sample=sample,
        success=stable_retreat_success,
        termination_cfg=termination_cfg,
    )

    # 4. Calculate named reward terms with the same prior/successor phase gates.
    rewards = _compute_reward_terms(
        state=state,
        sample=sample,
        reward_cfg=reward_cfg,
        termination_cfg=termination_cfg,
        press_steps=press_steps,
        phase=phase,
        first_contact=first_contact,
        first_contact_stable_now=first_contact_stable_now,
        first_toggle=first_toggle,
        repeat_toggle=repeat_toggle,
        retreat_target=retreat_target,
        misalignment_steps=misalignment_steps,
        success=termination.success,
        termination=termination,
        confirm_steps=confirm_steps,
    )

    # 5. Assemble successor state and cumulative episode metrics.
    successor = PressTaskState(
        press_steps=press_steps,
        toggle_latched=toggle_latched,
        lamp_state=lamp_state,
        phase=phase,
        initial_range_m=state.initial_range_m,
        previous_range_m=current_range,
        previous_button_position_m=sample.button_position_m,
        previous_compression_m=sample.pusher_compression_m,
        previous_contact_flag=sample.contact_flag,
        previous_action=sample.action,
        first_contact_seen=first_contact_seen,
        first_contact_stable=first_contact_stable,
        repeat_toggle_seen=repeat_toggle_seen,
        toggle_time_s=toggle_time,
        peak_compression_m=peak_compression,
        retreat_stable_steps=retreat_stable_steps,
        misalignment_steps=misalignment_steps,
    )
    metrics = PressStepMetrics(
        toggled_this_step=first_toggle,
        first_contact_stable=first_contact_stable,
        toggle_time_s=toggle_time,
        peak_compression_overshoot_m=_maximum(
            peak_compression
            - evaluation_cfg.compression_target_m
            - evaluation_cfg.compression_overshoot_tolerance_m,
            0.0,
        ),
        contacted=first_contact_seen,
        toggled=toggle_time >= 0.0,
        repeat_toggled=repeat_toggle_seen,
        retreated=termination.success,
    )
    return PressTaskResult(
        state=successor,
        rewards=rewards,
        termination=termination,
        metrics=metrics,
    )

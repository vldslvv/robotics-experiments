"""Backend-neutral frozen transition for the ground-truth press task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "PressRewardTerms",
    "PressStepMetrics",
    "PressTaskInput",
    "PressTaskResult",
    "PressTaskState",
    "PressTerminationFlags",
    "initial_press_task_state",
    "reset_press_task_state",
    "step_press_task",
]

_PHASE_PRECONTACT = 0
_PHASE_CONTACT = 1
_PHASE_TOGGLED = 2


@dataclass(frozen=True)
class PressTaskState:
    """Per-environment latch, phase, history, and evaluation progress."""

    press_steps: Any
    toggle_latched: Any
    lamp_state: Any
    phase: Any
    initial_range_m: Any
    previous_range_m: Any
    previous_button_position_m: Any
    previous_compression_m: Any
    previous_contact_flag: Any
    previous_action: Any
    first_contact_seen: Any
    first_contact_stable: Any
    toggle_time_s: Any
    peak_compression_m: Any
    retreat_stable_steps: Any
    misalignment_steps: Any


@dataclass(frozen=True)
class PressTaskInput:
    """One batched sample consumed by :func:`step_press_task`."""

    relative_switch_pos_m: Any
    body_linear_velocity_mps: Any
    body_angular_velocity_radps: Any
    body_z_world_z: Any
    altitude_m: Any
    pusher_compression_m: Any
    pusher_compression_rate_mps: Any
    button_position_m: Any
    button_rate_mps: Any
    contact_flag: Any
    action: Any
    elapsed_s: Any
    time_out: Any


@dataclass(frozen=True)
class PressRewardTerms:
    """Named signed reward contributions and their total."""

    alive: Any
    stability: Any
    approach: Any
    gentle_contact: Any
    compression_progress: Any
    button_progress: Any
    latch_progress: Any
    toggle: Any
    retreat: Any
    retreat_alignment: Any
    retreat_overspeed: Any
    success: Any
    action_energy: Any
    action_delta: Any
    contact_oscillation: Any
    scraping: Any
    misalignment: Any
    failure: Any
    timeout: Any
    total: Any


@dataclass(frozen=True)
class PressTerminationFlags:
    """Distinct successful, failure, and timeout outcomes."""

    success: Any
    crash: Any
    altitude: Any
    relative_envelope: Any
    time_out: Any
    terminated: Any


@dataclass(frozen=True)
class PressStepMetrics:
    """Per-step facts accumulated by the rollout evaluator."""

    toggled_this_step: Any
    first_contact_this_step: Any
    first_contact_stable: Any
    toggle_time_s: Any
    peak_compression_overshoot_m: Any
    contacted: Any
    toggled: Any
    retreated: Any


@dataclass(frozen=True)
class PressTaskResult:
    """Successor state plus reward, termination, and metric outputs."""

    state: PressTaskState
    rewards: PressRewardTerms
    termination: PressTerminationFlags
    metrics: PressStepMetrics


def _is_torch(value: Any) -> bool:
    return type(value).__module__.startswith("torch")


def _where(condition: Any, yes: Any, no: Any) -> Any:
    if _is_torch(condition):
        import torch

        return torch.where(condition, yes, no)
    return np.where(condition, yes, no)


def _zeros_like(*, value: Any, boolean: bool = False, integer: bool = False) -> Any:
    if _is_torch(value):
        import torch

        dtype = torch.bool if boolean else torch.long if integer else value.dtype
        return torch.zeros_like(value, dtype=dtype)
    dtype = bool if boolean else np.int64 if integer else None
    return np.zeros_like(value, dtype=dtype)


def _full_like(*, value: Any, fill: float, integer: bool = False) -> Any:
    if _is_torch(value):
        import torch

        return torch.full_like(value, fill, dtype=torch.long if integer else value.dtype)
    return np.full_like(value, fill, dtype=np.int64 if integer else None)


def _norm(value: Any) -> Any:
    if _is_torch(value):
        import torch

        return torch.linalg.norm(value, dim=-1)
    return np.linalg.norm(value, axis=-1)


def _mean_square(value: Any) -> Any:
    if _is_torch(value):
        return value.square().mean(dim=-1)
    return np.square(value).mean(axis=-1)


def _square(value: Any) -> Any:
    if _is_torch(value):
        return value.square()
    return np.square(value)


def _maximum(value: Any, minimum: float) -> Any:
    if _is_torch(value):
        return value.clamp_min(minimum)
    return np.maximum(value, minimum)


def _minimum(value: Any, maximum: float) -> Any:
    if _is_torch(value):
        return value.clamp_max(maximum)
    return np.minimum(value, maximum)


def _elementwise_minimum(*, left: Any, right: Any) -> Any:
    if _is_torch(left):
        import torch

        return torch.minimum(left, right)
    return np.minimum(left, right)


def initial_press_task_state(*, relative_switch_range_m: Any, action: Any) -> PressTaskState:
    """Create a zero-progress state matching a NumPy or torch batch."""
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
        toggle_time_s=_full_like(value=relative_switch_range_m, fill=-1.0),
        peak_compression_m=zeros,
        retreat_stable_steps=integers,
        misalignment_steps=integers,
    )


def reset_press_task_state(
    *, state: PressTaskState, env_ids: Any, relative_switch_range_m: Any, action: Any
) -> PressTaskState:
    """Return ``state`` with only the selected environments reset."""
    fresh = initial_press_task_state(
        relative_switch_range_m=relative_switch_range_m,
        action=action,
    )

    def replaced(current: Any, replacement: Any) -> Any:
        successor = current.clone() if _is_torch(current) else current.copy()
        successor[env_ids] = replacement
        return successor

    return PressTaskState(
        **{
            field: replaced(getattr(state, field), getattr(fresh, field))
            for field in PressTaskState.__dataclass_fields__
        }
    )


def _advance_latch(
    *, state: PressTaskState, button_position_m: Any, press_threshold_m: float, confirm_steps: int
) -> tuple[Any, Any, Any, Any]:
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
    *, sample: PressTaskInput, success: Any, termination_cfg: Any
) -> PressTerminationFlags:
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


def step_press_task(
    *,
    state: PressTaskState,
    sample: PressTaskInput,
    reward_cfg: Any,
    termination_cfg: Any,
    press_threshold_m: float,
    confirm_steps: int,
) -> PressTaskResult:
    """Advance the latch, phase, reward, termination, and metrics exactly once."""
    press_steps, toggle_latched, lamp_state, toggled = _advance_latch(
        state=state,
        button_position_m=sample.button_position_m,
        press_threshold_m=press_threshold_m,
        confirm_steps=confirm_steps,
    )
    current_range = sample.relative_switch_pos_m[..., 0]
    lateral_error = _norm(sample.relative_switch_pos_m[..., 1:3])
    linear_speed = _norm(sample.body_linear_velocity_mps)
    angular_speed = _norm(sample.body_angular_velocity_radps)
    first_contact = sample.contact_flag & ~state.first_contact_seen
    first_contact_stable_now = (linear_speed <= termination_cfg.first_contact_speed_max_mps) & (
        angular_speed <= termination_cfg.first_contact_angular_speed_max_radps
    )
    first_contact_seen = state.first_contact_seen | sample.contact_flag
    first_contact_stable = state.first_contact_stable | (first_contact & first_contact_stable_now)
    phase = _where(first_contact_seen, _PHASE_CONTACT, _PHASE_PRECONTACT)
    first_toggle = toggled & (state.toggle_time_s < 0.0)
    toggle_time = _where(
        first_toggle,
        sample.elapsed_s,
        state.toggle_time_s,
    )
    phase = _where(toggle_time >= 0.0, _PHASE_TOGGLED, phase)
    peak_compression = _where(
        sample.pusher_compression_m > state.peak_compression_m,
        sample.pusher_compression_m,
        state.peak_compression_m,
    )
    misaligned = sample.contact_flag & (lateral_error > reward_cfg.misalignment_tolerance_m)
    misalignment_steps = _where(
        misaligned,
        state.misalignment_steps + 1,
        _zeros_like(value=state.misalignment_steps),
    )
    retreat_target = _maximum(state.initial_range_m, termination_cfg.retreat_distance_m)
    retreat_candidate = (
        (state.phase == _PHASE_TOGGLED)
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
    success = retreat_stable_steps >= termination_cfg.stable_retreat_steps
    termination = _termination(
        sample=sample,
        success=success,
        termination_cfg=termination_cfg,
    )
    success = termination.success

    precontact = state.phase == _PHASE_PRECONTACT
    contact_phase = (phase == _PHASE_CONTACT) & ~first_toggle
    toggled_phase = state.phase == _PHASE_TOGGLED
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

    alive = _full_like(value=current_range, fill=reward_cfg.alive_weight)
    upright_alignment = _minimum(_maximum(sample.body_z_world_z, 0.0), 1.0)
    stability = -reward_cfg.stability_weight * _square(1.0 - upright_alignment)
    approach = reward_cfg.approach_progress_weight * approach_progress * precontact
    gentle_contact_stable = first_contact_stable_now & (
        linear_speed <= reward_cfg.gentle_contact_speed_max_mps
    )
    gentle_contact = (
        reward_cfg.gentle_contact_weight
        * first_contact
        * _where(
            gentle_contact_stable,
            1.0,
            -1.0,
        )
    )
    compression_term = reward_cfg.compression_progress_weight * compression_progress * contact_phase
    button_term = reward_cfg.button_progress_weight * button_progress * contact_phase
    latch_term = reward_cfg.latch_progress_weight * latch_progress * (state.toggle_time_s < 0.0)
    # Generic contact never earns this reward: only the N-step button latch can.
    toggle = reward_cfg.toggle_weight * first_toggle
    retreat = reward_cfg.retreat_progress_weight * retreat_progress * toggled_phase
    retreat_alignment = (
        -reward_cfg.retreat_alignment_weight * _square(lateral_error) * toggled_phase
    )
    retreat_overspeed = (
        -reward_cfg.retreat_overspeed_weight
        * _maximum(linear_speed - termination_cfg.retreat_speed_max_mps, 0.0) ** 2
        * toggled_phase
    )
    success_term = reward_cfg.success_weight * success
    action_energy = -reward_cfg.action_energy_weight * _mean_square(sample.action)
    action_delta = -reward_cfg.action_delta_weight * _mean_square(
        sample.action - state.previous_action
    )
    contact_oscillation = (
        -reward_cfg.contact_oscillation_weight * contact_changed * state.first_contact_seen
    )
    scraping = -reward_cfg.scraping_weight * scraping_speed * sample.contact_flag
    misalignment = (
        -reward_cfg.misalignment_weight
        * _maximum(lateral_error - reward_cfg.misalignment_tolerance_m, 0.0)
        * (misalignment_steps >= confirm_steps)
    )
    failure = -reward_cfg.failure_weight * (
        termination.crash | termination.altitude | termination.relative_envelope
    )
    timeout = -reward_cfg.timeout_weight * sample.time_out
    total = (
        alive
        + stability
        + approach
        + gentle_contact
        + compression_term
        + button_term
        + latch_term
        + toggle
        + retreat
        + retreat_alignment
        + retreat_overspeed
        + success_term
        + action_energy
        + action_delta
        + contact_oscillation
        + scraping
        + misalignment
        + failure
        + timeout
    )
    rewards = PressRewardTerms(
        alive=alive,
        stability=stability,
        approach=approach,
        gentle_contact=gentle_contact,
        compression_progress=compression_term,
        button_progress=button_term,
        latch_progress=latch_term,
        toggle=toggle,
        retreat=retreat,
        retreat_alignment=retreat_alignment,
        retreat_overspeed=retreat_overspeed,
        success=success_term,
        action_energy=action_energy,
        action_delta=action_delta,
        contact_oscillation=contact_oscillation,
        scraping=scraping,
        misalignment=misalignment,
        failure=failure,
        timeout=timeout,
        total=total,
    )
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
        toggle_time_s=toggle_time,
        peak_compression_m=peak_compression,
        retreat_stable_steps=retreat_stable_steps,
        misalignment_steps=misalignment_steps,
    )
    metrics = PressStepMetrics(
        toggled_this_step=first_toggle,
        first_contact_this_step=first_contact,
        first_contact_stable=first_contact_stable,
        toggle_time_s=toggle_time,
        peak_compression_overshoot_m=_maximum(
            peak_compression
            - reward_cfg.compression_target_m
            - reward_cfg.compression_overshoot_tolerance_m,
            0.0,
        ),
        contacted=first_contact_seen,
        toggled=toggle_time >= 0.0,
        retreated=success,
    )
    return PressTaskResult(
        state=successor,
        rewards=rewards,
        termination=termination,
        metrics=metrics,
    )

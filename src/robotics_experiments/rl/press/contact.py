"""Batched contact, button latch and press-and-retreat task transitions."""

from dataclasses import dataclass, fields

import torch

from robotics_experiments.config import PressTaskConfig
from robotics_experiments.rl.press.task import PressApproachResult

__all__ = [
    "PressContactState",
    "PressContactResult",
    "initial_press_contact_state",
    "reset_press_contact_state",
    "step_press_contact",
]


@dataclass(frozen=True)
class PressContactState:
    """Episode memory; every tensor contains one value per environment."""

    press_steps: torch.Tensor
    toggle_latched: torch.Tensor
    toggle_count: torch.Tensor
    contact_seen: torch.Tensor
    first_contact_stable: torch.Tensor
    stable_contact_steps: torch.Tensor
    stable_retreat_steps: torch.Tensor
    longest_retreat_steps: torch.Tensor
    closest_range_m: torch.Tensor
    maximum_button_m: torch.Tensor
    best_retreat_error_m: torch.Tensor
    peak_compression_m: torch.Tensor


@dataclass(frozen=True)
class PressContactResult:
    state: PressContactState
    rewards: dict[str, torch.Tensor]
    success: torch.Tensor
    collision: torch.Tensor
    failure: torch.Tensor
    timeout: torch.Tensor
    progress: torch.Tensor

    @property
    def reward(self) -> torch.Tensor:
        return torch.stack(tuple(self.rewards.values())).sum(dim=0)


def initial_press_contact_state(*, ranges: torch.Tensor) -> PressContactState:
    integer = torch.zeros_like(ranges, dtype=torch.long)
    flag = torch.zeros_like(ranges, dtype=torch.bool)
    zero = torch.zeros_like(ranges)
    return PressContactState(
        press_steps=integer.clone(),
        toggle_latched=flag.clone(),
        toggle_count=integer.clone(),
        contact_seen=flag.clone(),
        first_contact_stable=flag.clone(),
        stable_contact_steps=integer.clone(),
        stable_retreat_steps=integer.clone(),
        longest_retreat_steps=integer.clone(),
        closest_range_m=ranges.clone(),
        maximum_button_m=zero.clone(),
        best_retreat_error_m=torch.full_like(ranges, float("inf")),
        peak_compression_m=zero.clone(),
    )


def reset_press_contact_state(
    *, state: PressContactState, env_ids: torch.Tensor, ranges: torch.Tensor
) -> PressContactState:
    fresh = initial_press_contact_state(ranges=ranges)
    values = {}
    for field in fields(state):
        value = getattr(state, field.name).clone()
        value[env_ids] = getattr(fresh, field.name)
        values[field.name] = value
    return PressContactState(**values)


def step_press_contact(
    *,
    state: PressContactState,
    flight: PressApproachResult[torch.Tensor],
    button_position_m: torch.Tensor,
    compression_m: torch.Tensor,
    intended_contact: torch.Tensor,
    contact_seen: torch.Tensor,
    first_contact_stable: torch.Tensor,
    action: torch.Tensor,
    config: PressTaskConfig,
    press_threshold_m: float,
    confirm_steps: int,
) -> PressContactResult:
    """Advance once per policy tick, matching the production switch-latch cadence.

    Button displacement is privileged reward/evaluation evidence. The actor sees
    its existing compression, contact and lamp observations. Progress rewards
    are bounded by episode records, so repeated tapping cannot farm bonuses.
    """
    contact = config.contact
    pressed = button_position_m >= press_threshold_m
    press_steps = torch.where(pressed, state.press_steps + 1, 0)
    toggled = (press_steps >= confirm_steps) & ~state.toggle_latched
    toggle_latched = pressed & (state.toggle_latched | toggled)
    toggle_count = state.toggle_count + toggled.long()
    first_toggle = toggled & (state.toggle_count == 0)
    lamp_on = toggle_count.remainder(2) == 1
    gentle = (
        intended_contact
        & (flight.linear_speed_mps <= config.goal.linear_speed_max_mps)
        & (flight.angular_speed_radps <= config.goal.angular_speed_max_radps)
        & (flight.alignment_error_m <= config.goal.alignment_error_max_m)
        & (flight.attitude_error_rad <= config.goal.relative_attitude_error_max_rad)
    )
    stable_contact_steps = torch.where(gentle & ~lamp_on, state.stable_contact_steps + 1, 0)
    retreat = lamp_on & ~intended_contact & flight.strict_goal
    stable_retreat_steps = torch.where(retreat, state.stable_retreat_steps + 1, 0)
    longest_retreat_steps = torch.maximum(state.longest_retreat_steps, stable_retreat_steps)
    closest_range = torch.minimum(state.closest_range_m, flight.range_m)
    maximum_button = torch.maximum(state.maximum_button_m, button_position_m.clamp_min(0))
    peak_compression = torch.maximum(state.peak_compression_m, compression_m.clamp_min(0))
    retreat_error = flight.range_error_m + flight.alignment_error_m
    previous_retreat_error = torch.where(first_toggle, retreat_error, state.best_retreat_error_m)
    best_retreat_error = torch.where(
        lamp_on, torch.minimum(previous_retreat_error, retreat_error), previous_retreat_error
    )
    collision = flight.termination.collision
    failure = (
        collision
        | flight.termination.flight_failure
        | (peak_compression > contact.compression_max_m)
        | (toggle_count > 1)
    )
    contact_only = config.task == "contact"
    if contact_only:
        failure |= toggle_count != 0
        success = (stable_contact_steps >= contact.stable_steps) & first_contact_stable & ~failure
    else:
        success = (
            (stable_retreat_steps >= config.goal.stable_steps) & (toggle_count == 1) & ~failure
        )
    timeout = flight.termination.time_out & ~success & ~failure
    advance = (state.closest_range_m - closest_range).clamp_min(0)
    button_advance = (
        maximum_button.clamp(max=press_threshold_m)
        - state.maximum_button_m.clamp(max=press_threshold_m)
    ).clamp_min(0) / press_threshold_m
    retreat_advance = torch.where(lamp_on, previous_retreat_error - best_retreat_error, 0.0)
    # Contact and press use different objectives while preserving the actor schema.
    rewards = {
        "approach_progress": contact.approach_progress_weight * advance * ~state.contact_seen,
        "contact_hold": contact.contact_hold_weight * gentle.float() * contact_only,
        "button_progress": contact.button_progress_weight * button_advance * (not contact_only),
        "toggle": contact.toggle_weight * first_toggle.float() * (not contact_only),
        "retreat_progress": contact.retreat_progress_weight * retreat_advance * (not contact_only),
        "hold_progress": config.reward.hold_progress_weight
        * (longest_retreat_steps - state.longest_retreat_steps),
        "stabilization": -config.reward.stabilization_weight
        * (
            flight.alignment_error_m / config.goal.alignment_error_max_m
            + flight.attitude_error_rad / config.goal.relative_attitude_error_max_rad
        ),
        "action_effort": -config.reward.action_effort_weight * action.square().mean(dim=-1),
        "episode_outcome": config.reward.terminal_success_weight * success.float()
        - config.reward.terminal_failure_weight * (failure | timeout).float()
        - contact.step_cost,
    }
    # A bounded physical progress scalar, independent of PPO reward weights.
    approach_progress = (
        1
        - (flight.range_m - contact.target_range_m).clamp_min(0)
        / config.goal.velocity_reference.range_center_m
    ).clamp(0, 1)
    if contact_only:
        progress = approach_progress + (stable_contact_steps / contact.stable_steps).clamp(0, 1)
    else:
        progress = (
            approach_progress
            + contact_seen.float()
            + (maximum_button / press_threshold_m).clamp(0, 1)
            + (toggle_count > 0).float()
            + (longest_retreat_steps / config.goal.stable_steps).clamp(0, 1)
        )
    successor = PressContactState(
        press_steps=press_steps,
        toggle_latched=toggle_latched,
        toggle_count=toggle_count,
        contact_seen=contact_seen,
        first_contact_stable=first_contact_stable,
        stable_contact_steps=stable_contact_steps,
        stable_retreat_steps=stable_retreat_steps,
        longest_retreat_steps=longest_retreat_steps,
        closest_range_m=closest_range,
        maximum_button_m=maximum_button,
        best_retreat_error_m=best_retreat_error,
        peak_compression_m=peak_compression,
    )
    return PressContactResult(successor, rewards, success, collision, failure, timeout, progress)

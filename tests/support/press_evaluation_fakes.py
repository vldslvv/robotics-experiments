"""Deterministic vector rollout used by press evaluator tests."""

from dataclasses import fields
from types import SimpleNamespace

import torch

from robotics_experiments.rl.press import PressApproachRewardTerms

__all__ = ["PressEvaluationEnv", "ZeroPressPolicy"]

_REWARD_NAMES = tuple(field.name for field in fields(PressApproachRewardTerms)) + ("total",)


class ZeroPressPolicy:
    def __call__(self, observation, /):
        return torch.zeros((observation.shape[0], 4))


class PressEvaluationEnv:
    device = "cpu"
    max_episode_length = 80
    step_dt = 0.5

    episode_count = 3
    hold_steps = 60
    warmup_steps = 12

    def __init__(self, *, success_ends: tuple[int, int]):
        self.success_ends = success_ends
        self.seeds: list[int] = []
        self.step_count = 0
        self.press_approach_result = None

    def reset(self, *, seed):
        self.seeds.append(seed)
        self.step_count = 0
        return {"policy": torch.zeros((self.episode_count, 35))}, {}

    def step(self, _action, /):
        self.step_count += 1
        step = self.step_count
        # Early recovery, hold warmup, failed episodes and post-completion
        # samples are deliberately much larger than the qualifying values.
        values = torch.tensor(
            [
                step / 1000 if end - self.hold_steps + self.warmup_steps < step <= end else 9.0
                for end in (*self.success_ends, 0)
            ]
        )
        success = torch.tensor([step == end for end in (*self.success_ends, 0)])
        timeout = torch.tensor(
            [step == self.max_episode_length and end == 0 for end in (*self.success_ends, 0)]
        )
        clear = torch.zeros(self.episode_count, dtype=torch.bool)
        # The third episode fails physically before either successful hold.
        collision = clear.clone()
        collision[2] = step == 20
        stable_steps = torch.tensor(
            [
                max(0, min(self.hold_steps, step - end + self.hold_steps)) if end else 0
                for end in (*self.success_ends, 0)
            ]
        )
        self.press_approach_result = SimpleNamespace(
            state=SimpleNamespace(
                closest_range_m=torch.full((self.episode_count,), 0.41),
                normalized_pose_error=torch.ones(self.episode_count),
                stable_goal_steps=stable_steps,
                longest_stable_goal_steps=stable_steps,
            ),
            range_m=torch.full((self.episode_count,), 0.41),
            strict_goal=stable_steps > 0,
            hover_evaluation=SimpleNamespace(
                position_error_enu_m=values[:, None].expand(-1, 3).clone(),
                horizontal_position_error_m=values,
                position_error_m=values,
                attitude_error_rpy_rad=values[:, None].expand(-1, 3).clone(),
                tilt_error_rad=values,
                orientation_error_rad=values,
                position_utilization=torch.ones(self.episode_count),
                attitude_utilization=torch.zeros(self.episode_count),
            ),
            termination=SimpleNamespace(
                success=success,
                collision=collision,
                body_collision=collision,
                pusher_collision=clear,
                flight_failure=clear,
                range_undershoot=clear,
                time_out=timeout,
            ),
            rewards=SimpleNamespace(
                **{name: torch.ones(self.episode_count) for name in _REWARD_NAMES}
            ),
        )
        return (
            {"policy": torch.zeros((self.episode_count, 35))},
            torch.zeros(self.episode_count),
            success | collision,
            timeout,
            {},
        )

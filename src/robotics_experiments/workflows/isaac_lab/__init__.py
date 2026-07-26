"""Public contract for the Isaac Lab workflow integration.

This package root is the only supported import surface for code outside
``robotics_experiments.workflows.isaac_lab``. It stays lightweight at import
time: ordinary torch-backed helpers are imported eagerly, while Isaac Lab task
modules are loaded only when their exported symbols are requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from robotics_experiments.workflows.isaac_lab.press_evaluation import (
    PressEpisodeMetrics,
    PressEvaluationResult,
    PressEvaluationSummary,
    evaluate_press_policy,
    summarize_press_episodes,
)
from robotics_experiments.workflows.isaac_lab.registration import (
    DRONE_HOVER_TASK_ID,
    SWITCH_PRESS_TASK_ID,
    make_isaac_lab_env,
    register_isaac_lab_tasks,
)

if TYPE_CHECKING:
    from robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env import (
        DroneHoverEnv,
        DroneHoverEnvCfg,
        make_drone_hover_env_cfg,
    )
    from robotics_experiments.workflows.isaac_lab.tasks.flight import _FlightActuationParams
    from robotics_experiments.workflows.isaac_lab.tasks.switch_press_env import (
        SwitchPressEnv,
        SwitchPressEnvCfg,
        make_switch_press_env_cfg,
    )

__all__ = [
    "DRONE_HOVER_TASK_ID",
    "DroneHoverEnv",
    "DroneHoverEnvCfg",
    "PressEpisodeMetrics",
    "PressEvaluationResult",
    "PressEvaluationSummary",
    "SWITCH_PRESS_TASK_ID",
    "SwitchPressEnv",
    "SwitchPressEnvCfg",
    "compute_lab_rotor_omega",
    "evaluate_press_policy",
    "make_drone_hover_env_cfg",
    "make_isaac_lab_env",
    "make_switch_press_env_cfg",
    "register_isaac_lab_tasks",
    "summarize_press_episodes",
]


def __getattr__(name: str):
    if name in {"DroneHoverEnv", "DroneHoverEnvCfg", "make_drone_hover_env_cfg"}:
        from robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env import (
            DroneHoverEnv,
            DroneHoverEnvCfg,
            make_drone_hover_env_cfg,
        )

        return {
            "DroneHoverEnv": DroneHoverEnv,
            "DroneHoverEnvCfg": DroneHoverEnvCfg,
            "make_drone_hover_env_cfg": make_drone_hover_env_cfg,
        }[name]
    if name in {"SwitchPressEnv", "SwitchPressEnvCfg", "make_switch_press_env_cfg"}:
        from robotics_experiments.workflows.isaac_lab.tasks.switch_press_env import (
            SwitchPressEnv,
            SwitchPressEnvCfg,
            make_switch_press_env_cfg,
        )

        return {
            "SwitchPressEnv": SwitchPressEnv,
            "SwitchPressEnvCfg": SwitchPressEnvCfg,
            "make_switch_press_env_cfg": make_switch_press_env_cfg,
        }[name]
    if name == "compute_lab_rotor_omega":
        from robotics_experiments.workflows.isaac_lab.tasks.flight import action_to_rotor_omega

        return action_to_rotor_omega
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:

    def compute_lab_rotor_omega(
        params: _FlightActuationParams,
        action: torch.Tensor,
        attitude_wxyz: torch.Tensor,
        linear_velocity_world: torch.Tensor,
        angular_velocity_body: torch.Tensor,
    ) -> torch.Tensor: ...

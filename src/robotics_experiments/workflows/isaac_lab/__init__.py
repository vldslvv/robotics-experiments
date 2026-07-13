"""Public contract for the Isaac Lab workflow integration.

This package root is the only supported import surface for code outside
``robotics_experiments.workflows.isaac_lab``. It stays lightweight at import
time: Isaac Lab task modules and torch helpers are loaded only when their
exported symbols are requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robotics_experiments.workflows.isaac_lab.registration import (
    DRONE_HOVER_TASK_ID,
    register_isaac_lab_tasks,
)

if TYPE_CHECKING:
    import torch

    from robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env import (
        DroneHoverEnv,
        DroneHoverEnvCfg,
        make_drone_hover_env_cfg,
    )
    from robotics_experiments.workflows.isaac_lab.tasks.flight import _FlightActuationParams

__all__ = [
    "DRONE_HOVER_TASK_ID",
    "DroneHoverEnv",
    "DroneHoverEnvCfg",
    "compute_lab_rotor_omega",
    "make_drone_hover_env_cfg",
    "register_isaac_lab_tasks",
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

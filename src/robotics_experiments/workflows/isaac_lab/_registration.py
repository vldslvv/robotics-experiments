"""Gymnasium registration helpers for Isaac Lab tasks."""

from __future__ import annotations

from importlib.util import find_spec

__all__ = ["register_isaac_lab_tasks"]

_DRONE_HOVER_TASK_ID = "Isaac-Drone-Hover-Direct-v0"


def _dependency_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _gym_has_task(gym, task_id: str) -> bool:
    return task_id in gym.registry


def register_isaac_lab_tasks() -> bool:
    """Register this project's Isaac Lab tasks when dependencies are available.

    Returns:
        True when Gymnasium/Isaac Lab are available and the task is registered;
        False when called in a host environment without those dependencies.
    """

    if not _dependency_available("gymnasium") or not _dependency_available("isaaclab"):
        return False

    import gymnasium as gym

    if _gym_has_task(gym, _DRONE_HOVER_TASK_ID):
        return True

    gym.register(
        id=_DRONE_HOVER_TASK_ID,
        entry_point=(
            "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env:" "DroneHoverEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env:" "DroneHoverEnvCfg"
            )
        },
    )
    return True

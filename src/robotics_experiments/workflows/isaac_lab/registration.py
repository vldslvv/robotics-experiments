"""Gymnasium registration contract for Isaac Lab tasks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.util import find_spec

import gymnasium as gym

from robotics_experiments.config import ExperimentConfig
from robotics_experiments.controller import ControllerSettings
from robotics_experiments.flight_parameters import FlightEnvelope
from robotics_experiments.rl import PolicyContract, PolicyInterface

__all__ = [
    "DRONE_HOVER_TASK_ID",
    "SWITCH_PRESS_TASK_ID",
    "make_isaac_lab_env",
    "register_isaac_lab_tasks",
]

DRONE_HOVER_TASK_ID = "Isaac-Drone-Hover-Direct-v1"
SWITCH_PRESS_TASK_ID = "Isaac-Switch-Press-Direct-v1"


@dataclass(frozen=True)
class _TaskRegistration:
    entry_point: str
    cfg_entry_point: str
    cfg_factory: str


_TASKS = {
    DRONE_HOVER_TASK_ID: _TaskRegistration(
        entry_point=(
            "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env:" "DroneHoverEnv"
        ),
        cfg_entry_point=(
            "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env:" "DroneHoverEnvCfg"
        ),
        cfg_factory=(
            "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env:"
            "make_drone_hover_env_cfg"
        ),
    ),
    SWITCH_PRESS_TASK_ID: _TaskRegistration(
        entry_point=(
            "robotics_experiments.workflows.isaac_lab.tasks.switch_press_env:" "SwitchPressEnv"
        ),
        cfg_entry_point=(
            "robotics_experiments.workflows.isaac_lab.tasks.switch_press_env:" "SwitchPressEnvCfg"
        ),
        cfg_factory=(
            "robotics_experiments.workflows.isaac_lab.tasks.switch_press_env:"
            "make_switch_press_env_cfg"
        ),
    ),
}


def _dependency_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _gym_has_task(task_id: str) -> bool:
    return task_id in gym.registry


def register_isaac_lab_tasks() -> bool:
    """Register this project's Isaac Lab tasks when dependencies are available.

    Returns:
        True when Gymnasium/Isaac Lab are available and the tasks are registered;
        False when called in a host environment without those dependencies.
    """

    if not _dependency_available("isaaclab"):
        return False

    for task_id, task in _TASKS.items():
        if _gym_has_task(task_id):
            continue
        gym.register(
            id=task_id,
            entry_point=task.entry_point,
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": task.cfg_entry_point},
        )
    return True


def make_isaac_lab_env(
    *,
    task_id: str,
    experiment_config: ExperimentConfig,
    controller_settings: ControllerSettings,
    seed: int | None = None,
    num_envs: int | None = None,
    policy_contract: PolicyContract | None = None,
    flight_envelope: FlightEnvelope | None = None,
) -> tuple[gym.Env, object]:
    """Build a task, optionally interpreting actions through an artifact contract."""
    if not register_isaac_lab_tasks():
        raise RuntimeError("Isaac Lab task registration failed (gymnasium/isaaclab missing)")
    try:
        registration = _TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"unknown robotics-experiments Isaac Lab task {task_id!r}") from exc
    expected_interface = {
        DRONE_HOVER_TASK_ID: PolicyInterface.HOVER,
        SWITCH_PRESS_TASK_ID: PolicyInterface.PRESS,
    }[task_id]
    if policy_contract is not None and policy_contract.policy_interface != expected_interface:
        raise ValueError(
            f"task {task_id!r} requires policy_interface={expected_interface.value!r}, "
            f"artifact provides {policy_contract.policy_interface!r}"
        )
    module_name, factory_name = registration.cfg_factory.split(":", maxsplit=1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    factory_kwargs = {
        "experiment_config": experiment_config,
        "policy_contract": policy_contract,
    }
    env_cfg = factory(**factory_kwargs)
    if seed is not None and hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    env_kwargs = {
        "cfg": env_cfg,
        "experiment_config": experiment_config,
        "policy_contract": policy_contract,
        "controller_settings": controller_settings,
    }
    if task_id == SWITCH_PRESS_TASK_ID:
        if flight_envelope is None:
            raise ValueError("the press task requires a pinned flight envelope")
        env_kwargs["flight_envelope"] = flight_envelope
    env = gym.make(task_id, **env_kwargs).unwrapped
    return env, env_cfg

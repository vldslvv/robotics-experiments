import importlib
import importlib.abc
import importlib.machinery
import sys
import types

import pytest


class _BlockedDependencyFinder(importlib.abc.MetaPathFinder):
    def __init__(self, module_roots: set[str]):
        self._module_roots = module_roots

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", maxsplit=1)[0] in self._module_roots:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def _drop_modules(*prefixes: str) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def test_isaac_lab_package_imports_without_isaac_lab(monkeypatch):
    _drop_modules("gymnasium", "isaaclab", "robotics_experiments.workflows.isaac_lab")
    blocker = _BlockedDependencyFinder({"gymnasium", "isaaclab"})
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    module = importlib.import_module("robotics_experiments.workflows.isaac_lab")

    assert module.__all__ == [
        "DRONE_HOVER_TASK_ID",
        "DroneHoverEnv",
        "DroneHoverEnvCfg",
        "SWITCH_PRESS_TASK_ID",
        "SwitchPressEnv",
        "SwitchPressEnvCfg",
        "compute_lab_rotor_omega",
        "make_drone_hover_env_cfg",
        "make_isaac_lab_env",
        "make_switch_press_env_cfg",
        "register_isaac_lab_tasks",
    ]
    assert module.DRONE_HOVER_TASK_ID == "Isaac-Drone-Hover-Direct-v0"
    assert module.SWITCH_PRESS_TASK_ID == "Isaac-Switch-Press-Direct-v0"
    assert module.register_isaac_lab_tasks() is False


def test_registration_uses_expected_gymnasium_entry_points(monkeypatch):
    from robotics_experiments.workflows.isaac_lab import (
        DRONE_HOVER_TASK_ID,
        SWITCH_PRESS_TASK_ID,
        register_isaac_lab_tasks,
    )

    fake_gym = types.ModuleType("gymnasium")
    fake_gym.__spec__ = importlib.machinery.ModuleSpec("gymnasium", loader=None)
    fake_gym.registry = {}
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.__spec__ = importlib.machinery.ModuleSpec("isaaclab", loader=None)

    def register(*, id, entry_point, disable_env_checker, kwargs):
        fake_gym.registry[id] = {
            "entry_point": entry_point,
            "disable_env_checker": disable_env_checker,
            "kwargs": kwargs,
        }

    fake_gym.register = register
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)

    assert register_isaac_lab_tasks() is True
    assert register_isaac_lab_tasks() is True

    spec = fake_gym.registry[DRONE_HOVER_TASK_ID]
    assert spec == {
        "entry_point": (
            "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env:" "DroneHoverEnv"
        ),
        "disable_env_checker": True,
        "kwargs": {
            "env_cfg_entry_point": (
                "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env:" "DroneHoverEnvCfg"
            )
        },
    }
    fake_gym.registry.pop(SWITCH_PRESS_TASK_ID)
    assert register_isaac_lab_tasks() is True
    assert fake_gym.registry[DRONE_HOVER_TASK_ID] is spec
    assert fake_gym.registry[SWITCH_PRESS_TASK_ID] == {
        "entry_point": (
            "robotics_experiments.workflows.isaac_lab.tasks.switch_press_env:" "SwitchPressEnv"
        ),
        "disable_env_checker": True,
        "kwargs": {
            "env_cfg_entry_point": (
                "robotics_experiments.workflows.isaac_lab.tasks.switch_press_env:"
                "SwitchPressEnvCfg"
            )
        },
    }


def test_evaluation_contract_is_forwarded_to_task_config_and_environment(monkeypatch):
    registration = importlib.import_module("robotics_experiments.workflows.isaac_lab.registration")
    contract = types.SimpleNamespace(policy_interface="press-v1")
    calls = {}
    env_cfg = types.SimpleNamespace(scene=types.SimpleNamespace(num_envs=1))
    factory_module = types.SimpleNamespace(
        make_switch_press_env_cfg=lambda **kwargs: calls.setdefault("factory", kwargs) and env_cfg
    )
    fake_gym = types.ModuleType("gymnasium")

    def make(task_id, **kwargs):
        calls["gym"] = (task_id, kwargs)
        return types.SimpleNamespace(unwrapped="environment")

    fake_gym.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    monkeypatch.setattr(registration, "register_isaac_lab_tasks", lambda: True)
    monkeypatch.setattr(
        registration.importlib,
        "import_module",
        lambda _name: factory_module,
    )

    env, returned_cfg = registration.make_isaac_lab_env(
        task_id=registration.SWITCH_PRESS_TASK_ID,
        experiment_config="training-yaml",
        seed=123,
        num_envs=8,
        policy_contract=contract,
    )

    assert env == "environment"
    assert returned_cfg is env_cfg
    assert calls["factory"]["policy_contract"] is contract
    assert calls["gym"][1]["policy_contract"] is contract
    assert env_cfg.scene.num_envs == 8


def test_drone_hover_task_import_and_config_when_isaac_lab_available():
    pytest.importorskip("isaaclab")
    pytest.importorskip("gymnasium")

    module = importlib.import_module("robotics_experiments.workflows.isaac_lab")
    from robotics_experiments.config import load_config
    from robotics_experiments.workflows.isaac_lab import (
        DroneHoverEnv,
        DroneHoverEnvCfg,
        make_drone_hover_env_cfg,
    )
    from robotics_experiments.workflows.isaac_lab.tasks._flight_env import (
        _episode_step_count,
    )

    assert module.DroneHoverEnv is DroneHoverEnv
    assert module.DroneHoverEnvCfg is DroneHoverEnvCfg
    cfg = make_drone_hover_env_cfg(experiment_config=load_config("configs/takeoff_quad.yaml"))
    assert cfg.action_space == 4
    assert cfg.observation_space == 13
    assert not hasattr(cfg, "physics_rate_hz")
    assert cfg.sim.dt == pytest.approx(1.0 / 250.0)
    assert (
        _episode_step_count(
            episode_length_s=cfg.episode_length_s,
            policy_rate_hz=cfg.policy_rate_hz,
        )
        == 100
    )

import importlib
import importlib.abc
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

    assert module.__all__ == ["register_isaac_lab_tasks"]
    assert module.register_isaac_lab_tasks() is False


def test_registration_uses_expected_gymnasium_entry_points(monkeypatch):
    import robotics_experiments.workflows.isaac_lab._registration as registration
    from robotics_experiments.workflows.isaac_lab._registration import _DRONE_HOVER_TASK_ID

    fake_gym = types.ModuleType("gymnasium")
    fake_gym.registry = {}

    def register(*, id, entry_point, disable_env_checker, kwargs):
        fake_gym.registry[id] = {
            "entry_point": entry_point,
            "disable_env_checker": disable_env_checker,
            "kwargs": kwargs,
        }

    fake_gym.register = register
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gym)
    monkeypatch.setattr(registration, "_dependency_available", lambda module_name: True)

    assert registration.register_isaac_lab_tasks() is True
    assert registration.register_isaac_lab_tasks() is True

    spec = fake_gym.registry[_DRONE_HOVER_TASK_ID]
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


def test_drone_hover_task_import_and_config_when_isaac_lab_available():
    pytest.importorskip("isaaclab")
    pytest.importorskip("gymnasium")

    module = importlib.import_module(
        "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env"
    )

    assert module.__all__ == ["DroneHoverEnvCfg", "DroneHoverEnv"]
    cfg = module.DroneHoverEnvCfg()
    assert cfg.action_space == 4
    assert cfg.observation_space == 13

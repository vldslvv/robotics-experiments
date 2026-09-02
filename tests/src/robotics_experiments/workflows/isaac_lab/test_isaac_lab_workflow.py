import ast
import importlib
import importlib.abc
import importlib.machinery
import sys
import types

import pytest

from tests.support.paths import REPO_ROOT


def test_press_episode_counters_exist_before_phase_configuration():
    path = (
        REPO_ROOT
        / "src"
        / "robotics_experiments"
        / "workflows"
        / "isaac_lab"
        / "tasks"
        / "switch_press_env.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    env_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SwitchPressEnv"
    )
    methods = {node.name: node for node in env_class.body if isinstance(node, ast.FunctionDef)}

    def assigned_attributes(method, /):
        return {
            node.targets[0].attr
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        }

    assert "_completed_episode_count" in assigned_attributes(methods["__init__"])
    assert "_training_minimum_range_m" in assigned_attributes(methods["__init__"])
    assert "_training_minimum_range_m" in assigned_attributes(methods["configure_training_phase"])


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
    _drop_modules("isaaclab", "robotics_experiments.workflows.isaac_lab")
    blocker = _BlockedDependencyFinder({"isaaclab"})
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    module = importlib.import_module("robotics_experiments.workflows.isaac_lab")

    assert module.__all__ == [
        "DRONE_HOVER_TASK_ID",
        "DroneHoverEnv",
        "DroneHoverEnvCfg",
        "PressApproachEpisodeMetrics",
        "PressApproachEvaluationResult",
        "PressApproachEvaluationSummary",
        "SWITCH_PRESS_TASK_ID",
        "SwitchPressEnv",
        "SwitchPressEnvCfg",
        "compute_lab_rotor_omega",
        "evaluate_press_approach_policy",
        "make_drone_hover_env_cfg",
        "make_isaac_lab_env",
        "make_switch_press_env_cfg",
        "press_policy_evaluation",
        "register_isaac_lab_tasks",
        "summarize_press_approach_episodes",
    ]
    assert module.DRONE_HOVER_TASK_ID == "Isaac-Drone-Hover-Direct-v1"
    assert module.SWITCH_PRESS_TASK_ID == "Isaac-Switch-Press-Direct-v1"
    assert module.register_isaac_lab_tasks() is False


def test_registration_uses_expected_gymnasium_entry_points(monkeypatch):
    registration = importlib.import_module("robotics_experiments.workflows.isaac_lab.registration")
    DRONE_HOVER_TASK_ID = registration.DRONE_HOVER_TASK_ID
    SWITCH_PRESS_TASK_ID = registration.SWITCH_PRESS_TASK_ID

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
    monkeypatch.setattr(registration, "gym", fake_gym)
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)

    assert registration.register_isaac_lab_tasks() is True
    assert registration.register_isaac_lab_tasks() is True

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
    assert registration.register_isaac_lab_tasks() is True
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
    from robotics_experiments.rl import PolicyInterface

    contract = types.SimpleNamespace(policy_interface=PolicyInterface.PRESS)
    flight_parameters = object()
    hover_reset_samples = object()
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
    monkeypatch.setattr(registration, "gym", fake_gym)
    monkeypatch.setattr(registration, "register_isaac_lab_tasks", lambda: True)
    monkeypatch.setattr(
        registration.importlib,
        "import_module",
        lambda _name: factory_module,
    )

    env, returned_cfg = registration.make_isaac_lab_env(
        task_id=registration.SWITCH_PRESS_TASK_ID,
        experiment_config="training-yaml",
        controller_settings="aligned-controller",
        seed=123,
        num_envs=8,
        policy_contract=contract,
        flight_parameters=flight_parameters,
        hover_reset_samples=hover_reset_samples,
    )

    assert env == "environment"
    assert returned_cfg is env_cfg
    assert calls["factory"]["policy_contract"] is contract
    assert "use_training_reset_distribution" not in calls["factory"]
    assert calls["gym"][1]["policy_contract"] is contract
    assert calls["gym"][1]["controller_settings"] == "aligned-controller"
    assert calls["gym"][1]["flight_parameters"] is flight_parameters
    assert calls["gym"][1]["hover_reset_samples"] is hover_reset_samples
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
    assert cfg.policy_config.actor.hidden_dims == (128, 128)
    assert cfg.policy_config.critic.hidden_dims == (128, 128)
    assert not hasattr(cfg, "physics_rate_hz")
    assert cfg.sim.dt == pytest.approx(1.0 / 250.0)
    assert not cfg.sim.physx.enable_enhanced_determinism
    assert not cfg.drone_cfg.spawn.rigid_props.enable_gyroscopic_forces
    assert (
        _episode_step_count(
            episode_length_s=cfg.episode_length_s,
            policy_rate_hz=cfg.policy_rate_hz,
        )
        == 100
    )

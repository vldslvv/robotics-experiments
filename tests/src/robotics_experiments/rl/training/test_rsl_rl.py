from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

from robotics_experiments.rl.training.rsl_rl import build_runner, export_policy, train


class _IsaacLabRslModule(types.ModuleType):
    RslRlVecEnvWrapper: Mock


class _RslRunnersModule(types.ModuleType):
    OnPolicyRunner: Mock


def _training_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_steps_per_env=24,
        save_interval=50,
        init_noise_std=0.7,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[256, 128],
        activation="elu",
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        clip_param=0.2,
        value_loss_coef=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def test_build_runner_uses_rsl_rl_5_actor_critic_schema(monkeypatch) -> None:
    wrapped = object()
    runner = object()
    wrapper_factory = Mock(return_value=wrapped)
    runner_factory = Mock(return_value=runner)

    isaaclab_rl = types.ModuleType("isaaclab_rl")
    isaaclab_rl_rsl = _IsaacLabRslModule("isaaclab_rl.rsl_rl")
    isaaclab_rl_rsl.RslRlVecEnvWrapper = wrapper_factory
    rsl_rl = types.ModuleType("rsl_rl")
    rsl_rl_runners = _RslRunnersModule("rsl_rl.runners")
    rsl_rl_runners.OnPolicyRunner = runner_factory
    monkeypatch.setitem(sys.modules, "isaaclab_rl", isaaclab_rl)
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", isaaclab_rl_rsl)
    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_rl)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", rsl_rl_runners)

    env = object()
    assert build_runner(env, _training_cfg(), log_dir="logs", device="cuda:0") is runner
    wrapper_factory.assert_called_once_with(env, clip_actions=1.0)
    cfg = runner_factory.call_args.args[1]
    assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["policy"]}
    assert cfg["actor"] == {
        "class_name": "MLPModel",
        "hidden_dims": [128, 64],
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.7,
            "std_type": "scalar",
        },
    }
    assert cfg["critic"] == {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": None,
    }
    assert "policy" not in cfg


def test_train_returns_rsl_rl_5_logger_directory() -> None:
    runner = Mock()
    runner.logger.log_dir = "logs/run"

    result = train(runner, 12)

    runner.learn.assert_called_once_with(
        num_learning_iterations=12,
        init_at_random_ep_len=True,
    )
    assert result.iterations == 12
    assert result.log_dir == "logs/run"


def test_export_policy_uses_runner_public_jit_exporter(tmp_path) -> None:
    runner = Mock()
    metadata = {"obs_dim": 13, "action_dim": 4}

    export_policy(runner, tmp_path, metadata=metadata)

    runner.export_policy_to_jit.assert_called_once_with(str(tmp_path), "policy.pt")
    assert json.loads((tmp_path / "policy_meta.json").read_text()) == metadata

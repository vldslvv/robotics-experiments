"""Host tests for shared deterministic press-evaluation aggregation."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.paths import REPO_ROOT


def _contract():
    from robotics_experiments.rl import PolicyContract, PolicyInterface

    return PolicyContract(
        policy_interface=PolicyInterface.PRESS_V1,
        obs_schema_version=1,
        obs_dim=34,
        policy_rate_hz=17.0,
        action_kind="velocity",
        action_dim=4,
        max_body_velocity_mps=0.5,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_fraction=0.4,
    )


def _load_eval_press_module():
    pytest.importorskip("torch")
    script = REPO_ROOT / "isaac" / "runtime" / "standalone" / "eval_press.py"
    spec = importlib.util.spec_from_file_location("eval_press", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluation_aggregation_uses_metric_denominators() -> None:
    from robotics_experiments.workflows.isaac_lab import (
        PressEpisodeMetrics,
        summarize_press_episodes,
    )

    summary = summarize_press_episodes(
        episodes=(
            PressEpisodeMetrics(True, True, True, True, 1.0, 0.001),
            PressEpisodeMetrics(True, False, True, False, 2.0, 0.004),
            PressEpisodeMetrics(False, False, False, False, float("nan"), 0.0),
        )
    )

    assert summary.episodes == 3
    assert summary.confirmed_toggle_rate == pytest.approx(2.0 / 3.0)
    assert summary.stable_first_contact_rate == pytest.approx(0.5)
    assert summary.post_toggle_retreat_rate == pytest.approx(0.5)
    assert summary.full_completion_rate == pytest.approx(1.0 / 3.0)
    assert summary.median_toggle_time_s == pytest.approx(1.5)
    assert summary.peak_compression_overshoot_m == pytest.approx(0.004)


def test_seed_rollout_reseeds_existing_environment_and_preserves_reward_denominator() -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.press import PressRewardTerms
    from robotics_experiments.workflows.isaac_lab import evaluate_press_policy

    reward_names = tuple(field.name for field in fields(PressRewardTerms)) + ("total",)

    class FakePolicy:
        def __call__(self, observation):
            return torch.zeros((observation.shape[0], 4))

    class FakeEnv:
        device = "cpu"
        max_episode_length = 2

        def __init__(self):
            self.seeds = []
            self.step_count = 0
            self.press_task_result = None

        def reset(self, *, seed):
            self.seeds.append(seed)
            self.step_count = 0
            return {"policy": torch.zeros((2, 34))}, {}

        def step(self, _action):
            self.step_count += 1
            first_step = self.step_count == 1
            self.press_task_result = SimpleNamespace(
                metrics=SimpleNamespace(
                    contacted=torch.tensor([True, False]),
                    first_contact_stable=torch.tensor([True, False]),
                    toggled_this_step=torch.tensor([first_step, False]),
                    toggled=torch.tensor([True, False]),
                    retreated=torch.tensor([not first_step, False]),
                    toggle_time_s=torch.tensor([0.5, float("nan")]),
                    peak_compression_overshoot_m=torch.tensor([0.002, 0.0]),
                ),
                rewards=SimpleNamespace(**{name: torch.ones(2) for name in reward_names}),
            )
            done = torch.full((2,), not first_step)
            return {"policy": torch.zeros((2, 34))}, torch.zeros(2), done, done, {}

    env = FakeEnv()
    result = evaluate_press_policy(
        env=env,
        policy=FakePolicy(),
        seeds=(202,),
        episodes_per_seed=2,
        report=lambda _message: None,
    )

    assert env.seeds == [202]
    assert result.summary.confirmed_toggle_rate == 0.5
    assert result.summary.post_toggle_retreat_rate == 1.0
    assert {reward.mean for reward in result.reward_means} == {1.0}


def test_evaluation_reuses_one_environment_for_all_seeds(monkeypatch) -> None:
    evaluator = _load_eval_press_module()
    import robotics_experiments.workflows.isaac_lab as workflow
    from robotics_experiments.rl.training.selection import RewardMean

    class FakeEnv:
        device = "cpu"

        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    class FakePolicy:
        def __init__(self):
            self.eval_count = 0

        def eval(self):
            self.eval_count += 1

    env = FakeEnv()
    policy = FakePolicy()
    factory_calls = []
    evaluation_calls = []

    def fake_factory(**kwargs):
        factory_calls.append(kwargs)
        return env, SimpleNamespace(
            observation_schema_version=1,
            observation_space=34,
            policy_interface="press-v1",
        )

    result = workflow.PressEvaluationResult(
        summary=workflow.PressEvaluationSummary(
            episodes=3,
            confirmed_toggle_rate=1.0,
            stable_first_contact_rate=1.0,
            post_toggle_retreat_rate=1.0,
            full_completion_rate=1.0,
            peak_compression_overshoot_m=0.0,
            median_toggle_time_s=1.0,
        ),
        reward_means=(RewardMean(name="toggle", mean=1.0),),
    )

    def fake_evaluate(**kwargs):
        evaluation_calls.append(kwargs)
        return result

    monkeypatch.setattr(evaluator, "make_isaac_lab_env", fake_factory)
    monkeypatch.setattr(evaluator, "evaluate_press_policy", fake_evaluate)
    contract = _contract()
    monkeypatch.setattr(
        evaluator,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(contract=contract),
    )
    monkeypatch.setattr(evaluator.torch.jit, "load", lambda *_args, **_kwargs: policy)
    config = SimpleNamespace(
        rl=SimpleNamespace(
            training=SimpleNamespace(task_id="press-task"),
            press=SimpleNamespace(
                evaluation=SimpleNamespace(
                    seeds=(101, 202, 303),
                    episodes_per_seed=1,
                    success_rate_min=0.8,
                    first_contact_stability_min=0.85,
                    retreat_rate_min=0.95,
                )
            ),
        )
    )

    observed = evaluator._run(config=config, artifact_dir=Path("artifact"))  # noqa: SLF001

    assert observed == result
    assert len(factory_calls) == 1
    assert evaluation_calls[0]["seeds"] == (101, 202, 303)
    assert evaluation_calls[0]["env"] is env
    assert policy.eval_count == 1
    assert env.close_count == 1


def test_evaluation_closes_environment_when_rollout_fails(monkeypatch) -> None:
    evaluator = _load_eval_press_module()

    close_calls: list[bool] = []
    env = SimpleNamespace(device="cpu", close=lambda: close_calls.append(True))
    monkeypatch.setattr(
        evaluator,
        "make_isaac_lab_env",
        lambda **_kwargs: (
            env,
            SimpleNamespace(
                observation_schema_version=1,
                observation_space=34,
                policy_interface="press-v1",
            ),
        ),
    )
    contract = _contract()
    monkeypatch.setattr(
        evaluator,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(contract=contract),
    )
    monkeypatch.setattr(
        evaluator.torch.jit,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(eval=lambda: None),
    )
    monkeypatch.setattr(
        evaluator,
        "evaluate_press_policy",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("rollout failed")),
    )
    config = SimpleNamespace(
        rl=SimpleNamespace(
            training=SimpleNamespace(task_id="press-task"),
            press=SimpleNamespace(
                evaluation=SimpleNamespace(
                    seeds=(101,),
                    episodes_per_seed=1,
                    success_rate_min=0.8,
                    first_contact_stability_min=0.85,
                    retreat_rate_min=0.95,
                )
            ),
        )
    )

    with pytest.raises(RuntimeError, match="rollout failed"):
        evaluator._run(config=config, artifact_dir=Path("artifact"))  # noqa: SLF001

    assert close_calls == [True]

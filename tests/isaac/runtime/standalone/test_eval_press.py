"""Host tests for deterministic press-evaluation aggregation."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.paths import REPO_ROOT


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
    evaluator = _load_eval_press_module()
    episode = evaluator._EpisodeMetrics  # noqa: SLF001 - focused script helper test.
    summary = evaluator._summarize_episodes(  # noqa: SLF001 - focused script helper test.
        [
            episode(True, True, True, True, 1.0, 0.001),
            episode(True, False, True, False, 2.0, 0.004),
            episode(False, False, False, False, float("nan"), 0.0),
        ]
    )

    assert summary.episodes == 3
    assert summary.press_success_rate == pytest.approx(2.0 / 3.0)
    assert summary.first_contact_stability == pytest.approx(0.5)
    assert summary.retreat_rate == pytest.approx(0.5)
    assert summary.median_time_to_toggle_s == pytest.approx(1.5)
    assert summary.peak_compression_overshoot_m == pytest.approx(0.004)


def test_seed_rollout_reseeds_existing_environment() -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.press import PressRewardTerms

    evaluator = _load_eval_press_module()
    reward_names = tuple(field.name for field in fields(PressRewardTerms))

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
                rewards=SimpleNamespace(**{name: torch.zeros(2) for name in reward_names}),
            )
            done = torch.full((2,), not first_step)
            return {"policy": torch.zeros((2, 34))}, torch.zeros(2), done, done, {}

    env = FakeEnv()
    episodes, rewards = evaluator._run_seed(  # noqa: SLF001 - focused script helper test.
        env=env,
        policy=FakePolicy(),
        seed=202,
        episode_count=2,
    )

    assert env.seeds == [202]
    assert episodes[0].toggled
    assert episodes[0].retreated
    assert not episodes[1].contacted
    assert {count for _value, count in rewards.values()} == {4}


def test_evaluation_reuses_one_environment_for_all_seeds(monkeypatch) -> None:
    evaluator = _load_eval_press_module()
    import robotics_experiments.rl.press as press
    import robotics_experiments.workflows.isaac_lab as workflow

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
    rollout_seeds = []
    policy_loads = []

    def fake_factory(**kwargs):
        factory_calls.append(kwargs)
        return env, SimpleNamespace(
            observation_schema_version=1,
            observation_space=34,
            policy_interface="press-v1",
        )

    def fake_run_seed(*, env, policy, seed, episode_count):
        del env, policy
        rollout_seeds.append((seed, episode_count))
        episode = evaluator._EpisodeMetrics(  # noqa: SLF001 - focused script helper test.
            True, True, True, True, 1.0, 0.0
        )
        return [episode], {}

    monkeypatch.setattr(workflow, "make_isaac_lab_env", fake_factory)
    monkeypatch.setattr(press, "load_vector_policy", lambda **kwargs: policy_loads.append(kwargs))
    monkeypatch.setattr(evaluator.torch.jit, "load", lambda *_args, **_kwargs: policy)
    monkeypatch.setattr(evaluator, "_run_seed", fake_run_seed)
    config = SimpleNamespace(
        rl=SimpleNamespace(
            training=SimpleNamespace(task_id="press-task"),
            action=object(),
            policy_rate_hz=20.0,
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

    evaluator._run(config=config, artifact_dir=Path("artifact"))  # noqa: SLF001

    assert len(factory_calls) == 1
    assert factory_calls[0]["seed"] == 101
    assert rollout_seeds == [(101, 1), (202, 1), (303, 1)]
    assert len(policy_loads) == 1
    assert policy.eval_count == 1
    assert env.close_count == 1


def test_evaluation_closes_environment_when_rollout_fails(monkeypatch) -> None:
    evaluator = _load_eval_press_module()
    import robotics_experiments.rl.press as press
    import robotics_experiments.workflows.isaac_lab as workflow

    close_calls: list[bool] = []
    env = SimpleNamespace(device="cpu", close=lambda: close_calls.append(True))
    monkeypatch.setattr(
        workflow,
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
    monkeypatch.setattr(press, "load_vector_policy", lambda **_kwargs: None)
    monkeypatch.setattr(
        evaluator.torch.jit,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(eval=lambda: None),
    )
    monkeypatch.setattr(
        evaluator,
        "_run_seed",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("rollout failed")),
    )
    config = SimpleNamespace(
        rl=SimpleNamespace(
            training=SimpleNamespace(task_id="press-task"),
            action=object(),
            policy_rate_hz=20.0,
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

"""Host tests for deterministic approach evaluation and publication."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.alignment import ControllerSettings
from tests.support.paths import REPO_ROOT


def _contract():
    from robotics_experiments.rl import PolicyContract, PolicyInterface

    return PolicyContract(
        policy_interface=PolicyInterface.PRESS_V1,
        obs_schema_version=1,
        obs_dim=34,
        policy_rate_hz=20.0,
        action_kind="velocity",
        action_dim=4,
        max_body_velocity_mps=2.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_fraction=0.4,
        curriculum_stage="approach",
    )


def _controller_settings() -> ControllerSettings:
    return ControllerSettings(
        command_delay_s=0.2,
        kp_vel=(2.0, 2.0, 3.0),
        horizontal_velocity_error_gain_scale=1.0,
        horizontal_velocity_error_gain_start_mps=0.15,
        horizontal_velocity_error_gain_full_mps=0.30,
        ki_vel=(0.4, 0.4, 2.0),
        kd_vel=(1.5, 1.5, 1.5),
        velocity_filter_hz=10.0,
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 4.0),
        hover_thrust_fraction=0.5,
        hover_thrust_normalized=0.0,
        kp_att=(6.0, 6.0, 2.0),
        attitude_yaw_weight=0.4,
        kp_rate=(1.2, 1.2, 0.4),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.003, 0.003, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        rate_output_scale_nm=(1.0, 1.0, 1.0),
        rate_derivative_filter_hz=30.0,
        rate_integral_limit_nm=(0.3, 0.3, 0.2),
        max_tilt_cmd_deg=30.0,
        min_collective_thrust_fraction=0.05,
        max_collective_thrust_fraction=1.0,
        horizontal_thrust_margin_fraction=0.3,
        thruster_tau_inc_s=0.0,
        thruster_tau_dec_s=0.0,
        thruster_max_thrust_rate_nps=100000.0,
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


def _goal():
    return SimpleNamespace(
        range_min_m=0.40,
        range_max_m=0.42,
        alignment_error_max_m=0.03,
        linear_speed_max_mps=0.10,
        body_z_alignment_min=0.965926,
        velocity_reference=SimpleNamespace(
            range_center_m=0.41,
            wall_normal_gain_per_s=1.25,
            wall_normal_min_mps=-0.04,
            wall_normal_max_mps=0.08,
            tangential_gain_per_s=2.0,
            tangential_speed_max_mps=0.10,
        ),
    )


def test_approach_evaluation_aggregation() -> None:
    from robotics_experiments.workflows.isaac_lab import (
        PressApproachEpisodeMetrics,
        summarize_press_approach_episodes,
    )

    summary = summarize_press_approach_episodes(
        episodes=(
            PressApproachEpisodeMetrics(
                True, False, False, False, False, False, 0.405, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
            ),
            PressApproachEpisodeMetrics(
                False, True, True, False, False, False, 0.39, None, 1.0, 0.2, 0.3, 0.4, 0.5, 1.0
            ),
            PressApproachEpisodeMetrics(
                False, True, False, True, False, False, 0.40, None, 0.5, 0.3, 0.4, 0.5, 0.6, 1.0
            ),
            PressApproachEpisodeMetrics(
                False, False, False, False, True, False, 0.44, None, 1.0, 0.4, 0.5, 0.6, 0.7, 1.0
            ),
            PressApproachEpisodeMetrics(
                False, False, False, False, False, True, 0.43, None, 0.5, 0.5, 0.6, 0.7, 0.8, 1.0
            ),
        )
    )
    assert summary.episodes == 5
    assert summary.approach_success_rate == 0.2
    assert summary.collision_rate == 0.4
    assert summary.body_collision_rate == 0.2
    assert summary.pusher_collision_rate == 0.2
    assert summary.other_flight_failure_rate == 0.2
    assert summary.timeout_rate == 0.2
    assert summary.mean_closest_range_m == pytest.approx(0.413)
    assert summary.min_closest_range_m == pytest.approx(0.39)
    assert summary.median_completion_time_s == 1.0
    assert summary.mean_terminal_range_cost == pytest.approx(0.6)
    assert summary.mean_trajectory_alignment_cost == pytest.approx(0.3)
    assert summary.mean_velocity_tracking_cost == pytest.approx(0.4)
    assert summary.mean_completion_time_fraction == pytest.approx(0.9)


def test_trajectory_cost_tracks_full_contact_frame_target_velocity() -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.workflows.isaac_lab import press_evaluation

    observation = torch.zeros((1, 34))
    observation[:, :3] = torch.tensor([[0.45, 0.03, 0.04]])
    observation[:, 3:7] = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    observation[:, 10:13] = torch.tensor([[0.05, 0.06, 0.08]])
    observation[:, 19] = 1.0
    costs = press_evaluation._trajectory_costs(  # noqa: SLF001
        observation=observation,
        action=torch.zeros((1, 4)),
        goal_config=_goal(),
    )

    assert costs[1].item() == pytest.approx(0.0, abs=1.0e-6)
    observation[:, 12] = -0.02
    mismatched = press_evaluation._trajectory_costs(  # noqa: SLF001
        observation=observation,
        action=torch.zeros((1, 4)),
        goal_config=_goal(),
    )
    assert mismatched[1].item() == pytest.approx(1.0)


def test_seed_rollout_reseeds_and_enumerates_every_reward() -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.press import PressApproachRewardTerms
    from robotics_experiments.workflows.isaac_lab import evaluate_press_approach_policy

    reward_names = tuple(field.name for field in fields(PressApproachRewardTerms)) + ("total",)

    class FakePolicy:
        def __call__(self, observation):
            return torch.zeros((observation.shape[0], 4))

    class FakeEnv:
        device = "cpu"
        max_episode_length = 2
        step_dt = 0.05

        def __init__(self):
            self.seeds = []
            self.step_count = 0
            self.press_approach_result = None

        def reset(self, *, seed):
            self.seeds.append(seed)
            self.step_count = 0
            return {"policy": torch.zeros((2, 34))}, {}

        def step(self, _action):
            self.step_count += 1
            first = self.step_count == 1
            success = torch.tensor([first, False])
            timeout = torch.tensor([False, not first])
            self.press_approach_result = SimpleNamespace(
                state=SimpleNamespace(
                    closest_range_m=torch.tensor([0.41, 0.45 if first else 0.43]),
                    previous_range_m=torch.tensor([0.41, 0.45 if first else 0.43]),
                ),
                termination=SimpleNamespace(
                    success=success,
                    collision=torch.zeros(2, dtype=torch.bool),
                    body_collision=torch.zeros(2, dtype=torch.bool),
                    pusher_collision=torch.zeros(2, dtype=torch.bool),
                    flight_failure=torch.zeros(2, dtype=torch.bool),
                    time_out=timeout,
                ),
                rewards=SimpleNamespace(**{name: torch.ones(2) for name in reward_names}),
            )
            terminated = success
            truncated = timeout
            return (
                {"policy": torch.zeros((2, 34))},
                torch.zeros(2),
                terminated,
                truncated,
                {},
            )

    env = FakeEnv()
    result = evaluate_press_approach_policy(
        env=env,
        policy=FakePolicy(),
        seeds=(202,),
        episodes_per_seed=2,
        goal_config=_goal(),
        report=lambda _message: None,
    )
    assert env.seeds == [202]
    assert result.summary.approach_success_rate == 0.5
    assert result.summary.timeout_rate == 0.5
    assert result.summary.median_completion_time_s == pytest.approx(0.05)
    assert {reward.name for reward in result.reward_means} == set(reward_names)
    assert {reward.mean for reward in result.reward_means} == {1.0}


def _config():
    return SimpleNamespace(
        rl=SimpleNamespace(
            training=SimpleNamespace(task_id="press-task"),
            press=SimpleNamespace(
                curriculum_stage="approach",
                goal=_goal(),
                evaluation=SimpleNamespace(
                    seeds=(101, 202, 303),
                    episodes_per_seed=1,
                    success_rate_min=0.95,
                    collision_rate_max=0.01,
                ),
            ),
        )
    )


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

    monkeypatch.setattr(
        evaluator,
        "make_isaac_lab_env",
        fake_factory,
    )
    result = workflow.PressApproachEvaluationResult(
        summary=workflow.PressApproachEvaluationSummary(
            episodes=3,
            approach_success_rate=1.0,
            collision_rate=0.0,
            body_collision_rate=0.0,
            pusher_collision_rate=0.0,
            other_flight_failure_rate=0.0,
            timeout_rate=0.0,
            mean_closest_range_m=0.41,
            min_closest_range_m=0.405,
            median_completion_time_s=1.0,
            mean_terminal_range_cost=0.0,
            mean_trajectory_alignment_cost=0.1,
            mean_velocity_tracking_cost=0.1,
            mean_upright_error_cost=0.1,
            mean_action_delta_cost=0.1,
            mean_completion_time_fraction=0.1,
        ),
        reward_means=(RewardMean(name="success", mean=1.0),),
    )

    def fake_evaluate(**kwargs):
        evaluation_calls.append(kwargs)
        return result

    monkeypatch.setattr(evaluator, "evaluate_press_approach_policy", fake_evaluate)
    monkeypatch.setattr(
        evaluator,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(
            contract=_contract(),
            controller_settings=_controller_settings(),
        ),
    )
    monkeypatch.setattr(evaluator.torch.jit, "load", lambda *_args, **_kwargs: policy)

    observed = evaluator._run(  # noqa: SLF001 - standalone entrypoint seam
        config=_config(),
        artifact_dir=Path("artifact"),
    )
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
    monkeypatch.setattr(
        evaluator,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(
            contract=_contract(),
            controller_settings=_controller_settings(),
        ),
    )
    monkeypatch.setattr(
        evaluator.torch.jit,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(eval=lambda: None),
    )
    monkeypatch.setattr(
        evaluator,
        "evaluate_press_approach_policy",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("rollout failed")),
    )
    with pytest.raises(RuntimeError, match="rollout failed"):
        evaluator._run(  # noqa: SLF001 - standalone entrypoint seam
            config=_config(),
            artifact_dir=Path("artifact"),
        )
    assert close_calls == [True]


def test_evaluation_rejects_stage_mismatch(monkeypatch) -> None:
    evaluator = _load_eval_press_module()
    contract = replace(_contract(), curriculum_stage="future")
    monkeypatch.setattr(
        evaluator,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(contract=contract),
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluator._run(  # noqa: SLF001 - standalone entrypoint seam
            config=_config(),
            artifact_dir=Path("artifact"),
        )

"""Host tests for deterministic approach evaluation and publication."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.controller import ControllerSettings
from robotics_experiments.flight_parameters import (
    SignalStatistics,
    load_flight_parameters,
    summarize_signal,
)
from tests.support.paths import REPO_ROOT
from tests.support.press_evaluation_fakes import PressEvaluationEnv, ZeroPressPolicy

_FLIGHT_PARAMETERS = load_flight_parameters(
    REPO_ROOT / "tests/fixtures/flight_parameters/flight_parameters.json"
)


def _contract():
    from robotics_experiments.rl import PolicyContract, PolicyInterface

    return PolicyContract(
        policy_interface=PolicyInterface.PRESS,
        obs_schema_version=2,
        obs_dim=34,
        policy_rate_hz=20.0,
        action_kind="velocity",
        action_dim=4,
        max_body_velocity_mps=2.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_normalized=0.57,
        curriculum_stage="approach",
    )


def _controller_settings() -> ControllerSettings:
    return ControllerSettings(
        command_delay_s=0.2,
        velocity_controller_rate_hz=125.0,
        attitude_controller_rate_hz=187.5,
        kp_vel=(2.0, 2.0, 3.0),
        horizontal_velocity_error_gain_scale=1.0,
        horizontal_velocity_error_gain_start_mps=0.15,
        horizontal_velocity_error_gain_full_mps=0.30,
        horizontal_velocity_error_gain_end_mps=0.80,
        horizontal_velocity_braking_gain_scale=1.0,
        horizontal_velocity_braking_gain_start_mps=0.05,
        horizontal_velocity_braking_gain_full_mps=0.07,
        horizontal_velocity_braking_gain_end_mps=0.10,
        ki_vel=(0.4, 0.4, 2.0),
        kd_vel=(1.5, 1.5, 1.5),
        velocity_observation_scale=(1.0, 1.0, 1.0),
        velocity_observation_hold_scale=(1.0, 1.0, 1.0),
        velocity_observation_bias_mps=(0.0, 0.0, 0.0),
        velocity_observation_bias_command_start_mps=0.0,
        velocity_observation_bias_command_full_mps=0.01,
        velocity_estimator_bias_decay_hz=0.0,
        velocity_estimator_bias_hold_decay_hz=(0.0, 0.0, 0.0),
        velocity_filter_hz=(10.0, 10.0, 10.0),
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 4.0),
        hover_thrust_normalized=0.57,
        ctbr_effective_collective_scale=1.0,
        kp_att=(6.0, 6.0, 2.0),
        attitude_yaw_weight=0.4,
        kp_rate=(1.2, 1.2, 0.4),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.003, 0.003, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        angular_velocity_filter_hz=0.0,
        rate_derivative_filter_hz=30.0,
        yaw_torque_filter_hz=2.0,
        rate_integral_limit=(0.3, 0.3, 0.2),
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
        alignment_error_max_m=0.01,
        relative_attitude_error_max_rad=0.017453292519943295,
        linear_speed_max_mps=0.03,
        angular_speed_max_radps=0.05,
        body_z_alignment_min=0.965926,
        stable_steps=100,
        velocity_reference=SimpleNamespace(
            range_center_m=0.41,
            wall_normal_prediction_horizon_s=0.4,
            wall_normal_alignment_gate_m=0.03,
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

    common = {
        "hover_envelope_occupancy_rate": 0.5,
        "hover_envelope_utilization_mean": 0.4,
        "position_envelope_utilization_mean": 0.3,
        "attitude_envelope_utilization_mean": 0.2,
    }
    summary = summarize_press_approach_episodes(
        episodes=(
            PressApproachEpisodeMetrics(
                success=True,
                collision=False,
                other_flight_failure=False,
                range_undershoot=False,
                timeout=False,
                longest_hover_duration_s=30.0,
                closest_range_m=0.405,
                **common,
            ),
            PressApproachEpisodeMetrics(
                success=False,
                collision=True,
                other_flight_failure=False,
                range_undershoot=True,
                timeout=False,
                longest_hover_duration_s=6.0,
                closest_range_m=0.39,
                **common,
            ),
        ),
    )
    assert summary.episodes == 2
    assert summary.success_rate == 0.5
    assert summary.collision_rate == 0.5
    assert summary.range_undershoot_rate == 0.5
    assert summary.episode_longest_hover_duration_mean_s == 18.0
    assert summary.episode_longest_hover_duration_max_s == 30.0
    assert summary.min_closest_range_m == pytest.approx(0.39)


@pytest.mark.parametrize("success_ends", [(60, 60), (64, 72), (60, 72), (0, 0)])
def test_seed_rollout_reseeds_and_uses_each_successful_hold_window(*, success_ends) -> None:
    from robotics_experiments.rl.press import PressApproachRewardTerms
    from robotics_experiments.workflows.isaac_lab import evaluate_press_approach_policy

    reward_names = tuple(field.name for field in fields(PressApproachRewardTerms)) + ("total",)
    batch_size = 3
    hold_steps = 60
    warmup_steps = 12

    env = PressEvaluationEnv(success_ends=success_ends)
    reports: list[str] = []
    result = evaluate_press_approach_policy(
        env=env,
        policy=ZeroPressPolicy(),
        seeds=(202, 303),
        episodes_per_seed=batch_size,
        hold_steps=hold_steps,
        flight_parameters=replace(_FLIGHT_PARAMETERS, policy_warmup_s=6.0),
        report=reports.append,
        trace_worst_episode=True,
    )
    assert env.seeds == [202, 303]
    successful_count = sum(end > 0 for end in success_ends)
    assert result.summary.episodes == 6
    assert result.summary.success_rate == pytest.approx(successful_count / batch_size)
    assert result.summary.timeout_rate == pytest.approx((2 - successful_count) / batch_size)
    assert result.summary.collision_rate == pytest.approx(1 / batch_size)
    if successful_count:
        expected_values = [
            step / 1000
            for end in success_ends
            for step in range(end - hold_steps + warmup_steps + 1, end + 1)
        ]
        assert len(result.hover_statistics) == 10
        for statistics in result.hover_statistics:
            expected = summarize_signal(
                name=statistics.name, unit=statistics.unit, values=expected_values * 2
            )
            for field in fields(statistics):
                if field.name not in {"name", "unit"}:
                    assert getattr(statistics, field.name) == pytest.approx(
                        getattr(expected, field.name)
                    )
    else:
        assert not result.hover_statistics
    assert {reward.name for reward in result.reward_means} == set(reward_names)
    assert {reward.mean for reward in result.reward_means} == {1.0}
    assert any("worst trace seed=202 episode=0 step=0" in message for message in reports)
    assert any("normalized_target_error_enu=" in message for message in reports)
    assert not any("range_before_m=" in message or "lateral_m=" in message for message in reports)


def _config():
    return SimpleNamespace(
        rl=SimpleNamespace(
            training=SimpleNamespace(task_id="press-task"),
            policy_rate_hz=20.0,
            press=SimpleNamespace(
                selected_stage=lambda: SimpleNamespace(name="approach", hold_s=5.0),
                goal=_goal(),
                acceptance=SimpleNamespace(
                    success_rate_min=0.95,
                    collision_rate_max=0.01,
                    minimum_range_m=0.40,
                ),
                evaluation=SimpleNamespace(
                    validation_seeds=(101, 202, 303),
                    test_seeds=(404, 505, 606),
                    episodes_per_seed=1,
                ),
            ),
        )
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "field",
    [
        "policy_observation",
        "raw_action",
        "position_error_enu_x",
        "closest_range_m",
        "normalized_pose_error",
        "reward.total",
        "reward",
    ],
)
def test_rollout_rejects_non_finite_values(*, monkeypatch, field: str, value: float) -> None:
    from robotics_experiments.workflows.isaac_lab import evaluate_press_approach_policy

    env = PressEvaluationEnv(success_ends=(60, 72))
    reset = env.reset
    step = env.step
    policy = ZeroPressPolicy()

    def invalid_reset(*, seed):
        observation, info = reset(seed=seed)
        observation["policy"][1, 0] = value
        return observation, info

    def invalid_policy(observation, /):
        action = policy(observation)
        action[1, 0] = value
        return action

    def invalid_step(action, /):
        transition = step(action)
        result = env.press_approach_result
        targets = {
            "position_error_enu_x": result.hover_evaluation.position_error_enu_m[:, 0],
            "closest_range_m": result.state.closest_range_m,
            "normalized_pose_error": result.state.normalized_pose_error,
            "reward.total": result.rewards.total,
            "reward": transition[1],
        }
        targets[field][1] = value
        return transition

    if field == "policy_observation":
        monkeypatch.setattr(env, "reset", invalid_reset)
    elif field != "raw_action":
        monkeypatch.setattr(env, "step", invalid_step)
    with pytest.raises(ValueError, match=rf"non-finite {field}: seed=202 episode=1 step=0"):
        evaluate_press_approach_policy(
            env=env,
            policy=invalid_policy if field == "raw_action" else policy,
            seeds=(202, 303),
            episodes_per_seed=3,
            hold_steps=60,
            flight_parameters=_FLIGHT_PARAMETERS,
            report=lambda _message: None,
            trace_worst_episode=False,
        )
    assert env.seeds == [202]
    assert env.step_count == (0 if field in {"policy_observation", "raw_action"} else 1)


def test_rollout_ignores_non_finite_values_after_episode_completion(*, monkeypatch) -> None:
    from robotics_experiments.workflows.isaac_lab import evaluate_press_approach_policy

    env = PressEvaluationEnv(success_ends=(60, 72))
    step = env.step

    def step_with_invalid_completed_episode(action, /):
        transition = step(action)
        if env.step_count > 20:
            env.press_approach_result.hover_evaluation.position_error_enu_m[2] = float("nan")
            env.press_approach_result.rewards.total[2] = float("inf")
        if env.step_count >= 20:
            transition[0]["policy"][2] = float("nan")
        return transition

    monkeypatch.setattr(env, "step", step_with_invalid_completed_episode)
    result = evaluate_press_approach_policy(
        env=env,
        policy=lambda observation: observation[:, :4],
        seeds=(202,),
        episodes_per_seed=3,
        hold_steps=60,
        flight_parameters=_FLIGHT_PARAMETERS,
        report=lambda _message: None,
        trace_worst_episode=False,
    )
    assert result.summary.success_rate == pytest.approx(2 / 3)
    assert all(reward.mean == 1.0 for reward in result.reward_means)


def test_hover_statistics_pool_samples_across_episodes_and_seeds(*, monkeypatch) -> None:
    from robotics_experiments.workflows.isaac_lab import evaluate_press_approach_policy

    env = PressEvaluationEnv(success_ends=(60, 72))
    step = env.step

    def constant_error_step(action, /):
        transition = step(action)
        error = env.press_approach_result.hover_evaluation.position_error_enu_m
        error[:] = 0.0 if env.seeds[-1] == 202 else 2.0
        return transition

    monkeypatch.setattr(env, "step", constant_error_step)
    result = evaluate_press_approach_policy(
        env=env,
        policy=ZeroPressPolicy(),
        seeds=(202, 303),
        episodes_per_seed=3,
        hold_steps=60,
        flight_parameters=_FLIGHT_PARAMETERS,
        report=lambda _message: None,
        trace_worst_episode=False,
    )
    statistics = next(
        item for item in result.hover_statistics if item.name == "position_error_enu_x"
    )
    assert statistics.maximum_absolute == 2.0
    assert statistics.rmse == pytest.approx(2.0**0.5)
    assert statistics.p95_absolute == 2.0
    assert statistics.p99_absolute == 2.0


def test_evaluation_reuses_one_environment_for_all_seeds(monkeypatch, tmp_path: Path) -> None:
    evaluator = _load_eval_press_module()
    monkeypatch.setattr(evaluator, "_press_comparison_fingerprint", lambda **_kwargs: "comparison")
    import robotics_experiments.workflows.isaac_lab as workflow
    from robotics_experiments.rl.evaluation import RewardMean

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
            observation_schema_version=2,
            observation_space=34,
            policy_interface="press",
        )

    monkeypatch.setattr(
        workflow,
        "make_isaac_lab_env",
        fake_factory,
    )
    result = workflow.PressApproachEvaluationResult(
        summary=workflow.PressApproachEvaluationSummary(
            episodes=3,
            success_rate=1.0,
            collision_rate=0.0,
            other_flight_failure_rate=0.0,
            range_undershoot_rate=0.0,
            timeout_rate=0.0,
            hover_envelope_occupancy_rate=1.0,
            hover_envelope_utilization_mean=0.0,
            position_envelope_utilization_mean=0.0,
            attitude_envelope_utilization_mean=0.0,
            episode_longest_hover_duration_mean_s=30.0,
            episode_longest_hover_duration_max_s=30.0,
            min_closest_range_m=0.405,
        ),
        reward_means=(RewardMean(name="terminal_outcome", mean=1.0),),
        hover_statistics=tuple(
            SignalStatistics(
                name=limit.name,
                unit=limit.unit,
                minimum=0.0,
                maximum=0.0,
                maximum_absolute=0.0,
                rmse=0.0,
                p95_absolute=0.0,
                p99_absolute=0.0,
            )
            for limit in _FLIGHT_PARAMETERS.hover.acceptance_limits
        ),
    )

    def fake_evaluate(**kwargs):
        evaluation_calls.append(kwargs)
        return result

    monkeypatch.setattr(workflow, "evaluate_press_approach_policy", fake_evaluate)
    policy_bytes = b"policy"
    import hashlib

    digest = hashlib.sha256(policy_bytes).hexdigest()
    monkeypatch.setattr(
        evaluator,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(
            contract=_contract(),
            controller_settings=_controller_settings(),
            flight_parameters=_FLIGHT_PARAMETERS,
            values={
                "comparison_fingerprint": "comparison",
                "selected_checkpoint_iteration": 10,
                "selected_policy_digest": digest,
                "test_evaluation": {
                    "policy_digest": digest,
                    "workload": {
                        "role": "test",
                        "seeds": [404, 505, 606],
                        "episodes_per_seed": 1,
                        "episode_count": 3,
                    },
                },
            },
        ),
    )
    monkeypatch.setattr(evaluator.torch.jit, "load", lambda *_args, **_kwargs: policy)
    monkeypatch.setattr(
        evaluator,
        "prepare_policy_inputs",
        lambda **_kwargs: SimpleNamespace(
            flight_parameters=_FLIGHT_PARAMETERS,
            hover_reset_samples="hover-reset-samples",
        ),
    )
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "policy.pt").write_bytes(policy_bytes)

    observed = evaluator._run(  # noqa: SLF001 - standalone entrypoint
        config=_config(),
        artifact_dir=artifact_dir,
    )
    assert observed == result
    assert len(factory_calls) == 1
    assert evaluation_calls[0]["seeds"] == (404, 505, 606)
    assert evaluation_calls[0]["env"] is env
    assert policy.eval_count == 1
    assert env.close_count == 1


def test_evaluation_closes_environment_when_rollout_fails(monkeypatch) -> None:
    evaluator = _load_eval_press_module()
    monkeypatch.setattr(evaluator, "_press_comparison_fingerprint", lambda **_kwargs: "comparison")
    import robotics_experiments.workflows.isaac_lab as workflow

    close_calls: list[bool] = []
    env = SimpleNamespace(device="cpu", close=lambda: close_calls.append(True))
    monkeypatch.setattr(
        workflow,
        "make_isaac_lab_env",
        lambda **_kwargs: (
            env,
            SimpleNamespace(
                observation_schema_version=2,
                observation_space=34,
                policy_interface="press",
            ),
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(
            contract=_contract(),
            controller_settings=_controller_settings(),
            flight_parameters=_FLIGHT_PARAMETERS,
            values={
                "comparison_fingerprint": "comparison",
                "selected_checkpoint_iteration": 10,
            },
        ),
    )
    monkeypatch.setattr(
        evaluator.torch.jit,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(eval=lambda: None),
    )
    monkeypatch.setattr(
        evaluator,
        "prepare_policy_inputs",
        lambda **_kwargs: SimpleNamespace(
            flight_parameters=_FLIGHT_PARAMETERS,
            hover_reset_samples="hover-reset-samples",
        ),
    )
    monkeypatch.setattr(
        workflow,
        "evaluate_press_approach_policy",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("rollout failed")),
    )
    with pytest.raises(RuntimeError, match="rollout failed"):
        evaluator._run(  # noqa: SLF001 - standalone entrypoint
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
        evaluator._run(  # noqa: SLF001 - standalone entrypoint
            config=_config(),
            artifact_dir=Path("artifact"),
        )


def test_hover_feedback_signs_frames_and_limits():
    import math

    import torch

    from robotics_experiments.config import load_config

    module = _load_eval_press_module()
    config = load_config(REPO_ROOT / "configs/train_press_hover_quad.yaml")
    policy = module._HoverFeedback(  # noqa: SLF001 - host test of the CLI feedback policy
        config=config,
        flight_parameters=_FLIGHT_PARAMETERS,
    )
    observation = torch.zeros((5, 34), dtype=torch.float64)
    observation[:, 6] = 1.0
    observation[:, 19] = 1.0
    observation[0, 0] = 1.0
    observation[1, 0] = 1.0
    observation[1, 18:20] = math.sqrt(0.5)
    observation[2, 5:7] = torch.tensor([math.sin(0.1), math.cos(0.1)])
    observation[3, :3] = 1e6
    observation[3, 5:7] = torch.tensor([math.sin(1.0), math.cos(1.0)])
    observation[4, 10:13] = torch.tensor([0.1, -0.1, 0.1])
    action = policy(observation)
    assert action[0, 0] < 0.0
    assert action[0, 1:].tolist() == [0.0, 0.0, 0.0]
    assert action[1, 0].item() == pytest.approx(0.0, abs=1e-12)
    assert action[1, 1].item() == pytest.approx(-action[0, 0].item())
    assert action[2, 3] < 0.0
    assert action[3].tolist() == [-1.0, -1.0, -1.0, -1.0]
    assert action[4, :3].tolist() == pytest.approx([-0.04, 0.04, -0.04])
    assert torch.equal(action, policy(observation))


def test_hover_feasibility_reuses_production_evaluation_and_pinned_resets(monkeypatch, tmp_path):
    import robotics_experiments.workflows.isaac_lab as workflow
    from robotics_experiments.config import load_config

    module = _load_eval_press_module()
    config = load_config(REPO_ROOT / "configs/train_press_hover_quad.yaml")
    samples = object()
    inputs = SimpleNamespace(
        controller=SimpleNamespace(
            selected_settings=_controller_settings(), run_id="controller-run"
        ),
        controller_path=tmp_path / "controller.json",
        controller_digest="controller-digest",
        flight_parameters=_FLIGHT_PARAMETERS,
        flight_parameters_path=tmp_path / "flights",
        flight_parameters_digest="flight-digest",
        hover_reset_samples=samples,
    )
    monkeypatch.setattr(module, "prepare_policy_inputs", lambda **_kwargs: inputs)
    monkeypatch.setattr(module, "press_policy_contract", lambda **_kwargs: _contract())
    calls = []
    closed = []
    env = SimpleNamespace(close=lambda: closed.append(True))

    def factory(**kwargs):
        calls.append(kwargs)
        return env, object()

    def evaluate(**kwargs):
        calls.append(kwargs)
        raise ValueError("recorded evaluation failure")

    monkeypatch.setattr(workflow, "make_isaac_lab_env", factory)
    monkeypatch.setattr(workflow, "evaluate_press_approach_policy", evaluate)
    output = tmp_path / "feasibility"
    with pytest.raises(ValueError, match="recorded evaluation failure"):
        module._run_feasibility(config=config, output_dir=output)  # noqa: SLF001 - CLI workflow test
    assert closed == [True]
    assert calls[0]["hover_reset_samples"] is samples
    assert calls[0]["training"] is False
    assert calls[0]["experiment_config"] is config
    assert calls[1]["env"] is env
    assert calls[1]["seeds"] == (101, 202, 303)
    assert calls[1]["episodes_per_seed"] == 100
    assert calls[1]["hold_steps"] == 600
    assert "recorded evaluation failure" in (output / "feasibility.json").read_text()

"""Host tests for deterministic approach evaluation and publication."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.controller import ControllerSettings
from robotics_experiments.flight_parameters import SignalStatistics, load_flight_parameters
from tests.support.paths import REPO_ROOT

_FLIGHT_PARAMETERS = load_flight_parameters(
    REPO_ROOT / "tests/fixtures/flight_parameters/flight_parameters.json"
)


def _contract():
    from robotics_experiments.rl import PolicyContract, PolicyInterface

    return PolicyContract(
        policy_interface=PolicyInterface.PRESS,
        obs_schema_version=1,
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
            observation = torch.zeros((2, 34))
            observation[:, 6] = 1.0
            observation[:, 19] = 1.0
            return {"policy": observation}, {}

        def step(self, _action):
            self.step_count += 1
            first = self.step_count == 1
            success = torch.tensor([first, False])
            timeout = torch.tensor([False, not first])
            self.press_approach_result = SimpleNamespace(
                state=SimpleNamespace(
                    closest_range_m=torch.tensor([0.41, 0.45 if first else 0.43]),
                    normalized_pose_error=torch.tensor([0.0, 2.0]),
                    stable_goal_steps=torch.tensor([100 if first else 0, 0]),
                    longest_stable_goal_steps=torch.tensor([100 if first else 0, 0]),
                ),
                range_m=torch.tensor([0.41, 0.45 if first else 0.43]),
                range_error_m=torch.tensor([0.0, 0.04 if first else 0.02]),
                alignment_error_m=torch.zeros(2),
                attitude_error_rad=torch.zeros(2),
                linear_speed_mps=torch.zeros(2),
                angular_speed_radps=torch.zeros(2),
                contact_clear=torch.ones(2, dtype=torch.bool),
                strict_goal=torch.tensor([first, False]),
                hover_evaluation=SimpleNamespace(
                    position_error_enu_m=torch.zeros((2, 3)),
                    horizontal_position_error_m=torch.zeros(2),
                    position_error_m=torch.zeros(2),
                    attitude_error_rpy_rad=torch.zeros((2, 3)),
                    tilt_error_rad=torch.zeros(2),
                    orientation_error_rad=torch.zeros(2),
                    position_utilization=torch.tensor([0.0, 2.0]),
                    attitude_utilization=torch.zeros(2),
                ),
                termination=SimpleNamespace(
                    success=success,
                    collision=torch.zeros(2, dtype=torch.bool),
                    body_collision=torch.zeros(2, dtype=torch.bool),
                    pusher_collision=torch.zeros(2, dtype=torch.bool),
                    flight_failure=torch.zeros(2, dtype=torch.bool),
                    range_undershoot=torch.zeros(2, dtype=torch.bool),
                    time_out=timeout,
                ),
                rewards=SimpleNamespace(**{name: torch.ones(2) for name in reward_names}),
            )
            terminated = success
            truncated = timeout
            observation = torch.zeros((2, 34))
            observation[:, 6] = 1.0
            observation[:, 19] = 1.0
            return (
                {"policy": observation},
                torch.zeros(2),
                terminated,
                truncated,
                {},
            )

    env = FakeEnv()
    reports: list[str] = []
    result = evaluate_press_approach_policy(
        env=env,
        policy=FakePolicy(),
        seeds=(202,),
        episodes_per_seed=2,
        hold_steps=100,
        flight_parameters=_FLIGHT_PARAMETERS,
        report=reports.append,
        trace_worst_episode=True,
    )
    assert env.seeds == [202]
    assert result.summary.success_rate == 0.5
    assert result.summary.timeout_rate == 0.5
    assert len(result.hover_statistics) == 10
    assert all(statistics.maximum_absolute == 0.0 for statistics in result.hover_statistics)
    assert {reward.name for reward in result.reward_means} == set(reward_names)
    assert {reward.mean for reward in result.reward_means} == {1.0}
    assert any(
        "worst trace seed=202 episode=0 closest_range_m=0.410000" in message for message in reports
    )
    assert any("worst trace seed=202 episode=0 step=0" in message for message in reports)


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
            observation_schema_version=1,
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
                observation_schema_version=1,
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

"""Tests for the strict policy-qualification monitoring registry."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.rl.evaluation import EvaluationMetric, PolicyEvaluation
from robotics_experiments.rl.training import rsl_rl
from robotics_experiments.rl.training._monitoring import (
    EVALUATION_METRICS,
    evaluation_metric_names,
    metric_registry_payload,
    write_evaluation_scalars,
    write_system_scalars,
)


class _Writer:
    def __init__(self) -> None:
        self.events: list[tuple[str, float, int]] = []
        self.options: list[dict[str, object]] = []
        self.log_dir = "logs"

    def add_scalar(self, tag, scalar_value, global_step, /, *args, **kwargs) -> None:
        del args
        self.events.append((str(tag), float(scalar_value), int(global_step)))
        self.options.append(dict(kwargs))


def _evaluation(*, undefined_completion_time: bool = False) -> PolicyEvaluation:
    values = {
        "approach_success_rate": 0.96,
        "collision_rate": 0.01,
        "other_flight_failure_rate": 0.0,
        "range_undershoot_rate": 0.0,
        "timeout_rate": 0.03,
        "completed_episodes": 300.0,
        "range_pass_rate": 0.9,
        "alignment_pass_rate": 0.8,
        "attitude_pass_rate": 0.95,
        "linear_speed_pass_rate": 0.7,
        "angular_speed_pass_rate": 0.85,
        "contact_clear_pass_rate": 0.99,
        "complete_goal_pass_rate": 0.6,
        "mean_max_stable_goal_fraction": 0.8,
        "maximum_stable_goal_fraction": 1.0,
        "mean_normalized_pose_error": 0.5,
        "minimum_closest_range_m": 0.405,
        "mean_closest_range_m": 0.41,
        "all_traces_finite": 1.0,
        "held_out_task_cost": 0.25,
        "body_collision_rate": 0.006,
        "pusher_collision_rate": 0.004,
        "median_successful_completion_time_s": None if undefined_completion_time else 1.5,
        "mean_terminal_range_cost": 0.1,
        "mean_trajectory_alignment_cost": 0.2,
        "mean_relative_attitude_error_cost": 0.25,
        "mean_velocity_tracking_cost": 0.3,
        "mean_angular_velocity_cost": 0.35,
        "mean_upright_error_cost": 0.4,
        "mean_action_delta_cost": 0.5,
        "mean_completion_time_fraction": 0.6,
    }
    values.update(
        {
            spec.raw_name: 0.01
            for spec in EVALUATION_METRICS
            if spec.raw_name is not None and spec.raw_name.startswith("flight_parameters.")
        }
    )
    return PolicyEvaluation(
        iteration=10,
        checkpoint_path=Path("model_10.pt"),
        policy_digest="digest",
        metrics=tuple(EvaluationMetric(name, values[name]) for name in evaluation_metric_names()),
        reward_means=(),
        source="fresh-validation",
    )


def test_validation_and_test_share_exact_ordered_suffixes_and_report_values() -> None:
    writer = _Writer()
    evaluation = _evaluation()
    validation_tags = write_evaluation_scalars(
        writer=writer,
        role="val",
        evaluation=evaluation,
        all_checks_passed=True,
        step=17,
    )
    test_tags = write_evaluation_scalars(
        writer=writer,
        role="test",
        evaluation=evaluation,
        all_checks_passed=True,
        step=17,
    )
    expected_suffixes = tuple(spec.suffix for spec in EVALUATION_METRICS)
    assert tuple(tag.removeprefix("val/") for tag in validation_tags) == expected_suffixes
    assert tuple(tag.removeprefix("test/") for tag in test_tags) == expected_suffixes
    report_values = {metric.name: metric.value for metric in evaluation.metrics}
    event_values = {tag: value for tag, value, _step in writer.events}
    for spec in EVALUATION_METRICS:
        expected = 1.0 if spec.raw_name is None else report_values[spec.raw_name]
        assert event_values[f"val/{spec.suffix}"] == expected
        assert event_values[f"test/{spec.suffix}"] == expected
    assert writer.options
    assert all(
        options == {"new_style": True, "double_precision": True} for options in writer.options
    )


def test_metric_registry_names_rates_units_and_unavailable_values() -> None:
    registry = metric_registry_payload()
    assert registry["revision"] == 16
    assert all(spec.suffix == spec.suffix.lower() for spec in EVALUATION_METRICS)
    assert all(not spec.suffix.endswith("_percent") for spec in EVALUATION_METRICS)
    rate_specs = tuple(spec for spec in EVALUATION_METRICS if spec.suffix.endswith("_rate"))
    assert rate_specs
    assert all(spec.unit == "fraction" for spec in rate_specs)
    assert next(spec.unit for spec in EVALUATION_METRICS if spec.suffix.endswith("_range_m")) == "m"
    writer = _Writer()
    tags = write_evaluation_scalars(
        writer=writer,
        role="val",
        evaluation=_evaluation(undefined_completion_time=True),
        all_checks_passed=False,
        step=0,
    )
    assert "val/outcome/median_successful_completion_time_s" not in tags
    assert all(tag.startswith(("train/", "val/", "test/", "system/")) for tag in tags)


def test_rate_outside_fraction_range_is_rejected() -> None:
    evaluation = _evaluation()
    metrics = list(evaluation.metrics)
    metrics[0] = EvaluationMetric("approach_success_rate", 96.0)
    with pytest.raises(ValueError, match="must be in"):
        write_evaluation_scalars(
            writer=_Writer(),
            role="val",
            evaluation=PolicyEvaluation(
                iteration=evaluation.iteration,
                checkpoint_path=evaluation.checkpoint_path,
                policy_digest=evaluation.policy_digest,
                metrics=tuple(metrics),
                reward_means=(),
                source=evaluation.source,
            ),
            all_checks_passed=False,
            step=0,
        )


def test_system_metric_tags_are_complete_ordered_and_namespaced() -> None:
    summary = {
        "gpu_utilization_rate_mean": 0.5,
        "gpu_utilization_rate_p95": 0.7,
        "gpu_memory_used_gib_peak": 3.0,
        "gpu_memory_utilization_rate_peak": 0.4,
        "gpu_power_w_mean": 120.0,
        "gpu_temperature_c_max": 60.0,
        "process_resident_memory_gib_peak": 2.0,
        "process_cpu_cores_used_mean": 1.5,
        "duration_s": 10.0,
        "work_items_per_s": 30.0,
    }
    writer = _Writer()
    tags = write_system_scalars(writer=writer, role="val", summary=summary, step=20)
    assert tags == (
        "system/val/gpu_utilization_rate_mean",
        "system/val/gpu_utilization_rate_p95",
        "system/val/gpu_memory_used_gib_peak",
        "system/val/gpu_memory_utilization_rate_peak",
        "system/val/gpu_power_w_mean",
        "system/val/gpu_temperature_c_max",
        "system/val/process_resident_memory_gib_peak",
        "system/val/process_cpu_cores_used_mean",
        "system/val/duration_s",
        "system/val/work_items_per_s",
    )


def test_rsl_metrics_are_remapped_and_x_coordinate_never_rewinds() -> None:
    class BaseLogger:
        def __init__(self) -> None:
            self.writer = _Writer()
            self.log_dir = None
            self.disable_logs = False

        def process_env_step(self, *args, **kwargs) -> None:
            del args, kwargs

        def log(self, *, it: int, **kwargs) -> None:
            del kwargs
            self.writer.add_scalar("Loss/value", 1.0, it)
            self.writer.add_scalar("Loss/surrogate", 2.0, it)
            self.writer.add_scalar("Loss/entropy", 3.0, it)
            self.writer.add_scalar("Policy/mean_noise_std", 0.1, it)
            self.writer.add_scalar("Train/mean_reward", 4.0, it)
            self.writer.add_scalar("Train/mean_episode_length", 5.0, it)
            self.writer.add_scalar("Train/mean_reward/time", 4.0, it * 6)
            self.writer.add_scalar("Train/mean_episode_length/time", 5.0, it * 6)
            self.writer.add_scalar("Perf/collection_time", 0.2, it)
            self.writer.add_scalar("Perf/learning_time", 0.3, it)

    writer = _Writer()
    phase = SimpleNamespace(
        learning_rate=1.0e-4,
        range_undershoot_min_m=0.38,
    )
    runner = SimpleNamespace(logger=BaseLogger())
    rsl_rl.configure_monitoring(runner=runner, writer=writer, phase=phase, phase_number=1)
    runner.logger.log(it=170)
    runner.logger.writer.configure_phase(
        phase=SimpleNamespace(
            learning_rate=3.0e-5,
            range_undershoot_min_m=0.39,
        ),
        phase_number=2,
    )
    runner.logger.log(it=141)
    runner.logger.writer.configure_phase(
        phase=SimpleNamespace(
            learning_rate=1.0e-5,
            range_undershoot_min_m=0.40,
        ),
        phase_number=3,
    )
    runner.logger.log(it=141)
    assert rsl_rl.monitoring_step(runner=runner) == 3
    checkpoint_events = [
        (value, step)
        for tag, value, step in writer.events
        if tag == "train/schedule/checkpoint_update"
    ]
    assert checkpoint_events == [(170.0, 1), (141.0, 2), (141.0, 3)]
    assert all(
        tag.startswith(("train/", "val/", "test/", "system/"))
        for tag, _value, _step in writer.events
    )
    assert not any(
        tag.startswith(
            ("Training/", "Validation/", "Acceptance/", "Loss/", "Policy/", "Perf/", "Train/")
        )
        for tag, _value, _step in writer.events
    )


def test_embedded_runner_logging_surface_uses_the_same_registry() -> None:
    writer = _Writer()
    phase = SimpleNamespace(
        learning_rate=1.0e-4,
        range_undershoot_min_m=None,
    )
    runner = SimpleNamespace()
    rsl_rl.configure_monitoring(runner=runner, writer=writer, phase=phase, phase_number=1)
    runner.writer.add_scalar("Loss/value_function", 1.0, 7)
    runner.writer.add_scalar("Perf/collection time", 0.2, 7)
    runner.writer.add_scalar("Perf/learning_time", 0.3, 7)
    assert rsl_rl.monitoring_step(runner=runner) == 1
    assert runner.log_dir == "logs"
    assert {tag for tag, _value, _step in writer.events} >= {
        "train/optimization/value_loss",
        "train/performance/rollout_time_s",
        "train/performance/optimizer_update_time_s",
        "train/performance/total_update_time_s",
    }


def test_training_chunk_advances_when_restored_learner_iteration_repeats() -> None:
    writer = _Writer()
    phase = SimpleNamespace(
        learning_rate=1.0e-4,
        range_undershoot_min_m=None,
    )
    runner = SimpleNamespace(current_learning_iteration=0)
    rsl_rl.configure_monitoring(
        runner=runner,
        writer=writer,
        phase=phase,
        phase_number=1,
    )

    def learn(*, num_learning_iterations: int, init_at_random_ep_len: bool) -> None:
        del num_learning_iterations, init_at_random_ep_len
        runner.writer.add_scalar("Loss/value_function", 1.0, 1)
        runner.writer.add_scalar("Perf/collection time", 0.2, 1)
        runner.writer.add_scalar("Perf/learning_time", 0.3, 1)

    runner.learn = learn
    rsl_rl.train_chunk(
        runner=runner,
        completed_iterations=0,
        iterations=1,
        randomize_initial_episode_lengths=True,
    )
    runner.writer.configure_phase(phase=phase, phase_number=2)
    rsl_rl.train_chunk(
        runner=runner,
        completed_iterations=0,
        iterations=1,
        randomize_initial_episode_lengths=False,
    )

    assert rsl_rl.monitoring_step(runner=runner) == 2
    assert [
        (value, step)
        for tag, value, step in writer.events
        if tag == "train/schedule/checkpoint_update"
    ] == [(1.0, 1), (1.0, 2)]

"""Tests for deterministic evaluation progress, acceptance, and publication."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from robotics_experiments.config import RlPlateauCriteriaConfig
from robotics_experiments.flight_parameters import SignalStatistics, load_flight_parameters
from robotics_experiments.rl.evaluation import EvaluationMetric, PolicyEvaluation, RewardMean
from robotics_experiments.rl.press import (
    create_press_acceptance_criteria,
    evaluate_press_acceptance,
)
from robotics_experiments.rl.press._fingerprint import _press_comparison_fingerprint
from robotics_experiments.rl.training.progress import (
    PlateauState,
    advance_plateau,
    select_best_evaluation,
)
from robotics_experiments.rl.training.publication import publish_policy_labels
from tests.support.paths import REPO_ROOT

_FLIGHT_PARAMETERS = load_flight_parameters(
    REPO_ROOT / "tests/fixtures/flight_parameters/flight_parameters.json"
)


def _acceptance_criteria(config):
    return create_press_acceptance_criteria(
        config=config.rl.press.acceptance,
        flight_parameters=_FLIGHT_PARAMETERS,
    )


def _criteria(*, patience: int = 10) -> RlPlateauCriteriaConfig:
    return RlPlateauCriteriaConfig(
        metric_name="evaluation_diagnostic_cost",
        direction="minimize",
        minimum_delta=0.003,
        patience_evaluations=patience,
    )


def _evaluation(*, iteration: int, cost: float) -> PolicyEvaluation:
    return PolicyEvaluation(
        iteration=iteration,
        checkpoint_path=Path(f"logs/model_{iteration}.pt"),
        policy_digest=f"digest-{iteration}",
        metrics=(EvaluationMetric("evaluation_diagnostic_cost", cost),),
        reward_means=(),
        source="fresh",
    )


def _press_result(**overrides):
    from robotics_experiments.workflows.isaac_lab import (
        PressApproachEvaluationResult,
        PressApproachEvaluationSummary,
    )

    values = {
        "episodes": 300,
        "success_rate": 0.95,
        "collision_rate": 0.01,
        "other_flight_failure_rate": 0.0,
        "range_undershoot_rate": 0.0,
        "timeout_rate": 0.04,
        "hover_envelope_occupancy_rate": 0.95,
        "hover_envelope_utilization_mean": 0.1,
        "position_envelope_utilization_mean": 0.1,
        "attitude_envelope_utilization_mean": 0.1,
        "episode_longest_hover_duration_mean_s": 28.5,
        "episode_longest_hover_duration_max_s": 30.0,
        "min_closest_range_m": 0.405,
    }
    values.update(overrides)
    return PressApproachEvaluationResult(
        summary=PressApproachEvaluationSummary(**values),
        reward_means=(),
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


def _scored(**overrides) -> PolicyEvaluation:
    from robotics_experiments.workflows.isaac_lab import press_policy_evaluation

    return press_policy_evaluation(
        iteration=10,
        checkpoint_path=Path("logs/model_10.pt"),
        policy_digest="digest",
        result=_press_result(**overrides),
        source="fresh",
    )


def test_unavailable_hover_statistics_keep_the_complete_null_metric_contract() -> None:
    from robotics_experiments.rl.training._monitoring import evaluation_metric_names
    from robotics_experiments.workflows.isaac_lab import press_policy_evaluation

    evaluation = press_policy_evaluation(
        iteration=10,
        checkpoint_path=Path("logs/model_10.pt"),
        policy_digest="digest",
        result=replace(_press_result(), hover_statistics=()),
        source="fresh",
    )
    hover_metrics = tuple(
        metric for metric in evaluation.metrics if metric.name.startswith("flight_parameters.")
    )
    assert tuple(metric.name for metric in evaluation.metrics) == evaluation_metric_names()
    assert len(hover_metrics) == 40
    assert all(metric.value is None for metric in hover_metrics)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["summary", "reward", "hover_statistic"])
def test_evaluation_rejects_non_finite_aggregates(*, field: str, value: float) -> None:
    from robotics_experiments.workflows.isaac_lab import press_policy_evaluation

    result = _press_result()
    if field == "summary":
        result = replace(
            result, summary=replace(result.summary, hover_envelope_utilization_mean=value)
        )
        name = "hover_envelope_utilization_mean"
    elif field == "reward":
        result = replace(result, reward_means=(RewardMean(name="total", mean=value),))
        name = "reward.total"
    else:
        statistics = replace(result.hover_statistics[0], rmse=value)
        result = replace(result, hover_statistics=(statistics,))
        name = f"flight_parameters.{statistics.name}.rmse"
    with pytest.raises(
        ValueError, match=f"non-finite evaluation {name}: source=fresh-test iteration=10"
    ):
        press_policy_evaluation(
            iteration=10,
            checkpoint_path=Path("checkpoint.pt"),
            policy_digest="digest",
            result=result,
            source="fresh-test",
        )


@pytest.mark.parametrize("stage_name", ["hover", "approach"])
def test_comparison_fingerprint_tracks_effective_press_settings(*, stage_name: str) -> None:
    from robotics_experiments.config import load_config
    from robotics_experiments.rl import PolicyContract, PolicyInterface

    config = load_config(REPO_ROOT / "configs" / f"train_press_{stage_name}_quad.yaml")
    contract = PolicyContract(
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
        curriculum_stage=stage_name,
    )
    fingerprint = _press_comparison_fingerprint(
        config=config,
        contract=contract,
        flight_parameters_fingerprint="envelope-a",
    )
    assert fingerprint == _press_comparison_fingerprint(
        config=config,
        contract=replace(contract, max_body_velocity_mps=3.0),
        flight_parameters_fingerprint="envelope-a",
    )
    assert fingerprint != _press_comparison_fingerprint(
        config=config,
        contract=replace(contract, obs_dim=35),
        flight_parameters_fingerprint="envelope-a",
    )
    assert fingerprint != _press_comparison_fingerprint(
        config=config,
        contract=contract,
        flight_parameters_fingerprint="envelope-b",
    )
    assert fingerprint == _press_comparison_fingerprint(
        config=replace(config, rl=replace(config.rl, episode_length_s=50.0)),
        contract=contract,
        flight_parameters_fingerprint="envelope-a",
    )
    curriculum = replace(
        config.rl.press.curriculum,
        stages=[
            replace(stage, episode_length_s=50.0) if stage.name == stage_name else stage
            for stage in config.rl.press.curriculum.stages
        ],
    )
    assert fingerprint != _press_comparison_fingerprint(
        config=replace(
            config, rl=replace(config.rl, press=replace(config.rl.press, curriculum=curriculum))
        ),
        contract=contract,
        flight_parameters_fingerprint="envelope-a",
    )


def test_acceptance_uses_only_named_evaluation_metrics() -> None:
    from robotics_experiments.config import load_config

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    evaluation = _scored(
        collision_rate=0.0,
        min_closest_range_m=0.399,
    )
    acceptance = evaluate_press_acceptance(
        criteria=_acceptance_criteria(config),
        evaluation=evaluation,
    )
    assert not acceptance.passed
    assert acceptance.reasons == ("minimum closest range",)
    assert len(acceptance.checks) == 45


def test_acceptance_includes_configured_range_safety_boundary() -> None:
    from robotics_experiments.config import load_config

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    acceptance = evaluate_press_acceptance(
        criteria=_acceptance_criteria(config),
        evaluation=_scored(
            collision_rate=0.0,
            min_closest_range_m=0.40,
        ),
    )
    assert acceptance.passed


@pytest.mark.parametrize(
    ("overrides", "failed_gate"),
    [
        ({"success_rate": 0.94}, "success rate"),
        ({"collision_rate": 0.02}, "collision rate"),
        ({"min_closest_range_m": 0.399}, "minimum closest range"),
        ({"other_flight_failure_rate": 0.01}, "other flight-failure rate"),
    ],
)
def test_every_acceptance_check_is_individually_binding(overrides, failed_gate: str) -> None:
    from robotics_experiments.config import load_config

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    acceptance = evaluate_press_acceptance(
        criteria=_acceptance_criteria(config),
        evaluation=_scored(**overrides),
    )
    assert not acceptance.passed
    assert failed_gate in acceptance.reasons
    assert {check.name for check in acceptance.checks if not check.passed} == set(
        acceptance.reasons
    )


@pytest.mark.parametrize("metric_name", ["max", "rmse", "p95", "p99"])
def test_every_flight_parameter_statistic_is_binding(metric_name: str) -> None:
    from robotics_experiments.config import load_config

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    evaluation = _scored(
        collision_rate=0.0,
    )
    registered_name = f"flight_parameters.position_error.{metric_name}"
    limit = next(
        item for item in _FLIGHT_PARAMETERS.hover.acceptance_limits if item.name == "position_error"
    )
    threshold = {
        "max": limit.maximum_absolute,
        "rmse": limit.rmse,
        "p95": limit.p95_absolute,
        "p99": limit.p99_absolute,
    }[metric_name]
    metrics = tuple(
        EvaluationMetric(metric.name, threshold + 0.001)
        if metric.name == registered_name
        else metric
        for metric in evaluation.metrics
    )
    acceptance = evaluate_press_acceptance(
        criteria=_acceptance_criteria(config),
        evaluation=replace(evaluation, metrics=metrics),
    )
    assert f"flight parameters position_error {metric_name}" in acceptance.reasons


def test_missing_flight_parameter_statistic_is_a_json_null_failure() -> None:
    from robotics_experiments.config import load_config

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    evaluation = _scored()
    missing_name = "flight_parameters.position_error.max"
    acceptance = evaluate_press_acceptance(
        criteria=_acceptance_criteria(config),
        evaluation=replace(
            evaluation,
            metrics=tuple(metric for metric in evaluation.metrics if metric.name != missing_name),
        ),
    )
    check = next(
        item for item in acceptance.checks if item.name == "flight parameters position_error max"
    )
    assert not check.passed
    assert check.actual is None
    assert json.loads(json.dumps(asdict(check), allow_nan=False))["actual"] is None


def test_plateau_accumulates_small_improvements_against_material_reference() -> None:
    state = PlateauState(reference_value=1.0, stale_evaluations=0)
    first = advance_plateau(
        criteria=_criteria(),
        state=state,
        evaluation=_evaluation(iteration=10, cost=0.998),
    )
    assert not first.made_progress
    assert first.state.reference_value == 1.0
    second = advance_plateau(
        criteria=_criteria(),
        state=first.state,
        evaluation=_evaluation(iteration=20, cost=0.997),
    )
    assert second.made_progress
    assert second.state == PlateauState(reference_value=0.997, stale_evaluations=0)


def test_plateau_requires_all_ten_consecutive_stale_evaluations() -> None:
    state = PlateauState(reference_value=1.0, stale_evaluations=0)
    for iteration in range(1, 10):
        update = advance_plateau(
            criteria=_criteria(),
            state=state,
            evaluation=_evaluation(iteration=iteration, cost=1.0),
        )
        state = update.state
        assert not update.plateau_reached
    update = advance_plateau(
        criteria=_criteria(),
        state=state,
        evaluation=_evaluation(iteration=10, cost=1.0),
    )
    assert update.plateau_reached


def test_run_selection_minimizes_cost_and_prefers_earlier_exact_tie() -> None:
    selected = select_best_evaluation(
        criteria=_criteria(),
        evaluations=(
            _evaluation(iteration=10, cost=0.5),
            _evaluation(iteration=20, cost=0.4),
            _evaluation(iteration=30, cost=0.4),
        ),
        eligible_policy_digests=frozenset(),
    )
    assert selected.iteration == 20


def test_validation_passing_candidate_ranks_ahead_of_lower_cost_failure() -> None:
    passing = _evaluation(iteration=20, cost=0.5)
    selected = select_best_evaluation(
        criteria=_criteria(),
        evaluations=(_evaluation(iteration=10, cost=0.1), passing),
        eligible_policy_digests=frozenset({passing.policy_digest}),
    )
    assert selected is passing


def _report(*, fingerprint: str, value: float) -> str:
    import json

    return json.dumps(
        {
            "comparison_fingerprint": fingerprint,
            "selected_metric": {
                "name": "evaluation_diagnostic_cost",
                "direction": "minimize",
                "value": value,
            },
        }
    )


def test_latest_always_advances_and_best_requires_lower_comparable_cost(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"
    for run, value in ((first, 0.5), (second, 0.6), (third, 0.4)):
        run.mkdir()
        (run / "training_report.json").write_text(_report(fingerprint="same", value=value))
    assert publish_policy_labels(
        policy_root=tmp_path,
        run_dir=first,
        comparison_fingerprint="same",
        metric_name="evaluation_diagnostic_cost",
        metric_value=0.5,
        direction="minimize",
    )
    assert not publish_policy_labels(
        policy_root=tmp_path,
        run_dir=second,
        comparison_fingerprint="same",
        metric_name="evaluation_diagnostic_cost",
        metric_value=0.6,
        direction="minimize",
    )
    assert (tmp_path / "latest").resolve() == second
    assert (tmp_path / "best").resolve() == first
    assert publish_policy_labels(
        policy_root=tmp_path,
        run_dir=third,
        comparison_fingerprint="same",
        metric_name="evaluation_diagnostic_cost",
        metric_value=0.4,
        direction="minimize",
    )
    assert (tmp_path / "best").resolve() == third


def test_plateau_metric_can_be_replaced_declaratively() -> None:
    criteria = RlPlateauCriteriaConfig(
        metric_name="success",
        direction="maximize",
        minimum_delta=0.01,
        patience_evaluations=3,
    )
    evaluation = PolicyEvaluation(
        iteration=1,
        checkpoint_path=Path("model_1.pt"),
        policy_digest="digest",
        metrics=(EvaluationMetric("success", 0.51),),
        reward_means=(),
        source="fresh",
    )
    update = advance_plateau(
        criteria=criteria,
        state=PlateauState(reference_value=0.5, stale_evaluations=2),
        evaluation=evaluation,
    )
    assert update.made_progress
    assert not update.plateau_reached

"""Tests for deterministic evaluation progress, acceptance, and publication."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from robotics_experiments.config import RlPlateauCriteriaConfig
from robotics_experiments.rl.training._fingerprint import _press_comparison_fingerprint
from robotics_experiments.rl.training.progress import (
    EvaluationMetric,
    PlateauState,
    PolicyEvaluation,
    advance_plateau,
    select_best_evaluation,
)
from robotics_experiments.rl.training.selection import publish_policy_labels
from tests.support.paths import REPO_ROOT


def _criteria(*, patience: int = 10) -> RlPlateauCriteriaConfig:
    return RlPlateauCriteriaConfig(
        metric_name="held_out_task_cost",
        direction="minimize",
        minimum_delta=0.003,
        patience_evaluations=patience,
    )


def _evaluation(*, iteration: int, cost: float) -> PolicyEvaluation:
    return PolicyEvaluation(
        iteration=iteration,
        checkpoint_path=Path(f"logs/model_{iteration}.pt"),
        policy_digest=f"digest-{iteration}",
        metrics=(EvaluationMetric("held_out_task_cost", cost),),
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
        "approach_success_rate": 0.95,
        "collision_rate": 0.01,
        "body_collision_rate": 0.006,
        "pusher_collision_rate": 0.004,
        "other_flight_failure_rate": 0.0,
        "range_undershoot_rate": 0.0,
        "timeout_rate": 0.04,
        "range_pass_rate": 1.0,
        "alignment_pass_rate": 1.0,
        "attitude_pass_rate": 1.0,
        "linear_speed_pass_rate": 1.0,
        "angular_speed_pass_rate": 1.0,
        "contact_clear_pass_rate": 1.0,
        "complete_goal_pass_rate": 0.95,
        "mean_normalized_pose_error": 0.1,
        "mean_max_stable_goal_fraction": 0.95,
        "maximum_stable_goal_fraction": 1.0,
        "mean_closest_range_m": 0.41,
        "min_closest_range_m": 0.405,
        "median_completion_time_s": 2.0,
        "mean_terminal_range_cost": 0.0,
        "mean_trajectory_alignment_cost": 0.1,
        "mean_relative_attitude_error_cost": 0.1,
        "mean_velocity_tracking_cost": 0.1,
        "mean_angular_velocity_cost": 0.1,
        "mean_upright_error_cost": 0.1,
        "mean_action_delta_cost": 0.1,
        "mean_completion_time_fraction": 0.2,
    }
    values.update(overrides)
    return PressApproachEvaluationResult(
        summary=PressApproachEvaluationSummary(**values),
        reward_means=(),
    )


def _scored(**overrides) -> PolicyEvaluation:
    from robotics_experiments.config import load_config
    from robotics_experiments.workflows.isaac_lab import press_policy_evaluation

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    return press_policy_evaluation(
        iteration=10,
        checkpoint_path=Path("logs/model_10.pt"),
        policy_digest="digest",
        result=_press_result(**overrides),
        evaluation_config=config.rl.press.evaluation,
        source="fresh",
    )


def test_comparison_fingerprint_excludes_candidate_owned_action_scales() -> None:
    from robotics_experiments.config import load_config
    from robotics_experiments.rl import PolicyContract, PolicyInterface

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
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
        curriculum_stage="approach",
    )
    fingerprint = _press_comparison_fingerprint(
        config=config,
        contract=contract,
        flight_envelope_fingerprint="envelope-a",
    )
    assert fingerprint == _press_comparison_fingerprint(
        config=config,
        contract=replace(contract, max_body_velocity_mps=3.0),
        flight_envelope_fingerprint="envelope-a",
    )
    assert fingerprint != _press_comparison_fingerprint(
        config=config,
        contract=replace(contract, obs_dim=35),
        flight_envelope_fingerprint="envelope-a",
    )
    assert fingerprint != _press_comparison_fingerprint(
        config=config,
        contract=contract,
        flight_envelope_fingerprint="envelope-b",
    )


@pytest.mark.parametrize(
    ("field", "better", "worse"),
    [
        ("approach_success_rate", 0.99, 0.95),
        ("collision_rate", 0.0, 0.01),
        ("other_flight_failure_rate", 0.0, 0.01),
        ("timeout_rate", 0.0, 0.04),
        ("mean_terminal_range_cost", 0.0, 0.1),
        ("mean_trajectory_alignment_cost", 0.0, 0.1),
        ("mean_relative_attitude_error_cost", 0.0, 0.1),
        ("mean_velocity_tracking_cost", 0.0, 0.1),
        ("mean_angular_velocity_cost", 0.0, 0.1),
        ("mean_upright_error_cost", 0.0, 0.1),
        ("mean_action_delta_cost", 0.0, 0.1),
        ("mean_completion_time_fraction", 0.0, 0.2),
    ],
)
def test_raw_held_out_task_cost_preserves_every_configured_component(
    field: str,
    better: float,
    worse: float,
) -> None:
    assert _scored(**{field: better}).metric("held_out_task_cost") < _scored(
        **{field: worse}
    ).metric("held_out_task_cost")


def test_acceptance_is_separate_from_evaluation_cost() -> None:
    from robotics_experiments.config import load_config
    from robotics_experiments.workflows.isaac_lab import evaluate_press_acceptance

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    evaluation = _scored(
        collision_rate=0.0,
        body_collision_rate=0.0,
        pusher_collision_rate=0.0,
        min_closest_range_m=0.399,
    )
    assert evaluation.metric("held_out_task_cost") >= 0.0
    acceptance = evaluate_press_acceptance(
        evaluation=evaluation,
        evaluation_config=config.rl.press.evaluation,
        minimum_range_m=config.rl.press.goal.range_min_m,
    )
    assert not acceptance.passed
    assert acceptance.reasons == ("minimum closest range",)
    assert len(acceptance.gates) == 5


@pytest.mark.parametrize(
    ("overrides", "failed_gate"),
    [
        ({"median_completion_time_s": float("nan")}, "trace finiteness"),
        ({"approach_success_rate": 0.94}, "approach success rate"),
        ({"collision_rate": 0.02}, "collision rate"),
        ({"min_closest_range_m": 0.399}, "minimum closest range"),
        ({"other_flight_failure_rate": 0.01}, "other flight-failure rate"),
    ],
)
def test_every_acceptance_gate_is_individually_binding(overrides, failed_gate: str) -> None:
    from robotics_experiments.config import load_config
    from robotics_experiments.workflows.isaac_lab import evaluate_press_acceptance

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    acceptance = evaluate_press_acceptance(
        evaluation=_scored(**overrides),
        evaluation_config=config.rl.press.evaluation,
        minimum_range_m=config.rl.press.goal.range_min_m,
    )
    assert not acceptance.passed
    assert failed_gate in acceptance.reasons
    assert {gate.name for gate in acceptance.gates if not gate.passed} == set(acceptance.reasons)


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
                "name": "held_out_task_cost",
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
        metric_name="held_out_task_cost",
        metric_value=0.5,
        direction="minimize",
    )
    assert not publish_policy_labels(
        policy_root=tmp_path,
        run_dir=second,
        comparison_fingerprint="same",
        metric_name="held_out_task_cost",
        metric_value=0.6,
        direction="minimize",
    )
    assert (tmp_path / "latest").resolve() == second
    assert (tmp_path / "best").resolve() == first
    assert publish_policy_labels(
        policy_root=tmp_path,
        run_dir=third,
        comparison_fingerprint="same",
        metric_name="held_out_task_cost",
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

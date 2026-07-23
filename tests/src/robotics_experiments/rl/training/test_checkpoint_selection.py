"""Tests for task-neutral selection and approach-stage ranking."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.rl.training.selection import (
    CandidateEvaluation,
    CheckpointCandidate,
    CheckpointSelection,
    MetricValue,
    RewardMean,
    enumerate_checkpoints,
    publish_policy_labels,
    select_checkpoint,
)
from tests.support.paths import REPO_ROOT


def _load_selector_module():
    pytest.importorskip("torch")
    script = REPO_ROOT / "isaac" / "runtime" / "standalone" / "select_rl_checkpoint.py"
    spec = importlib.util.spec_from_file_location("select_rl_checkpoint", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(
    *,
    success: float = 0.95,
    collision: float = 0.01,
    body_collision: float = 0.006,
    pusher_collision: float = 0.004,
    other_failure: float = 0.0,
    timeout: float = 0.04,
    closest: float = 0.41,
    min_closest: float = 0.405,
    completion_time: float | None = 2.0,
    terminal_range_cost: float = 0.0,
    alignment_cost: float = 0.1,
    velocity_tracking_cost: float = 0.1,
    upright_cost: float = 0.1,
    action_delta_cost: float = 0.1,
    completion_fraction: float = 0.2,
):
    from robotics_experiments.workflows.isaac_lab import (
        PressApproachEvaluationResult,
        PressApproachEvaluationSummary,
    )

    return PressApproachEvaluationResult(
        summary=PressApproachEvaluationSummary(
            episodes=300,
            approach_success_rate=success,
            collision_rate=collision,
            body_collision_rate=body_collision,
            pusher_collision_rate=pusher_collision,
            other_flight_failure_rate=other_failure,
            timeout_rate=timeout,
            mean_closest_range_m=closest,
            min_closest_range_m=min_closest,
            median_completion_time_s=completion_time,
            mean_terminal_range_cost=terminal_range_cost,
            mean_trajectory_alignment_cost=alignment_cost,
            mean_velocity_tracking_cost=velocity_tracking_cost,
            mean_upright_error_cost=upright_cost,
            mean_action_delta_cost=action_delta_cost,
            mean_completion_time_fraction=completion_fraction,
        ),
        reward_means=(RewardMean(name="success", mean=success),),
    )


def _evaluation(*, selector, iteration: int, result=None) -> CandidateEvaluation:
    return selector._score_press_result(  # noqa: SLF001 - focused script helper test
        candidate=CheckpointCandidate(iteration, Path(f"model_{iteration}.pt")),
        result=result or _result(),
        evaluation_config=SimpleNamespace(
            success_rate_min=0.95,
            collision_rate_max=0.01,
            collision_penalty=4.0,
            flight_failure_penalty=3.0,
            timeout_penalty=2.0,
            non_success_penalty=1.0,
            terminal_range_weight=0.25,
            trajectory_alignment_weight=0.20,
            velocity_tracking_weight=0.20,
            upright_error_weight=0.10,
            action_delta_weight=0.10,
            completion_time_weight=0.15,
        ),
        goal_range_min_m=0.40,
    )


def test_checkpoint_enumeration_is_numeric(tmp_path: Path) -> None:
    for name in ("model_50.pt", "model_0.pt", "model_299.pt", "model_latest.pt"):
        (tmp_path / name).touch()
    assert [candidate.iteration for candidate in enumerate_checkpoints(logs_dir=tmp_path)] == [
        0,
        50,
        299,
    ]


def test_comparison_fingerprint_excludes_candidate_owned_action_scales() -> None:
    selector = _load_selector_module()
    from robotics_experiments.config import load_config
    from robotics_experiments.rl import PolicyContract, PolicyInterface

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    contract = PolicyContract(
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

    fingerprint = selector._comparison_fingerprint(  # noqa: SLF001
        config=config,
        contract=contract,
    )

    assert fingerprint == selector._comparison_fingerprint(  # noqa: SLF001
        config=config,
        contract=replace(contract, max_body_velocity_mps=3.0),
    )
    assert fingerprint != selector._comparison_fingerprint(  # noqa: SLF001
        config=config,
        contract=replace(contract, obs_dim=35),
    )


def test_approach_acceptance_uses_success_and_collision_gates() -> None:
    selector = _load_selector_module()
    assert _evaluation(selector=selector, iteration=0).accepted
    assert not _evaluation(
        selector=selector,
        iteration=1,
        result=_result(success=0.949),
    ).accepted
    assert not _evaluation(
        selector=selector,
        iteration=2,
        result=_result(collision=0.011),
    ).accepted
    assert not _evaluation(
        selector=selector,
        iteration=3,
        result=_result(min_closest=0.399),
    ).accepted
    assert not _evaluation(
        selector=selector,
        iteration=4,
        result=_result(other_failure=0.001),
    ).accepted


@pytest.mark.parametrize(
    ("better", "worse"),
    [
        (_result(success=0.99), _result(success=0.95)),
        (_result(collision=0.01), _result(collision=0.02)),
        (_result(other_failure=0.01), _result(other_failure=0.02)),
        (_result(timeout=0.01), _result(timeout=0.02)),
        (_result(terminal_range_cost=0.1), _result(terminal_range_cost=0.2)),
        (_result(alignment_cost=0.1), _result(alignment_cost=0.2)),
        (
            _result(velocity_tracking_cost=0.1),
            _result(velocity_tracking_cost=0.2),
        ),
        (_result(upright_cost=0.1), _result(upright_cost=0.2)),
        (_result(action_delta_cost=0.1), _result(action_delta_cost=0.2)),
        (_result(completion_fraction=0.1), _result(completion_fraction=0.2)),
    ],
)
def test_held_out_quality_ranks_every_component(better, worse) -> None:
    selector = _load_selector_module()
    assert (
        _evaluation(
            selector=selector,
            iteration=100,
            result=better,
        ).ranking_values
        > _evaluation(
            selector=selector,
            iteration=100,
            result=worse,
        ).ranking_values
    )


def test_ranking_final_tie_breaker_prefers_earlier_checkpoint() -> None:
    selector = _load_selector_module()
    assert (
        _evaluation(
            selector=selector,
            iteration=50,
        ).ranking_values
        > _evaluation(selector=selector, iteration=100).ranking_values
    )


def test_quality_ranking_is_strict_json_serializable_without_completion() -> None:
    selector = _load_selector_module()
    evaluation = _evaluation(
        selector=selector,
        iteration=0,
        result=_result(success=0.0, completion_time=None),
    )
    json.dumps(evaluation.ranking_values, allow_nan=False)


def test_quality_ranking_rejects_non_finite_components() -> None:
    selector = _load_selector_module()

    with pytest.raises(ValueError, match="non-finite quality score"):
        _evaluation(
            selector=selector,
            iteration=0,
            result=_result(alignment_cost=float("nan")),
        )


def test_advisory_acceptance_requires_finite_completed_traces() -> None:
    selector = _load_selector_module()

    evaluation = _evaluation(
        selector=selector,
        iteration=0,
        result=_result(closest=float("nan")),
    )

    assert not evaluation.accepted


def test_advisory_acceptance_does_not_override_quality_score() -> None:
    candidates = (
        CheckpointCandidate(0, Path("model_0.pt")),
        CheckpointCandidate(50, Path("model_50.pt")),
    )
    evaluations = (
        CandidateEvaluation(
            candidates[0],
            False,
            (MetricValue("score", 10.0),),
            (10.0, 10.0),
            (10.0,),
            (),
        ),
        CandidateEvaluation(
            candidates[1],
            True,
            (MetricValue("score", 1.0),),
            (1.0, 1.0),
            (1.0,),
            (),
        ),
    )
    selection = select_checkpoint(
        policy_interface="test-v1",
        candidates=candidates,
        selection_method="test",
        scorer=lambda _candidates: evaluations,
    )
    assert selection.selected.iteration == 0


def test_selector_publishes_best_rejected_fallback() -> None:
    candidates = tuple(
        CheckpointCandidate(iteration, Path(f"model_{iteration}.pt"))
        for iteration in (100, 150, 200)
    )
    evaluations = tuple(
        CandidateEvaluation(
            candidate,
            False,
            (MetricValue("score", ranking),),
            (ranking,),
            (ranking,),
            (),
        )
        for candidate, ranking in zip(candidates, (0.1, 0.3, 0.0), strict=True)
    )
    selection = select_checkpoint(
        policy_interface="test-v1",
        candidates=candidates,
        selection_method="test",
        scorer=lambda _candidates: evaluations,
    )
    assert selection.selected.iteration == 150


def test_press_selection_evaluates_every_candidate_with_one_environment(
    monkeypatch,
) -> None:
    selector = _load_selector_module()
    candidates = tuple(
        CheckpointCandidate(iteration, Path(f"model_{iteration}.pt")) for iteration in (0, 50, 299)
    )
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return _evaluation(
            selector=selector,
            iteration=kwargs["candidate"].iteration,
        )

    monkeypatch.setattr(selector, "_press_evaluation", fake_evaluate)
    selection = selector._select_candidates(  # noqa: SLF001
        policy_interface="press-v1",
        candidates=candidates,
        runner=object(),
        env=object(),
        seeds=(101, 202, 303),
        evaluation_config=SimpleNamespace(
            episodes_per_seed=100,
            success_rate_min=0.95,
            collision_rate_max=0.01,
            selection_metric="held-out-press-task-quality-v2",
        ),
        goal_config=SimpleNamespace(range_min_m=0.40),
        device="cuda:0",
    )
    assert [call["candidate"] for call in calls] == list(candidates)
    assert all(call["seeds"] == (101, 202, 303) for call in calls)
    assert selection.selected in candidates


def test_no_scorer_warns_and_selects_final_checkpoint() -> None:
    candidates = (
        CheckpointCandidate(0, Path("model_0.pt")),
        CheckpointCandidate(299, Path("model_299.pt")),
    )
    with pytest.warns(RuntimeWarning, match="no checkpoint scorer"):
        selection = select_checkpoint(
            policy_interface="hover-v1",
            candidates=candidates,
        )
    assert selection.selection_method == "final-no-scorer"
    assert selection.selected.iteration == 299


def test_approach_report_records_stage_and_v2_thresholds() -> None:
    selector = _load_selector_module()
    candidate = CheckpointCandidate(150, Path("/tmp/run/logs/model_150.pt"))
    evaluation = _evaluation(
        selector=selector,
        iteration=150,
        result=_result(success=0.1, collision=0.2),
    )
    selection = CheckpointSelection(
        selection_method="held-out-press-task-quality-v2",
        evaluations=(evaluation,),
        selected=candidate,
    )
    report = selector._selection_report(  # noqa: SLF001
        selection=selection,
        run_dir=Path("/tmp/run"),
        policy_name="press-approach",
        policy_interface="press-v1",
        curriculum_stage="approach",
        seeds=(101, 202, 303),
        episodes_per_seed=100,
        evaluation_config=SimpleNamespace(
            success_rate_min=0.95,
            collision_rate_max=0.01,
        ),
        goal_range_min_m=0.40,
        all_candidates=(candidate,),
        comparison_fingerprint="comparison",
    )
    assert report["schema_version"] == 7
    assert report["curriculum_stage"] == "approach"
    assert report["selected_advisory_accepted"] is False
    assert report["selected_score"] == evaluation.ranking_values[0]
    assert report["comparison_fingerprint"] == "comparison"
    assert report["thresholds"] == {
        "approach_success_rate_min": 0.95,
        "collision_rate_max": 0.01,
        "min_closest_range_m": 0.40,
    }
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("accepted", [False, True])
def test_publication_records_evaluation_and_advances_latest(
    monkeypatch,
    tmp_path: Path,
    accepted: bool,
) -> None:
    selector = _load_selector_module()
    import robotics_experiments.rl.training.rsl_rl as training

    policy_root = tmp_path / "press-approach"
    old_run = policy_root / "old"
    run_dir = policy_root / tmp_path.name
    checkpoint = run_dir / "logs" / "model_50.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    old_run.mkdir()
    (policy_root / "latest").symlink_to(old_run.name)
    monkeypatch.setattr(training, "load_checkpoint", lambda **_kwargs: None)

    def fake_export(_runner, out_dir, *, metadata):
        (out_dir / "policy.pt").write_text("policy")
        (out_dir / "policy_meta.json").write_text(json.dumps(metadata))

    monkeypatch.setattr(training, "export_policy", fake_export)
    evaluation = CandidateEvaluation(
        candidate=CheckpointCandidate(50, checkpoint),
        accepted=accepted,
        metrics=(
            MetricValue("approach_success_rate", 0.75),
            MetricValue("collision_rate", 0.04),
            MetricValue("body_collision_rate", 0.03),
            MetricValue("pusher_collision_rate", 0.01),
        ),
        normalized_margins=(0.9375, 1.2),
        ranking_values=(0.9375, 1.2, -0.1, -0.1, -2.0, -50.0),
        reward_means=(RewardMean("success", 0.125),),
    )
    selection = CheckpointSelection(
        "held-out-press-task-quality-v2",
        (evaluation,),
        evaluation.candidate,
    )
    report = {
        "selected_iteration": 50,
        "selected_score": evaluation.ranking_values[0],
        "selected_advisory_accepted": accepted,
        "comparison_fingerprint": "comparison",
        "seeds": [101, 202, 303],
        "episodes_per_seed": 100,
        "thresholds": {
            "approach_success_rate_min": 0.95,
            "collision_rate_max": 0.01,
        },
    }
    best_updated = selector._publish_selection(  # noqa: SLF001
        selection=selection,
        report=report,
        metadata={
            "policy_interface": "press-v1",
            "curriculum_stage": "approach",
        },
        runner=object(),
        run_dir=run_dir,
        policy_root=policy_root,
        device="cpu",
    )
    assert best_updated
    assert os.readlink(policy_root / "latest") == run_dir.name
    assert os.readlink(policy_root / "best") == run_dir.name
    metadata = json.loads((run_dir / "policy_meta.json").read_text())
    assert metadata["selected_checkpoint_iteration"] == 50
    assert metadata["checkpoint_evaluation"]["advisory_accepted"] is accepted
    assert metadata["checkpoint_evaluation"]["score"] == evaluation.ranking_values[0]
    assert metadata["checkpoint_evaluation"]["reward_means"] == {"success": 0.125}


def test_export_failure_preserves_previous_latest(monkeypatch, tmp_path: Path) -> None:
    selector = _load_selector_module()
    import robotics_experiments.rl.training.rsl_rl as training

    policy_root = tmp_path / "press-approach"
    old_run = policy_root / "old"
    run_dir = policy_root / tmp_path.name
    checkpoint = run_dir / "logs" / "model_50.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    old_run.mkdir()
    (policy_root / "latest").symlink_to(old_run.name)
    monkeypatch.setattr(training, "load_checkpoint", lambda **_kwargs: None)
    monkeypatch.setattr(
        training,
        "export_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("export failed")),
    )
    selection = CheckpointSelection(
        "held-out-press-task-quality-v2",
        (),
        CheckpointCandidate(50, checkpoint),
    )
    with pytest.raises(RuntimeError, match="export failed"):
        selector._publish_selection(  # noqa: SLF001
            selection=selection,
            report={"selected_iteration": 50},
            metadata={
                "policy_interface": "press-v1",
                "curriculum_stage": "approach",
            },
            runner=object(),
            run_dir=run_dir,
            policy_root=policy_root,
            device="cpu",
        )
    assert os.readlink(policy_root / "latest") == old_run.name


def test_latest_always_advances_while_best_requires_strict_improvement(
    tmp_path: Path,
) -> None:
    policy_root = tmp_path / "press-approach"

    def run(name: str, score: float, fingerprint: str = "comparison") -> Path:
        run_dir = policy_root / name
        run_dir.mkdir(parents=True)
        (run_dir / "checkpoint_selection.json").write_text(
            json.dumps(
                {
                    "comparison_fingerprint": fingerprint,
                    "selected_score": score,
                }
            )
        )
        return run_dir

    first = run("first", 1.0)
    assert publish_policy_labels(
        policy_root=policy_root,
        run_dir=first,
        comparison_fingerprint="comparison",
        score=1.0,
    )
    worse = run("worse", 0.5)
    assert not publish_policy_labels(
        policy_root=policy_root,
        run_dir=worse,
        comparison_fingerprint="comparison",
        score=0.5,
    )
    assert os.readlink(policy_root / "latest") == worse.name
    assert os.readlink(policy_root / "best") == first.name

    equal = run("equal", 1.0)
    assert not publish_policy_labels(
        policy_root=policy_root,
        run_dir=equal,
        comparison_fingerprint="comparison",
        score=1.0,
    )
    assert os.readlink(policy_root / "latest") == equal.name
    assert os.readlink(policy_root / "best") == first.name

    better = run("better", 1.1)
    assert publish_policy_labels(
        policy_root=policy_root,
        run_dir=better,
        comparison_fingerprint="comparison",
        score=1.1,
    )
    assert os.readlink(policy_root / "latest") == better.name
    assert os.readlink(policy_root / "best") == better.name


def test_new_comparison_generation_initializes_best(tmp_path: Path) -> None:
    policy_root = tmp_path / "press-approach"
    old = policy_root / "old"
    old.mkdir(parents=True)
    (old / "checkpoint_selection.json").write_text(
        json.dumps(
            {
                "comparison_fingerprint": "old-comparison",
                "selected_score": 100.0,
            }
        )
    )
    (policy_root / "best").symlink_to(old.name)
    current = policy_root / "current"
    current.mkdir()
    (current / "checkpoint_selection.json").write_text(
        json.dumps(
            {
                "comparison_fingerprint": "new-comparison",
                "selected_score": -100.0,
            }
        )
    )

    assert publish_policy_labels(
        policy_root=policy_root,
        run_dir=current,
        comparison_fingerprint="new-comparison",
        score=-100.0,
    )
    assert os.readlink(policy_root / "latest") == current.name
    assert os.readlink(policy_root / "best") == current.name

"""Tests for task-neutral checkpoint selection and press ranking."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
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
    toggle: float = 0.8,
    contact: float = 0.85,
    retreat: float = 0.95,
    completion: float = 0.8,
    overshoot: float = 0.01,
    toggle_time: float | None = 2.0,
):
    from robotics_experiments.workflows.isaac_lab import (
        PressEvaluationResult,
        PressEvaluationSummary,
    )

    return PressEvaluationResult(
        summary=PressEvaluationSummary(
            episodes=300,
            confirmed_toggle_rate=toggle,
            stable_first_contact_rate=contact,
            post_toggle_retreat_rate=retreat,
            full_completion_rate=completion,
            peak_compression_overshoot_m=overshoot,
            median_toggle_time_s=toggle_time,
        ),
        reward_means=(RewardMean(name="toggle", mean=toggle),),
    )


def _evaluation(
    *,
    selector,
    iteration: int,
    result=None,
) -> CandidateEvaluation:
    config = SimpleNamespace(
        success_rate_min=0.8,
        first_contact_stability_min=0.85,
        retreat_rate_min=0.95,
    )
    return selector._score_press_result(  # noqa: SLF001 - focused script helper test.
        candidate=CheckpointCandidate(
            iteration=iteration,
            path=Path(f"model_{iteration}.pt"),
        ),
        result=result or _result(),
        evaluation_config=config,
    )


def test_checkpoint_enumeration_is_numeric_and_includes_zero_and_final(
    tmp_path: Path,
) -> None:
    for name in ("model_50.pt", "model_0.pt", "model_299.pt", "model_latest.pt"):
        (tmp_path / name).touch()

    candidates = enumerate_checkpoints(logs_dir=tmp_path)

    assert [candidate.iteration for candidate in candidates] == [0, 50, 299]


def test_press_acceptance_and_balanced_maximin_ranking() -> None:
    selector = _load_selector_module()
    rejected = _evaluation(selector=selector, iteration=0, result=_result(toggle=0.79))
    unbalanced = _evaluation(
        selector=selector,
        iteration=50,
        result=_result(toggle=0.8, contact=1.0, retreat=1.0),
    )
    balanced = _evaluation(
        selector=selector,
        iteration=100,
        result=_result(toggle=0.88, contact=0.935, retreat=1.0),
    )

    assert not rejected.accepted
    assert unbalanced.accepted
    assert balanced.ranking_values > unbalanced.ranking_values


@pytest.mark.parametrize(
    ("better_result", "worse_result"),
    [
        (_result(completion=0.9), _result(completion=0.8)),
        (_result(overshoot=0.005), _result(overshoot=0.01)),
        (_result(toggle_time=1.0), _result(toggle_time=2.0)),
    ],
)
def test_press_ranking_applies_metric_tie_breakers(
    better_result,
    worse_result,
) -> None:
    selector = _load_selector_module()

    better = _evaluation(selector=selector, iteration=100, result=better_result)
    worse = _evaluation(selector=selector, iteration=100, result=worse_result)

    assert better.ranking_values > worse.ranking_values


def test_press_ranking_breaks_exact_ties_by_earlier_iteration() -> None:
    selector = _load_selector_module()

    earlier = _evaluation(selector=selector, iteration=50)
    later = _evaluation(selector=selector, iteration=100)

    assert earlier.ranking_values > later.ranking_values


def test_press_ranking_without_a_toggle_time_is_strict_json_serializable() -> None:
    selector = _load_selector_module()

    evaluation = _evaluation(
        selector=selector,
        iteration=0,
        result=_result(toggle=0.0, toggle_time=None),
    )

    json.dumps(evaluation.ranking_values, allow_nan=False)


def test_accepted_candidate_always_outranks_rejected_candidate() -> None:
    candidates = (
        CheckpointCandidate(0, Path("model_0.pt")),
        CheckpointCandidate(50, Path("model_50.pt")),
    )
    evaluations = (
        CandidateEvaluation(
            candidate=candidates[0],
            accepted=False,
            metrics=(MetricValue("score", 10.0),),
            normalized_margins=(10.0, 10.0, 10.0),
            ranking_values=(10.0,),
            reward_means=(),
        ),
        CandidateEvaluation(
            candidate=candidates[1],
            accepted=True,
            metrics=(MetricValue("score", 1.0),),
            normalized_margins=(1.0, 1.0, 1.0),
            ranking_values=(1.0,),
            reward_means=(),
        ),
    )
    selection = select_checkpoint(
        policy_interface="test-eligible-v1",
        candidates=candidates,
        selection_method="test",
        scorer=lambda _candidates: evaluations,
    )

    assert selection.selected is not None
    assert selection.selected.iteration == 50


def test_scorer_selects_highest_ranked_fallback_when_none_are_accepted() -> None:
    candidates = (
        CheckpointCandidate(100, Path("model_100.pt")),
        CheckpointCandidate(150, Path("model_150.pt")),
        CheckpointCandidate(200, Path("model_200.pt")),
    )
    evaluations = tuple(
        CandidateEvaluation(
            candidate=candidate,
            accepted=False,
            metrics=(MetricValue("score", ranking),),
            normalized_margins=(ranking, ranking, ranking),
            ranking_values=(ranking,),
            reward_means=(),
        )
        for candidate, ranking in zip(candidates, (0.1, 0.3, 0.0), strict=True)
    )
    selection = select_checkpoint(
        policy_interface="test-fallback-v1",
        candidates=candidates,
        selection_method="test-maximin-v2",
        scorer=lambda _candidates: evaluations,
    )

    assert selection.selected is not None
    assert selection.selected.iteration == 150


def test_press_selection_evaluates_every_candidate_with_one_runner_and_environment(
    monkeypatch,
) -> None:
    selector = _load_selector_module()
    candidates = tuple(
        CheckpointCandidate(iteration, Path(f"model_{iteration}.pt")) for iteration in (0, 50, 299)
    )
    runner = object()
    env = object()
    calls = []

    def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return _evaluation(
            selector=selector,
            iteration=kwargs["candidate"].iteration,
        )

    monkeypatch.setattr(selector, "_press_evaluation", fake_evaluate)
    config = SimpleNamespace(
        episodes_per_seed=100,
        success_rate_min=0.8,
        first_contact_stability_min=0.85,
        retreat_rate_min=0.95,
    )

    selection = selector._select_candidates(  # noqa: SLF001
        policy_interface="press-v1",
        candidates=candidates,
        runner=runner,
        env=env,
        seeds=(101, 202, 303),
        evaluation_config=config,
        device="cuda:0",
    )

    assert [call["candidate"] for call in calls] == list(candidates)
    assert all(call["runner"] is runner and call["env"] is env for call in calls)
    assert all(call["seeds"] == (101, 202, 303) for call in calls)
    assert all(call["episodes_per_seed"] == 100 for call in calls)
    assert selection.selected is not None


def test_no_scorer_warns_and_selects_final_checkpoint() -> None:
    candidates = (
        CheckpointCandidate(0, Path("model_0.pt")),
        CheckpointCandidate(299, Path("model_299.pt")),
    )

    with pytest.warns(RuntimeWarning, match="no checkpoint scorer"):
        selection = select_checkpoint(
            policy_interface="hover-without-scorer-v1",
            candidates=candidates,
        )

    assert selection.selection_method == "final-no-scorer"
    assert selection.selected is not None
    assert selection.selected.iteration == 299


@pytest.mark.parametrize("accepted", [False, True])
def test_publish_records_exact_evaluation_and_advances_relative_latest(
    monkeypatch,
    tmp_path: Path,
    accepted: bool,
) -> None:
    selector = _load_selector_module()
    import robotics_experiments.rl.training.rsl_rl as training

    policy_root = tmp_path / "press"
    old_run = policy_root / "old"
    run_dir = policy_root / tmp_path.name
    checkpoint = run_dir / "logs" / "model_50.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    old_run.mkdir()
    (policy_root / "latest").symlink_to(old_run.name)
    loaded = []

    def fake_export(_runner, out_dir, *, metadata):
        (out_dir / "policy.pt").write_text("policy")
        (out_dir / "policy_meta.json").write_text(json.dumps(metadata))

    monkeypatch.setattr(
        training,
        "load_checkpoint",
        lambda **kwargs: loaded.append(kwargs["checkpoint_path"]),
    )
    monkeypatch.setattr(training, "export_policy", fake_export)
    evaluation = CandidateEvaluation(
        candidate=CheckpointCandidate(iteration=50, path=checkpoint),
        accepted=accepted,
        metrics=(
            MetricValue("confirmed_toggle_rate", 0.75),
            MetricValue("median_toggle_time_s", None),
        ),
        normalized_margins=(0.9375, 1.0, 1.0),
        ranking_values=(0.9375, 1.0, 1.0, 0.75, -0.01, -sys.float_info.max, -50.0),
        reward_means=(RewardMean("toggle", 0.125),),
    )
    selection = CheckpointSelection(
        selection_method="press-gated-maximin-v2",
        evaluations=(evaluation,),
        selected=evaluation.candidate,
    )
    thresholds = {
        "confirmed_toggle_rate_min": 0.8,
        "stable_first_contact_rate_min": 0.85,
        "post_toggle_retreat_rate_min": 0.95,
    }
    report = {
        "selected_iteration": 50,
        "selected_accepted": accepted,
        "seeds": [101, 202, 303],
        "episodes_per_seed": 100,
        "thresholds": thresholds,
    }

    selector._publish_selection(  # noqa: SLF001 - focused publication contract test.
        selection=selection,
        report=report,
        metadata={"policy_interface": "press-v1"},
        runner=object(),
        run_dir=run_dir,
        policy_root=policy_root,
        device="cpu",
    )

    assert loaded == [checkpoint]
    assert os.readlink(policy_root / "latest") == run_dir.name
    metadata = json.loads((run_dir / "policy_meta.json").read_text())
    assert metadata["selected_checkpoint_iteration"] == 50
    assert metadata["selection_method"] == "press-gated-maximin-v2"
    assert metadata["checkpoint_evaluation"] == {
        "accepted": accepted,
        "seeds": [101, 202, 303],
        "episodes_per_seed": 100,
        "thresholds": thresholds,
        "metrics": {
            "confirmed_toggle_rate": 0.75,
            "median_toggle_time_s": None,
        },
        "normalized_margins": [0.9375, 1.0, 1.0],
        "ranking_values": [
            0.9375,
            1.0,
            1.0,
            0.75,
            -0.01,
            -sys.float_info.max,
            -50.0,
        ],
        "reward_means": {"toggle": 0.125},
    }
    assert (run_dir / "checkpoint_selection.json").is_file()


def test_schema_two_report_marks_selected_fallback() -> None:
    selector = _load_selector_module()
    candidate = CheckpointCandidate(150, Path("/tmp/run/logs/model_150.pt"))
    evaluation = _evaluation(
        selector=selector,
        iteration=150,
        result=_result(toggle=1 / 300, contact=1.0, retreat=0.0),
    )
    selection = CheckpointSelection(
        selection_method="press-gated-maximin-v2",
        evaluations=(evaluation,),
        selected=candidate,
    )

    report = selector._selection_report(  # noqa: SLF001
        selection=selection,
        run_dir=Path("/tmp/run"),
        policy_name="press",
        policy_interface="press-v1",
        seeds=(101, 202, 303),
        episodes_per_seed=100,
        evaluation_config=SimpleNamespace(
            success_rate_min=0.8,
            first_contact_stability_min=0.85,
            retreat_rate_min=0.95,
        ),
        all_candidates=(candidate,),
    )

    assert report["schema_version"] == 2
    assert report["selected_iteration"] == 150
    assert report["selected_accepted"] is False
    assert report["candidates"][0]["normalized_margins"] == list(evaluation.normalized_margins)
    json.dumps(report, allow_nan=False)


def test_export_failure_preserves_previous_latest(monkeypatch, tmp_path: Path) -> None:
    selector = _load_selector_module()
    import robotics_experiments.rl.training.rsl_rl as training

    policy_root = tmp_path / "press"
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
        selection_method="press-gated-maximin-v2",
        evaluations=(),
        selected=CheckpointCandidate(iteration=50, path=checkpoint),
    )

    with pytest.raises(RuntimeError, match="export failed"):
        selector._publish_selection(  # noqa: SLF001
            selection=selection,
            report={"selected_iteration": 50},
            metadata={"policy_interface": "press-v1"},
            runner=object(),
            run_dir=run_dir,
            policy_root=policy_root,
            device="cpu",
        )

    assert os.readlink(policy_root / "latest") == old_run.name

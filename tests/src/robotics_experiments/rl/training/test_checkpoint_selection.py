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
    MetricValue,
    RewardMean,
    enumerate_checkpoints,
    register_checkpoint_scorer,
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


def test_registered_scorer_selects_only_eligible_candidates() -> None:
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
    register_checkpoint_scorer(
        policy_interface="test-eligible-v1",
        selection_method="test",
        scorer=lambda _candidates: evaluations,
    )

    selection = select_checkpoint(
        policy_interface="test-eligible-v1",
        candidates=candidates,
    )

    assert selection.selected is not None
    assert selection.selected.iteration == 50


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


def test_no_passing_press_candidate_retains_report_and_previous_latest(
    tmp_path: Path,
) -> None:
    selector = _load_selector_module()
    candidate = CheckpointCandidate(0, tmp_path / "run" / "logs" / "model_0.pt")
    evaluation = selector._score_press_result(  # noqa: SLF001
        candidate=candidate,
        result=_result(toggle=0.1, contact=0.1, retreat=0.1),
        evaluation_config=SimpleNamespace(
            success_rate_min=0.8,
            first_contact_stability_min=0.85,
            retreat_rate_min=0.95,
        ),
    )
    register_checkpoint_scorer(
        policy_interface="press-no-pass-v1",
        selection_method="press-acceptance-maximin-v1",
        scorer=lambda _candidates: (evaluation,),
    )
    selection = select_checkpoint(
        policy_interface="press-no-pass-v1",
        candidates=(candidate,),
    )
    policy_root = tmp_path / "press"
    run_dir = policy_root / tmp_path.name
    old_run = policy_root / "old"
    run_dir.mkdir(parents=True)
    old_run.mkdir()
    (policy_root / "latest").symlink_to(old_run.name)
    report_path = run_dir / "checkpoint_selection.json"

    selector._atomic_write_report(  # noqa: SLF001
        path=report_path,
        report={"selected_iteration": None},
    )

    assert selection.selected is None
    assert json.loads(report_path.read_text())["selected_iteration"] is None
    assert os.readlink(policy_root / "latest") == old_run.name


def test_publish_reloads_winner_and_advances_relative_latest_only_after_export(
    monkeypatch,
    tmp_path: Path,
) -> None:
    selector = _load_selector_module()
    import robotics_experiments.rl.training.rsl_rl as training
    from robotics_experiments.rl.training.selection import CheckpointSelection, SelectedCheckpoint

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
    selection = CheckpointSelection(
        selection_method="press-acceptance-maximin-v1",
        candidates=(),
        selected=SelectedCheckpoint(iteration=50, path=checkpoint),
    )

    selector._publish_selection(  # noqa: SLF001 - focused publication contract test.
        selection=selection,
        report={"selected_iteration": 50},
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
    assert metadata["selection_method"] == "press-acceptance-maximin-v1"
    assert (run_dir / "checkpoint_selection.json").is_file()


def test_export_failure_preserves_previous_latest(monkeypatch, tmp_path: Path) -> None:
    selector = _load_selector_module()
    import robotics_experiments.rl.training.rsl_rl as training
    from robotics_experiments.rl.training.selection import CheckpointSelection, SelectedCheckpoint

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
        selection_method="press-acceptance-maximin-v1",
        candidates=(),
        selected=SelectedCheckpoint(iteration=50, path=checkpoint),
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

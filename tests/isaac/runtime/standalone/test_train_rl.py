"""Unit tests for merged scratch training and its RSL-RL adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.config import RlTrainingPhaseConfig
from robotics_experiments.flight_parameters import load_flight_parameters
from robotics_experiments.rl.evaluation import EvaluationMetric, PolicyEvaluation
from robotics_experiments.rl.press import create_press_acceptance_criteria
from tests.support.paths import REPO_ROOT


def _load_train_rl_module():
    pytest.importorskip("torch")
    script = REPO_ROOT / "isaac" / "runtime" / "standalone" / "train_rl.py"
    spec = importlib.util.spec_from_file_location("train_rl", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _phase(
    *,
    maximum: int = 20,
    interval: int = 10,
) -> RlTrainingPhaseConfig:
    return RlTrainingPhaseConfig(
        max_iterations=maximum,
        evaluation_interval=interval,
        learning_rate=1.0e-4,
        schedule="fixed",
        entropy_coef=0.0,
    )


def _inputs(tmp_path: Path):
    flight_parameters = load_flight_parameters(REPO_ROOT / "tests/fixtures/flight_parameters")
    return SimpleNamespace(
        controller=SimpleNamespace(selected_settings=SimpleNamespace(hover_thrust_normalized=0.5)),
        controller_path=tmp_path / "resolved/controller.json",
        controller_digest="controller-digest",
        flight_parameters=flight_parameters,
        flight_parameters_path=tmp_path / "resolved/flight-parameters",
        flight_parameters_digest="flight-parameters-digest",
        hover_reset_samples=SimpleNamespace(summary=lambda **_kwargs: {"sample_count": 1.0}),
    )


def _acceptance_criteria(config):
    return create_press_acceptance_criteria(
        config=config.rl.press.acceptance,
        flight_parameters=load_flight_parameters(REPO_ROOT / "tests/fixtures/flight_parameters"),
    )


def test_train_rl_uses_the_requested_artifact_run_id(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    policy_root = tmp_path / "training" / "press-approach"
    run_dir = train_rl._make_run_dir(  # noqa: SLF001 - focused script helper test
        out_root=policy_root,
        run_id="requested-run",
    )
    assert run_dir == policy_root / "requested-run"
    assert not (policy_root / "latest").exists()


@pytest.mark.parametrize("policy_name", ["", ".", "..", "/hover", "hover/latest"])
def test_train_rl_rejects_invalid_policy_names(policy_name: str) -> None:
    train_rl = _load_train_rl_module()
    with pytest.raises(ValueError, match="policy_name"):
        train_rl._validate_policy_name(policy_name)  # noqa: SLF001


def test_procedural_smoke_keeps_every_phase_boundary_at_one_update() -> None:
    train_rl = _load_train_rl_module()
    phases = train_rl._procedural_smoke_phases(  # noqa: SLF001
        phases=(_phase(maximum=20, interval=10), _phase(maximum=30, interval=5))
    )
    assert tuple(phase.max_iterations for phase in phases) == (1, 1)
    assert tuple(phase.evaluation_interval for phase in phases) == (1, 1)


def test_later_press_stage_uses_the_previous_ordered_accepted_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config
    from robotics_experiments.rl import PolicyContract, PolicyInterface

    config = load_config(REPO_ROOT / "configs/train_press_approach_quad.yaml")
    contract = PolicyContract(
        policy_interface=PolicyInterface.PRESS,
        obs_schema_version=2,
        obs_dim=34,
        policy_rate_hz=20.0,
        action_kind="velocity",
        action_dim=4,
        max_body_velocity_mps=0.5,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_normalized=0.57,
        curriculum_stage="approach",
    )
    policy_root = tmp_path / ".build/training/press-hover"
    run_dir = policy_root / "accepted-hover"
    checkpoint = run_dir / "attempt/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "training_report.json").write_text(
        json.dumps(
            {
                "status": "accepted",
                "acceptance": {"passed": True},
                "selected_checkpoint": "attempt/checkpoint.pt",
            }
        )
    )
    policy_root.mkdir(exist_ok=True)
    (policy_root / "best").symlink_to(run_dir.name)
    monkeypatch.setattr(
        train_rl.rl,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(
            contract=train_rl.replace(contract, curriculum_stage="hover"),
            controller_digest="controller-digest",
            flight_parameters_digest="flight-parameters-digest",
            values={"selected_policy_digest": "policy-digest"},
        ),
    )

    selected = train_rl._previous_stage_checkpoint(  # noqa: SLF001
        config=config,
        repo_root=tmp_path,
        contract=contract,
        controller_digest="controller-digest",
        flight_parameters_digest="flight-parameters-digest",
    )

    assert selected is not None
    assert selected[0] == checkpoint
    assert selected[1]["stage"] == "hover"


def test_report_schema_records_disjoint_300_episode_workloads(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    acceptance_criteria = _acceptance_criteria(config)
    report = train_rl._training_report(  # noqa: SLF001
        run_dir=tmp_path,
        config=config,
        phases=tuple(config.rl.training.phases),
        comparison_fingerprint="fingerprint",
        attempts=[],
        selected=None,
        selected_attempt_index=None,
        selected_policy_path=None,
        test_evaluation=None,
        acceptance=None,
        status="training",
        acceptance_criteria=acceptance_criteria,
        hover_reset_distribution={"source": "flight_parameters_policy_window"},
    )
    assert report["schema_version"] == 23
    assert report["reward_configuration"] == {
        "hover_stabilization": "boundary_weighted",
        "position_progress_weight": 0.1,
        "stabilization_weight": 0.01,
        "outside_envelope_weight": 0.03,
        "predictive_barrier_weight": 0.1,
        "action_effort_weight": 1.0,
        "hold_progress_weight": 0.01,
        "terminal_success_weight": 1.5,
        "terminal_failure_weight": 3.0,
    }
    assert report["hover_reset_distribution"] == {"source": "flight_parameters_policy_window"}
    assert report["report_kind"] == "campaign"
    assert report["attempt_execution"] == "isolated_process"
    assert report["workloads"] == {
        "validation": {
            "role": "validation",
            "seeds": [101, 202, 303],
            "episodes_per_seed": 100,
            "episode_count": 300,
        },
        "test": {
            "role": "test",
            "seeds": [404, 505, 606],
            "episodes_per_seed": 100,
            "episode_count": 300,
        },
    }
    assert report["acceptance_criteria"]["minimum_range_m"] == 0.40
    assert len(report["acceptance_criteria"]["hover_limits"]) == 10


def test_attempt_report_preserves_unavailable_metrics_as_null(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    evaluation = train_rl._evaluation_from_payload(  # noqa: SLF001
        payload={
            "metrics": {"hover_position_rmse_m": None},
            "reward_means": {},
        },
        iteration=1,
        checkpoint_path=tmp_path / "model.pt",
        policy_digest="digest",
        source="fresh-validation",
    )

    assert evaluation.metrics[0].value is None


def test_attempt_worker_command_forwards_pinned_inputs_and_one_hydra_seed() -> None:
    train_rl = _load_train_rl_module()
    command = train_rl._attempt_worker_command(  # noqa: SLF001
        args=SimpleNamespace(
            procedural_smoke=True,
            scene="@config",
            strict=True,
        ),
        attempt_index=2,
        controller_artifact_digest="controller-digest",
        controller_artifact_path=Path("resolved/controller.json"),
        flight_parameters_artifact_digest="flight-parameters-digest",
        flight_parameters_artifact_path=Path("resolved/flight-parameters"),
        seed=44,
    )
    assert command[:2] == [train_rl.sys.executable, str(Path(train_rl.__file__).resolve())]
    assert command[2:] == [
        "--attempt-worker",
        "--attempt-index",
        "2",
        "--attempt-seed",
        "44",
        "--controller-artifact",
        "resolved/controller.json",
        "--controller-artifact-digest",
        "controller-digest",
        "--flight-parameters-artifact",
        "resolved/flight-parameters",
        "--flight-parameters-artifact-digest",
        "flight-parameters-digest",
        "--procedural-smoke",
        "--scene",
        "@config",
        "--strict",
    ]


def test_campaign_parent_prepares_and_pins_input_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config

    config_path = REPO_ROOT / "configs" / "train_press_hover_quad.yaml"
    config = load_config(config_path)
    inputs = _inputs(tmp_path)
    campaign: dict[str, object] = {}
    monkeypatch.setattr(train_rl, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        train_rl.rl,
        "prepare_policy_inputs",
        lambda **_kwargs: inputs,
    )
    monkeypatch.setattr(
        train_rl,
        "_run_policy_workflow",
        lambda **kwargs: campaign.update(kwargs),
    )

    train_rl._run(  # noqa: SLF001
        args=SimpleNamespace(attempt_worker=False),
        config=config,
        config_path=config_path,
    )

    assert campaign["inputs"] is inputs
    assert campaign["acceptance_criteria"] == _acceptance_criteria(config)


def test_invalid_campaign_input_fails_before_run_creation(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config

    config_path = REPO_ROOT / "configs" / "train_press_hover_quad.yaml"
    config = load_config(config_path)
    monkeypatch.setattr(train_rl, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        train_rl.rl,
        "prepare_policy_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("unsupported controller artifact")),
    )
    monkeypatch.setattr(
        train_rl,
        "_run_policy_workflow",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("campaign must not start")),
    )

    with pytest.raises(SystemExit) as raised:
        train_rl._run(  # noqa: SLF001
            args=SimpleNamespace(attempt_worker=False),
            config=config,
            config_path=config_path,
        )

    assert raised.value.code == 2
    assert capsys.readouterr().err.endswith(
        "input artifact validation FAILED: unsupported controller artifact\n"
    )
    assert not (tmp_path / ".build").exists()


def test_campaign_retries_in_distinct_processes_and_stops_after_validation_pass(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config

    config_path = REPO_ROOT / "configs" / "train_press_approach_quad.yaml"
    config = load_config(config_path)
    policy_root = tmp_path / "training" / "press-approach"
    run_dir = policy_root / "campaign"
    commands: list[list[str]] = []

    def make_run_dir(**_kwargs):
        run_dir.mkdir(parents=True)
        return run_dir

    def run_worker(command, **kwargs):
        commands.append(command)
        attempt_index = int(command[command.index("--attempt-index") + 1])
        seed = int(command[command.index("--attempt-seed") + 1])
        passed = attempt_index == 1
        attempt_dir = run_dir / f"attempt_{attempt_index + 1:02d}_seed_{seed}"
        candidate_dir = attempt_dir / "evaluated" / "candidate_000"
        checkpoint = attempt_dir / "checkpoints" / "model.pt"
        candidate_dir.mkdir(parents=True)
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        (candidate_dir / "policy.pt").write_bytes(f"policy-{seed}".encode())
        (candidate_dir / "policy_meta.json").write_text("{}\n")
        digest = f"digest-{seed}"
        evaluation = PolicyEvaluation(
            iteration=1,
            checkpoint_path=checkpoint,
            policy_digest=digest,
            metrics=(EvaluationMetric("hover_envelope_utilization_mean", 2.0 - attempt_index),),
            reward_means=(),
            source="fresh-validation",
        )
        validation = {"passed": passed, "checks": [], "reasons": []}
        attempt = {
            "attempt_index": attempt_index,
            "training_seed": seed,
            "worker_pid": train_rl.os.getpid() + 100 + attempt_index,
            "evaluations": [
                {
                    "iteration": 1,
                    "checkpoint": str(checkpoint.relative_to(run_dir)),
                    "policy": str((candidate_dir / "policy.pt").relative_to(run_dir)),
                    "policy_digest": digest,
                    "source": "fresh-validation",
                    "phase_index": 2,
                    "metrics": {"hover_envelope_utilization_mean": 2.0 - attempt_index},
                    "reward_means": {},
                }
            ],
            "phase_results": [],
            "selected_policy_digest": digest,
            "selected_checkpoint": str(checkpoint.relative_to(run_dir)),
            "validation": validation,
        }
        report = train_rl._attempt_report(  # noqa: SLF001
            run_dir=run_dir,
            config=config,
            phases=(config.rl.training.phases[0],),
            comparison_fingerprint="fingerprint",
            attempt=attempt,
            selected=evaluation,
            selected_policy_path=candidate_dir / "policy.pt",
            status="validation_passed" if passed else "validation_failed",
        )
        train_rl._write_json(  # noqa: SLF001
            path=attempt_dir / "attempt_report.json",
            payload=report,
        )
        assert kwargs["env"]["RL_CAMPAIGN_RUN_DIR"] == str(run_dir)
        return SimpleNamespace(returncode=0)

    finalized = {}
    monkeypatch.setattr(train_rl, "_make_run_dir", make_run_dir)
    monkeypatch.setattr(train_rl, "probe_telemetry", lambda: {"gpu_utilization_rate": 0.5})
    monkeypatch.setattr(train_rl.subprocess, "run", run_worker)
    monkeypatch.setattr(
        train_rl,
        "_run_final_test",
        lambda **kwargs: (kwargs["selection"].evaluation, {}),
    )
    monkeypatch.setattr(
        train_rl,
        "evaluate_press_acceptance",
        lambda **_kwargs: train_rl.AcceptanceResult(passed=True, checks=(), reasons=()),
    )
    monkeypatch.setattr(
        train_rl,
        "_record_final_test",
        lambda **kwargs: finalized.update(kwargs) or {},
    )
    monkeypatch.setattr(
        train_rl,
        "_publish_selected",
        lambda **_kwargs: False,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch.utils.tensorboard",
        SimpleNamespace(SummaryWriter=lambda **_kwargs: SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.delenv("RL_RUN_ID", raising=False)
    train_rl._run_policy_workflow(  # noqa: SLF001
        args=SimpleNamespace(
            procedural_smoke=False,
            scene=None,
            strict=False,
        ),
        config=config,
        config_path=config_path,
        inputs=_inputs(tmp_path),
        acceptance_criteria=_acceptance_criteria(config),
        comparison_fingerprint="fingerprint",
        attempt_seeds=(42, 43, 44),
        policy_root=policy_root,
        repo_root=tmp_path,
    )
    assert [command[command.index("--attempt-seed") + 1] for command in commands] == [
        "42",
        "43",
    ]
    assert finalized["selected_attempt_index"] == 1
    assert [attempt["training_seed"] for attempt in finalized["attempts"]] == [42, 43]
    assert len({attempt["worker_pid"] for attempt in finalized["attempts"]}) == 2


def test_campaign_stops_retrying_when_final_phase_did_not_improve(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    evaluation = PolicyEvaluation(
        iteration=10,
        checkpoint_path=tmp_path / "model.pt",
        policy_digest="selected",
        metrics=(),
        reward_means=(),
        source="fresh-validation",
    )
    selection = train_rl._AttemptSelection(  # noqa: SLF001
        evaluation=evaluation,
        artifact_dir=tmp_path,
        validation=SimpleNamespace(passed=False),
    )
    assert train_rl._final_phase_failed_to_improve(  # noqa: SLF001
        attempt={"evaluations": [{"policy_digest": "selected", "phase_index": 1}]},
        selection=selection,
        phase_count=3,
    )
    assert not train_rl._final_phase_failed_to_improve(  # noqa: SLF001
        attempt={"evaluations": [{"policy_digest": "selected", "phase_index": 2}]},
        selection=selection,
        phase_count=3,
    )


def test_campaign_parent_preserves_active_attempt_on_interruption(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config

    config_path = REPO_ROOT / "configs" / "train_press_approach_quad.yaml"
    config = load_config(config_path)
    policy_root = tmp_path / "training" / "press-approach"
    run_dir = policy_root / "campaign"

    def make_run_dir(**_kwargs):
        run_dir.mkdir(parents=True)
        return run_dir

    def interrupt(command, **_kwargs):
        attempt_index = int(command[command.index("--attempt-index") + 1])
        seed = int(command[command.index("--attempt-seed") + 1])
        attempt_dir = run_dir / f"attempt_{attempt_index + 1:02d}_seed_{seed}"
        attempt_dir.mkdir()
        (attempt_dir / "attempt_report.json").write_text(
            '{"schema_version": 21, "report_kind": "attempt", "status": "training"}\n'
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(train_rl, "_make_run_dir", make_run_dir)
    monkeypatch.setattr(train_rl, "probe_telemetry", lambda: {"gpu_utilization_rate": 0.5})
    monkeypatch.setattr(train_rl.subprocess, "run", interrupt)
    monkeypatch.delenv("RL_RUN_ID", raising=False)
    with pytest.raises(KeyboardInterrupt):
        train_rl._run_policy_workflow(  # noqa: SLF001
            args=SimpleNamespace(
                procedural_smoke=False,
                scene=None,
                strict=False,
            ),
            config=config,
            config_path=config_path,
            inputs=_inputs(tmp_path),
            acceptance_criteria=_acceptance_criteria(config),
            comparison_fingerprint="fingerprint",
            attempt_seeds=(42, 43, 44),
            policy_root=policy_root,
            repo_root=tmp_path,
        )
    report = json.loads((run_dir / "training_report.json").read_text())
    assert report["status"] == "interrupted"
    assert report["attempts"] == []
    assert report["active_attempt"] == {
        "attempt_index": 0,
        "training_seed": 42,
        "report": "attempt_01_seed_42/attempt_report.json",
    }
    assert (run_dir / report["active_attempt"]["report"]).is_file()
    assert not (policy_root / "latest").exists()
    assert not (policy_root / "best").exists()


def test_campaign_parent_reports_missing_worker_report_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config

    config_path = REPO_ROOT / "configs" / "train_press_hover_quad.yaml"
    config = load_config(config_path)
    policy_root = tmp_path / "training" / "press-hover"
    run_dir = policy_root / "campaign"

    def make_run_dir(**_kwargs):
        run_dir.mkdir(parents=True)
        return run_dir

    monkeypatch.setattr(train_rl, "_make_run_dir", make_run_dir)
    monkeypatch.setattr(train_rl, "probe_telemetry", lambda: {"gpu_utilization_rate": 0.5})
    monkeypatch.setattr(
        train_rl.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.delenv("RL_RUN_ID", raising=False)

    with pytest.raises(RuntimeError, match="exited without an attempt report"):
        train_rl._run_policy_workflow(  # noqa: SLF001
            args=SimpleNamespace(
                procedural_smoke=False,
                scene=None,
                strict=False,
            ),
            config=config,
            config_path=config_path,
            inputs=_inputs(tmp_path),
            acceptance_criteria=_acceptance_criteria(config),
            comparison_fingerprint="fingerprint",
            attempt_seeds=(42,),
            policy_root=policy_root,
            repo_root=tmp_path,
        )

    report = json.loads((run_dir / "training_report.json").read_text())
    assert report["status"] == "attempt_failed"
    assert report["active_attempt"]["report"] == "attempt_01_seed_42/attempt_report.json"
    attempt_report = json.loads((run_dir / report["active_attempt"]["report"]).read_text())
    assert attempt_report == {
        "schema_version": 23,
        "report_kind": "attempt",
        "status": "failed",
        "artifact_run_id": "campaign",
        "attempt_index": 0,
        "training_seed": 42,
    }


def test_interrupt_handlers_raise_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_rl = _load_train_rl_module()
    installed: dict[int, object] = {}
    monkeypatch.setattr(
        train_rl.signal,
        "signal",
        lambda signal_number, handler, /: installed.__setitem__(signal_number, handler),
    )

    train_rl._install_interrupt_handlers()  # noqa: SLF001

    assert set(installed) == {
        train_rl.signal.SIGHUP,
        train_rl.signal.SIGINT,
        train_rl.signal.SIGTERM,
    }
    for handler in installed.values():
        with pytest.raises(KeyboardInterrupt):
            handler(0, None)  # type: ignore[operator]


def test_training_tensorboard_metrics_are_small_and_namespaced() -> None:
    source = (
        REPO_ROOT / "src/robotics_experiments/workflows/isaac_lab/tasks/switch_press_env.py"
    ).read_text()
    assert '"train/02_progress/hover_envelope_occupancy_rate"' in source
    assert '"train/02_progress/hover_envelope_utilization_mean"' in source
    assert '"train/03_reward/per_step_mean"' in source
    assert '"train/cumulative_outcome/' not in source
    assert '"train/target/' not in source
    assert "_percent" not in source
    assert "100.0 * self._" not in source


def test_attempt_teardown_drops_runner_closures_before_closing_environment() -> None:
    source = (REPO_ROOT / "isaac" / "runtime" / "standalone" / "train_rl.py").read_text()
    teardown = source.split('_report(f"attempt teardown START', maxsplit=1)[1].split(
        'f"attempt teardown COMPLETE', maxsplit=1
    )[0]
    assert teardown.index("evaluate_candidate = None") < teardown.index("runner = None")
    assert teardown.index("runner = None") < teardown.index("env.close()")
    assert teardown.index("env = None") < teardown.index("torch.cuda.empty_cache()")


def test_evaluation_log_reports_plateau_only_after_the_decision(capsys, tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    evaluation = PolicyEvaluation(
        iteration=20,
        checkpoint_path=tmp_path / "model_20.pt",
        policy_digest="digest",
        metrics=(
            EvaluationMetric("episode_longest_hover_duration_mean_s", 15.0),
            EvaluationMetric("success_rate", 0.95),
            EvaluationMetric("collision_rate", 0.01),
            EvaluationMetric("other_flight_failure_rate", 0.0),
            EvaluationMetric("timeout_rate", 0.04),
        ),
        reward_means=(),
        source="fresh-validation",
    )
    artifact_dir = tmp_path / "candidate"
    train_rl._report_evaluation(  # noqa: SLF001
        evaluation=evaluation,
        phase_index=0,
        phase_iteration=20,
        plateau_state=train_rl.PlateauState(reference_value=0.3, stale_evaluations=0),
        run_dir=tmp_path,
        artifact_dir=artifact_dir,
        validation=SimpleNamespace(checks=(), passed=False),
    )
    metrics_log = capsys.readouterr().err
    assert "longest_hover_mean_s=15.000" in metrics_log
    assert "progress=" not in metrics_log
    assert "stale=" not in metrics_log

    train_rl._report_plateau(  # noqa: SLF001
        evaluation=evaluation,
        made_progress=True,
        plateau_state=train_rl.PlateauState(reference_value=0.25, stale_evaluations=0),
        plateau_reached=False,
    )
    plateau_log = capsys.readouterr().err
    assert "progress=true reference=0.250000 stale=0 reached=false" in plateau_log


def test_git_metadata_marks_mounted_repository_safe(monkeypatch, tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    calls = []
    outputs = iter(("abc123\n", " M changed.py\n"))

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(train_rl.subprocess, "run", fake_run)
    metadata = train_rl._git_source_metadata(repo_root=tmp_path)  # noqa: SLF001
    safe_directory = f"safe.directory={tmp_path.resolve()}"
    assert calls[0][0][:3] == ["git", "-c", safe_directory]
    assert calls[1][0][:3] == ["git", "-c", safe_directory]
    assert metadata == {"source_commit": "abc123", "source_dirty": True}


def test_checkpoint_restore_loads_optimizer_and_iteration(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    calls = []
    runner = SimpleNamespace(
        current_learning_iteration=99,
        load=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    checkpoint = tmp_path / "model_299.pt"
    torch.save({"iter": 17}, checkpoint)
    restored = rsl_rl.restore_checkpoint(
        runner=runner,
        checkpoint_path=checkpoint,
        device="cpu",
    )
    assert restored == 17
    assert runner.current_learning_iteration == 17
    assert calls == [((str(checkpoint),), {"load_optimizer": True, "map_location": "cpu"})]


def test_publication_copies_the_exact_tested_actor_and_only_then_labels(tmp_path: Path) -> None:
    import hashlib

    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config
    from robotics_experiments.rl.press import AcceptanceResult

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    run_dir = tmp_path / "campaign"
    selected_dir = run_dir / "attempt_01_seed_42" / "evaluated" / "candidate_001"
    selected_dir.mkdir(parents=True)
    selected_policy = selected_dir / "policy.pt"
    selected_policy.write_bytes(b"immutable actor bytes")
    digest = hashlib.sha256(selected_policy.read_bytes()).hexdigest()
    selected = PolicyEvaluation(
        iteration=10,
        checkpoint_path=run_dir / "attempt_01_seed_42" / "checkpoints" / "model.pt",
        policy_digest=digest,
        metrics=(EvaluationMetric("hover_envelope_utilization_mean", 0.25),),
        reward_means=(),
        source="fresh-validation",
    )
    tested = PolicyEvaluation(
        iteration=10,
        checkpoint_path=selected.checkpoint_path,
        policy_digest=digest,
        metrics=(EvaluationMetric("hover_envelope_utilization_mean", 0.3),),
        reward_means=(),
        source="fresh-test",
    )
    acceptance = AcceptanceResult(passed=True, checks=(), reasons=())
    report = {
        "comparison_fingerprint": "fingerprint",
        "selected_policy_digest": digest,
        "test_evaluation": {"policy_digest": digest},
        "selected_metric": {
            "name": "hover_envelope_utilization_mean",
            "direction": "minimize",
            "value": 0.25,
        },
    }
    train_rl._publish_selected(  # noqa: SLF001
        run_dir=run_dir,
        policy_root=tmp_path,
        metadata={},
        report=report,
        selected=selected,
        selected_policy=selected_policy,
        test_evaluation=tested,
        acceptance=acceptance,
        comparison_fingerprint="fingerprint",
        config=config,
    )
    assert (run_dir / "policy.pt").read_bytes() == b"immutable actor bytes"
    assert hashlib.sha256((run_dir / "policy.pt").read_bytes()).hexdigest() == digest
    published_metadata = json.loads((run_dir / "policy_meta.json").read_text())
    published_report = json.loads((run_dir / "training_report.json").read_text())
    assert published_metadata["selected_policy_digest"] == digest
    assert published_metadata["test_evaluation"]["policy_digest"] == digest
    assert published_report["selected_policy_digest"] == digest
    assert published_report["test_evaluation"]["policy_digest"] == digest
    assert (tmp_path / "latest").resolve() == run_dir


def test_failed_test_cannot_publish_any_label(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config
    from robotics_experiments.rl.press import AcceptanceResult

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    policy = run_dir / "candidate.pt"
    policy.write_bytes(b"actor")
    selected = PolicyEvaluation(
        iteration=1,
        checkpoint_path=run_dir / "model.pt",
        policy_digest="irrelevant",
        metrics=(EvaluationMetric("evaluation_diagnostic_cost", 1.0),),
        reward_means=(),
        source="fresh-validation",
    )
    with pytest.raises(ValueError, match="failed test"):
        train_rl._publish_selected(  # noqa: SLF001
            run_dir=run_dir,
            policy_root=tmp_path,
            metadata={},
            report={},
            selected=selected,
            selected_policy=policy,
            test_evaluation=selected,
            acceptance=AcceptanceResult(passed=False, checks=(), reasons=("failure",)),
            comparison_fingerprint="fingerprint",
            config=config,
        )
    assert not (tmp_path / "latest").exists()
    assert not (tmp_path / "best").exists()


def test_initial_checkpoint_has_rsl_state_without_calling_runner_logger(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer),
        current_learning_iteration=0,
    )
    checkpoint = tmp_path / "model_0.pt"
    rsl_rl.save_checkpoint(runner=runner, checkpoint_path=checkpoint, iteration=0)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["iter"] == 0
    assert payload["infos"] is None
    assert payload["model_state_dict"].keys() == policy.state_dict().keys()
    assert payload["optimizer_state_dict"]["param_groups"]


def test_phase_configuration_preserves_optimizer_and_updates_values() -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-4)
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            optimizer=optimizer,
            learning_rate=1.0e-4,
            entropy_coef=0.001,
            schedule="adaptive",
        )
    )
    phase = _phase()
    original_optimizer = runner.alg.optimizer
    rsl_rl.configure_phase(runner=runner, phase=phase)
    assert runner.alg.optimizer is original_optimizer
    assert runner.alg.optimizer.param_groups[0]["lr"] == phase.learning_rate
    assert runner.alg.schedule == "fixed"
    assert runner.alg.entropy_coef == 0.0


def test_refinement_restores_then_configures_then_resets(monkeypatch, tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    torch = pytest.importorskip("torch")
    calls = []
    runner = object()

    class FakeEnv:
        def reset(self):
            calls.append(("reset", torch.is_inference_mode_enabled()))

    checkpoint = tmp_path / "model.pt"

    def restore_checkpoint(**kwargs):
        calls.append(("restore", kwargs))
        return 17

    monkeypatch.setattr(
        train_rl.training,
        "restore_checkpoint",
        restore_checkpoint,
    )
    monkeypatch.setattr(
        train_rl.training,
        "configure_phase",
        lambda **kwargs: calls.append(("configure", kwargs)),
    )
    phase = _phase()
    restored = train_rl._restore_phase_branch(  # noqa: SLF001
        runner=runner,
        env=FakeEnv(),
        parent_checkpoint=checkpoint,
        phase=phase,
        phase_number=2,
        device="cuda:0",
    )
    assert restored == 17
    assert [call[0] for call in calls] == [
        "restore",
        "configure",
        "reset",
    ]
    assert calls[0][1]["checkpoint_path"] == checkpoint
    assert calls[1][1]["phase"] is phase
    assert calls[1][1]["phase_number"] == 2
    assert calls[-1] == ("reset", True)


def test_training_chunks_do_not_repeat_boundary_iteration_numbers() -> None:
    from robotics_experiments.rl.training import rsl_rl

    starts: list[tuple[int, dict[str, int | bool]]] = []

    runner = SimpleNamespace(current_learning_iteration=0, log_dir="logs")

    def learn(
        *,
        num_learning_iterations: int,
        init_at_random_ep_len: bool,
    ) -> None:
        starts.append(
            (
                runner.current_learning_iteration,
                {
                    "num_learning_iterations": num_learning_iterations,
                    "init_at_random_ep_len": init_at_random_ep_len,
                },
            )
        )

    runner.learn = learn
    first = rsl_rl.train_chunk(
        runner=runner,
        completed_iterations=0,
        iterations=10,
        randomize_initial_episode_lengths=True,
    )
    second = rsl_rl.train_chunk(
        runner=runner,
        completed_iterations=10,
        iterations=5,
        randomize_initial_episode_lengths=False,
    )
    assert starts == [
        (1, {"num_learning_iterations": 10, "init_at_random_ep_len": True}),
        (11, {"num_learning_iterations": 5, "init_at_random_ep_len": False}),
    ]
    assert first.iterations == 10
    assert second.iterations == 5


def test_training_adapter_disables_internal_periodic_checkpoint_cadence() -> None:
    from robotics_experiments.rl.training import rsl_rl

    training_cfg = SimpleNamespace(
        num_steps_per_env=32,
        gamma=0.995,
        lam=0.95,
        clip_param=0.2,
        value_loss_coef=1.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    policy_cfg = SimpleNamespace(
        activation="elu",
        actor=SimpleNamespace(
            hidden_dims=[256, 256, 128],
            initial_noise_std=0.2,
            output_initialization="scratch-neutral-output",
        ),
        critic=SimpleNamespace(hidden_dims=[256, 256, 128]),
    )
    runner_cfg = rsl_rl._train_cfg(  # noqa: SLF001 - adapter mapping contract
        training_config=training_cfg,
        phase=_phase(),
        policy_config=policy_cfg,
    )
    assert runner_cfg["save_interval"] == 2**31 - 1
    assert runner_cfg["algorithm"]["learning_rate"] == 1.0e-4


def test_scratch_neutral_output_initialization_is_near_zero_and_unbiased() -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    actor = torch.nn.Sequential(
        torch.nn.Linear(34, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 4),
    )
    runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(actor=actor)))
    rsl_rl.initialize_actor_output(runner=runner, method="scratch-neutral-output")
    output = actor[-1]
    assert torch.count_nonzero(output.bias) == 0
    action_mean = actor(torch.zeros((128, 34)))
    assert float(action_mean.abs().max()) < 0.01


def test_export_contains_only_actor_and_metadata(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    actor = torch.nn.Sequential(torch.nn.Linear(34, 4))
    critic = torch.nn.Sequential(torch.nn.Linear(34, 1))
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=SimpleNamespace(actor=actor, critic=critic))
    )
    metadata = {"obs_dim": 34, "marker": "contract"}
    rsl_rl.export_policy(runner, tmp_path, metadata=metadata)
    module = torch.jit.load(str(tmp_path / "policy.pt"))
    assert module(torch.zeros((2, 34))).shape == (2, 4)
    assert json.loads((tmp_path / "policy_meta.json").read_text()) == metadata

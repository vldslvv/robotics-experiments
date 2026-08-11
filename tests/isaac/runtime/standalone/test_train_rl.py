"""Unit tests for merged scratch training and its RSL-RL adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.config import RlTrainingPhaseConfig
from robotics_experiments.rl.training.progress import EvaluationMetric, PolicyEvaluation
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


def _phase(*, maximum: int = 20, interval: int = 10) -> RlTrainingPhaseConfig:
    return RlTrainingPhaseConfig(
        max_iterations=maximum,
        evaluation_interval=interval,
        learning_rate=1.0e-4,
        schedule="fixed",
        entropy_coef=0.0,
        hard_case_fraction=0.0,
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


def test_phase_cap_preserves_order_and_reduces_terminal_evaluation_interval() -> None:
    train_rl = _load_train_rl_module()
    phases = train_rl._capped_phases(  # noqa: SLF001
        phases=(_phase(maximum=20, interval=10), _phase(maximum=20, interval=10)),
        maximum_iterations=25,
    )
    assert tuple(phase.max_iterations for phase in phases) == (20, 5)
    assert tuple(phase.evaluation_interval for phase in phases) == (10, 5)


def test_procedural_smoke_keeps_every_phase_boundary_at_one_update() -> None:
    train_rl = _load_train_rl_module()
    phases = train_rl._procedural_smoke_phases(  # noqa: SLF001
        phases=(_phase(maximum=20, interval=10), _phase(maximum=30, interval=5))
    )
    assert tuple(phase.max_iterations for phase in phases) == (1, 1)
    assert tuple(phase.evaluation_interval for phase in phases) == (1, 1)


def test_evaluation_cache_round_trip_rebinds_candidate_identity(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    original = PolicyEvaluation(
        iteration=10,
        checkpoint_path=Path("logs/model_10.pt"),
        policy_digest="policy-digest",
        metrics=(EvaluationMetric("held_out_task_cost", 0.4),),
        reward_means=(),
        source="fresh",
    )
    cache_path = tmp_path / "cache.json"
    train_rl._write_cached_evaluation(  # noqa: SLF001
        cache_path=cache_path,
        evaluation=original,
        policy_content_digest="content-digest",
        source_run_id="source",
    )
    loaded = train_rl._read_cached_evaluation(  # noqa: SLF001
        cache_path=cache_path,
        iteration=20,
        checkpoint_path=Path("logs/model_20.pt"),
        policy_digest="new-raw-policy-digest",
        policy_content_digest="content-digest",
    )
    assert loaded is not None
    assert loaded.iteration == 20
    assert loaded.checkpoint_path == Path("logs/model_20.pt")
    assert loaded.policy_digest == "new-raw-policy-digest"
    assert loaded.source == "cache"
    assert loaded.metric("held_out_task_cost") == 0.4


def test_validation_cache_identity_includes_role_and_exact_workload(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    common = {
        "cache_dir": tmp_path,
        "policy_content_digest": "content",
        "comparison_fingerprint": "fingerprint",
        "controller_digest": "controller",
        "episodes_per_seed": 100,
    }
    validation = train_rl._cache_path(  # noqa: SLF001
        **common,
        workload_role="validation",
        seeds=(101, 202, 303),
    )
    test = train_rl._cache_path(  # noqa: SLF001
        **common,
        workload_role="test",
        seeds=(404, 505, 606),
    )
    assert validation != test


def test_report_schema_records_disjoint_300_episode_workloads(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
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
    )
    assert report["schema_version"] == 2
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


def test_training_tensorboard_rates_are_fraction_valued_and_namespaced() -> None:
    source = (
        REPO_ROOT / "src/robotics_experiments/workflows/isaac_lab/tasks/switch_press_env.py"
    ).read_text()
    assert '"Training/cumulative_collision_rate"' in source
    assert '"Training/cumulative_flight_failure_rate"' in source
    assert "_percent" not in source
    assert "100.0 * self._" not in source


def test_policy_content_digest_ignores_only_generated_serialization_id(
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    paths = (tmp_path / "first.pt", tmp_path / "second.pt")
    for path, serialization_id in zip(paths, (b"first", b"second"), strict=True):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("policy/data/0", b"model parameters")
            archive.writestr("policy/code/model.py", b"def forward(): pass")
            archive.writestr("policy/.data/serialization_id", serialization_id)
    assert train_rl._file_sha256(paths[0]) != train_rl._file_sha256(paths[1])  # noqa: SLF001
    assert train_rl._policy_content_sha256(  # noqa: SLF001
        paths[0]
    ) == train_rl._policy_content_sha256(paths[1])  # noqa: SLF001


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
    from robotics_experiments.rl.training.progress import AcceptanceResult

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
        metrics=(EvaluationMetric("held_out_task_cost", 0.25),),
        reward_means=(),
        source="fresh-validation",
    )
    tested = PolicyEvaluation(
        iteration=10,
        checkpoint_path=selected.checkpoint_path,
        policy_digest=digest,
        metrics=(EvaluationMetric("held_out_task_cost", 0.3),),
        reward_means=(),
        source="fresh-test",
    )
    acceptance = AcceptanceResult(passed=True, gates=(), reasons=())
    report = {
        "comparison_fingerprint": "fingerprint",
        "selected_policy_digest": digest,
        "test_evaluation": {"policy_digest": digest},
        "selected_metric": {
            "name": "held_out_task_cost",
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
    from robotics_experiments.rl.training.progress import AcceptanceResult

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad.yaml")
    run_dir = tmp_path / "campaign"
    run_dir.mkdir()
    policy = run_dir / "candidate.pt"
    policy.write_bytes(b"actor")
    selected = PolicyEvaluation(
        iteration=1,
        checkpoint_path=run_dir / "model.pt",
        policy_digest="irrelevant",
        metrics=(EvaluationMetric("held_out_task_cost", 1.0),),
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
            acceptance=AcceptanceResult(passed=False, gates=(), reasons=("gate",)),
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
        def set_training_hard_case_fraction(self, value):
            calls.append(("mixture", value))

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
        device="cuda:0",
    )
    assert restored == 17
    assert [call[0] for call in calls] == ["restore", "configure", "mixture", "reset"]
    assert calls[0][1]["checkpoint_path"] == checkpoint
    assert calls[1][1]["phase"] is phase
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

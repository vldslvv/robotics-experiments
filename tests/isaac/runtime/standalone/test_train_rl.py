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


def test_checkpoint_load_skips_optimizer_state(tmp_path: Path) -> None:
    from robotics_experiments.rl.training import rsl_rl

    calls = []
    runner = SimpleNamespace(load=lambda *args, **kwargs: calls.append((args, kwargs)))
    checkpoint = tmp_path / "model_299.pt"
    rsl_rl.load_checkpoint(
        runner=runner,
        checkpoint_path=checkpoint,
        device="cuda:0",
    )
    assert calls == [((str(checkpoint),), {"load_optimizer": False, "map_location": "cuda:0"})]


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
    rsl_rl.save_checkpoint(runner=runner, checkpoint_path=checkpoint)
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

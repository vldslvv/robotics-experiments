"""Unit tests for the RL training entrypoint's artifact path contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.paths import REPO_ROOT


def _load_train_rl_module():
    pytest.importorskip("torch")
    script = REPO_ROOT / "isaac" / "runtime" / "standalone" / "train_rl.py"
    spec = importlib.util.spec_from_file_location("train_rl", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_rl_uses_the_requested_artifact_run_id(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    training_root = tmp_path / "training"
    policy_root = train_rl._policy_out_root(  # noqa: SLF001 - helper contract.
        training_root,
        "hover",
    )
    requested_run_id = tmp_path.name

    run_dir = train_rl._make_run_dir(  # noqa: SLF001 - helper contract.
        policy_root,
        requested_run_id,
    )

    assert run_dir.parent == training_root / "hover"
    assert run_dir.name == requested_run_id
    assert not (policy_root / "latest").exists()


@pytest.mark.parametrize("policy_name", ["", ".", "..", "/hover", "hover/latest", "hover.latest"])
def test_train_rl_rejects_invalid_policy_names(policy_name: str, tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()

    with pytest.raises(ValueError, match="policy_name"):
        train_rl._policy_out_root(  # noqa: SLF001 - helper contract.
            tmp_path,
            policy_name,
        )


def test_hover_artifact_metadata_remains_task_owned_and_unchanged() -> None:
    train_rl = _load_train_rl_module()
    action = SimpleNamespace(
        kind="velocity",
        action_dim=4,
        max_body_velocity_mps=3.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_fraction=0.4,
    )
    config = SimpleNamespace(rl=SimpleNamespace(action=action, policy_rate_hz=20.0))
    env_cfg = SimpleNamespace(
        observation_schema_version=1,
        observation_space=13,
        policy_interface="hover-v1",
    )

    metadata = train_rl._artifact_metadata(  # noqa: SLF001
        config=config,
        env_cfg=env_cfg,
        source_metadata={"source_commit": "abc123", "source_dirty": False},
        artifact_run_id="run-1",
    )

    assert metadata["obs_schema_version"] == 1
    assert metadata["obs_dim"] == 13
    assert metadata["policy_interface"] == "hover-v1"
    assert metadata["policy_rate_hz"] == 20.0
    assert metadata["source_commit"] == "abc123"
    assert metadata["source_dirty"] is False
    assert metadata["artifact_run_id"] == "run-1"


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


def test_checkpoint_export_resolves_existing_policy_run(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    out_root = tmp_path / "training" / "press"
    checkpoint = out_root / "run-1" / "logs" / "model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    run_dir = train_rl._checkpoint_run_dir(  # noqa: SLF001 - helper contract.
        checkpoint_path=checkpoint,
        out_root=out_root,
    )

    assert run_dir == out_root / "run-1"


def test_checkpoint_export_rejects_checkpoint_outside_policy_root(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    checkpoint = tmp_path / "other" / "run-1" / "logs" / "model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    with pytest.raises(ValueError, match="outside policy artifact root"):
        train_rl._checkpoint_run_dir(  # noqa: SLF001 - helper contract.
            checkpoint_path=checkpoint,
            out_root=tmp_path / "training" / "press",
        )


def test_checkpoint_export_reuses_contract_captured_at_training_start(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    metadata = {
        "policy_interface": "press-v1",
        "obs_schema_version": 1,
        "obs_dim": 34,
        "policy_rate_hz": 17.0,
        "action_kind": "velocity",
        "action_dim": 4,
        "action_scales": {
            "max_body_velocity_mps": 0.5,
            "max_yaw_rate_radps": 1.5,
            "max_body_rate_radps": 6.0,
            "hover_thrust_fraction": 0.4,
        },
        "source_commit": "captured",
        "source_dirty": False,
    }
    train_rl._write_artifact_metadata(  # noqa: SLF001
        run_dir=tmp_path,
        metadata=metadata,
    )

    stored, contract = train_rl._stored_artifact_metadata(run_dir=tmp_path)  # noqa: SLF001

    assert stored == metadata
    assert contract.policy_rate_hz == 17.0
    assert contract.max_body_velocity_mps == 0.5


def test_checkpoint_export_rejects_run_without_stored_contract(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()

    with pytest.raises(ValueError, match="original training contract cannot be recovered"):
        train_rl._stored_artifact_metadata(run_dir=tmp_path)  # noqa: SLF001


def test_training_persists_contract_before_ppo_without_exporting_or_publishing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    action = SimpleNamespace(
        kind="velocity",
        action_dim=4,
        max_body_velocity_mps=0.75,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_fraction=0.4,
    )
    training_cfg = SimpleNamespace(
        task_id="press-task",
        policy_name="press",
        seed=7,
        max_iterations=2,
    )
    config = SimpleNamespace(
        rl=SimpleNamespace(
            action=action,
            policy_rate_hz=17.0,
            training=training_cfg,
        )
    )
    env = SimpleNamespace(close=lambda: None)
    env_cfg = SimpleNamespace(
        observation_schema_version=1,
        observation_space=34,
        policy_interface="press-v1",
    )
    runner = object()
    before_ppo = {}
    export_calls = []

    monkeypatch.setattr(train_rl, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(train_rl, "_build_env", lambda **_kwargs: (env, env_cfg))
    monkeypatch.setattr(
        train_rl,
        "_git_source_metadata",
        lambda **_kwargs: {"source_commit": "captured", "source_dirty": False},
    )
    import robotics_experiments.rl.training.rsl_rl as training

    def build_runner(*_args, log_dir, **_kwargs):
        run_dir = Path(log_dir).parent
        before_ppo.update(json.loads((run_dir / "policy_meta.json").read_text()))
        return runner

    monkeypatch.setattr(training, "build_runner", build_runner)
    monkeypatch.setattr(
        training,
        "train",
        lambda *_args: SimpleNamespace(iterations=2),
    )
    monkeypatch.setattr(
        training,
        "export_policy",
        lambda _runner, _run_dir, *, metadata: export_calls.append(metadata),
    )
    args = SimpleNamespace(
        seed=None,
        max_iterations=None,
        num_envs=None,
        export_checkpoint=None,
    )

    train_rl._run(args=args, config=config)  # noqa: SLF001

    assert before_ppo["policy_rate_hz"] == 17.0
    assert before_ppo["action_scales"]["max_body_velocity_mps"] == 0.75
    assert before_ppo["artifact_run_id"]
    assert export_calls == []
    assert not (tmp_path / ".build" / "training" / "press" / "latest").exists()


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


def test_deterministic_actor_returns_actor_mean_in_evaluation_mode() -> None:
    from robotics_experiments.rl.training import rsl_rl

    class FakeActor:
        def __init__(self) -> None:
            self.evaluating = False

        def eval(self) -> None:
            self.evaluating = True

    actor = FakeActor()
    runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(actor=actor)))

    observed = rsl_rl.deterministic_actor(runner=runner)

    assert observed is actor
    assert actor.evaluating

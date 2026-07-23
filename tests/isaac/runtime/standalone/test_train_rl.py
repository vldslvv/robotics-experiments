"""Unit tests for the RL training entrypoint's artifact path contract."""

from __future__ import annotations

import importlib.util
import os
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


def test_train_rl_paths_are_namespaced_by_policy_name(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    training_root = tmp_path / "training"
    policy_root = train_rl._policy_out_root(  # noqa: SLF001 - helper contract.
        training_root,
        "hover",
    )

    run_dir = train_rl._make_run_dir(policy_root)  # noqa: SLF001 - helper contract.
    train_rl._update_latest_symlink(  # noqa: SLF001 - helper contract.
        policy_root,
        run_dir,
    )

    assert run_dir.parent == training_root / "hover"
    assert run_dir.name != "latest"
    assert os.readlink(policy_root / "latest") == run_dir.name


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
    )

    assert metadata["obs_schema_version"] == 1
    assert metadata["obs_dim"] == 13
    assert metadata["policy_interface"] == "hover-v1"
    assert metadata["policy_rate_hz"] == 20.0
    assert metadata["source_commit"] == "abc123"
    assert metadata["source_dirty"] is False


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

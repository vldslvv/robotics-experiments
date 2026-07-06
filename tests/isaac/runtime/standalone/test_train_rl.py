"""Unit tests for the RL training entrypoint's artifact path contract."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

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

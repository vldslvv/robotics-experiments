"""Tests for the reusable training/evaluation power-off wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.support.paths import REPO_ROOT

_SCRIPT = REPO_ROOT / "make/scripts/run_training_evaluation_then_shutdown.sh"


def _write_executable(*, path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_wrapper(
    *,
    tmp_path: Path,
    train_exit: int,
    eval_exit: int = 0,
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.log"
    config = tmp_path / "experiment.yaml"
    config.write_text("defaults: []\n", encoding="utf-8")

    _write_executable(
        path=fake_bin / "sudo",
        body="""#!/usr/bin/env bash
printf 'sudo %s\\n' "$*" >>"${WRAPPER_CALLS}"
exit 0
""",
    )
    _write_executable(
        path=fake_bin / "make",
        body="""#!/usr/bin/env bash
printf 'make %s\\n' "$*" >>"${WRAPPER_CALLS}"
if [ "$1" = "train-test" ]; then
    exit "${TRAIN_EXIT}"
fi
exit "${EVAL_EXIT}"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EVAL_EXIT": str(eval_exit),
            "TRAIN_EXIT": str(train_exit),
            "WRAPPER_CALLS": str(calls),
        }
    )
    return subprocess.run(
        [
            "bash",
            str(_SCRIPT),
            "--hydra-config",
            str(config),
            "--train-target",
            "train-test",
            "--eval-target",
            "eval-test",
            "--train-args",
            "--iterations 3",
            "--eval-args",
            "--episodes 4",
            "--label",
            "test-run",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_wrapper_runs_training_evaluation_and_requests_power_off(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path=tmp_path, train_exit=0)

    assert result.returncode == 0
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert (
        "make train-test HYDRA_CONFIG="
        f"{tmp_path}/experiment.yaml TRAIN_ARGS=--iterations 3"
    ) in calls
    assert (
        "make eval-test HYDRA_CONFIG="
        f"{tmp_path}/experiment.yaml EVAL_ARGS=--episodes 4"
    ) in calls
    assert "sudo -n shutdown -P now" in calls
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert "train_exit=0 eval_exit=0 overall_exit=0" in status


@pytest.mark.parametrize("train_exit", [1, 42])
def test_wrapper_skips_evaluation_after_training_failure(
    *,
    tmp_path: Path,
    train_exit: int,
) -> None:
    result = _run_wrapper(tmp_path=tmp_path, train_exit=train_exit)

    assert result.returncode == train_exit
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert "make train-test " in calls
    assert "make eval-test " not in calls
    assert "sudo -n shutdown -P now" in calls
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert f"train_exit={train_exit} eval_exit=not-run overall_exit={train_exit}" in status


def test_wrapper_requests_power_off_after_evaluation_failure(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path=tmp_path, train_exit=0, eval_exit=23)

    assert result.returncode == 23
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert "make train-test " in calls
    assert "make eval-test " in calls
    assert "sudo -n shutdown -P now" in calls
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert "train_exit=0 eval_exit=23 overall_exit=23" in status

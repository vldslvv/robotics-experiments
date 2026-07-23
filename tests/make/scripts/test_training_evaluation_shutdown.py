"""Tests for the reusable training/evaluation wrapper."""

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
    select_exit: int = 0,
    eval_exit: int = 0,
    shutdown: bool = False,
    shutdown_exit: int = 0,
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
if [ "$1" = "-n" ] && [ "$2" = "shutdown" ]; then
    exit "${SHUTDOWN_EXIT}"
fi
exit 0
""",
    )
    _write_executable(
        path=fake_bin / "sync",
        body="""#!/usr/bin/env bash
printf 'sync\\n' >>"${WRAPPER_CALLS}"
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
if [ "$1" = "select-test" ]; then
    exit "${SELECT_EXIT}"
fi
exit "${EVAL_EXIT}"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "EVAL_EXIT": str(eval_exit),
            "SELECT_EXIT": str(select_exit),
            "SHUTDOWN_EXIT": str(shutdown_exit),
            "TRAIN_EXIT": str(train_exit),
            "WRAPPER_CALLS": str(calls),
        }
    )
    command = [
        "bash",
        str(_SCRIPT),
        "--hydra-config",
        str(config),
        "--train-target",
        "train-test",
        "--eval-target",
        "eval-test",
        "--select-target",
        "select-test",
        "--train-args",
        "--iterations 3",
        "--eval-args",
        "--episodes 4",
        "--select-args",
        "--selection-note test",
        "--label",
        "test-run",
        "--log-dir",
        str(tmp_path / "logs"),
    ]
    if shutdown:
        command.append("--shutdown")
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_wrapper_runs_three_stages_with_one_artifact_id_without_shutdown_by_default(
    tmp_path: Path,
) -> None:
    result = _run_wrapper(tmp_path=tmp_path, train_exit=0)

    assert result.returncode == 0
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    call_lines = [line for line in calls.splitlines() if line.startswith("make ")]
    assert [line.split()[1] for line in call_lines] == [
        "train-test",
        "select-test",
        "eval-test",
    ]
    artifact_ids = {
        part.removeprefix("RL_RUN_ID=")
        for line in call_lines
        for part in line.split()
        if part.startswith("RL_RUN_ID=")
    }
    assert len(artifact_ids) == 1
    assert "TRAIN_ARGS=--iterations 3" in call_lines[0]
    assert "SELECT_ARGS=--selection-note test" in call_lines[1]
    assert "EVAL_ARGS=--episodes 4" in call_lines[2]
    assert "sudo " not in calls
    assert "sync" not in calls
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert "artifact_run_id=" in status
    assert "select_target=select-test" in status
    assert "train_exit=0 select_exit=0 eval_exit=0 overall_exit=0" in status
    assert "shutdown=disabled shutdown_exit=not-requested" in status
    console = next((tmp_path / "logs").glob("*.console.log")).read_text(encoding="utf-8")
    assert "shutdown=disabled" in console


def test_wrapper_requests_power_off_after_success(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path=tmp_path, train_exit=0, shutdown=True)

    assert result.returncode == 0
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert "make train-test " in calls
    assert "make select-test " in calls
    assert "make eval-test " in calls
    assert "sync" in calls
    assert "sudo -n shutdown -P now" in calls
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert "train_exit=0 select_exit=0 eval_exit=0 overall_exit=0" in status
    assert "shutdown=enabled shutdown_exit=0" in status


@pytest.mark.parametrize("shutdown", [False, True])
def test_wrapper_skips_evaluation_after_training_failure(
    *,
    tmp_path: Path,
    shutdown: bool,
) -> None:
    result = _run_wrapper(tmp_path=tmp_path, train_exit=42, shutdown=shutdown)

    assert result.returncode == 42
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert "make train-test " in calls
    assert "make select-test " not in calls
    assert "make eval-test " not in calls
    assert ("sudo -n shutdown -P now" in calls) is shutdown
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert "train_exit=42 select_exit=not-run eval_exit=not-run overall_exit=42" in status


@pytest.mark.parametrize("shutdown", [False, True])
def test_wrapper_skips_evaluation_after_selection_failure(
    *,
    tmp_path: Path,
    shutdown: bool,
) -> None:
    result = _run_wrapper(
        tmp_path=tmp_path,
        train_exit=0,
        select_exit=31,
        shutdown=shutdown,
    )

    assert result.returncode == 31
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert "make train-test " in calls
    assert "make select-test " in calls
    assert "make eval-test " not in calls
    assert ("sudo -n shutdown -P now" in calls) is shutdown
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert "train_exit=0 select_exit=31 eval_exit=not-run overall_exit=31" in status


def test_wrapper_requests_power_off_after_evaluation_failure(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path=tmp_path, train_exit=0, eval_exit=23, shutdown=True)

    assert result.returncode == 23
    calls = (tmp_path / "calls.log").read_text(encoding="utf-8")
    assert "make train-test " in calls
    assert "make eval-test " in calls
    assert "sudo -n shutdown -P now" in calls
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert "train_exit=0 select_exit=0 eval_exit=23 overall_exit=23" in status


def test_wrapper_returns_shutdown_failure(tmp_path: Path) -> None:
    result = _run_wrapper(
        tmp_path=tmp_path,
        train_exit=0,
        shutdown=True,
        shutdown_exit=17,
    )

    assert result.returncode == 17
    status = next((tmp_path / "logs").glob("*.status")).read_text(encoding="utf-8")
    assert "train_exit=0 select_exit=0 eval_exit=0 overall_exit=17" in status
    assert "shutdown=enabled shutdown_exit=17" in status


def test_wrapper_help_reports_shutdown_disabled_by_default() -> None:
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--select-target select-rl-checkpoint" in result.stdout
    assert "--shutdown     disabled" in result.stdout

from __future__ import annotations

import subprocess

from tests.support.paths import REPO_ROOT


def test_make_check_runs_complete_non_mutating_host_gate() -> None:
    result = subprocess.run(
        ["make", "--dry-run", "check"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [
        "uv lock --check",
        "uv run --extra dev ruff check",
        "uv run --extra dev ruff format --check",
        "uv run --extra dev mypy src tests",
        "uv run --extra dev vulture --config pyproject.toml",
        "uv run --extra dev pytest",
    ]
    positions = [result.stdout.index(command) for command in commands]
    assert positions == sorted(positions)


def test_pull_request_workflow_runs_make_check() -> None:
    workflow = (REPO_ROOT / ".github/workflows/host-check.yml").read_text()

    assert "pull_request:" in workflow
    assert "uv sync --extra dev --locked" in workflow
    assert "run: make check" in workflow

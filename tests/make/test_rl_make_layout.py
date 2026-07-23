"""Make ownership tests for RL, Isaac Lab, and press workflows."""

from __future__ import annotations

import re
import subprocess

from tests.support.paths import REPO_ROOT

_MOVED_TARGETS = (
    "train-rl",
    "select-rl-checkpoint",
    "eval-press",
    "test-integration-press-hover-goto",
    "test-integration-press-contact-chain",
    "test-integration-press-policy",
    "test-integration-press-routine",
)


def test_rl_targets_are_defined_only_in_rl_makefile() -> None:
    makefiles = (REPO_ROOT / "Makefile", *(REPO_ROOT / "make").glob("*.mk"))
    definitions: dict[str, list[str]] = {target: [] for target in _MOVED_TARGETS}
    for path in makefiles:
        text = path.read_text()
        for target in _MOVED_TARGETS:
            if re.search(rf"^{re.escape(target)}(?:\s*:|:)", text, flags=re.MULTILINE):
                definitions[target].append(path.name)

    assert definitions == {target: ["rl.mk"] for target in _MOVED_TARGETS}


def test_root_makefile_includes_rl_makefile_and_training_makefile_is_removed() -> None:
    root_makefile = (REPO_ROOT / "Makefile").read_text()

    assert "include make/rl.mk" in root_makefile
    assert "make/training.mk" not in root_makefile
    assert not (REPO_ROOT / "make" / "training.mk").exists()
    assert "RL, Isaac Lab, and press workflow targets:" in root_makefile


def test_integration_aggregate_retains_moved_lab_targets_and_make_resolves_them() -> None:
    integration_makefile = (REPO_ROOT / "make" / "integration-tests.mk").read_text()
    assert "test-integration-press-hover-goto" in integration_makefile
    assert "test-integration-press-contact-chain" in integration_makefile

    result = subprocess.run(
        ["make", "-n", "-o", "generate-drone-usd", "test-integration-press-hover-goto"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "run_isaac_integration_test.sh" in result.stdout

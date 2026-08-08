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
    "test-integration-press-policy-transfer",
    "test-integration-press-approach-routine",
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
    assert "include make/alignment.mk" in root_makefile
    assert "make/training.mk" not in root_makefile
    assert not (REPO_ROOT / "make" / "training.mk").exists()
    assert "RL, Isaac Lab, and press workflow targets:" in root_makefile
    assert "Controller alignment targets:" in root_makefile


def test_alignment_targets_are_defined_only_in_alignment_makefile() -> None:
    targets = (
        "alignment-capture-reference",
        "alignment-benchmark-capacity",
        "alignment-sweep-target",
        "alignment-fit",
        "align-controllers",
        "test-integration-alignment-target",
    )
    makefiles = (REPO_ROOT / "Makefile", *(REPO_ROOT / "make").glob("*.mk"))
    definitions: dict[str, list[str]] = {target: [] for target in targets}
    for path in makefiles:
        text = path.read_text()
        for target in targets:
            if re.search(rf"^{re.escape(target)}:", text, flags=re.MULTILINE):
                definitions[target].append(path.name)
    assert definitions == {target: ["alignment.mk"] for target in targets}


def test_alignment_campaign_runs_integration_before_publication() -> None:
    alignment_makefile = (REPO_ROOT / "make" / "alignment.mk").read_text()
    campaign = alignment_makefile.split("align-controllers:", maxsplit=1)[1].split(
        ".PHONY: test-integration-alignment-target", maxsplit=1
    )[0]

    integration = campaign.index("test-integration-alignment-target")
    finalization = campaign.index("--finalize")
    assert "CONTROLLER_ARTIFACT='$(ALIGNMENT_RUN_DIR)/controller.json'" in campaign
    assert integration < finalization


def test_alignment_capacity_benchmark_flag_reaches_container_runner() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "-o",
            "generate-drone-usd",
            "alignment-benchmark-capacity",
            "ALIGNMENT_RUN_ID=test-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ALIGNMENT_CAPACITY_BENCHMARK='1'" in result.stdout
    runner = (REPO_ROOT / "make/scripts/run_isaac_integration_test.sh").read_text()
    assert '-e "ALIGNMENT_CAPACITY_BENCHMARK=${ALIGNMENT_CAPACITY_BENCHMARK}"' in runner


def test_legacy_temporal_targets_and_variables_are_removed() -> None:
    text = "\n".join(
        path.read_text() for path in (REPO_ROOT / "Makefile", *(REPO_ROOT / "make").glob("*.mk"))
    )
    assert "test-integration-press-temporal" not in text
    assert "TEMPORAL_" not in text


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


def test_training_automatically_runs_checkpoint_selection_with_one_run_id() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "-o",
            "generate-drone-usd",
            "train-rl",
            "RL_RUN_ID=layout-test",
            "HYDRA_CONFIG=configs/train_press_approach_quad.yaml",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("run_isaac_integration_test.sh") == 2
    assert "ISAAC_TEST_SCRIPT='/isaac-sim/proj/isaac/runtime/standalone/train_rl.py'" in (
        result.stdout
    )
    assert (
        "ISAAC_TEST_SCRIPT=" "'/isaac-sim/proj/isaac/runtime/standalone/select_rl_checkpoint.py'"
    ) in result.stdout
    assert result.stdout.count('RL_RUN_ID="$run_id"') == 2

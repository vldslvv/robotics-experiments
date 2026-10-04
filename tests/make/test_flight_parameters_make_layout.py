"""Make ownership tests for the task-neutral flight-parameter workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.support.paths import REPO_ROOT


def test_public_campaign_routes_one_optional_flight_count() -> None:
    result = subprocess.run(
        ["make", "-n", "flight-parameters", "FLIGHT_COUNT=100"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--flight-count '100'" in result.stdout
    assert "run_flight_parameters_campaign.py" in result.stdout


def test_capture_routes_host_and_container_paths() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "-o",
            "generate-assets",
            "test-integration-flight-parameters-sample",
            "HYDRA_CONFIG=configs/flight_parameters_x500.yaml",
            "FLIGHT_PARAMETERS_RUN_ID=test-run",
            "FLIGHT_PARAMETERS_FLIGHT_INDEX=2",
            "FLIGHT_COUNT=10",
            "FLIGHT_PARAMETERS_TRIAL_DIR=.build/test-flight",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "FLIGHT_PARAMETERS_FLIGHT_INDEX='2'" in result.stdout
    assert f"FLIGHT_PARAMETERS_TRIAL_DIR='{REPO_ROOT}/.build/test-flight'" in result.stdout
    assert (
        "FLIGHT_PARAMETERS_TRIAL_DIR_CONTAINER='/isaac-sim/proj/.build/test-flight'"
        in result.stdout
    )
    assert "flight_parameters_capture.py" in result.stdout


def test_alignment_prepares_the_existing_reference_from_flight_parameters() -> None:
    makefile = (REPO_ROOT / "make/alignment.mk").read_text()

    assert "alignment-prepare-reference:" in makefile
    assert "prepare_alignment_reference.py" in makefile
    assert "mkdir -p '$(ALIGNMENT_RUN_DIR)'" not in makefile


def test_alignment_input_failure_is_clean_and_creates_no_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "alignment-run"
    result = subprocess.run(
        [
            sys.executable,
            "make/scripts/prepare_alignment_reference.py",
            "--flight-parameters",
            str(tmp_path / "missing-flight-parameters"),
            "--definition",
            "configs/alignment/controllers/px4_sitl_x500_velocity.yaml",
            "--run-dir",
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("alignment input artifact validation failed:")
    assert "Traceback" not in result.stderr
    assert not run_dir.exists()

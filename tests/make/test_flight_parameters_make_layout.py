"""Make ownership tests for the task-neutral flight-parameter workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.support.paths import REPO_ROOT


def test_root_makefile_includes_isolated_flight_parameter_targets() -> None:
    root = (REPO_ROOT / "Makefile").read_text()
    makefile = (REPO_ROOT / "make/flight-parameters.mk").read_text()
    runner = (REPO_ROOT / "make/scripts/run_px4_integration_test.sh").read_text()

    assert "include make/flight-parameters.mk" in root
    assert "measure-flight-parameters:" in makefile
    assert "test-integration-flight-parameters-sample:" in makefile
    assert "test-integration-flight-parameters-response-sample:" in makefile
    assert "FLIGHT_PARAMETERS_RESPONSE_DEFINITION_CONTAINER" in makefile
    assert (
        '-e "FLIGHT_PARAMETERS_RESPONSE_DEFINITION=${FLIGHT_PARAMETERS_RESPONSE_DEFINITION}"'
        in runner
    )
    assert (
        '-e "FLIGHT_PARAMETERS_RESPONSE_DIR_CONTAINER=${FLIGHT_PARAMETERS_RESPONSE_DIR_CONTAINER}"'
        in runner
    )
    assert "FLIGHT_PARAMETERS_OUTPUT_ROOT ?= .build/flight-parameters" in makefile


def test_hover_sample_routes_host_and_container_artifact_paths() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "-o",
            "generate-assets",
            "test-integration-flight-parameters-sample",
            "HYDRA_CONFIG=configs/flight_parameters_x500_hover.yaml",
            "FLIGHT_PARAMETERS_RUN_ID=test-run",
            "FLIGHT_PARAMETERS_TRIAL_INDEX=1",
            "FLIGHT_PARAMETERS_TRIAL_DIR=.build/test-flight-trial",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "flight_parameters_capture.py" in result.stdout
    assert f"mkdir -p '{REPO_ROOT}/.build/test-flight-trial'" in result.stdout
    assert f"FLIGHT_PARAMETERS_TRIAL_DIR='{REPO_ROOT}/.build/test-flight-trial'" in result.stdout
    assert (
        "FLIGHT_PARAMETERS_TRIAL_DIR_CONTAINER=" "'/isaac-sim/proj/.build/test-flight-trial'"
    ) in result.stdout
    assert "configs/flight_parameters_x500_hover.yaml" in result.stdout


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


def test_response_sample_routes_flight_config_and_repetition() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "-o",
            "generate-assets",
            "test-integration-flight-parameters-response-sample",
            "HYDRA_CONFIG=configs/alignment_px4_x500_velocity.yaml",
            "FLIGHT_PARAMETERS_RESPONSE_REPETITION=2",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "FLIGHT_PARAMETERS_RESPONSE_REPETITION='2'" in result.stdout
    assert (
        "FLIGHT_PARAMETERS_CONFIG='/isaac-sim/proj/configs/flight/holybro_x500_v2_hover.yaml'"
        in result.stdout
    )
    runner = (REPO_ROOT / "make/scripts/run_px4_integration_test.sh").read_text()
    assert (
        '-e "FLIGHT_PARAMETERS_RESPONSE_REPETITION=${FLIGHT_PARAMETERS_RESPONSE_REPETITION}"'
        in runner
    )

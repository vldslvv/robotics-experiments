"""Make ownership tests for the task-neutral flight-parameter workflow."""

from __future__ import annotations

import subprocess

from tests.support.paths import REPO_ROOT


def test_root_makefile_includes_isolated_flight_parameter_targets() -> None:
    root = (REPO_ROOT / "Makefile").read_text()
    makefile = (REPO_ROOT / "make/flight-parameters.mk").read_text()

    assert "include make/flight-parameters.mk" in root
    assert "measure-flight-envelope:" in makefile
    assert "test-integration-flight-parameters-sample:" in makefile


def test_hover_sample_routes_host_and_container_artifact_paths() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "-o",
            "generate-drone-usd",
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

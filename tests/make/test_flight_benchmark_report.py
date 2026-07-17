from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTER = REPO_ROOT / "make/scripts/flight_benchmark_report.py"


def _write_json(path: Path, payload: Any, /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_repeat(
    *,
    root: Path,
    workload: str,
    repetition: int,
    sample_ms: float,
    gpu_model: str = "Test GPU",
) -> None:
    output = root / f"{workload}-{repetition}" / "benchmark"
    metadata = {
        "methodology_version": 1,
        "workload": workload,
        "scene": "test.usda",
        "physics_hz": 250.0,
        "rendering_hz": 60.0,
        "solver_type": "TGS",
        "vehicle_count": 1,
        "vehicle_assets": ["drone.usd"],
        "vehicle_ids": [0],
        "target_physics_ticks": 5_000 if workload == "physics_only" else None,
    }
    environment = {
        "cpu_model": "Test CPU",
        "cpu_governor": "performance",
        "logical_cpus": 16,
        "gpu_models": [gpu_model],
        "isaac_version": "5.1.0",
        "isaac_image": "test:5.1.0",
        "kernel": "test",
    }
    _write_json(
        output / "step_intervals.json",
        {
            "schema_version": 1,
            "metadata": metadata,
            "environment": environment,
            "physics_step_count": 5_000,
            "sim_elapsed_s": 20.0,
            "wall_elapsed_s": 20.0,
            "real_time_factor": 1.0,
            "physics_step_interval_samples_ms": [sample_ms] * 100,
        },
    )
    _write_json(
        output / f"metrics_{workload}.json",
        [
            {
                "phase_name": "measured",
                "measurements": [
                    {
                        "name": f"{workload} measured App_Update Frametime Samples",
                        "value": [sample_ms] * 100,
                        "type": "list",
                    },
                    {
                        "name": f"{workload} measured Physics Frametime Samples",
                        "value": [sample_ms / 2.0] * 100,
                        "type": "list",
                    },
                ],
                "metadata": [],
            }
        ],
    )


def _write_raw(*, root: Path, sample_ms: float, gpu_model: str = "Test GPU") -> None:
    for workload in ("physics_only", "px4_flight"):
        for repetition in range(1, 4):
            _write_repeat(
                root=root,
                workload=workload,
                repetition=repetition,
                sample_ms=sample_ms,
                gpu_model=gpu_model,
            )


def _run_build(raw: Path, result: Path, baseline: Path, /) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "build",
            "--raw-root",
            str(raw),
            "--run-root",
            str(result),
            "--baseline",
            str(baseline),
            "--isaac-image",
            "test:5.1.0",
            "--isaac-image-id",
            "sha256:test",
            "--px4-ref",
            "test-ref",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _accept(result: Path, baseline: Path, /) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "accept",
            "--result",
            str(result / "report.json"),
            "--baseline",
            str(baseline),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_accept_and_pass(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    raw = tmp_path / "raw-initial"
    result = tmp_path / "result-initial"
    _write_raw(root=raw, sample_ms=4.0)

    first = _run_build(raw, result, baseline)
    assert first.returncode == 2
    assert json.loads((result / "report.json").read_text())["status"] == "no_baseline"
    assert _accept(result, baseline).returncode == 0

    next_raw = tmp_path / "raw-next"
    next_result = tmp_path / "result-next"
    _write_raw(root=next_raw, sample_ms=4.1)
    second = _run_build(next_raw, next_result, baseline)
    assert second.returncode == 0
    assert json.loads((next_result / "report.json").read_text())["status"] == "pass"


def test_regression_fails_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    raw = tmp_path / "raw-initial"
    result = tmp_path / "result-initial"
    _write_raw(root=raw, sample_ms=4.0)
    assert _run_build(raw, result, baseline).returncode == 2
    assert _accept(result, baseline).returncode == 0

    regressed_raw = tmp_path / "raw-regressed"
    regressed_result = tmp_path / "result-regressed"
    _write_raw(root=regressed_raw, sample_ms=5.0)
    regressed = _run_build(regressed_raw, regressed_result, baseline)
    assert regressed.returncode == 1
    report = json.loads((regressed_result / "report.json").read_text())
    assert report["status"] == "fail"
    assert any(not item["passed"] for item in report["comparisons"])


def test_hardware_mismatch_is_incomparable(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    raw = tmp_path / "raw-initial"
    result = tmp_path / "result-initial"
    _write_raw(root=raw, sample_ms=4.0)
    assert _run_build(raw, result, baseline).returncode == 2
    assert _accept(result, baseline).returncode == 0

    changed_raw = tmp_path / "raw-changed"
    changed_result = tmp_path / "result-changed"
    _write_raw(root=changed_raw, sample_ms=4.0, gpu_model="Different GPU")
    changed = _run_build(changed_raw, changed_result, baseline)
    assert changed.returncode == 2
    report = json.loads((changed_result / "report.json").read_text())
    assert report["status"] == "incomparable"

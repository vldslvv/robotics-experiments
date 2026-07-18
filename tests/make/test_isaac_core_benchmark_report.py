from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTER = REPO_ROOT / "make/scripts/isaac_core_benchmark_report.py"
WORKLOADS = (
    "kit_only_unlimited",
    "render_only_unlimited",
    "scene_physics_unlimited",
    "drone_physics_unlimited",
    "drone_articulation_unlimited",
    "drone_articulation_realtime",
)


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
    output = root / f"{workload}-{repetition}" / "benchmark" / workload
    physics_expected = workload not in {"kit_only_unlimited", "render_only_unlimited"}
    articulation = workload.startswith("drone_articulation_")
    metadata = {
        "methodology_version": 3,
        "workload": workload,
        "scene": "test.usda",
        "physics_hz": 250.0,
        "rendering_hz": 60.0,
        "solver_type": "TGS",
        "vehicle_count": 1,
        "vehicle_assets": ["drone.usd"],
        "vehicle_ids": [0],
        "target_app_updates": None if physics_expected else 600,
        "target_physics_ticks": 5_000 if physics_expected else None,
        "rate_limited": workload == "drone_articulation_realtime",
        "renderer": "RaytracedLighting",
        "render_resolution": [1280, 720],
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
    sample_groups = {
        "physics_step_interval": [sample_ms] * 100 if physics_expected else [],
        "app_update_interval": [sample_ms] * 100,
        "render_update_interval": [],
    }
    if articulation:
        sample_groups.update(
            {
                "articulation_state_read": [sample_ms / 10.0] * 100,
                "rotor_visual_update": [sample_ms / 20.0] * 100,
                "dynamics_apply": [sample_ms / 5.0] * 100,
            }
        )
    _write_json(
        output / "samples.json",
        {
            "schema_version": 2,
            "metadata": metadata,
            "environment": environment,
            "app_update_count": 600,
            "physics_step_count": 5_000 if physics_expected else 0,
            "sim_elapsed_s": 20.0 if physics_expected else 0.0,
            "wall_elapsed_s": 20.0,
            "real_time_factor": 1.0 if physics_expected else 0.0,
            "samples_ms": sample_groups,
        },
    )
    measurements = [
        {
            "name": f"{workload} measured App_Update Frametime Samples",
            "value": [sample_ms] * 100,
            "type": "list",
        },
        {
            "name": f"{workload} measured GPU Frametime Samples",
            "value": [sample_ms / 2.0] * 100,
            "type": "list",
        },
    ]
    if physics_expected:
        measurements.append(
            {
                "name": f"{workload} measured Physics Frametime Samples",
                "value": [sample_ms / 2.0] * 100,
                "type": "list",
            }
        )
    _write_json(
        output / f"metrics_{workload}.json",
        [{"phase_name": "measured", "measurements": measurements, "metadata": []}],
    )


def _write_raw(*, root: Path, sample_ms: float, gpu_model: str = "Test GPU") -> None:
    for workload in WORKLOADS:
        for repetition in range(1, 4):
            _write_repeat(
                root=root,
                workload=workload,
                repetition=repetition,
                sample_ms=sample_ms,
                gpu_model=gpu_model,
            )


def _write_profile(root: Path, /) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "isaac_core.tracy").write_bytes(b"test trace")


def _run_build(
    raw: Path,
    profile: Path,
    result: Path,
    baseline: Path,
    /,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "build",
            "--raw-root",
            str(raw),
            "--profile-root",
            str(profile),
            "--run-root",
            str(result),
            "--baseline",
            str(baseline),
            "--isaac-image",
            "test:5.1.0",
            "--isaac-image-id",
            "sha256:test",
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
    profile = tmp_path / "profile"
    _write_profile(profile)
    raw = tmp_path / "raw-initial"
    result = tmp_path / "result-initial"
    _write_raw(root=raw, sample_ms=4.0)

    first = _run_build(raw, profile, result, baseline)
    assert first.returncode == 2
    report = json.loads((result / "report.json").read_text())
    assert report["status"] == "no_baseline"
    assert report["profile_trace"].endswith("isaac_core.tracy")
    assert "articulation_io" in report["attribution_deltas"]
    assert _accept(result, baseline).returncode == 0

    next_raw = tmp_path / "raw-next"
    next_result = tmp_path / "result-next"
    _write_raw(root=next_raw, sample_ms=4.1)
    second = _run_build(next_raw, profile, next_result, baseline)
    assert second.returncode == 0
    assert json.loads((next_result / "report.json").read_text())["status"] == "pass"


def test_regression_fails_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    profile = tmp_path / "profile"
    _write_profile(profile)
    raw = tmp_path / "raw-initial"
    result = tmp_path / "result-initial"
    _write_raw(root=raw, sample_ms=4.0)
    assert _run_build(raw, profile, result, baseline).returncode == 2
    assert _accept(result, baseline).returncode == 0

    regressed_raw = tmp_path / "raw-regressed"
    regressed_result = tmp_path / "result-regressed"
    _write_raw(root=regressed_raw, sample_ms=5.0)
    regressed = _run_build(regressed_raw, profile, regressed_result, baseline)
    assert regressed.returncode == 1
    report = json.loads((regressed_result / "report.json").read_text())
    assert report["status"] == "fail"
    assert any(not item["passed"] for item in report["comparisons"])


def test_hardware_mismatch_is_incomparable(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    profile = tmp_path / "profile"
    _write_profile(profile)
    raw = tmp_path / "raw-initial"
    result = tmp_path / "result-initial"
    _write_raw(root=raw, sample_ms=4.0)
    assert _run_build(raw, profile, result, baseline).returncode == 2
    assert _accept(result, baseline).returncode == 0

    changed_raw = tmp_path / "raw-changed"
    changed_result = tmp_path / "result-changed"
    _write_raw(root=changed_raw, sample_ms=4.0, gpu_model="Different GPU")
    changed = _run_build(changed_raw, profile, changed_result, baseline)
    assert changed.returncode == 2
    report = json.loads((changed_result / "report.json").read_text())
    assert report["status"] == "incomparable"


def test_missing_profile_trace_is_invalid(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_raw(root=raw, sample_ms=4.0)
    result = tmp_path / "result"
    completed = _run_build(raw, tmp_path / "empty-profile", result, tmp_path / "baseline")
    assert completed.returncode == 2
    report = json.loads((result / "report.json").read_text())
    assert report["status"] == "invalid"
    assert "exactly one .tracy" in report["reason"]


def test_accept_requires_preserved_profile_trace(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    profile = tmp_path / "profile"
    result = tmp_path / "result"
    _write_raw(root=raw, sample_ms=4.0)
    _write_profile(profile)
    assert _run_build(raw, profile, result, tmp_path / "baseline").returncode == 2

    (result / "profile" / "isaac_core.tracy").unlink()
    completed = _accept(result, tmp_path / "baseline")
    assert completed.returncode != 0
    assert "profiler trace is missing or empty" in completed.stderr


def test_incomplete_sample_target_is_invalid(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    profile = tmp_path / "profile"
    result = tmp_path / "result"
    _write_raw(root=raw, sample_ms=4.0)
    _write_profile(profile)
    sample_path = next(raw.rglob("kit_only_unlimited/samples.json"))
    payload = json.loads(sample_path.read_text())
    payload["app_update_count"] = 599
    _write_json(sample_path, payload)

    completed = _run_build(raw, profile, result, tmp_path / "baseline")
    assert completed.returncode == 2
    report = json.loads((result / "report.json").read_text())
    assert report["status"] == "invalid"
    assert "expected at least 600" in report["reason"]

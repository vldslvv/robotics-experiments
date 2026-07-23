from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.support.paths import REPO_ROOT

REPORTER = REPO_ROOT / "make/scripts/flight_benchmark_report.py"
PHASES = ("connected_disarmed", "armed_takeoff", "airborne")
_REPORT_ENV = os.environ.copy()
_REPORT_ENV["PYTHONPATH"] = str(REPO_ROOT / "src")


def _write_json(path: Path, payload: Any, /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_repeat(*, root: Path, repetition: int, frame_ms: float) -> None:
    phases = {}
    component_samples = []
    resource_samples = []
    sim_start = 1.0
    for phase in PHASES:
        sim_duration = 5.0 if phase == "airborne" else 1.0
        frames = []
        for frame in range(1, 11):
            frames.append(
                {
                    "frame": frame,
                    "interval_ms": frame_ms,
                    "physics_steps": 4,
                    "sim_elapsed_s": 0.016,
                }
            )
            component_samples.extend(
                [
                    {
                        "phase": phase,
                        "frame": frame,
                        "metric": "bridge_receive",
                        "entity": "fleet",
                        "duration_ms": frame_ms / 2.0,
                    },
                    {
                        "phase": phase,
                        "frame": frame,
                        "metric": "bridge_send",
                        "entity": "vehicle-0",
                        "duration_ms": frame_ms / 8.0,
                    },
                ]
            )
            resource_samples.append(
                {
                    "phase": phase,
                    "frame": frame,
                    "cpu_percent": 150.0,
                    "rss_mb": 2048.0,
                }
            )
        phases[phase] = {
            "start_sim_time_s": sim_start,
            "end_sim_time_s": sim_start + sim_duration,
            "wall_elapsed_s": sim_duration * 2.0,
            "app_frames": frames,
            "render_intervals_ms": [frame_ms] * 10,
        }
        sim_start += sim_duration
    environment = {
        "cpu_model": "Test CPU",
        "cpu_governor": "performance",
        "logical_cpus": 16,
        "gpu_models": ["Test GPU"],
        "isaac_version": "5.1.0",
        "isaac_image": "test:5.1.0",
        "isaac_image_id": "sha256:isaac",
        "px4_image": "px4:test",
        "px4_image_id": "sha256:px4",
        "source_commit": "abc",
        "kernel": "test",
    }
    metadata = {
        "methodology_version": 1,
        "run_kind": "measured",
        "repetition": repetition,
        "scene": "@isaac/assets/LightSwitchWarehouse/Warehouse.usd",
        "config": "/isaac-sim/proj/configs/circle_land_quad_quad.yaml",
        "physics_hz": 250.0,
        "rendering_hz": 60.0,
        "solver_type": "TGS",
        "renderer": "RaytracedLighting",
        "render_resolution": [1280, 720],
        "vehicle_count": 2,
        "vehicle_labels": ["x500_01", "x500_02"],
        "vehicle_ids": [0, 1],
        "airborne_duration_s": 5.0,
        "measurement_level": "full",
        "config_sha256": "config",
    }
    _write_json(
        root / "samples.json",
        {
            "schema_version": 1,
            "status": "complete",
            "reason": None,
            "metadata": metadata,
            "environment": environment,
            "phases": phases,
            "component_samples": component_samples,
            "resource_samples": resource_samples,
        },
    )
    measurements = []
    for phase in PHASES:
        measurements.append(
            {
                "phase_name": phase,
                "measurements": [
                    {
                        "name": "flight App_Update Frametime Samples",
                        "value": [frame_ms] * 10,
                    },
                    {
                        "name": "flight GPU Frametime Samples",
                        "value": [frame_ms / 2.0] * 10,
                    },
                    {
                        "name": "flight Physics Frametime Samples",
                        "value": [frame_ms / 3.0] * 10,
                    },
                ],
            }
        )
    _write_json(root / "nvidia/metrics_flight.json", measurements)
    stats = [
        {"Name": "test-isaac", "CPUPerc": "200.0%", "MemUsage": "2GiB / 32GiB"},
        {"Name": "test-px4-0", "CPUPerc": "25.0%", "MemUsage": "100MiB / 32GiB"},
        {"Name": "test-px4-1", "CPUPerc": "24.0%", "MemUsage": "101MiB / 32GiB"},
    ]
    (root / "container-stats.jsonl").write_text(
        "\n".join(json.dumps(sample) for sample in stats) + "\n",
        encoding="utf-8",
    )


def _build(*, root: Path, frame_ms: float) -> subprocess.CompletedProcess[str]:
    for repetition in range(1, 4):
        _write_repeat(
            root=root / f"raw/repeat-{repetition}",
            repetition=repetition,
            frame_ms=frame_ms,
        )
    (root / "profile").mkdir(parents=True)
    (root / "profile/flight.tracy").write_bytes(b"trace")
    _write_json(
        root / "fingerprint.json",
        {
            "config_sha256": "config",
            "vehicle_labels": ["x500_01", "x500_02"],
            "px4_revision": "px4-ref",
            "source_commit": "abc",
            "isaac_image": "test:5.1.0",
            "isaac_image_id": "sha256:isaac",
            "px4_image": "px4:test",
            "px4_image_id": "sha256:px4",
        },
    )
    return subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "build",
            "--raw-root",
            str(root / "raw"),
            "--profile-root",
            str(root / "profile"),
            "--run-root",
            str(root),
            "--fingerprint",
            str(root / "fingerprint.json"),
        ],
        env=_REPORT_ENV,
        check=False,
        capture_output=True,
        text=True,
    )


def _fingerprint(*, config: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "fingerprint",
            "--config",
            str(config),
            "--output",
            str(output),
            "--isaac-image",
            "isaac:test",
            "--isaac-image-id",
            "sha256:isaac",
            "--px4-image",
            "px4:test",
            "--px4-image-id",
            "sha256:px4",
            "--px4-revision",
            "px4-ref",
            "--source-commit",
            "abc",
        ],
        env=_REPORT_ENV,
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_build_validates_and_summarizes_full_flight(tmp_path: Path) -> None:
    root = tmp_path / "flight"
    completed = _build(root=root, frame_ms=20.0)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["diagnostic_only"] is True
    assert "px4_image_id" not in report["contract"]
    assert report["environment"]["px4_image_id"] == "sha256:px4"
    airborne = report["metrics"]["phases"]["airborne"]
    assert airborne["app_frame_ms"]["max"] == 20.0
    assert airborne["frame_budget_violation_percent"] == 100.0
    assert airborne["real_time_factor"] == 0.5
    assert report["metrics"]["container_resources"]["px4_fleet"]["cpu_percent"]["p95"] > 0


def test_incomplete_phase_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "flight"
    assert _build(root=root, frame_ms=10.0).returncode == 0
    sample = root / "raw/repeat-1/samples.json"
    payload = json.loads(sample.read_text())
    del payload["phases"]["airborne"]
    _write_json(sample, payload)
    (root / "report.json").unlink()
    completed = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "build",
            "--raw-root",
            str(root / "raw"),
            "--profile-root",
            str(root / "profile"),
            "--run-root",
            str(root),
            "--fingerprint",
            str(root / "fingerprint.json"),
        ],
        env=_REPORT_ENV,
        check=False,
    )
    assert completed.returncode == 2
    assert json.loads((root / "report.json").read_text())["status"] == "invalid"


def test_build_accepts_lifecycle_overhead_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "flight"
    for repetition in range(1, 4):
        repeat_root = root / f"raw/repeat-{repetition}"
        _write_repeat(root=repeat_root, repetition=repetition, frame_ms=20.0)
        sample = repeat_root / "samples.json"
        payload = json.loads(sample.read_text())
        payload["metadata"]["measurement_level"] = "lifecycle"
        payload["component_samples"] = []
        payload["resource_samples"] = []
        for phase in payload["phases"].values():
            phase["app_frames"] = []
            phase["render_intervals_ms"] = []
        _write_json(sample, payload)
        for metrics in (repeat_root / "nvidia").glob("*.json"):
            metrics.unlink()
    (root / "profile").mkdir(parents=True)
    (root / "profile/flight.tracy").write_bytes(b"trace")
    _write_json(
        root / "fingerprint.json",
        {
            "config_sha256": "config",
            "vehicle_labels": ["x500_01", "x500_02"],
            "px4_revision": "px4-ref",
            "source_commit": "abc",
            "isaac_image": "test:5.1.0",
            "isaac_image_id": "sha256:isaac",
            "px4_image": "px4:test",
            "px4_image_id": "sha256:px4",
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "build",
            "--raw-root",
            str(root / "raw"),
            "--profile-root",
            str(root / "profile"),
            "--run-root",
            str(root),
            "--fingerprint",
            str(root / "fingerprint.json"),
        ],
        env=_REPORT_ENV,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((root / "report.json").read_text())
    assert report["contract"]["measurement_level"] == "lifecycle"
    assert "nvidia_ms" not in report["metrics"]["phases"]["airborne"]


def test_build_rejects_repetition_environment_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "flight"
    assert _build(root=root, frame_ms=10.0).returncode == 0
    sample = root / "raw/repeat-2/samples.json"
    payload = json.loads(sample.read_text())
    payload["environment"]["cpu_governor"] = "powersave"
    _write_json(sample, payload)
    (root / "report.json").unlink()

    completed = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "build",
            "--raw-root",
            str(root / "raw"),
            "--profile-root",
            str(root / "profile"),
            "--run-root",
            str(root),
            "--fingerprint",
            str(root / "fingerprint.json"),
        ],
        env=_REPORT_ENV,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "different environments" in json.loads((root / "report.json").read_text())["reason"]


def test_compare_writes_cross_version_json_and_markdown(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    assert _build(root=reference, frame_ms=10.0).returncode == 0
    assert _build(root=candidate, frame_ms=20.0).returncode == 0
    output = tmp_path / "comparison"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "compare",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output-root",
            str(output),
        ],
        env=_REPORT_ENV,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    comparison = json.loads(next(output.glob("*.json")).read_text())
    assert comparison["status"] == "complete"
    assert comparison["airborne_component_p95"][0]["metric"].endswith("bridge_receive.all.p95")
    assert comparison["airborne_deltas_ms"]["app"] == 10.0
    assert next(output.glob("*.md")).is_file()


def test_fingerprint_hashes_resolved_config_and_preserves_vehicle_order(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    shutil.copytree(REPO_ROOT / "configs", config_root)
    config = config_root / "flight_benchmark.yaml"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    completed = _fingerprint(config=config, output=first)
    assert completed.returncode == 0, completed.stderr
    assert _fingerprint(config=config, output=second).returncode == 0
    first_payload = json.loads(first.read_text())
    assert first_payload["config_sha256"] == json.loads(second.read_text())["config_sha256"]
    assert first_payload["vehicle_labels"] == ["x500_01", "x500_02"]

    inherited = config_root / "experiment/sim/default.yaml"
    inherited.write_text(
        inherited.read_text().replace("physics_hz: 250.00", "physics_hz: 240.00"),
        encoding="utf-8",
    )
    assert _fingerprint(config=config, output=second).returncode == 0
    assert first_payload["config_sha256"] != json.loads(second.read_text())["config_sha256"]


def test_compare_represents_zero_baseline_as_absolute_delta(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    assert _build(root=reference, frame_ms=10.0).returncode == 0
    assert _build(root=candidate, frame_ms=20.0).returncode == 0
    reference_report = reference / "report.json"
    payload = json.loads(reference_report.read_text())
    payload["metrics"]["phases"]["airborne"]["app_frame_ms"]["p50"] = 0.0
    _write_json(reference_report, payload)

    output = tmp_path / "comparison"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "compare",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output-root",
            str(output),
        ],
        env=_REPORT_ENV,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    comparison = json.loads(next(output.glob("*.json")).read_text())
    metric = next(
        item
        for item in comparison["metrics"]
        if item["metric"] == "phases.airborne.app_frame_ms.p50"
    )
    assert metric["delta"] == 20.0
    assert metric["percent_change"] is None
    assert metric["ratio"] is None


def test_compare_rejects_different_numeric_metric_trees(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    assert _build(root=reference, frame_ms=10.0).returncode == 0
    assert _build(root=candidate, frame_ms=10.0).returncode == 0
    candidate_report = candidate / "report.json"
    payload = json.loads(candidate_report.read_text())
    del payload["metrics"]["phases"]["airborne"]["app_frame_ms"]["max"]
    _write_json(candidate_report, payload)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPORTER),
            "compare",
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output-root",
            str(tmp_path / "comparison"),
        ],
        env=_REPORT_ENV,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "different numeric metric trees" in completed.stderr

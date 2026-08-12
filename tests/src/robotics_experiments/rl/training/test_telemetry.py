"""Tests for policy-training telemetry sampling and evidence."""

from __future__ import annotations

import subprocess
import time
from types import SimpleNamespace

import pytest

from robotics_experiments.rl.training import _telemetry
from robotics_experiments.rl.training._telemetry import TelemetrySampler, probe_telemetry


def _gpu_result() -> SimpleNamespace:
    return SimpleNamespace(stdout="50, 2048, 8192, 100, 60\n70, 1024, 8192, 80, 55\n")


def test_telemetry_aggregates_fractional_gpu_and_process_metrics(monkeypatch) -> None:
    monkeypatch.setattr(_telemetry.subprocess, "run", lambda *_args, **_kwargs: _gpu_result())
    sampler = TelemetrySampler(sample_interval_s=10.0)
    sampler.start(role="train")
    interval = sampler.finish(work_items=320)
    assert interval["status"] == "complete"
    assert len(interval["samples"]) == 2
    summary = interval["summary"]
    assert summary["gpu_utilization_rate_mean"] == pytest.approx(0.6)
    assert summary["gpu_utilization_rate_p95"] == pytest.approx(0.6)
    assert summary["gpu_memory_used_gib_peak"] == pytest.approx(3.0)
    assert summary["gpu_memory_utilization_rate_peak"] == pytest.approx(3.0 / 16.0)
    assert summary["gpu_power_w_mean"] == pytest.approx(180.0)
    assert summary["gpu_temperature_c_max"] == 60.0
    assert summary["work_items_per_s"] > 0.0


def test_sampler_failure_is_reported_with_raw_evidence(monkeypatch) -> None:
    calls = 0

    def run(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise subprocess.CalledProcessError(1, "nvidia-smi")
        return _gpu_result()

    monkeypatch.setattr(_telemetry.subprocess, "run", run)
    sampler = TelemetrySampler(sample_interval_s=0.001)
    sampler.start(role="val")
    time.sleep(0.01)
    interval = sampler.finish(work_items=300)
    assert interval["status"] == "failed"
    assert interval["samples"]
    assert "nvidia-smi" in interval["error"]
    assert interval["summary"] == {}


def test_preflight_probe_fails_before_rollout_when_nvidia_smi_is_unavailable(
    monkeypatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(_telemetry.subprocess, "run", fail)
    with pytest.raises(FileNotFoundError, match="nvidia-smi"):
        probe_telemetry()

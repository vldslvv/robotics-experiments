"""Diagnostic GPU and process telemetry for policy-training intervals."""

from __future__ import annotations

import math
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["TelemetrySampler", "probe_telemetry"]

_GIB = 1024.0**3
_NVIDIA_QUERY = (
    "nvidia-smi",
    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
    "--format=csv,noheader,nounits",
)


def _gpu_sample() -> dict[str, float]:
    result = subprocess.run(
        _NVIDIA_QUERY,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        tuple(float(value.strip()) for value in line.split(","))
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    if not rows or any(len(row) != 5 for row in rows):
        raise RuntimeError("nvidia-smi returned no complete GPU telemetry rows")
    memory_used_mib = sum(row[1] for row in rows)
    memory_total_mib = sum(row[2] for row in rows)
    if memory_total_mib <= 0.0:
        raise RuntimeError("nvidia-smi returned non-positive total GPU memory")
    return {
        "gpu_utilization_rate": statistics.fmean(row[0] for row in rows) / 100.0,
        "gpu_memory_used_gib": memory_used_mib / 1024.0,
        "gpu_memory_utilization_rate": memory_used_mib / memory_total_mib,
        "gpu_power_w": sum(row[3] for row in rows),
        "gpu_temperature_c": max(row[4] for row in rows),
    }


def _process_sample(*, process_id: int) -> tuple[float, float]:
    stat = Path(f"/proc/{process_id}/stat").read_text(encoding="utf-8")
    fields = stat.rsplit(")", maxsplit=1)[1].split()
    cpu_time_s = (float(fields[11]) + float(fields[12])) / os.sysconf("SC_CLK_TCK")
    resident_pages = int(Path(f"/proc/{process_id}/statm").read_text(encoding="utf-8").split()[1])
    resident_gib = resident_pages * os.sysconf("SC_PAGE_SIZE") / _GIB
    return cpu_time_s, resident_gib


def probe_telemetry() -> dict[str, float]:
    """Require usable GPU and process telemetry before campaign rollout."""
    gpu = _gpu_sample()
    _cpu_time_s, resident_gib = _process_sample(process_id=os.getpid())
    return gpu | {"process_resident_memory_gib": resident_gib}


@dataclass(frozen=True)
class _IntervalState:
    role: str
    started_at: float
    samples: tuple[dict[str, float], ...]
    error: str


class TelemetrySampler:
    """Sample one process and all visible NVIDIA GPUs during one interval."""

    def __init__(self, *, sample_interval_s: float, process_id: int | None = None) -> None:
        if not math.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
            raise ValueError("telemetry sample interval must be finite and positive")
        self._sample_interval_s = sample_interval_s
        self._process_id = process_id if process_id is not None else os.getpid()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._role = ""
        self._started_at = math.nan
        self._samples: list[dict[str, float]] = []
        self._error = ""
        self._last_wall_time = math.nan
        self._last_cpu_time = math.nan

    def _sample(self) -> None:
        sampled_at = time.monotonic()
        gpu = _gpu_sample()
        cpu_time, resident_gib = _process_sample(process_id=self._process_id)
        if math.isnan(self._last_wall_time):
            cpu_cores = 0.0
        else:
            elapsed = sampled_at - self._last_wall_time
            cpu_cores = (cpu_time - self._last_cpu_time) / elapsed
        self._last_wall_time = sampled_at
        self._last_cpu_time = cpu_time
        sample = {
            "elapsed_s": sampled_at - self._started_at,
            **gpu,
            "process_resident_memory_gib": resident_gib,
            "process_cpu_cores_used": max(0.0, cpu_cores),
        }
        with self._lock:
            self._samples.append(sample)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._sample_interval_s):
            try:
                self._sample()
            except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
                with self._lock:
                    self._error = f"telemetry sampling failed: {exc}"
                self._stop.set()
                return

    def start(self, *, role: str) -> None:
        """Start one train, validation, or test interval after a synchronous probe."""
        if role not in {"train", "val", "test"}:
            raise ValueError("telemetry role must be train, val, or test")
        if self._thread is not None:
            raise RuntimeError("a telemetry interval is already active")
        self._role = role
        self._started_at = time.monotonic()
        self._samples = []
        self._error = ""
        self._last_wall_time = math.nan
        self._last_cpu_time = math.nan
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _state(self) -> _IntervalState:
        with self._lock:
            return _IntervalState(
                role=self._role,
                started_at=self._started_at,
                samples=tuple(dict(sample) for sample in self._samples),
                error=self._error,
            )

    def finish(self, *, work_items: int) -> dict[str, Any]:
        """Stop an interval, retain evidence, and summarize its samples."""
        if self._thread is None:
            raise RuntimeError("no telemetry interval is active")
        self._stop.set()
        self._thread.join()
        self._thread = None
        state = self._state()
        if not state.error:
            try:
                self._sample()
            except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
                with self._lock:
                    self._error = f"telemetry sampling failed: {exc}"
                state = self._state()
            else:
                state = self._state()
        duration_s = time.monotonic() - state.started_at
        payload: dict[str, Any] = {
            "role": state.role,
            "status": "failed" if state.error else "complete",
            "error": state.error,
            "sample_interval_s": self._sample_interval_s,
            "work_items": work_items,
            "samples": list(state.samples),
            "summary": {},
        }
        if state.samples and not state.error:
            utilization = sorted(sample["gpu_utilization_rate"] for sample in state.samples)
            p95_index = max(0, math.ceil(0.95 * len(utilization)) - 1)
            payload["summary"] = {
                "gpu_utilization_rate_mean": statistics.fmean(utilization),
                "gpu_utilization_rate_p95": utilization[p95_index],
                "gpu_memory_used_gib_peak": max(
                    sample["gpu_memory_used_gib"] for sample in state.samples
                ),
                "gpu_memory_utilization_rate_peak": max(
                    sample["gpu_memory_utilization_rate"] for sample in state.samples
                ),
                "gpu_power_w_mean": statistics.fmean(
                    sample["gpu_power_w"] for sample in state.samples
                ),
                "gpu_temperature_c_max": max(
                    sample["gpu_temperature_c"] for sample in state.samples
                ),
                "process_resident_memory_gib_peak": max(
                    sample["process_resident_memory_gib"] for sample in state.samples
                ),
                "process_cpu_cores_used_mean": statistics.fmean(
                    sample["process_cpu_cores_used"] for sample in state.samples
                ),
                "duration_s": duration_s,
                "work_items_per_s": work_items / duration_s,
            }
        return payload

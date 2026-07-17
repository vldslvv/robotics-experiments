"""Standalone-only performance capture for Isaac integration workloads."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

__all__ = ["StandaloneBenchmarkCapture", "advance_physics_ticks"]

_METHODOLOGY_VERSION = 1


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    raise RuntimeError("benchmark could not determine the CPU model")


def _cpu_governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if not path.exists():
        raise RuntimeError("benchmark could not determine the CPU governor")
    return path.read_text(encoding="utf-8").strip()


def _isaac_version() -> str:
    from isaacsim.core.version import get_version

    version = get_version()
    if isinstance(version, (tuple, list)):
        return ".".join(str(part) for part in version[:3])
    return str(version)


def _gpu_models() -> list[str]:
    import torch

    models = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not models:
        raise RuntimeError("benchmark could not determine any CUDA GPU models")
    return models


def _portable_path(value: str, /) -> str:
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[4]
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _benchmark_metadata(*, context: Any, workload: str, target_physics_ticks: int | None) -> dict:
    vehicles = list(context.config.vehicles.items())
    return {
        "methodology_version": _METHODOLOGY_VERSION,
        "workload": workload,
        "scene": context.scene_override or context.config.scene,
        "physics_hz": float(context.config.sim.physics_hz),
        "rendering_hz": float(context.config.sim.rendering_hz),
        "solver_type": str(context.config.sim.solver_type),
        "vehicle_count": len(vehicles),
        "vehicle_assets": [
            _portable_path(str(vehicle.spawn.usd_path)) for _label, vehicle in vehicles
        ],
        "vehicle_ids": [int(vehicle.id) for _label, vehicle in vehicles],
        "target_physics_ticks": target_physics_ticks,
    }


def advance_physics_ticks(*, simulation_app: Any, minimum_ticks: int) -> int:
    """Advance until at least ``minimum_ticks`` PhysX callbacks have fired."""
    import omni.physx

    tick_count = 0

    def on_step(_dt: float) -> None:
        nonlocal tick_count
        tick_count += 1

    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(on_step)
    try:
        while tick_count < minimum_ticks:
            simulation_app.update()
    finally:
        del subscription
    return tick_count


class StandaloneBenchmarkCapture:
    """Capture one measured phase without changing production runtime state."""

    def __init__(
        self,
        *,
        context: Any,
        workload: str,
        target_physics_ticks: int | None,
    ) -> None:
        if context.benchmark_output_dir is None:
            raise ValueError("benchmark capture requires a benchmark output directory")

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.benchmark.services")
        from isaacsim.benchmark.services import BaseIsaacBenchmark

        self._output_dir = Path(context.benchmark_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = _benchmark_metadata(
            context=context,
            workload=workload,
            target_physics_ticks=target_physics_ticks,
        )

        import carb

        carb.settings.get_settings().set(
            "/exts/isaacsim.benchmark.services/metrics/metrics_output_folder",
            str(self._output_dir),
        )
        self._benchmark = BaseIsaacBenchmark(
            benchmark_name=workload,
            backend_type="JSONFileMetrics",
            report_generation=False,
            workflow_metadata={
                "metadata": [
                    {"name": key, "data": value}
                    for key, value in self._metadata.items()
                    if isinstance(value, (str, int, float)) and value is not None
                ]
            },
        )
        self._subscription = None
        self._started = False
        self._stopped = False
        self._last_step_ns = 0
        self._start_ns = 0
        self._step_count = 0
        self._sim_elapsed_s = 0.0
        self._step_intervals_ms: list[float] = []

    @property
    def started(self) -> bool:
        """Return whether capture has started."""
        return self._started

    @property
    def physics_step_count(self) -> int:
        """Return the number of physics callbacks observed in the phase."""
        return self._step_count

    def _on_physics_step(self, dt: float) -> None:
        now_ns = time.perf_counter_ns()
        if self._last_step_ns:
            self._step_intervals_ms.append((now_ns - self._last_step_ns) / 1_000_000.0)
        self._last_step_ns = now_ns
        self._step_count += 1
        self._sim_elapsed_s += float(dt)

    def start(self) -> None:
        """Start the measured benchmark phase."""
        if self._started:
            raise RuntimeError("benchmark capture already started")

        import omni.physx

        self._benchmark.set_phase("measured")
        self._start_ns = time.perf_counter_ns()
        self._subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )
        self._started = True

    def stop(self) -> None:
        """Stop capture and persist raw NVIDIA and project timing samples."""
        if not self._started:
            raise RuntimeError("benchmark capture was not started")
        if self._stopped:
            return

        elapsed_wall_s = (time.perf_counter_ns() - self._start_ns) / 1_000_000_000.0
        self._subscription = None
        self._benchmark.store_measurements()
        self._benchmark.stop()
        payload = {
            "schema_version": 1,
            "metadata": self._metadata,
            "environment": {
                "cpu_model": _cpu_model(),
                "cpu_governor": _cpu_governor(),
                "logical_cpus": os.cpu_count(),
                "gpu_models": _gpu_models(),
                "isaac_version": _isaac_version(),
                "isaac_image": os.environ.get("ISAAC_BENCHMARK_IMAGE", "unknown"),
                "kernel": platform.release(),
            },
            "physics_step_count": self._step_count,
            "sim_elapsed_s": self._sim_elapsed_s,
            "wall_elapsed_s": elapsed_wall_s,
            "real_time_factor": (
                self._sim_elapsed_s / elapsed_wall_s if elapsed_wall_s > 0.0 else 0.0
            ),
            "physics_step_interval_samples_ms": self._step_intervals_ms,
        }
        (self._output_dir / "step_intervals.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        self._stopped = True

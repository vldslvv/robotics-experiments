"""Standalone-only performance capture for Isaac integration workloads."""

from __future__ import annotations

import inspect
import json
import os
import platform
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["StandaloneBenchmarkCapture", "advance_physics_ticks"]

_METHODOLOGY_VERSION = 3


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


def _benchmark_metadata(
    *,
    context: Any,
    workload: str,
    target_app_updates: int | None,
    target_physics_ticks: int | None,
    rate_limited: bool,
    renderer: str,
    render_resolution: tuple[int, int],
) -> dict[str, Any]:
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
        "target_app_updates": target_app_updates,
        "target_physics_ticks": target_physics_ticks,
        "rate_limited": rate_limited,
        "renderer": renderer,
        "render_resolution": list(render_resolution),
    }


def _subscribe_physics_steps(callback: Callable[..., None], /) -> object:
    """Subscribe through the Isaac 6 API, falling back to the Isaac 5 API."""
    try:
        import omni.physics.core
    except ImportError:
        import omni.physx

        return omni.physx.get_physx_interface().subscribe_physics_step_events(callback)

    physics = omni.physics.core.get_physics_simulation_interface()
    return physics.subscribe_physics_on_step_events(
        on_update=callback,
        pre_step=False,
        order=0,
    )


def advance_physics_ticks(*, simulation_app: Any, minimum_ticks: int) -> int:
    """Advance until at least ``minimum_ticks`` physics callbacks have fired."""
    tick_count = 0

    def on_step(_dt: float, *_args: Any) -> None:
        nonlocal tick_count
        tick_count += 1

    subscription = _subscribe_physics_steps(on_step)
    try:
        while tick_count < minimum_ticks:
            simulation_app.update()
    finally:
        del subscription
    return tick_count


def _create_nvidia_benchmark(*, output_dir: Path, workload: str, metadata: dict[str, Any]) -> Any:
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.benchmark.services")
    import carb
    from isaacsim.benchmark.services import BaseIsaacBenchmark

    carb.settings.get_settings().set(
        "/exts/isaacsim.benchmark.services/metrics/metrics_output_folder",
        str(output_dir),
    )
    kwargs: dict[str, Any] = {
        "benchmark_name": workload,
        "backend_type": "JSONFileMetrics",
        "report_generation": False,
        "workflow_metadata": {
            "metadata": [
                {"name": key, "data": value}
                for key, value in metadata.items()
                if isinstance(value, (str, int, float)) and value is not None
            ]
        },
    }
    parameters = inspect.signature(BaseIsaacBenchmark).parameters
    if "recorders" in parameters:
        from isaacsim.benchmark.services import DEFAULT_RECORDERS

        kwargs["recorders"] = [*DEFAULT_RECORDERS, "gpu_frametime"]
    else:
        kwargs["gpu_frametime"] = True
    return BaseIsaacBenchmark(**kwargs)


class StandaloneBenchmarkCapture:
    """Capture one measured phase without changing production runtime state."""

    def __init__(
        self,
        *,
        context: Any,
        workload: str,
        target_app_updates: int | None,
        target_physics_ticks: int | None,
        rate_limited: bool,
        renderer: str,
        render_resolution: tuple[int, int],
        on_physics_step: Callable[["StandaloneBenchmarkCapture", float], None] | None = None,
    ) -> None:
        if context.benchmark_output_dir is None:
            raise ValueError("benchmark capture requires a benchmark output directory")
        if (target_app_updates is None) == (target_physics_ticks is None):
            raise ValueError("benchmark capture requires exactly one update or physics-tick target")

        self._output_dir = Path(context.benchmark_output_dir) / workload
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = _benchmark_metadata(
            context=context,
            workload=workload,
            target_app_updates=target_app_updates,
            target_physics_ticks=target_physics_ticks,
            rate_limited=rate_limited,
            renderer=renderer,
            render_resolution=render_resolution,
        )
        self._benchmark = _create_nvidia_benchmark(
            output_dir=self._output_dir,
            workload=workload,
            metadata=self._metadata,
        )
        self._on_tick = on_physics_step
        self._physics_subscription = None
        self._app_subscription = None
        self._render_subscription = None
        self._started = False
        self._stopped = False
        self._last_step_ns = 0
        self._last_app_ns = 0
        self._last_render_ns = 0
        self._start_ns = 0
        self._physics_step_count = 0
        self._app_update_count = 0
        self._sim_elapsed_s = 0.0
        self._samples_ms: dict[str, list[float]] = {
            "physics_step_interval": [],
            "app_update_interval": [],
            "render_update_interval": [],
        }

    @property
    def app_update_count(self) -> int:
        """Return the number of Kit app-update events observed in the phase."""
        return self._app_update_count

    @property
    def physics_step_count(self) -> int:
        """Return the number of physics callbacks observed in the phase."""
        return self._physics_step_count

    @contextmanager
    def measure_component(self, name: str, /) -> Iterator[None]:
        """Append one named project-component duration to this phase."""
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            self._samples_ms.setdefault(name, []).append(elapsed_ms)

    def _on_physics_step(self, dt: float, *_args: Any) -> None:
        now_ns = time.perf_counter_ns()
        if self._last_step_ns:
            self._samples_ms["physics_step_interval"].append(
                (now_ns - self._last_step_ns) / 1_000_000.0
            )
        self._last_step_ns = now_ns
        self._physics_step_count += 1
        self._sim_elapsed_s += float(dt)
        if self._on_tick is not None:
            self._on_tick(self, float(dt))

    def _on_app_update(self, _event: Any, /) -> None:
        now_ns = time.perf_counter_ns()
        if self._last_app_ns:
            self._samples_ms["app_update_interval"].append(
                (now_ns - self._last_app_ns) / 1_000_000.0
            )
        self._last_app_ns = now_ns
        self._app_update_count += 1

    def _on_render_update(self, _event: Any, /) -> None:
        now_ns = time.perf_counter_ns()
        if self._last_render_ns:
            self._samples_ms["render_update_interval"].append(
                (now_ns - self._last_render_ns) / 1_000_000.0
            )
        self._last_render_ns = now_ns

    def start(self) -> None:
        """Start the measured benchmark phase."""
        if self._started:
            raise RuntimeError("benchmark capture already started")

        import carb
        import omni.kit.app

        self._benchmark.set_phase("measured")
        carb.profiler.begin(1, f"Robotics Benchmark: {self._metadata['workload']}")
        self._start_ns = time.perf_counter_ns()
        self._physics_subscription = _subscribe_physics_steps(self._on_physics_step)
        dispatcher = carb.eventdispatcher.get_eventdispatcher()
        self._app_subscription = dispatcher.observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_PRE_UPDATE,
            on_event=self._on_app_update,
            observer_name=f"StandaloneBenchmarkCapture.{self._metadata['workload']}.app",
        )
        self._render_subscription = dispatcher.observe_event(
            event_name="runloop:rendering_0:update",
            on_event=self._on_render_update,
            observer_name=f"StandaloneBenchmarkCapture.{self._metadata['workload']}.render",
        )
        self._started = True

    def stop(self) -> None:
        """Stop capture and persist raw NVIDIA and project timing samples."""
        if not self._started:
            raise RuntimeError("benchmark capture was not started")
        if self._stopped:
            return

        elapsed_wall_s = (time.perf_counter_ns() - self._start_ns) / 1_000_000_000.0
        self._physics_subscription = None
        self._app_subscription = None
        self._render_subscription = None
        import carb

        carb.profiler.end(1)
        self._benchmark.store_measurements()
        self._benchmark.stop()
        payload = {
            "schema_version": 2,
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
            "app_update_count": self._app_update_count,
            "physics_step_count": self._physics_step_count,
            "sim_elapsed_s": self._sim_elapsed_s,
            "wall_elapsed_s": elapsed_wall_s,
            "real_time_factor": (
                self._sim_elapsed_s / elapsed_wall_s if elapsed_wall_s > 0.0 else 0.0
            ),
            "samples_ms": self._samples_ms,
        }
        (self._output_dir / "samples.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        self._stopped = True

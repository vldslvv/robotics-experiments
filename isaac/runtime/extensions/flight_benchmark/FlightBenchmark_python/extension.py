"""Opt-in Kit plugin wrapping the live full-GUI flight workload."""

from __future__ import annotations

import importlib
import inspect
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
import torch
from ExperimentRuntime_python import get_instance

from FlightBenchmark_python._capture import (
    FlightBenchmarkRecorder,
    FlightPhase,
    FlightPhaseController,
    FlightPhaseObservation,
)
from FlightBenchmark_python._instrumentation import InstrumentationSession
from robotics_experiments.config import load_config

__all__ = ["Extension"]

_LOG_PREFIX = "[FlightBenchmark]"
_OUTPUT_ENV = "FLIGHT_BENCHMARK_OUTPUT_DIR"
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_CONFIG_DIGEST_ENV = "FLIGHT_BENCHMARK_CONFIG_SHA256"
_MEASUREMENT_LEVEL_ENV = "FLIGHT_BENCHMARK_MEASUREMENT_LEVEL"
_MEASUREMENT_LEVELS = ("lifecycle", "nvidia", "full")


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.partition(":")[2].strip()
    raise RuntimeError("flight benchmark could not determine the CPU model")


def _cpu_governor() -> str:
    return (
        Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .read_text(encoding="utf-8")
        .strip()
    )


def _isaac_version() -> str:
    from isaacsim.core.version import get_version

    version = get_version()
    if isinstance(version, (tuple, list)):
        return ".".join(str(part) for part in version[:3])
    return str(version)


def _gpu_models() -> list[str]:
    models = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not models:
        raise RuntimeError("flight benchmark could not determine a CUDA GPU")
    return models


def _rss_mb() -> float:
    resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def _subscribe_physics_steps(callback: Any, /) -> object:
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


def _takeoff_cursors(config: Any, /) -> dict[str, int]:
    """Return the completed-step cursor for each configured takeoff."""
    cursors = {}
    for label, vehicle in config.vehicles.items():
        if not vehicle.bridge.enabled:
            continue
        takeoffs = [
            index for index, step in enumerate(vehicle.routine.steps) if step["kind"] == "takeoff"
        ]
        if len(takeoffs) != 1:
            raise ValueError(
                f"flight benchmark vehicle {label!r} requires exactly one takeoff step; "
                f"found {len(takeoffs)}"
            )
        cursors[label] = takeoffs[0] + 1
    if not cursors:
        raise ValueError("flight benchmark requires at least one bridge-enabled vehicle")
    return cursors


def _load_base_isaac_benchmark() -> tuple[Any, list[str] | None]:
    """Load the synchronous benchmark API across Isaac 5 and Isaac 6."""
    services = importlib.import_module("isaacsim.benchmark.services")
    benchmark_type = getattr(services, "BaseIsaacBenchmark", None)
    if benchmark_type is None:
        legacy = importlib.import_module("isaacsim.benchmark.services.base_isaac_benchmark")
        benchmark_type = legacy.BaseIsaacBenchmark
    default_recorders = getattr(services, "DEFAULT_RECORDERS", None)
    return benchmark_type, default_recorders


def _create_nvidia_benchmark(*, output_dir: Path, metadata: dict[str, Any]) -> Any:
    base_isaac_benchmark, default_recorders = _load_base_isaac_benchmark()

    carb.settings.get_settings().set(
        "/exts/isaacsim.benchmark.services/metrics/metrics_output_folder",
        str(output_dir),
    )
    kwargs: dict[str, Any] = {
        "benchmark_name": "flight",
        "backend_type": "JSONFileMetrics",
        "report_generation": False,
        "workflow_metadata": {
            "metadata": [
                {"name": key, "data": value}
                for key, value in metadata.items()
                if isinstance(value, (str, int, float))
            ]
        },
    }
    if "recorders" in inspect.signature(base_isaac_benchmark).parameters:
        if default_recorders is None:
            raise RuntimeError("Isaac benchmark service does not expose default recorders")
        kwargs["recorders"] = [*default_recorders, "gpu_frametime"]
    else:
        kwargs["gpu_frametime"] = True
    return base_isaac_benchmark(**kwargs)


class Extension(omni.ext.IExt):
    """Own the benchmark lifecycle around the dependent production extension."""

    def on_startup(self, ext_id: str) -> None:
        """Create capture resources and install wrappers before Play."""
        _ = ext_id
        output_dir = Path(os.environ[_OUTPUT_ENV])
        config_path = os.environ[_CONFIG_PATH_ENV]
        measurement_level = os.environ.get(_MEASUREMENT_LEVEL_ENV, "full")
        if measurement_level not in _MEASUREMENT_LEVELS:
            raise ValueError(
                f"{_MEASUREMENT_LEVEL_ENV} must be one of {_MEASUREMENT_LEVELS}; "
                f"got {measurement_level!r}"
            )
        config = load_config(config_path)
        airborne_duration_s = float(config.flight_benchmark.airborne_duration_s)
        if not math.isfinite(airborne_duration_s) or airborne_duration_s <= 0.0:
            raise ValueError(
                "flight_benchmark.airborne_duration_s must be positive and finite; "
                f"got {airborne_duration_s!r}"
            )
        self._takeoff_cursors = _takeoff_cursors(config)
        metadata = {
            "methodology_version": 1,
            "run_kind": os.environ.get("FLIGHT_BENCHMARK_RUN_KIND", "measured"),
            "repetition": int(os.environ.get("FLIGHT_BENCHMARK_REPETITION", "0")),
            "scene": str(config.scene),
            "config": config_path,
            "config_sha256": os.environ[_CONFIG_DIGEST_ENV],
            "physics_hz": float(config.sim.physics_hz),
            "rendering_hz": float(config.sim.rendering_hz),
            "solver_type": str(config.sim.solver_type),
            "renderer": "RaytracedLighting",
            "render_resolution": [1280, 720],
            "vehicle_count": len(self._takeoff_cursors),
            "vehicle_labels": list(self._takeoff_cursors),
            "vehicle_ids": [
                vehicle.id for vehicle in config.vehicles.values() if vehicle.bridge.enabled
            ],
            "airborne_duration_s": airborne_duration_s,
            "measurement_level": measurement_level,
        }
        environment = {
            "cpu_model": _cpu_model(),
            "cpu_governor": _cpu_governor(),
            "logical_cpus": os.cpu_count(),
            "gpu_models": _gpu_models(),
            "isaac_version": _isaac_version(),
            "isaac_image": os.environ.get("ISAAC_BENCHMARK_IMAGE", "unknown"),
            "isaac_image_id": os.environ.get("ISAAC_BENCHMARK_IMAGE_ID", "unknown"),
            "px4_image": os.environ.get("PX4_BENCHMARK_IMAGE", "unknown"),
            "px4_image_id": os.environ.get("PX4_BENCHMARK_IMAGE_ID", "unknown"),
            "source_commit": os.environ.get("FLIGHT_BENCHMARK_SOURCE_COMMIT", "unknown"),
            "kernel": platform.release(),
        }
        self._expected_vehicles = len(self._takeoff_cursors)
        self._recorder = FlightBenchmarkRecorder(
            output_path=output_dir / "samples.json",
            metadata=metadata,
            environment=environment,
        )
        self._controller = FlightPhaseController(airborne_duration_s=airborne_duration_s)
        capture_nvidia = measurement_level in ("nvidia", "full")
        self._capture_full = measurement_level == "full"
        self._nvidia = (
            _create_nvidia_benchmark(
                output_dir=output_dir / "nvidia",
                metadata=metadata,
            )
            if capture_nvidia
            else None
        )
        self._finished = False
        self._play_requested = False
        self._last_wall_ns = time.perf_counter_ns()
        self._last_cpu_ns = time.process_time_ns()
        self._instrumentation = InstrumentationSession(recorder=self._recorder)
        if self._capture_full:
            self._instrumentation.start()

        app = omni.kit.app.get_app()
        self._pre_update_sub = (
            app.get_pre_update_event_stream().create_subscription_to_pop(
                self._on_pre_update,
                name="flight_benchmark_pre_update",
            )
            if measurement_level != "lifecycle"
            else None
        )
        self._post_update_sub = app.get_post_update_event_stream().create_subscription_to_pop(
            self._on_post_update,
            name="flight_benchmark_post_update",
        )
        self._physics_sub = (
            _subscribe_physics_steps(self._on_physics_step)
            if measurement_level != "lifecycle"
            else None
        )
        carb.log_info(
            f"{_LOG_PREFIX} {measurement_level} measurement active; waiting for the scene"
        )

    def on_shutdown(self) -> None:
        """Release plugin-owned subscriptions, wrappers, and metric resources."""
        self._pre_update_sub = None
        self._post_update_sub = None
        self._physics_sub = None
        self._instrumentation.stop()
        if not self._finished:
            self._fail(reason="application shut down before benchmark completion")

    def _on_pre_update(self, event: Any) -> None:
        _ = event
        if self._finished:
            return
        self._recorder.begin_app_frame()
        if self._capture_full:
            now_wall_ns = time.perf_counter_ns()
            now_cpu_ns = time.process_time_ns()
            wall_delta_ns = now_wall_ns - self._last_wall_ns
            cpu_percent = (
                100.0 * (now_cpu_ns - self._last_cpu_ns) / wall_delta_ns
                if wall_delta_ns > 0
                else 0.0
            )
            self._last_wall_ns = now_wall_ns
            self._last_cpu_ns = now_cpu_ns
            self._recorder.record_resources(cpu_percent=cpu_percent, rss_mb=_rss_mb())

    def _on_post_update(self, event: Any) -> None:
        _ = event
        if self._finished:
            return
        try:
            if self._nvidia is not None:
                self._recorder.record_render_update()
            runtime = get_instance()
            status = runtime.read_status()
            if not self._play_requested:
                app = omni.kit.app.get_app()
                loading = omni.usd.get_context().get_stage_loading_status()[2]
                if app.is_app_ready() and status.scene_ready and loading == 0:
                    self._play_requested = True
                    timeline = omni.timeline.get_timeline_interface()
                    timeline.set_auto_update(True)
                    timeline.play()
                    timeline.commit()
                return
            self._observe(status=status)
        except Exception as exc:
            self._abort(reason=str(exc))

    def _on_physics_step(self, dt: float, *_args: Any) -> None:
        if not self._finished:
            self._recorder.record_physics_step(dt=dt)

    def _observe(self, *, status: Any) -> None:
        if status.failure is not None:
            sim_times_us = [vehicle.sim_time_us for vehicle in status.vehicles]
            raise RuntimeError(f"{status.failure}; vehicle_sim_times_us={sim_times_us}")
        if not status.vehicles:
            return
        if len(status.vehicles) != self._expected_vehicles:
            raise RuntimeError(
                f"expected {self._expected_vehicles} flying vehicles; "
                f"observed {len(status.vehicles)}"
            )
        labels = [vehicle.label for vehicle in status.vehicles]
        if labels != list(self._takeoff_cursors):
            raise RuntimeError(
                "runtime vehicle order differs from the configured bridge-enabled fleet: "
                f"expected {list(self._takeoff_cursors)}, observed {labels}"
            )

        bridge_statuses = [vehicle.bridge_status for vehicle in status.vehicles]
        sim_time_s = min(vehicle.sim_time_us for vehicle in status.vehicles) / 1_000_000.0
        observation = FlightPhaseObservation(
            all_connected=all(
                bridge is not None and bridge.is_connected for bridge in bridge_statuses
            ),
            any_armed=any(bridge is not None and bridge.is_armed for bridge in bridge_statuses),
            all_takeoffs_complete=all(
                vehicle.completed_steps >= self._takeoff_cursors[vehicle.label]
                for vehicle in status.vehicles
            ),
            sim_time_s=sim_time_s,
        )
        successor = self._controller.observe(observation=observation)
        if successor.phase is not self._controller.phase and successor.phase is not None:
            self._transition(phase=successor.phase, sim_time_s=sim_time_s)
        self._controller = successor
        if successor.complete:
            self._finish(sim_time_s=sim_time_s)
            self._quit(return_code=0)

    def _transition(self, *, phase: FlightPhase, sim_time_s: float) -> None:
        if self._nvidia is not None:
            if self._recorder.phase is not None:
                self._nvidia.store_measurements()
            self._nvidia.set_phase(phase.value)
        self._recorder.transition(phase=phase, sim_time_s=sim_time_s)

    def _finish(self, *, sim_time_s: float) -> None:
        self._recorder.finish(sim_time_s=sim_time_s)
        self._stop_nvidia()
        self._finished = True

    def _fail(self, *, reason: str) -> None:
        if self._finished:
            return
        status = get_instance().read_status()
        sim_time_s = (
            min(vehicle.sim_time_us for vehicle in status.vehicles) / 1_000_000.0
            if status.vehicles
            else 0.0
        )
        self._recorder.fail(reason=reason, sim_time_s=sim_time_s)
        self._stop_nvidia()
        self._finished = True

    def _abort(self, *, reason: str) -> None:
        carb.log_error(f"{_LOG_PREFIX} failed: {reason}")
        self._fail(reason=reason)
        self._quit(return_code=1)

    def _stop_nvidia(self) -> None:
        if self._nvidia is None:
            return
        if self._recorder.phase is not None:
            self._nvidia.store_measurements()
        self._nvidia.stop()

    def _quit(self, *, return_code: int) -> None:
        if os.environ.get("FLIGHT_BENCHMARK_RUN_KIND") == "profile":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(return_code)  # noqa: SLF001 - profiler capture ends on disconnect
        omni.kit.app.get_app().post_quit(return_code)

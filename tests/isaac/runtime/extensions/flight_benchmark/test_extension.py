from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.paths import REPO_ROOT


class _EventStream:
    def __init__(self):
        self.callbacks = []

    def create_subscription_to_pop(self, callback, name=None):
        _ = name
        self.callbacks.append(callback)
        return SimpleNamespace()


class _App:
    def __init__(self):
        self.pre = _EventStream()
        self.post = _EventStream()
        self.ready = False
        self.quit_codes = []

    def get_pre_update_event_stream(self):
        return self.pre

    def get_post_update_event_stream(self):
        return self.post

    def is_app_ready(self):
        return self.ready

    def post_quit(self, return_code=0):
        self.quit_codes.append(return_code)


def _import_extension(monkeypatch):
    root = REPO_ROOT / "isaac/runtime/extensions/flight_benchmark/FlightBenchmark_python"
    package = types.ModuleType("FlightBenchmark_python")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "FlightBenchmark_python", package)
    for name in list(sys.modules):
        if name.startswith("FlightBenchmark_python."):
            sys.modules.pop(name, None)

    app = _App()
    timeline = SimpleNamespace(auto_update=False, commit_count=0, play_count=0)
    timeline.set_auto_update = lambda enabled: setattr(timeline, "auto_update", enabled)
    timeline.play = lambda: setattr(timeline, "play_count", timeline.play_count + 1)
    timeline.commit = lambda: setattr(timeline, "commit_count", timeline.commit_count + 1)
    usd_context = SimpleNamespace(loading=0)
    usd_context.get_stage_loading_status = lambda: ("", 0, usd_context.loading)
    physx = SimpleNamespace(callback=None)
    physx.subscribe_physics_step_events = lambda callback: setattr(physx, "callback", callback)

    carb = types.ModuleType("carb")
    carb.log_info = lambda *args, **kwargs: None
    carb.log_error = lambda *args, **kwargs: None
    carb.settings = SimpleNamespace(get_settings=lambda: SimpleNamespace(set=lambda *args: None))
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_kit = types.ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: app
    omni_physx = types.ModuleType("omni.physx")
    omni_physx.get_physx_interface = lambda: physx
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.get_timeline_interface = lambda: timeline
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: usd_context
    omni.ext = omni_ext
    omni.kit = omni_kit
    omni.kit.app = omni_kit_app
    omni.physx = omni_physx
    omni.timeline = omni_timeline
    omni.usd = omni_usd

    runtime_status = SimpleNamespace(scene_ready=True, failure=None, vehicles=())
    runtime = SimpleNamespace(read_status=lambda: runtime_status)
    production = types.ModuleType("ExperimentRuntime_python")
    production.get_instance = lambda: runtime
    config = types.ModuleType("robotics_experiments.config")
    config.load_config = lambda path: SimpleNamespace(
        scene="scene.usd",
        sim=SimpleNamespace(physics_hz=250.0, rendering_hz=60.0, solver_type="TGS"),
        flight_benchmark=SimpleNamespace(airborne_duration_s=5.0),
        vehicles={
            "one": SimpleNamespace(
                id=0,
                bridge=SimpleNamespace(enabled=True),
                routine=SimpleNamespace(steps=({"kind": "arm"}, {"kind": "takeoff"})),
            ),
            "two": SimpleNamespace(
                id=1,
                bridge=SimpleNamespace(enabled=True),
                routine=SimpleNamespace(steps=({"kind": "arm"}, {"kind": "takeoff"})),
            ),
        },
    )
    for name, value in {
        "carb": carb,
        "omni": omni,
        "omni.ext": omni_ext,
        "omni.kit": omni_kit,
        "omni.kit.app": omni_kit_app,
        "omni.physx": omni_physx,
        "omni.timeline": omni_timeline,
        "omni.usd": omni_usd,
        "ExperimentRuntime_python": production,
        "robotics_experiments.config": config,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    module = importlib.import_module("FlightBenchmark_python.extension")
    return module, app, timeline, usd_context, physx


def test_plugin_owns_autoplay_and_restores_instrumentation(monkeypatch, tmp_path) -> None:
    module, app, timeline, usd_context, physx = _import_extension(monkeypatch)
    monkeypatch.setenv("FLIGHT_BENCHMARK_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("FLIGHT_BENCHMARK_CONFIG_SHA256", "config")
    monkeypatch.setenv("ROBOTICS_EXPERIMENT_CONFIG_PATH", "config.yaml")
    monkeypatch.setattr(module, "_cpu_model", lambda: "CPU")
    monkeypatch.setattr(module, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(module, "_gpu_models", lambda: ["GPU"])
    monkeypatch.setattr(module, "_isaac_version", lambda: "5.1.0")
    monkeypatch.setattr(module, "_rss_mb", lambda: 100.0)
    nvidia = SimpleNamespace(
        set_phase=lambda phase: None,
        store_measurements=lambda: None,
        stop=lambda: None,
    )
    monkeypatch.setattr(module, "_create_nvidia_benchmark", lambda **kwargs: nvidia)
    instrument_calls = []

    class FakeInstrumentation:
        def __init__(self, *, recorder):
            _ = recorder

        def start(self):
            instrument_calls.append("start")

        def stop(self):
            instrument_calls.append("stop")

    monkeypatch.setattr(module, "InstrumentationSession", FakeInstrumentation)
    extension = module.Extension()
    extension.on_startup("flight_benchmark")
    assert physx.callback is not None

    for callback in app.post.callbacks:
        callback(SimpleNamespace())
    assert timeline.play_count == 0

    app.ready = True
    usd_context.loading = 1
    for callback in app.post.callbacks:
        callback(SimpleNamespace())
    assert timeline.play_count == 0

    usd_context.loading = 0
    for callback in app.post.callbacks:
        callback(SimpleNamespace())
    assert timeline.auto_update
    assert timeline.play_count == 1
    assert timeline.commit_count == 1

    extension.on_shutdown()
    assert instrument_calls == ["start", "stop"]
    assert (tmp_path / "samples.json").is_file()


def test_plugin_loads_legacy_isaac_5_gui_benchmark_api(monkeypatch) -> None:
    module, _app, _timeline, _usd_context, _physx = _import_extension(monkeypatch)
    legacy_benchmark = object()
    modules = {
        "isaacsim.benchmark.services": SimpleNamespace(),
        "isaacsim.benchmark.services.base_isaac_benchmark": SimpleNamespace(
            BaseIsaacBenchmark=legacy_benchmark
        ),
    }
    monkeypatch.setattr(module.importlib, "import_module", lambda name: modules[name])

    benchmark_type, default_recorders = module.__dict__["_load_base_isaac_benchmark"]()

    assert benchmark_type is legacy_benchmark
    assert default_recorders is None


def test_physics_subscription_uses_isaac_6_api(monkeypatch) -> None:
    module, _app, _timeline, _usd_context, _physx = _import_extension(monkeypatch)
    calls = []

    def subscribe(**kwargs):
        calls.append(kwargs)
        return "subscription"

    interface = SimpleNamespace(subscribe_physics_on_step_events=subscribe)
    physics: Any = types.ModuleType("omni.physics")
    physics.__path__ = []
    physics_core: Any = types.ModuleType("omni.physics.core")
    physics_core.get_physics_simulation_interface = lambda: interface
    physics.core = physics_core
    omni: Any = sys.modules["omni"]
    omni.physics = physics
    monkeypatch.setitem(sys.modules, "omni.physics", physics)
    monkeypatch.setitem(sys.modules, "omni.physics.core", physics_core)

    def callback(*_args):
        return None

    assert module.__dict__["_subscribe_physics_steps"](callback) == "subscription"
    assert calls == [{"on_update": callback, "pre_step": False, "order": 0}]


def test_physics_callback_accepts_isaac_5_extra_arguments(monkeypatch) -> None:
    module, _app, _timeline, _usd_context, _physx = _import_extension(monkeypatch)
    extension = module.Extension.__new__(module.Extension)
    samples = []
    vars(extension).update(
        _finished=False,
        _recorder=SimpleNamespace(record_physics_step=lambda **sample: samples.append(sample)),
    )

    type(extension).__dict__["_on_physics_step"](extension, 0.004, object())

    assert samples == [{"dt": 0.004}]


def test_lifecycle_measurement_skips_nvidia_and_wrappers(monkeypatch, tmp_path) -> None:
    module, app, _timeline, _usd_context, physx = _import_extension(monkeypatch)
    monkeypatch.setenv("FLIGHT_BENCHMARK_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("FLIGHT_BENCHMARK_CONFIG_SHA256", "config")
    monkeypatch.setenv("FLIGHT_BENCHMARK_MEASUREMENT_LEVEL", "lifecycle")
    monkeypatch.setenv("ROBOTICS_EXPERIMENT_CONFIG_PATH", "config.yaml")
    monkeypatch.setattr(module, "_cpu_model", lambda: "CPU")
    monkeypatch.setattr(module, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(module, "_gpu_models", lambda: ["GPU"])
    monkeypatch.setattr(module, "_isaac_version", lambda: "5.1.0")
    monkeypatch.setattr(
        module,
        "_create_nvidia_benchmark",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    class FakeInstrumentation:
        def __init__(self, *, recorder):
            _ = recorder

        def start(self):
            raise AssertionError("lifecycle measurement installed wrappers")

        def stop(self):
            pass

    monkeypatch.setattr(module, "InstrumentationSession", FakeInstrumentation)
    extension = module.Extension()
    extension.on_startup("flight_benchmark")
    assert app.pre.callbacks == []
    assert physx.callback is None
    extension.on_shutdown()

    assert (tmp_path / "samples.json").is_file()


def test_profile_quit_uses_process_exit(monkeypatch) -> None:
    module, app, _timeline, _usd_context, _physx = _import_extension(monkeypatch)
    extension = module.Extension.__new__(module.Extension)
    quit_plugin = type(extension).__dict__["_quit"]

    quit_plugin(extension, return_code=0)
    assert app.quit_codes == [0]

    exit_codes: list[int] = []
    monkeypatch.setenv("FLIGHT_BENCHMARK_RUN_KIND", "profile")
    monkeypatch.setattr(module.os, "_exit", exit_codes.append)
    quit_plugin(extension, return_code=3)
    assert exit_codes == [3]
    assert app.quit_codes == [0]


def test_takeoff_cursors_follow_order_and_ignore_disabled_vehicles(monkeypatch) -> None:
    module, _app, _timeline, _usd_context, _physx = _import_extension(monkeypatch)
    config = SimpleNamespace(
        vehicles={
            "lead": SimpleNamespace(
                bridge=SimpleNamespace(enabled=True),
                routine=SimpleNamespace(
                    steps=({"kind": "arm"}, {"kind": "hover"}, {"kind": "takeoff"})
                ),
            ),
            "manual": SimpleNamespace(
                bridge=SimpleNamespace(enabled=False),
                routine=SimpleNamespace(steps=()),
            ),
        }
    )

    assert module.__dict__["_takeoff_cursors"](config) == {"lead": 3}


def test_takeoff_cursors_reject_ambiguous_routines(monkeypatch) -> None:
    module, _app, _timeline, _usd_context, _physx = _import_extension(monkeypatch)
    for steps in ((), ({"kind": "takeoff"}, {"kind": "takeoff"})):
        config = SimpleNamespace(
            vehicles={
                "drone": SimpleNamespace(
                    bridge=SimpleNamespace(enabled=True),
                    routine=SimpleNamespace(steps=steps),
                )
            }
        )
        with pytest.raises(ValueError, match="exactly one takeoff"):
            module.__dict__["_takeoff_cursors"](config)

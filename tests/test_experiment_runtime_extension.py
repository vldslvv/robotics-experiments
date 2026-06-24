"""Unit tests for the live ``experiment_runtime`` Kit shell.

The shell owns no flight logic: it wires Kit events to the host-agnostic
:class:`FleetOrchestrator` phases. These tests replace the orchestrator with a
recording fake and assert the event -> phase mapping (the orchestrator's own
logic is covered by ``test_fleet_orchestrator.py``).
"""

import importlib
import sys
import types
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace


class _TimelineEventType(IntEnum):
    PLAY = 1
    STOP = 2
    PAUSE = 3


class _StageEventType(IntEnum):
    OPENED = 1


class _EventStream:
    def __init__(self):
        self.callbacks = []

    def create_subscription_to_pop(self, callback, name=None):
        self.callbacks.append(callback)
        return SimpleNamespace(unsubscribe=lambda: None)


class _UsdContext:
    def __init__(self):
        self.stage_events = _EventStream()

    def get_stage_event_stream(self):
        return self.stage_events


class _Timeline:
    def __init__(self):
        self.events = _EventStream()

    def get_timeline_event_stream(self):
        return self.events


class _App:
    def __init__(self):
        self.updates = _EventStream()
        self.ready = False

    def get_update_event_stream(self):
        return self.updates

    def is_app_ready(self):
        return self.ready


class _FakeOrchestrator:
    """Records the phase calls the shell makes; configurable fleet."""

    last: "list[_FakeOrchestrator]" = []

    def __init__(self, env):
        self.env = env
        self.stage = None
        self.calls: list[str] = []
        self._drones: list = []
        type(self).last.append(self)

    def open_scene(self, scene_override=None):
        self.calls.append("open_scene")

    def adopt_current_stage(self):
        self.calls.append("adopt_current_stage")
        self.stage = object()

    def apply_settings(self):
        self.calls.append("apply_settings")

    def spawn_fleet(self):
        self.calls.append("spawn_fleet")

    def start_bridges(self):
        self.calls.append("start_bridges")

    def build_drivers(self, make_env):
        self.calls.append("build_drivers")

    def teardown(self):
        self.calls.append("teardown")

    @property
    def drones(self):
        return self._drones

    @property
    def flying(self):
        return []


def _drone_with_steps(steps):
    return SimpleNamespace(vehicle=SimpleNamespace(routine=SimpleNamespace(steps=list(steps))))


def _install_fake_modules(monkeypatch, usd_context, timeline, app):
    carb = types.ModuleType("carb")
    carb.log_info = lambda *a, **k: None
    carb.log_warn = lambda *a, **k: None
    carb.log_error = lambda *a, **k: None

    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_kit = types.ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: app
    omni_kit.app = omni_kit_app
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.TimelineEventType = _TimelineEventType
    omni_timeline.get_timeline_interface = lambda: timeline
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.StageEventType = _StageEventType
    omni_usd.get_context = lambda: usd_context
    omni.ext = omni_ext
    omni.kit = omni_kit
    omni.timeline = omni_timeline
    omni.usd = omni_usd

    fake_config = types.ModuleType("robotics_experiments.config")
    fake_config.get_config = lambda: SimpleNamespace()

    fake_runtime = types.ModuleType("robotics_experiments.runtime")
    fake_runtime.FleetOrchestrator = _FakeOrchestrator
    fake_runtime.FleetDriverEnv = lambda *a, **k: object()

    modules = {
        "carb": carb,
        "omni": omni,
        "omni.ext": omni_ext,
        "omni.kit": omni_kit,
        "omni.kit.app": omni_kit_app,
        "omni.timeline": omni_timeline,
        "omni.usd": omni_usd,
        "robotics_experiments.config": fake_config,
        "robotics_experiments.runtime": fake_runtime,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _import_extension(monkeypatch):
    for name in list(sys.modules):
        if name == "ExperimentRuntime_python" or name.startswith("ExperimentRuntime_python."):
            sys.modules.pop(name, None)
    _FakeOrchestrator.last = []
    usd_context = _UsdContext()
    timeline = _Timeline()
    app = _App()
    _install_fake_modules(monkeypatch, usd_context, timeline, app)
    extension_root = (
        Path(__file__).resolve().parents[1] / "isaac/runtime/extensions/experiment_runtime"
    )
    monkeypatch.syspath_prepend(str(extension_root))
    module = importlib.import_module("ExperimentRuntime_python.extension")
    return module, usd_context, timeline, app


def _started_extension(monkeypatch):
    module, usd_context, timeline, app = _import_extension(monkeypatch)
    extension = module.Extension()
    extension.on_startup("experiment_runtime")
    orch = _FakeOrchestrator.last[-1]
    return module, extension, orch, usd_context, timeline, app


def _fire_deferred_scene_open(extension):
    extension._open_scene_when_ready(SimpleNamespace())


def test_startup_defers_scene_open_and_subscribes(monkeypatch):
    _module, extension, orch, usd_context, timeline, app = _started_extension(monkeypatch)

    # The orchestrator is built and Kit streams are subscribed immediately, but
    # scene-open is deferred until Kit reports app readiness.
    assert orch.calls == []
    assert orch.stage is None
    assert usd_context.stage_events.callbacks  # subscribed to stage events
    assert timeline.events.callbacks  # subscribed to timeline events
    assert app.updates.callbacks == [extension._open_scene_when_ready]
    assert extension._scene_open_sub is not None


def test_deferred_scene_open_waits_for_app_ready(monkeypatch):
    _module, extension, orch, _usd_context, _timeline, app = _started_extension(monkeypatch)

    _fire_deferred_scene_open(extension)

    assert orch.calls == []
    assert extension._scene_open_sub is not None

    app.ready = True

    _fire_deferred_scene_open(extension)
    _fire_deferred_scene_open(extension)

    assert orch.calls == ["open_scene"]
    assert extension._scene_open_sub is None


def test_opened_adopts_stage_and_applies_settings(monkeypatch):
    _module, extension, orch, _usd_context, _timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_deferred_scene_open(extension)

    extension._on_stage_event(SimpleNamespace(type=int(_StageEventType.OPENED)))

    # A new stage tears down any running fleet, then adopts + applies settings.
    assert orch.calls == ["open_scene", "teardown", "adopt_current_stage", "apply_settings"]
    assert orch.stage is not None


def test_play_spawns_and_builds_drivers_when_routine_present(monkeypatch):
    _module, extension, orch, _usd_context, _timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_deferred_scene_open(extension)
    extension._on_stage_event(SimpleNamespace(type=int(_StageEventType.OPENED)))
    orch._drones = [_drone_with_steps([{"kind": "arm"}])]

    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    assert orch.calls[-4:] == ["teardown", "spawn_fleet", "start_bridges", "build_drivers"]
    # A control tick subscription is armed.
    assert app.updates.callbacks
    assert extension._update_sub is not None


def test_play_without_routine_builds_no_drivers(monkeypatch):
    _module, extension, orch, _usd_context, _timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_deferred_scene_open(extension)
    extension._on_stage_event(SimpleNamespace(type=int(_StageEventType.OPENED)))
    orch._drones = [_drone_with_steps([])]  # manual drone -> no driver

    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    assert "build_drivers" not in orch.calls
    assert app.updates.callbacks == [extension._open_scene_when_ready]
    assert extension._update_sub is None


def test_stop_drops_control_and_tears_down(monkeypatch):
    _module, extension, orch, _usd_context, _timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_deferred_scene_open(extension)
    extension._on_stage_event(SimpleNamespace(type=int(_StageEventType.OPENED)))
    orch._drones = [_drone_with_steps([{"kind": "arm"}])]
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.STOP)))

    assert orch.calls[-1] == "teardown"
    assert extension._update_sub is None


def test_shutdown_tears_down(monkeypatch):
    _module, extension, orch, _usd_context, _timeline, _app = _started_extension(monkeypatch)

    extension.on_shutdown()

    assert orch.calls[-1] == "teardown"

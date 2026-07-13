"""Unit tests for the live ``experiment_runtime`` Kit shell.

The shell owns no flight logic: it wires Kit events to the host-agnostic
:class:`FleetState` transitions. These tests replace the runtime with a
recording fake and assert the event -> transition mapping (the runtime's own
logic is covered by ``test_fleet_orchestrator.py``).
"""

import importlib
import sys
import types
from enum import IntEnum
from types import SimpleNamespace

from tests.support.paths import REPO_ROOT


class _TimelineEventType(IntEnum):
    PLAY = 1
    STOP = 2
    PAUSE = 3


class _StageEventType(IntEnum):
    OPENED = 1


class _Subscription:
    """Fake Kit subscription handle modelling RAII unsubscribe.

    Kit keeps a subscription alive only while the returned handle is held;
    releasing it (the extension sets its ``self._..._sub`` attribute to ``None``)
    unsubscribes. Mirroring that here makes ``stream.callbacks`` the observable
    for "is this subscription still armed" without reading the extension's
    private subscription attributes.
    """

    def __init__(self, stream, callback):
        self._stream = stream
        self._callback = callback

    def unsubscribe(self):
        if self._callback in self._stream.callbacks:
            self._stream.callbacks.remove(self._callback)

    def __del__(self):
        self.unsubscribe()


class _EventStream:
    def __init__(self):
        self.callbacks = []

    def create_subscription_to_pop(self, callback, name=None):
        self.callbacks.append(callback)
        return _Subscription(self, callback)


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
        # Public on the real orchestrator (a read-only property there); a plain
        # attribute here so a test can configure the fleet without private access.
        self.drones: list = []
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
    def flying(self):
        return []


def _drone_with_steps(steps):
    return SimpleNamespace(vehicle=SimpleNamespace(routine=SimpleNamespace(steps=list(steps))))


def _install_fake_modules(monkeypatch, usd_context, timeline, app):
    monkeypatch.setenv("ROBOTICS_EXPERIMENT_CONFIG_PATH", "configs/takeoff_quad.yaml")
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
    fake_config.load_config = lambda path: SimpleNamespace()

    fake_runtime = types.ModuleType("robotics_experiments.runtime")
    fake_runtime.FleetDriverEnv = lambda *a, **k: object()
    fake_runtime.create_fleet = lambda *, config: _FakeOrchestrator(config)
    fake_runtime.open_scene = lambda *, state: state.open_scene() or state
    fake_runtime.adopt_current_stage = lambda *, state: state.adopt_current_stage() or state
    fake_runtime.apply_settings = lambda *, state: state.apply_settings() or state
    fake_runtime.spawn_fleet = lambda *, state: state.spawn_fleet() or state
    fake_runtime.start_bridges = lambda *, state: state.start_bridges() or state
    fake_runtime.build_drivers = lambda *, state, make_env: state.build_drivers(make_env) or state
    fake_runtime.teardown = lambda *, state: state.teardown() or state
    fake_runtime.tick_drivers = lambda *, state, on_error: state

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
    extension_root = REPO_ROOT / "isaac/runtime/extensions/experiment_runtime"
    monkeypatch.syspath_prepend(str(extension_root))
    module = importlib.import_module("ExperimentRuntime_python.extension")
    return module, usd_context, timeline, app


def _started_extension(monkeypatch):
    module, usd_context, timeline, app = _import_extension(monkeypatch)
    extension = module.Extension()
    extension.on_startup("experiment_runtime")
    orch = _FakeOrchestrator.last[-1]
    return module, extension, orch, usd_context, timeline, app


def _fire_app_update(app):
    """Drive one Kit app-update tick by firing every registered update callback."""
    for callback in list(app.updates.callbacks):
        callback(SimpleNamespace())


def _fire_stage_opened(usd_context):
    event = SimpleNamespace(type=int(_StageEventType.OPENED))
    for callback in list(usd_context.stage_events.callbacks):
        callback(event)


def _fire_timeline(timeline, event_type):
    event = SimpleNamespace(type=int(event_type))
    for callback in list(timeline.events.callbacks):
        callback(event)


def test_startup_defers_scene_open_and_subscribes(monkeypatch):
    _module, _extension, orch, usd_context, timeline, app = _started_extension(monkeypatch)

    # The orchestrator is built and Kit streams are subscribed immediately, but
    # scene-open is deferred until Kit reports app readiness.
    assert orch.calls == []
    assert orch.stage is None
    assert usd_context.stage_events.callbacks  # subscribed to stage events
    assert timeline.events.callbacks  # subscribed to timeline events
    assert app.updates.callbacks  # deferred scene-open armed on app-update


def test_deferred_scene_open_waits_for_app_ready(monkeypatch):
    _module, _extension, orch, _usd_context, _timeline, app = _started_extension(monkeypatch)

    _fire_app_update(app)

    # App not ready yet: scene stays closed and the deferred callback is still armed.
    assert orch.calls == []
    assert app.updates.callbacks

    app.ready = True

    _fire_app_update(app)
    _fire_app_update(app)

    # Scene opens exactly once and the one-shot deferred callback releases itself.
    assert orch.calls == ["open_scene"]
    assert app.updates.callbacks == []


def test_opened_adopts_stage_and_applies_settings(monkeypatch):
    _module, _extension, orch, usd_context, _timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_app_update(app)

    _fire_stage_opened(usd_context)

    # A new stage tears down any running fleet, then adopts + applies settings.
    assert orch.calls == ["open_scene", "teardown", "adopt_current_stage", "apply_settings"]
    assert orch.stage is not None


def test_play_spawns_and_builds_drivers_when_routine_present(monkeypatch):
    _module, _extension, orch, usd_context, timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_app_update(app)
    _fire_stage_opened(usd_context)
    orch.drones = [_drone_with_steps([{"kind": "arm"}])]

    _fire_timeline(timeline, _TimelineEventType.PLAY)

    assert orch.calls[-5:] == [
        "adopt_current_stage",
        "teardown",
        "spawn_fleet",
        "start_bridges",
        "build_drivers",
    ]
    # A control tick subscription is armed (the one-shot scene-open already
    # released, so the lone remaining update callback is the control tick).
    assert app.updates.callbacks


def test_play_before_stage_open_does_not_spawn(monkeypatch):
    _module, _extension, orch, _usd_context, timeline, _app = _started_extension(monkeypatch)

    _fire_timeline(timeline, _TimelineEventType.PLAY)

    assert orch.calls == []


def test_play_without_routine_builds_no_drivers(monkeypatch):
    _module, _extension, orch, usd_context, timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_app_update(app)
    _fire_stage_opened(usd_context)
    orch.drones = [_drone_with_steps([])]  # manual drone -> no driver

    _fire_timeline(timeline, _TimelineEventType.PLAY)

    assert "build_drivers" not in orch.calls
    # No control tick is armed (and the one-shot scene-open already released).
    assert app.updates.callbacks == []


def test_play_re_adopts_current_stage_before_stage_mutation(monkeypatch):
    _module, _extension, orch, usd_context, timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_app_update(app)
    _fire_stage_opened(usd_context)
    opened_stage = orch.stage

    _fire_timeline(timeline, _TimelineEventType.PLAY)

    assert orch.stage is not opened_stage
    assert orch.calls[-4:] == [
        "adopt_current_stage",
        "teardown",
        "spawn_fleet",
        "start_bridges",
    ]


def test_stop_drops_control_and_tears_down(monkeypatch):
    _module, _extension, orch, usd_context, timeline, app = _started_extension(monkeypatch)
    app.ready = True
    _fire_app_update(app)
    _fire_stage_opened(usd_context)
    orch.drones = [_drone_with_steps([{"kind": "arm"}])]
    _fire_timeline(timeline, _TimelineEventType.PLAY)
    assert app.updates.callbacks  # control tick armed by PLAY

    _fire_timeline(timeline, _TimelineEventType.STOP)

    assert orch.calls[-1] == "teardown"
    # STOP releases the control tick subscription.
    assert app.updates.callbacks == []


def test_shutdown_tears_down(monkeypatch):
    _module, extension, orch, _usd_context, _timeline, _app = _started_extension(monkeypatch)

    extension.on_shutdown()

    assert orch.calls[-1] == "teardown"

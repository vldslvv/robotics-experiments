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
    def __init__(self, stage):
        self.stage = stage
        self.stage_events = _EventStream()

    def get_stage(self):
        return self.stage

    def get_stage_event_stream(self):
        return self.stage_events


class _Timeline:
    def __init__(self):
        self.events = _EventStream()

    def get_timeline_event_stream(self):
        return self.events


class _BridgeHost:
    instances = []

    def __init__(self, *, articulation_path, config):
        self.articulation_path = articulation_path
        self.config = config
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _DefaultPrim:
    def GetPath(self):
        return SimpleNamespace(pathString="/World")


class _Stage:
    def GetDefaultPrim(self):
        return _DefaultPrim()


def _install_fake_modules(monkeypatch, usd_context):
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.TimelineEventType = _TimelineEventType
    omni_timeline.get_timeline_interface = lambda: _Timeline()
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.StageEventType = _StageEventType
    omni_usd.get_context = lambda: usd_context
    omni.ext = omni_ext
    omni.timeline = omni_timeline
    omni.usd = omni_usd

    fake_config = types.ModuleType("robotics_experiments.config")
    fake_config.get_config = lambda: SimpleNamespace(
        drone=SimpleNamespace(prim_name="Drone"),
        px4=SimpleNamespace(enabled=False),
    )

    fake_drone = types.ModuleType("robotics_experiments.drone")
    fake_drone.spawn_drone = lambda stage, config=None: None
    fake_drone.despawn_drone = lambda stage: None

    fake_bridge_host = types.ModuleType("DroneSpawner_python.bridge_host")
    fake_bridge_host.BridgeHost = _BridgeHost

    modules = {
        "omni": omni,
        "omni.ext": omni_ext,
        "omni.timeline": omni_timeline,
        "omni.usd": omni_usd,
        "robotics_experiments.config": fake_config,
        "robotics_experiments.drone": fake_drone,
        "DroneSpawner_python.bridge_host": fake_bridge_host,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _import_extension(monkeypatch, stage=None):
    for name in list(sys.modules):
        if name == "DroneSpawner_python" or name.startswith("DroneSpawner_python."):
            sys.modules.pop(name, None)

    if stage is None:
        stage = object()
    usd_context = _UsdContext(stage)
    _install_fake_modules(monkeypatch, usd_context)
    extension_root = Path(__file__).resolve().parents[1] / "isaac/runtime/extensions/drone_spawner"
    monkeypatch.syspath_prepend(str(extension_root))
    module = importlib.import_module("DroneSpawner_python.extension")
    return module, usd_context


def test_startup_and_stage_open_do_not_spawn(monkeypatch):
    module, _usd_context = _import_extension(monkeypatch)
    calls = []
    monkeypatch.setattr(
        module,
        "spawn_drone",
        lambda stage, config=None: calls.append(("spawn", stage, config)),
    )
    monkeypatch.setattr(module, "despawn_drone", lambda stage: calls.append(("despawn", stage)))

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    extension._on_stage_event(SimpleNamespace(type=int(_StageEventType.OPENED)))

    assert calls == [("despawn", _usd_context.stage)]


def test_play_spawns_once_after_clearing_stale_runtime_drone(monkeypatch):
    stage = object()
    module, _usd_context = _import_extension(monkeypatch, stage=stage)
    calls = []
    monkeypatch.setattr(module, "despawn_drone", lambda stage: calls.append(("despawn", stage)))
    monkeypatch.setattr(
        module,
        "spawn_drone",
        lambda stage, config=None: calls.append(("spawn", stage, config)),
    )

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    expected_drone_config = sys.modules["robotics_experiments.config"].get_config().drone
    assert calls == [("despawn", stage), ("spawn", stage, expected_drone_config)]


def test_play_starts_enabled_bridge_after_spawn(monkeypatch):
    module, _usd_context = _import_extension(monkeypatch, stage=_Stage())
    calls = []
    monkeypatch.setattr(module, "despawn_drone", lambda stage: calls.append("despawn"))
    monkeypatch.setattr(
        module,
        "spawn_drone",
        lambda stage, config=None: calls.append(("spawn", config)),
    )
    monkeypatch.setattr(
        module,
        "_get_experiment_config",
        lambda: SimpleNamespace(
            drone=SimpleNamespace(prim_name="Drone"),
            px4=SimpleNamespace(enabled=True),
        ),
    )

    class BridgeHost:
        def __init__(self, *, articulation_path, config):
            calls.append(("bridge_init", articulation_path))

        def start(self):
            calls.append("bridge_start")

        def stop(self):
            calls.append("bridge_stop")

    monkeypatch.setattr(module, "BridgeHost", BridgeHost)

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    expected_drone_config = sys.modules["robotics_experiments.config"].get_config().drone
    assert calls == [
        "despawn",
        ("spawn", expected_drone_config),
        ("bridge_init", "/World/Drone"),
        "bridge_start",
    ]


def test_stop_pause_and_shutdown_stop_bridge_and_despawn(monkeypatch):
    stage = object()
    module, _usd_context = _import_extension(monkeypatch, stage=stage)
    calls = []
    monkeypatch.setattr(module, "despawn_drone", lambda stage: calls.append(("despawn", stage)))

    class BridgeHost:
        def stop(self):
            calls.append("bridge_stop")

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    bridge = BridgeHost()
    extension._bridge_host = bridge

    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PAUSE)))

    assert extension._bridge_host is None
    assert calls == ["bridge_stop", ("despawn", stage)]

    extension.on_shutdown()

    assert calls == ["bridge_stop", ("despawn", stage), ("despawn", stage)]

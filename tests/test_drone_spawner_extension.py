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
    instances: list["_BridgeHost"] = []

    def __init__(self, *, articulation_path, config, physics_hz):
        self.articulation_path = articulation_path
        self.config = config
        self.physics_hz = physics_hz
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _BridgeHostGroup:
    """Fake coordinator: delegates start/stop to its hosts (one subscription)."""

    def __init__(self, hosts):
        self.hosts = list(hosts)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        for host in self.hosts:
            host.start()

    def stop(self):
        self.stopped = True
        for host in self.hosts:
            host.stop()


def _vehicle(vehicle_id, prim_name, spawn_xy, *, enabled):
    return SimpleNamespace(
        id=vehicle_id,
        spawn_xy=spawn_xy,
        drone=SimpleNamespace(prim_name=prim_name),
        px4=SimpleNamespace(enabled=enabled, vehicle_id=vehicle_id, connection_baseport=4560),
    )


def _env(*vehicles, physics_hz=250.0):
    return SimpleNamespace(sim=SimpleNamespace(physics_hz=physics_hz), vehicles=list(vehicles))


class _DefaultPrim:
    def GetPath(self):
        return SimpleNamespace(pathString="/World")


class _Stage:
    def GetDefaultPrim(self):
        return _DefaultPrim()


def _install_fake_modules(monkeypatch, usd_context):
    carb = types.ModuleType("carb")
    carb.log_info = lambda *a, **k: None
    carb.log_warn = lambda *a, **k: None
    carb.log_error = lambda *a, **k: None
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
    fake_config.get_config = lambda: _env(
        _vehicle(0, "Drone", (0.0, 0.0), enabled=False),
    )

    fake_drone = types.ModuleType("robotics_experiments.drone")
    fake_drone.spawn_drone = (
        lambda stage, config=None, *, position=None, yaw=0.0: f"/World/{config.prim_name}"
    )
    fake_drone.despawn_drone = lambda stage, config=None: None

    fake_bridge_host = types.ModuleType("DroneSpawner_python.bridge_host")
    fake_bridge_host.BridgeHost = _BridgeHost
    fake_bridge_host.BridgeHostGroup = _BridgeHostGroup

    modules = {
        "carb": carb,
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
        lambda stage, config=None, *, position=None, yaw=0.0: calls.append(
            ("spawn", stage, config)
        ),
    )
    monkeypatch.setattr(
        module, "despawn_drone", lambda stage, config=None: calls.append(("despawn", stage))
    )

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    extension._on_stage_event(SimpleNamespace(type=int(_StageEventType.OPENED)))

    # Nothing has been spawned yet, so teardown despawns nothing.
    assert calls == []


def test_play_spawns_each_vehicle_after_clearing_stale_runtime_drone(monkeypatch):
    stage = _Stage()
    module, _usd_context = _import_extension(monkeypatch, stage=stage)
    calls = []
    monkeypatch.setattr(
        module, "despawn_drone", lambda stage, config=None: calls.append(("despawn", config))
    )
    monkeypatch.setattr(
        module,
        "spawn_drone",
        lambda stage, config=None, *, position=None, yaw=0.0: calls.append(
            ("spawn", config.prim_name, position)
        )
        or f"/World/{config.prim_name}",
    )
    v0 = _vehicle(0, "Drone", (0.0, 0.0), enabled=False)
    v1 = _vehicle(1, "Drone1", (2.0, 0.0), enabled=False)
    monkeypatch.setattr(module, "_get_experiment_config", lambda: _env(v0, v1))

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    assert calls == [
        ("spawn", "Drone", (0.0, 0.0)),
        ("spawn", "Drone1", (2.0, 0.0)),
    ]


def test_play_starts_one_bridge_per_enabled_vehicle(monkeypatch):
    module, _usd_context = _import_extension(monkeypatch, stage=_Stage())
    calls = []
    monkeypatch.setattr(module, "despawn_drone", lambda stage, config=None: None)
    monkeypatch.setattr(
        module,
        "spawn_drone",
        lambda stage, config=None, *, position=None, yaw=0.0: f"/World/{config.prim_name}",
    )
    v0 = _vehicle(0, "Drone", (0.0, 0.0), enabled=True)
    v1 = _vehicle(1, "Drone1", (2.0, 0.0), enabled=True)
    monkeypatch.setattr(module, "_get_experiment_config", lambda: _env(v0, v1, physics_hz=250.0))

    class BridgeHost:
        def __init__(self, *, articulation_path, config, physics_hz):
            self.config = config
            calls.append(("bridge_init", articulation_path, config.vehicle_id, physics_hz))

        def start(self):
            calls.append(("bridge_start", self.config.vehicle_id))

        def stop(self):
            calls.append(("bridge_stop", self.config.vehicle_id))

    monkeypatch.setattr(module, "BridgeHost", BridgeHost)

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    # All hosts are constructed first, then the group's single start() starts
    # each in order (the group owns the one physics-step subscription).
    assert calls == [
        ("bridge_init", "/World/Drone", 0, 250.0),
        ("bridge_init", "/World/Drone1", 1, 250.0),
        ("bridge_start", 0),
        ("bridge_start", 1),
    ]
    assert len(extension._bridge_group.hosts) == 2

    # STOP tears down both bridges and despawns both vehicles.
    despawned = []
    monkeypatch.setattr(
        module, "despawn_drone", lambda stage, config=None: despawned.append(config.prim_name)
    )
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.STOP)))

    assert ("bridge_stop", 0) in calls
    assert ("bridge_stop", 1) in calls
    assert extension._bridge_group.hosts == []
    assert despawned == ["Drone", "Drone1"]


def test_disabled_vehicle_spawns_without_a_bridge(monkeypatch):
    module, _usd_context = _import_extension(monkeypatch, stage=_Stage())
    monkeypatch.setattr(module, "despawn_drone", lambda stage, config=None: None)
    monkeypatch.setattr(
        module,
        "spawn_drone",
        lambda stage, config=None, *, position=None, yaw=0.0: f"/World/{config.prim_name}",
    )
    bridge_inits = []

    class BridgeHost:
        def __init__(self, *, articulation_path, config, physics_hz):
            bridge_inits.append(config.vehicle_id)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(module, "BridgeHost", BridgeHost)
    v0 = _vehicle(0, "Drone", (0.0, 0.0), enabled=True)
    v1 = _vehicle(1, "Drone1", (2.0, 0.0), enabled=False)
    monkeypatch.setattr(module, "_get_experiment_config", lambda: _env(v0, v1))

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    # Only the enabled vehicle gets a bridge; both vehicles were spawned.
    assert bridge_inits == [0]
    assert len(extension._bridge_group.hosts) == 1
    assert len(extension._spawned_drones) == 2


def test_shutdown_stops_bridges_and_despawns(monkeypatch):
    module, _usd_context = _import_extension(monkeypatch, stage=_Stage())
    calls = []
    monkeypatch.setattr(
        module,
        "despawn_drone",
        lambda stage, config=None: calls.append(("despawn", config.prim_name)),
    )

    class BridgeHost:
        def __init__(self, vehicle_id):
            self.vehicle_id = vehicle_id

        def stop(self):
            calls.append(("bridge_stop", self.vehicle_id))

    extension = module.Extension()
    extension.on_startup("drone_spawner")
    extension._bridge_group = module.BridgeHostGroup([BridgeHost(0)])
    extension._spawned_drones = [SimpleNamespace(prim_name="Drone")]

    extension.on_shutdown()

    assert calls == [("bridge_stop", 0), ("despawn", "Drone")]
    assert extension._bridge_group.hosts == []
    assert extension._spawned_drones == []

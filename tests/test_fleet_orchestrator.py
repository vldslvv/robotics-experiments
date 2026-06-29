"""Unit tests for the host-agnostic :class:`FleetOrchestrator`.

The orchestrator is the single production pipeline (open scene -> apply settings
-> spawn fleet -> start lockstep bridge group -> build per-drone routine drivers
-> teardown). These tests fake the Isaac collaborators (``BridgeHost``,
``IsaacArticulationIO``, ``spawn_drone``/``despawn_drone``, the command port and
driver) and assert the pipeline's structure: a bridge host + observer only for
``px4.enabled`` vehicles, a driver only for flying drones (non-empty
``routine.steps`` on an enabled vehicle), and a teardown that stops the group,
disconnects ports, shuts observers, and despawns everything.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


class _FakeBridgeHost:
    def __init__(self, *, articulation_path, vehicle, physics_hz):
        self.articulation_path = articulation_path
        self.vehicle = vehicle
        self.physics_hz = physics_hz
        self.is_connected = False
        self.sim_time_us = 0
        self.status = object()


class _FakeBridgeHostGroup:
    # Test-local fake: tracks the most recently constructed group so a test can
    # observe teardown's stop() without reaching into the orchestrator's private
    # ``_group`` handle.
    last: "_FakeBridgeHostGroup | None" = None

    def __init__(self, hosts):
        self.hosts = list(hosts)
        self.started = False
        self.stopped = False
        _FakeBridgeHostGroup.last = self

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _FakeObserver:
    def __init__(self, *, articulation_path, rotor_link_paths):
        self.articulation_path = articulation_path
        self.rotor_link_paths = rotor_link_paths
        self.shut = False

    def initialize(self):
        pass

    def shutdown(self):
        self.shut = True

    def read_state(self, *, sim_time_us):
        return object()


class _FakePort:
    def __init__(self, *, gcs_endpoint, heartbeat_period_s, label):
        self.gcs_endpoint = gcs_endpoint
        self.heartbeat_period_s = heartbeat_period_s
        self.label = label
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


class _FakeDriver:
    def __init__(self, params, port, env, *, label=""):
        self.params = params
        self.port = port
        self.env = env
        self.label = label
        self.is_done = False
        self.ticks = 0

    def tick(self):
        self.ticks += 1


def _import_orchestrator(monkeypatch):
    """Import the orchestrator with fake Isaac collaborators wired in.

    Returns the module and a record namespace capturing spawn/despawn calls.
    """
    carb = types.ModuleType("carb")
    carb.log_info = lambda *a, **k: None
    carb.log_warn = lambda *a, **k: None
    carb.log_error = lambda *a, **k: None
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_physx = types.ModuleType("omni.physx")
    omni_physx.get_physx_interface = lambda: SimpleNamespace(
        subscribe_physics_step_events=lambda cb: SimpleNamespace(unsubscribe=lambda: None)
    )
    omni_usd = types.ModuleType("omni.usd")
    omni.physx = omni_physx
    omni.usd = omni_usd
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physx", omni_physx)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)

    for name in list(sys.modules):
        if name == "robotics_experiments.runtime" or name.startswith(
            "robotics_experiments.runtime."
        ):
            sys.modules.pop(name, None)
    module = importlib.import_module("robotics_experiments.runtime.orchestrator")

    monkeypatch.setattr(module, "BridgeHost", _FakeBridgeHost)
    monkeypatch.setattr(module, "BridgeHostGroup", _FakeBridgeHostGroup)
    monkeypatch.setattr(module, "MavlinkCommandPort", _FakePort)
    monkeypatch.setattr(module, "RoutineDriver", _FakeDriver)
    monkeypatch.setattr(
        module, "DriverParams", SimpleNamespace(from_routine_config=lambda rc, **_kwargs: rc)
    )

    spawned: list = []
    despawned: list = []
    fake_drone = types.ModuleType("robotics_experiments.drone")
    fake_drone.spawn_drone = (
        lambda stage, config=None, *, position=None, yaw=0.0: spawned.append(
            (config.prim_name, position)
        )
        or f"/World/{config.prim_name}"
    )
    fake_drone.despawn_drone = lambda stage, config=None: despawned.append(config.prim_name)
    monkeypatch.setitem(sys.modules, "robotics_experiments.drone", fake_drone)

    fake_px4 = types.ModuleType("robotics_experiments.px4_bridge")
    fake_px4.IsaacArticulationIO = _FakeObserver
    monkeypatch.setitem(sys.modules, "robotics_experiments.px4_bridge", fake_px4)

    return module, SimpleNamespace(spawned=spawned, despawned=despawned)


def _vehicle(vid, prim, xy, *, enabled, steps=()):
    return SimpleNamespace(
        id=vid,
        prim_name=prim,
        spawn_xy=xy,
        drone=SimpleNamespace(prim_name=prim),
        px4=SimpleNamespace(
            enabled=enabled,
            articulation_path=None,
            home_lat_deg=47.3977,
            home_lon_deg=8.5456,
            home_alt_m=488.0,
        ),
        routine=SimpleNamespace(steps=list(steps), gcs_heartbeat_period_s=1.0),
        gcs_port=14550 + vid,
        mavlink_port=4560 + vid,
    )


def _env(*vehicles):
    return SimpleNamespace(
        scene="@scene.usd",
        sim=SimpleNamespace(physics_hz=250.0),
        vehicles=list(vehicles),
    )


def test_spawn_fleet_builds_host_only_for_enabled(monkeypatch):
    module, rec = _import_orchestrator(monkeypatch)
    env = _env(
        _vehicle(0, "Drone", (0.0, 0.0), enabled=True),
        _vehicle(1, "Drone1", (2.0, 0.0), enabled=False),
    )
    orch = module.FleetOrchestrator(env)
    orch.stage = object()

    orch.spawn_fleet()

    assert len(orch.drones) == 2
    assert [h.label for h in orch.bridged] == ["drone0"]
    assert orch.drones[1].host is None and orch.drones[1].observer is None
    assert rec.spawned == [("Drone", (0.0, 0.0)), ("Drone1", (2.0, 0.0))]
    assert len(orch.bridged) == 1


def test_spawn_fleet_requires_an_adopted_stage(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    orch = module.FleetOrchestrator(_env(_vehicle(0, "Drone", (0.0, 0.0), enabled=True)))
    with pytest.raises(RuntimeError):
        orch.spawn_fleet()


def test_build_drivers_only_for_enabled_with_steps(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    env = _env(
        _vehicle(0, "D0", (0.0, 0.0), enabled=True, steps=[{"kind": "arm"}]),  # flies
        _vehicle(1, "D1", (2.0, 0.0), enabled=True),  # manual (no steps)
        _vehicle(2, "D2", (4.0, 0.0), enabled=False, steps=[{"kind": "arm"}]),  # no bridge -> skip
    )
    orch = module.FleetOrchestrator(env)
    orch.stage = object()
    orch.spawn_fleet()

    orch.build_drivers(lambda handle: object())

    assert [h.label for h in orch.flying] == ["drone0"]
    assert orch.drones[0].port is not None
    assert orch.drones[0].port.gcs_endpoint == "udp:127.0.0.1:14550"
    assert orch.drones[1].port is None and orch.drones[1].driver is None
    assert orch.drones[2].port is None and orch.drones[2].driver is None


def test_all_done_reflects_flying_drivers(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    env = _env(_vehicle(0, "D0", (0.0, 0.0), enabled=True, steps=[{"kind": "arm"}]))
    orch = module.FleetOrchestrator(env)
    orch.stage = object()
    orch.spawn_fleet()
    orch.build_drivers(lambda handle: object())

    assert orch.all_done is False
    orch.drones[0].driver.is_done = True
    assert orch.all_done is True


def test_all_done_true_when_no_drone_flies(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    orch = module.FleetOrchestrator(_env(_vehicle(0, "D0", (0.0, 0.0), enabled=True)))
    orch.stage = object()
    orch.spawn_fleet()
    orch.build_drivers(lambda handle: object())

    assert orch.flying == []
    assert orch.all_done is True


def test_teardown_stops_group_disconnects_and_despawns(monkeypatch):
    module, rec = _import_orchestrator(monkeypatch)
    env = _env(
        _vehicle(0, "D0", (0.0, 0.0), enabled=True, steps=[{"kind": "arm"}]),
        _vehicle(1, "D1", (2.0, 0.0), enabled=False),
    )
    orch = module.FleetOrchestrator(env)
    orch.stage = object()
    orch.spawn_fleet()
    orch.build_drivers(lambda handle: object())
    group = _FakeBridgeHostGroup.last
    port = orch.drones[0].port
    observer = orch.drones[0].observer

    orch.teardown()

    assert group.stopped is True
    assert port.disconnected is True
    assert observer.shut is True
    assert sorted(rec.despawned) == ["D0", "D1"]
    assert orch.drones == []


def test_apply_settings_uses_simulation_api(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    applied: list = []
    fake_sim = types.ModuleType("robotics_experiments.simulation")
    fake_sim.apply_settings = lambda stage, settings: applied.append((stage, settings))
    monkeypatch.setitem(sys.modules, "robotics_experiments.simulation", fake_sim)
    env = _env()
    orch = module.FleetOrchestrator(env)
    stage = object()
    orch.stage = stage

    orch.apply_settings()

    assert applied == [(stage, env.sim)]

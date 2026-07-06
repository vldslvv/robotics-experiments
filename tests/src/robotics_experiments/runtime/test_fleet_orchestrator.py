"""Unit tests for the host-agnostic :class:`FleetOrchestrator`.

The orchestrator is the single production pipeline (open scene -> apply settings
-> spawn fleet -> start lockstep bridge group -> build per-drone routine drivers
-> teardown). These tests fake the Isaac collaborators (``BridgeHost``,
``IsaacArticulationIO``, ``spawn_drone``/``despawn_drone``, the command port and
driver) and assert the pipeline's structure: a bridge host + observer only for
``px4.enabled`` vehicles, a driver only for flying drones (non-empty
``routine.steps`` on an enabled vehicle), and a teardown that disconnects ports,
shuts observers, stops bridge hosts, and despawns everything.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


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


class _FakeDriverParams:
    @classmethod
    def from_routine_config(cls, routine, **_kwargs):
        return routine


def _import_orchestrator(monkeypatch):
    """Import the runtime contract with fake upstream collaborators wired in.

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

    fake_commanding = types.ModuleType("robotics_experiments.commanding")
    fake_commanding.CommandAck = object
    fake_commanding.DriverEnvironment = object
    fake_commanding.DriverParams = _FakeDriverParams
    fake_commanding.FailureKind = SimpleNamespace()
    fake_commanding.MavlinkCommandPort = _FakePort
    fake_commanding.RoutineDriver = _FakeDriver
    fake_commanding.RoutineObservation = object
    monkeypatch.setitem(sys.modules, "robotics_experiments.commanding", fake_commanding)

    for name in list(sys.modules):
        if name == "robotics_experiments.runtime" or name.startswith(
            "robotics_experiments.runtime."
        ):
            sys.modules.pop(name, None)
    module = importlib.import_module("robotics_experiments.runtime")

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
    fake_px4.__path__ = []
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
            connection_type="tcpin",
            connection_ip="localhost",
            lockstep=SimpleNamespace(
                first_heartbeat_timeout_s=5.0,
                lockstep_timeout_s=1.0,
            ),
            home_lat_deg=47.3977,
            home_lon_deg=8.5456,
            home_alt_m=488.0,
        ),
        routine=SimpleNamespace(steps=list(steps), gcs_heartbeat_period_s=1.0),
        gcs_port=14550 + vid,
        mavlink_port=4560 + vid,
        debug_log_path=f"/logs/{vid}_bridge.log",
    )


def _env(*vehicles):
    return SimpleNamespace(
        scene="@scene.usd",
        sim=SimpleNamespace(physics_hz=250.0),
        vehicles=list(vehicles),
        driver=SimpleNamespace(
            goto_min_heading_dist_m=0.25,
            prearm_backstop_floor_s=20.0,
            prearm_backstop_multiplier=5.0,
            drive_budget_multiplier=4.0,
        ),
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
    # The host receives the bridge config as one component plus the derived
    # identity straight from the resolved vehicle (never recomputed).
    params = orch.bridged[0].host.params
    assert params.px4 is env.vehicles[0].px4
    assert params.vehicle_id == 0
    assert params.mavlink_port == 4560
    assert params.debug_log_path == "/logs/0_bridge.log"
    assert params.articulation_path == "/World/Drone"
    assert params.physics_hz == 250.0


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

    orch.build_drivers(lambda spawned, port: object())

    assert [f.spawned.label for f in orch.flying] == ["drone0"]
    assert orch.flying[0].port.gcs_endpoint == "udp:127.0.0.1:14550"
    # The manual drone (no steps) and the disabled one (no bridge) fly nothing.
    assert len(orch.flying) == 1


def test_all_done_reflects_flying_drivers(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    env = _env(_vehicle(0, "D0", (0.0, 0.0), enabled=True, steps=[{"kind": "arm"}]))
    orch = module.FleetOrchestrator(env)
    orch.stage = object()
    orch.spawn_fleet()
    orch.build_drivers(lambda spawned, port: object())

    assert orch.all_done is False
    orch.flying[0].driver.is_done = True
    assert orch.all_done is True


def test_all_done_true_when_no_drone_flies(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    orch = module.FleetOrchestrator(_env(_vehicle(0, "D0", (0.0, 0.0), enabled=True)))
    orch.stage = object()
    orch.spawn_fleet()
    orch.build_drivers(lambda spawned, port: object())

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
    orch.build_drivers(lambda spawned, port: object())
    port = orch.flying[0].port
    observer = orch.drones[0].observer

    orch.teardown()

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

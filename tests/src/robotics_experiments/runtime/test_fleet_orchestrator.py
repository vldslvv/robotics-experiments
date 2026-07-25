"""Unit tests for the host-agnostic functional fleet lifecycle.

The orchestrator is the single production pipeline (open scene -> apply settings
-> spawn fleet -> start lockstep bridge group -> build per-drone routine drivers
-> teardown). These tests fake the Isaac collaborators (``BridgeHost``,
``IsaacArticulationIO``, ``spawn_drone``/``despawn_drone``, the command port and
driver) and assert the pipeline's structure: a bridge host + observer only for
``px4.enabled`` vehicles, a driver for every bridge-enabled drone, and a teardown
that disconnects ports, shuts observers, stops bridge hosts, and despawns
everything.
"""

import importlib
import sys
import types
from dataclasses import replace
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

    def read_status(self):
        return SimpleNamespace(failsafe=False)


class _FakeDriver:
    def __init__(self):
        self.is_done = False
        self.ticks = 0


def _import_orchestrator(monkeypatch):
    """Import the runtime contract with fake upstream collaborators wired in.

    Returns the module and a record namespace capturing spawn/despawn calls.
    """
    carb = types.ModuleType("carb")
    carb.log_info = lambda *a, **k: None
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
    fake_commanding.AttitudeRateSetpoint = object
    fake_commanding.CommandAck = object
    fake_commanding.DriverContext = SimpleNamespace
    fake_commanding.DriverEnvironment = object
    fake_commanding.DriverParams = SimpleNamespace
    fake_commanding.DriverTick = SimpleNamespace
    fake_commanding.EnvelopeGates = SimpleNamespace
    fake_commanding.FailureKind = SimpleNamespace()
    fake_commanding.MavlinkCommandPort = _FakePort
    fake_commanding.PolicyAdapter = object
    fake_commanding.PolicyInput = object
    fake_commanding.PolicyResult = object
    fake_commanding.RoutineDriverState = _FakeDriver
    fake_commanding.RoutineObservation = object
    fake_commanding.VelocitySetpoint = object
    creation_ticks = []
    fake_commanding.create_routine_driver = (
        lambda *, context, tick: creation_ticks.append((context, tick)) or _FakeDriver()
    )
    fake_commanding.routine_driver_is_done = lambda *, state: state.is_done
    driver_ticks = []
    fake_commanding.tick_routine_driver = (
        lambda *, context, state, tick: driver_ticks.append((context, state, tick)) or state
    )
    monkeypatch.setitem(sys.modules, "robotics_experiments.commanding", fake_commanding)

    fake_environment = types.ModuleType("robotics_experiments.environment")
    fake_environment.PressObservationIO = object
    fake_environment.create_switches_runtime = lambda *, config, stage: SimpleNamespace(
        config=config,
        stage=stage,
        phase="idle",
    )
    fake_environment.adopt_switches_stage = lambda *, state, stage: SimpleNamespace(
        config=state.config,
        stage=stage,
        phase="idle",
    )
    fake_environment.play_switches_runtime = lambda *, state: SimpleNamespace(
        config=state.config,
        stage=state.stage,
        phase="running",
    )
    fake_environment.step_switches_runtime = lambda *, state, on_toggle=None: state
    fake_environment.teardown_switches_runtime = lambda *, state: SimpleNamespace(
        config=state.config,
        stage=state.stage,
        phase="idle",
    )
    monkeypatch.setitem(sys.modules, "robotics_experiments.environment", fake_environment)

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
        lambda *, stage, config: spawned.append((config.prim_name, config.position_xy))
        or f"/World/{config.prim_name}"
    )
    fake_drone.despawn_drone = lambda *, stage, config: despawned.append(config.prim_name)
    monkeypatch.setitem(sys.modules, "robotics_experiments.drone", fake_drone)

    px4_bridge = importlib.import_module("robotics_experiments.px4_bridge")
    monkeypatch.setattr(px4_bridge, "IsaacArticulationIO", _FakeObserver)

    return module, SimpleNamespace(
        spawned=spawned,
        despawned=despawned,
        creation_ticks=creation_ticks,
        driver_ticks=driver_ticks,
    )


def _vehicle(vid, prim, xy, *, enabled, steps=()):
    return SimpleNamespace(
        id=vid,
        spawn=SimpleNamespace(prim_name=prim, position_xy=xy),
        bridge=SimpleNamespace(
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
        routine=SimpleNamespace(
            steps=list(steps),
            gcs_heartbeat_period_s=1.0,
            max_tilt_deg=35.0,
            max_lateral_drift_m=2.0,
            max_vertical_speed_m_s=8.0,
            max_downward_speed_m_s=-4.0,
            max_angular_speed_rad_s=8.0,
            hover_max_abs_vertical_speed_m_s=0.35,
            hover_max_abs_horizontal_speed_m_s=0.3,
            command_ack_timeout_s=10.0,
            failure_recovery_timeout_s=30.0,
            bridge_heartbeat_timeout_s=30.0,
            handshake_timeout_s=30.0,
            prearm_settle_s=1.0,
            takeoff_target_timeout_s=10.0,
            takeoff_timeout_s=30.0,
            hover_settle_s=2.0,
            goto_position_tol_m=0.5,
            goto_heading_tol_deg=25.0,
            goto_cross_track_m=2.0,
            goto_offboard_prime_ticks=3,
        ),
        gcs_port=14550 + vid,
        mavlink_port=4560 + vid,
        debug_log_path=f"/logs/{vid}_bridge.log",
    )


def _env(*vehicles):
    return SimpleNamespace(
        scene="@scene.usd",
        sim=SimpleNamespace(physics_hz=250.0),
        switches=SimpleNamespace(num_switches=1),
        rl=SimpleNamespace(
            policy_rate_hz=20.0,
            action=SimpleNamespace(),
        ),
        vehicles={f"drone{vehicle.id}": vehicle for vehicle in vehicles},
        driver=SimpleNamespace(
            control_rate_hz=50.0,
            offboard_setpoint_rate_hz=50.0,
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
    fleet = replace(module.create_fleet(config=env), stage=object())

    empty = fleet
    fleet = module.spawn_fleet(state=fleet)
    assert fleet is not empty
    assert empty.drones == ()

    bridged = [drone for drone in fleet.drones if drone.host is not None]
    assert len(fleet.drones) == 2
    assert [drone.label for drone in bridged] == ["drone0"]
    assert fleet.drones[1].host is None and fleet.drones[1].observer is None
    assert rec.spawned == [("Drone", (0.0, 0.0)), ("Drone1", (2.0, 0.0))]
    assert len(bridged) == 1
    # The host receives the bridge config as one component plus the derived
    # identity straight from the resolved vehicle (never recomputed).
    params = bridged[0].host.params
    assert params.bridge is env.vehicles["drone0"].bridge
    assert params.vehicle_id == 0
    assert params.mavlink_port == 4560
    assert params.debug_log_path == "/logs/0_bridge.log"
    assert params.articulation_path == "/World/Drone"
    assert params.physics_hz == 250.0


def test_spawn_fleet_with_all_bridges_disabled_builds_no_bridge_resources(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    fleet = replace(
        module.create_fleet(config=_env(_vehicle(0, "Drone", (0.0, 0.0), enabled=False))),
        stage=object(),
    )

    fleet = module.spawn_fleet(state=fleet)

    assert fleet.drones[0].host is None
    assert fleet.drones[0].observer is None


def test_spawn_fleet_requires_an_adopted_stage(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    fleet = module.create_fleet(config=_env(_vehicle(0, "Drone", (0.0, 0.0), enabled=True)))
    with pytest.raises(RuntimeError):
        module.spawn_fleet(state=fleet)


def test_policy_preflight_fails_before_stage_mutation(monkeypatch):
    module, rec = _import_orchestrator(monkeypatch)

    def fail_preflight(*, config, repo_root):
        del config, repo_root
        raise ValueError("incompatible policy metadata")

    orchestrator = importlib.import_module("robotics_experiments.runtime.orchestrator")
    monkeypatch.setattr(
        orchestrator,
        "_preflight_policy_artifacts",
        fail_preflight,
    )
    vehicle = _vehicle(
        0,
        "Drone",
        (0.0, 0.0),
        enabled=True,
        steps=[{"kind": "policy", "artifact": "missing"}],
    )
    fleet = replace(module.create_fleet(config=_env(vehicle)), stage=object())

    with pytest.raises(ValueError, match="incompatible policy metadata"):
        module.spawn_fleet(state=fleet)

    assert rec.spawned == []
    assert rec.despawned == []


def test_spawn_fleet_rolls_back_drones_after_observer_failure(monkeypatch):
    module, rec = _import_orchestrator(monkeypatch)
    env = _env(
        _vehicle(0, "D0", (0.0, 0.0), enabled=True),
        _vehicle(1, "D1", (2.0, 0.0), enabled=True),
    )
    fleet = replace(module.create_fleet(config=env), stage=object())
    orchestrator = importlib.import_module("robotics_experiments.runtime.orchestrator")
    monkeypatch.setattr(
        orchestrator,
        "_prepare_policy_observer",
        lambda *, state, spawned, preflight: (_ for _ in ()).throw(RuntimeError("observer failed")),
    )

    with pytest.raises(RuntimeError, match="observer failed"):
        module.spawn_fleet(state=fleet)

    assert rec.spawned == [("D0", (0.0, 0.0)), ("D1", (2.0, 0.0))]
    assert sorted(rec.despawned) == ["D0", "D1"]


def test_build_drivers_for_every_bridge_enabled_drone(monkeypatch):
    module, rec = _import_orchestrator(monkeypatch)
    env = _env(
        _vehicle(0, "D0", (0.0, 0.0), enabled=True, steps=[{"kind": "arm"}]),
        _vehicle(1, "D1", (2.0, 0.0), enabled=True),
        _vehicle(2, "D2", (4.0, 0.0), enabled=False, steps=[{"kind": "arm"}]),
        _vehicle(3, "D3", (6.0, 0.0), enabled=False),
    )
    fleet = replace(module.create_fleet(config=env), stage=object())
    fleet = module.spawn_fleet(state=fleet)

    fleet = module.build_drivers(state=fleet, make_env=lambda spawned, port: object())

    assert [f.spawned.label for f in fleet.flying] == ["drone0", "drone1"]
    assert fleet.flying[0].context.port.gcs_endpoint == "udp:127.0.0.1:14550"
    assert fleet.flying[1].context.params.steps == []
    assert not hasattr(fleet.flying[0], "port")
    assert len(rec.creation_ticks) == 2
    assert rec.creation_ticks[0][1] is rec.creation_ticks[1][1]


def test_tick_drivers_passes_context_and_progress_state(monkeypatch):
    module, rec = _import_orchestrator(monkeypatch)
    env = _env(_vehicle(0, "D0", (0.0, 0.0), enabled=True))
    fleet = replace(module.create_fleet(config=env), stage=object())
    fleet = module.spawn_fleet(state=fleet)
    fleet = module.build_drivers(state=fleet, make_env=lambda spawned, port: object())
    flying = fleet.flying[0]

    for _ in range(5):
        fleet = module.tick_drivers(state=fleet, physics_dt_s=0.004, monotonic_s=7.0)

    assert len(rec.driver_ticks) == 1
    context, driver, tick = rec.driver_ticks[0]
    assert (context, driver) == (flying.context, flying.driver)
    assert (tick.control_time_us, tick.monotonic_s) == (20_000, 7.0)
    assert fleet.flying[0] is flying
    assert fleet.flying[0].context is flying.context


def test_switch_latch_uses_artifact_selected_cadence(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    env = _env()
    fleet = replace(
        module.create_fleet(config=env),
        switch_rate_hz=10.0,
    )
    steps = []
    environment = sys.modules["robotics_experiments.environment"]
    environment.step_switches_runtime = (
        lambda *, state, on_toggle=None: steps.append(state) or state
    )

    for _ in range(250):
        fleet = module.tick_drivers(state=fleet, physics_dt_s=0.004, monotonic_s=7.0)

    assert len(steps) == 10
    assert env.rl.policy_rate_hz == 20.0


def test_tick_drivers_isolates_errors_and_shares_one_fleet_tick(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    env = _env(
        _vehicle(0, "D0", (0.0, 0.0), enabled=True),
        _vehicle(1, "D1", (2.0, 0.0), enabled=True),
        _vehicle(2, "D2", (4.0, 0.0), enabled=True),
    )
    fleet = replace(module.create_fleet(config=env), stage=object())
    fleet = module.spawn_fleet(state=fleet)
    fleet = module.build_drivers(state=fleet, make_env=lambda spawned, port: object())
    first_context = fleet.flying[0].context
    successful_ticks = []
    errors = []

    def transition(*, context, state, tick):
        if context is first_context:
            raise RuntimeError("broken driver")
        successful_ticks.append((context, tick))
        return state

    orchestrator = importlib.import_module("robotics_experiments.runtime.orchestrator")
    monkeypatch.setattr(orchestrator, "tick_routine_driver", transition)
    for _ in range(5):
        fleet = module.tick_drivers(
            state=fleet,
            physics_dt_s=0.004,
            monotonic_s=9.0,
            on_error=lambda flying, exc: errors.append((flying, exc)),
        )

    assert len(errors) == 1
    assert str(errors[0][1]) == "broken driver"
    assert len(successful_ticks) == 2
    assert successful_ticks[0][1].control_time_us == 20_000
    assert successful_ticks[0][1] is successful_ticks[1][1]


def test_teardown_resets_driver_cadence(monkeypatch):
    module, rec = _import_orchestrator(monkeypatch)
    env = _env(_vehicle(0, "D0", (0.0, 0.0), enabled=True))
    fleet = replace(module.create_fleet(config=env), stage=object())
    fleet = module.spawn_fleet(state=fleet)
    fleet = module.build_drivers(state=fleet, make_env=lambda spawned, port: object())
    for _ in range(3):
        fleet = module.tick_drivers(state=fleet, physics_dt_s=0.004)

    assert rec.driver_ticks == []
    fleet = module.teardown(state=fleet)
    assert fleet.driver_cadence.previous_time_ns == 0


def test_all_done_reflects_flying_drivers(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    env = _env(_vehicle(0, "D0", (0.0, 0.0), enabled=True, steps=[{"kind": "arm"}]))
    fleet = replace(module.create_fleet(config=env), stage=object())
    fleet = module.spawn_fleet(state=fleet)
    fleet = module.build_drivers(state=fleet, make_env=lambda spawned, port: object())

    assert module.fleet_is_done(state=fleet) is False
    fleet.flying[0].driver.is_done = True
    assert module.fleet_is_done(state=fleet) is True


def test_all_done_true_when_no_bridge_enabled_drone_flies(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    fleet = replace(
        module.create_fleet(config=_env(_vehicle(0, "D0", (0.0, 0.0), enabled=False))),
        stage=object(),
    )
    fleet = module.spawn_fleet(state=fleet)
    fleet = module.build_drivers(state=fleet, make_env=lambda spawned, port: object())

    assert fleet.flying == ()
    assert module.fleet_is_done(state=fleet) is True


def test_teardown_stops_group_disconnects_and_despawns(monkeypatch):
    module, rec = _import_orchestrator(monkeypatch)
    env = _env(
        _vehicle(0, "D0", (0.0, 0.0), enabled=True, steps=[{"kind": "arm"}]),
        _vehicle(1, "D1", (2.0, 0.0), enabled=True),
        _vehicle(2, "D2", (4.0, 0.0), enabled=False),
    )
    fleet = replace(module.create_fleet(config=env), stage=object())
    fleet = module.spawn_fleet(state=fleet)
    fleet = module.build_drivers(state=fleet, make_env=lambda spawned, port: object())
    ports = [flying.context.port for flying in fleet.flying]
    observer = fleet.drones[0].observer

    fleet = module.teardown(state=fleet)

    assert all(port.disconnected for port in ports)
    assert observer.shut is True
    assert sorted(rec.despawned) == ["D0", "D1", "D2"]
    assert fleet.drones == ()


def test_apply_settings_uses_simulation_api(monkeypatch):
    module, _rec = _import_orchestrator(monkeypatch)
    applied: list = []
    fake_sim = types.ModuleType("robotics_experiments.simulation")
    fake_sim.apply_settings = lambda stage, settings: applied.append((stage, settings))
    monkeypatch.setitem(sys.modules, "robotics_experiments.simulation", fake_sim)
    env = _env()
    stage = object()
    fleet = replace(module.create_fleet(config=env), stage=stage)

    fleet = module.apply_settings(state=fleet)

    assert applied == [(stage, env.sim)]

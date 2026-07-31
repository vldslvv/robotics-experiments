import importlib
import sys
import types
from dataclasses import dataclass, replace
from enum import IntEnum
from types import SimpleNamespace

import pytest


class _TimelineEventType(IntEnum):
    PLAY = 1
    STOP = 2
    PAUSE = 3


class _StageEventType(IntEnum):
    OPENED = 1


class _Physx:
    def __init__(self):
        self.callback = None
        self.subscribe_count = 0

    def subscribe_physics_step_events(self, callback):
        self.callback = callback
        self.subscribe_count += 1
        return SimpleNamespace(unsubscribe=lambda: None)


def _step(host, dt):
    """Drive one physics tick through the two-phase host API."""
    host.send_phase(dt)
    host.recv_phase()


def _install_fake_isaac(monkeypatch):
    physx = _Physx()
    carb = types.ModuleType("carb")
    carb.log_info = lambda message: None
    carb.log_warn = lambda message: None
    carb.log_error = lambda message: None

    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_physx = types.ModuleType("omni.physx")
    omni_physx.get_physx_interface = lambda: physx
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.TimelineEventType = _TimelineEventType
    omni_timeline.get_timeline_interface = lambda: SimpleNamespace(
        get_timeline_event_stream=lambda: SimpleNamespace(
            create_subscription_to_pop=lambda callback: SimpleNamespace(unsubscribe=lambda: None)
        )
    )
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.StageEventType = _StageEventType
    omni_usd.get_context = lambda: SimpleNamespace(
        get_stage=lambda: object(),
        get_stage_event_stream=lambda: SimpleNamespace(
            create_subscription_to_pop=lambda callback: SimpleNamespace(unsubscribe=lambda: None)
        ),
    )
    omni.physx = omni_physx
    omni.ext = omni_ext
    omni.timeline = omni_timeline
    omni.usd = omni_usd

    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.ext", omni_ext)
    monkeypatch.setitem(sys.modules, "omni.physx", omni_physx)
    monkeypatch.setitem(sys.modules, "omni.timeline", omni_timeline)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)

    fake_drone = types.ModuleType("robotics_experiments.drone")
    fake_drone.spawn_drone = lambda stage, config=None: None
    fake_drone.despawn_drone = lambda stage: None
    monkeypatch.setitem(sys.modules, "robotics_experiments.drone", fake_drone)
    return physx


def _install_fake_px4_bridge(monkeypatch, calls):
    # SimClockState transitions are pure Python with no Isaac deps and the host's sim-time
    # behavior is exercised here, so the fake reuses the real implementation.
    # Import it before the module is replaced in sys.modules below.
    from robotics_experiments.px4_bridge import (
        advance_sim_clock,
        bridge_time_us,
        create_sim_clock,
        sim_clock_is_settling,
        vehicle_log_context,
    )

    fake_px4_bridge = types.ModuleType("robotics_experiments.px4_bridge")
    fake_px4_bridge.__path__ = []

    class Px4MavlinkPort:
        def __init__(self, **kwargs):
            self.debug_log_path = kwargs["debug_log_path"]
            self.connected = False
            calls.append(("port", kwargs))

        def is_connected(self):
            return self.connected

        def read_status(self):
            return SimpleNamespace(is_connected=self.connected, is_armed=False)

    class PegasusQuadraticThrustModel:
        pass

    @dataclass(frozen=True)
    class Px4BridgeState:
        connected: bool
        port: Px4MavlinkPort
        thrust_model: PegasusQuadraticThrustModel
        pending: object | None = None

    def create_pegasus_thrust_model(**kwargs):
        calls.append(("thrust", kwargs))
        return PegasusQuadraticThrustModel()

    def create_px4_bridge(
        *,
        articulation_path,
        rotor_link_paths,
        thrust_model,
        port,
        max_tilt_angle_deg=180.0,
        rotor_spin_visual_scale=0.03,
        rotor_spin_signs=None,
    ):
        calls.append(
            (
                "bridge",
                articulation_path,
                rotor_link_paths,
                max_tilt_angle_deg,
                rotor_spin_visual_scale,
                rotor_spin_signs,
            )
        )
        return Px4BridgeState(connected=False, port=port, thrust_model=thrust_model)

    def start_px4_bridge(*, state):
        state.port.connected = True
        return replace(state, connected=True, pending=None)

    def stop_px4_bridge(*, state):
        state.port.connected = False
        return replace(state, connected=False, pending=None)

    def send_sensors(*, state, dt, sim_time_us):
        return replace(state, pending=(float(dt), int(sim_time_us)))

    def recv_and_apply(*, state):
        return replace(state, pending=None)

    def discover_px4_rotor_mapping(stage, articulation_path):
        calls.append(("discover", stage, articulation_path))
        return SimpleNamespace(
            rotor_link_paths=[
                "/World/Drone/body/propulsion/motor_4/prop",
                "/World/Drone/body/propulsion/motor_2/prop",
            ],
            rot_dir=[-1.0, 1.0],
        )

    class LockstepSettings:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_px4_bridge.LockstepSettings = LockstepSettings
    fake_px4_bridge.BridgeStatus = object
    fake_px4_bridge.DiscoveredRotorMapping = object
    fake_px4_bridge.PegasusQuadraticThrustModel = PegasusQuadraticThrustModel
    fake_px4_bridge.Px4BridgeState = Px4BridgeState
    fake_px4_bridge.Px4MavlinkPort = Px4MavlinkPort
    fake_px4_bridge.SimClockState = object
    fake_px4_bridge.advance_sim_clock = advance_sim_clock
    fake_px4_bridge.bridge_time_us = bridge_time_us
    fake_px4_bridge.create_px4_bridge = create_px4_bridge
    fake_px4_bridge.create_pegasus_thrust_model = create_pegasus_thrust_model
    fake_px4_bridge.create_sim_clock = create_sim_clock
    fake_px4_bridge.discover_px4_rotor_mapping = discover_px4_rotor_mapping
    fake_px4_bridge.recv_and_apply = recv_and_apply
    fake_px4_bridge.send_sensors = send_sensors
    fake_px4_bridge.sim_clock_is_settling = sim_clock_is_settling
    fake_px4_bridge.start_px4_bridge = start_px4_bridge
    fake_px4_bridge.stop_px4_bridge = stop_px4_bridge
    fake_px4_bridge.vehicle_log_context = vehicle_log_context
    monkeypatch.setitem(sys.modules, "robotics_experiments.px4_bridge", fake_px4_bridge)


def _import_bridge_host(monkeypatch):
    # Re-import fresh each test so the root contract lazily binds this test's
    # fake px4_bridge / Isaac modules installed in sys.modules above.
    for name in list(sys.modules):
        if name == "robotics_experiments.runtime" or name.startswith(
            "robotics_experiments.runtime."
        ):
            sys.modules.pop(name, None)
    return importlib.import_module("robotics_experiments.runtime")


def _config(**overrides):
    values = {
        "enabled": True,
        "articulation_path": "",
        "input_offset": [0.0, 0.0],
        "input_scaling": [1000.0, 1000.0],
        "zero_position_armed": [100.0, 100.0],
        "rotor_constant": [1.0, 1.0],
        "rolling_moment_coefficient": [1.0, 1.0],
        "min_rotor_velocity": [0.0, 0.0],
        "max_rotor_velocity": [1100.0, 1100.0],
        "linear_drag": [0.0, 0.0, 0.0],
        "connection_type": "tcpin",
        "connection_ip": "localhost",
        "connection_baseport": 4560,
        "lockstep": SimpleNamespace(
            lockstep_timeout_s=1.0,
            first_heartbeat_timeout_s=5.0,
            bootstrap_prime_ticks=20,
            first_actuator_timeout_s=5.0,
            max_sustained_actuator_lag_s=0.5,
            heartbeat_rate_hz=1.0,
        ),
        "startup_settle_time_s": 0.012,
        "gps_enabled": True,
        "gps_send_rate_hz": 250.0,
        "sensor_noise_enabled": True,
        "barometer_noise_std_pa": 1.0,
        "magnetometer_noise_scale": 1.0,
        "home_lat_deg": 47.3977,
        "home_lon_deg": 8.5456,
        "home_alt_m": 488.0,
        "debug_log_path": "",
        "debug_log_level": "DEBUG",
        "sensor_debug_log_interval": 50,
        "max_tilt_angle_deg": 60.0,
        "rotor_spin_visual_scale": 0.03,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _vehicle(*, vehicle_id=0, prim_name="Drone", spawn_xy=(0.0, 0.0), **px4_overrides):
    px4 = _config(**px4_overrides)
    return SimpleNamespace(
        id=vehicle_id,
        spawn=SimpleNamespace(prim_name=prim_name, position_xy=spawn_xy),
        bridge=px4,
        mavlink_port=px4.connection_baseport + vehicle_id,
        gcs_port=14550 + vehicle_id,
        debug_log_path=px4.debug_log_path.replace("{vehicle}", str(vehicle_id)),
        px4_sim_model="none_x500_v2",
        px4_sys_autostart=22000,
    )


def _make_host(module, vehicle, *, spawned_path="/World/Drone", physics_hz=250.0):
    """Construct a BridgeHost from the vehicle's params, as the orchestrator does."""
    return module.BridgeHost(
        module.BridgeHostParams(
            bridge=vehicle.bridge,
            vehicle_id=vehicle.id,
            mavlink_port=vehicle.mavlink_port,
            debug_log_path=vehicle.debug_log_path,
            articulation_path=vehicle.bridge.articulation_path or spawned_path,
            physics_hz=physics_hz,
        )
    )


def test_bridge_host_delays_px4_connection_until_startup_settle(monkeypatch):
    _install_fake_isaac(monkeypatch)
    _install_fake_px4_bridge(monkeypatch, [])
    module = _import_bridge_host(monkeypatch)

    # startup_settle_time_s defaults to 0.012 -> 12_000 us, i.e. three 0.004 s
    # ticks. The bridge must not come up until that window elapses.
    host = _make_host(module, _vehicle())

    host.start()

    # Observable through the public surface: not running / not connected while
    # settling, and no sim time accrues into the bridge before it starts.
    assert module.read_bridge_host_status(host=host).bridge_status is None

    _step(host, 0.004)
    _step(host, 0.004)

    assert module.read_bridge_host_status(host=host).bridge_status is None

    _step(host, 0.004)

    # Settle window elapsed: the bridge is now built and connected.
    assert module.read_bridge_host_status(host=host).bridge_status.is_connected is True

    _step(host, 0.004)

    # Public time is the same post-settle time sent to PX4 HIL.
    assert module.read_bridge_host_status(host=host).sim_time_us == 4000
    host.stop()
    assert module.read_bridge_host_status(host=host).sim_time_us == 4000


def test_bridge_host_rejects_send_before_start(monkeypatch):
    _install_fake_isaac(monkeypatch)
    _install_fake_px4_bridge(monkeypatch, [])
    module = _import_bridge_host(monkeypatch)
    host = _make_host(module, _vehicle())

    with pytest.raises(RuntimeError, match="send_phase called before start"):
        host.send_phase(0.004)


def test_bridge_host_starts_immediately_when_settle_time_is_zero(monkeypatch):
    _install_fake_isaac(monkeypatch)
    _install_fake_px4_bridge(monkeypatch, [])
    module = _import_bridge_host(monkeypatch)

    host = _make_host(module, _vehicle(startup_settle_time_s=0.0))

    host.start()

    # No settle delay: the bridge is up immediately, observable via public state.
    assert module.read_bridge_host_status(host=host).bridge_status.is_connected is True

    _step(host, 0.004)

    assert module.read_bridge_host_status(host=host).sim_time_us == 4000


def test_bridge_host_discards_bridge_pending_after_receive_error(monkeypatch):
    _install_fake_isaac(monkeypatch)
    _install_fake_px4_bridge(monkeypatch, [])
    module = _import_bridge_host(monkeypatch)
    host = _make_host(module, _vehicle(startup_settle_time_s=0.0))
    host.start()
    host.send_phase(0.004)

    bridge_host_module = importlib.import_module("robotics_experiments.runtime.bridge_host")
    receive_calls = 0

    def fail_receive(*, state):
        nonlocal receive_calls
        receive_calls += 1
        raise RuntimeError("injected receive failure")

    monkeypatch.setattr(bridge_host_module, "recv_and_apply", fail_receive)

    host.recv_phase()
    host.recv_phase()

    retained_bridge = host._state.bridge  # noqa: SLF001 - verify discarded transition state
    assert retained_bridge is not None
    assert retained_bridge.pending is None
    assert receive_calls == 1


def test_bridge_host_discards_bridge_pending_after_send_error(monkeypatch):
    _install_fake_isaac(monkeypatch)
    _install_fake_px4_bridge(monkeypatch, [])
    module = _import_bridge_host(monkeypatch)
    host = _make_host(module, _vehicle(startup_settle_time_s=0.0))
    host.start()
    host.send_phase(0.004)

    bridge_host_module = importlib.import_module("robotics_experiments.runtime.bridge_host")

    def fail_send(*, state, dt, sim_time_us):
        raise RuntimeError("injected send failure")

    receive_calls = 0

    def record_receive(*, state):
        nonlocal receive_calls
        receive_calls += 1
        return state

    monkeypatch.setattr(bridge_host_module, "send_sensors", fail_send)
    monkeypatch.setattr(bridge_host_module, "recv_and_apply", record_receive)

    host.send_phase(0.004)
    host.recv_phase()

    retained_bridge = host._state.bridge  # noqa: SLF001 - verify discarded transition state
    assert retained_bridge is not None
    assert retained_bridge.pending is None
    assert receive_calls == 0


def test_bridge_host_uses_discovered_rotor_mapping(monkeypatch):
    # Wiring-contract test. BridgeHost is an adapter: its observable output is
    # the Px4BridgeState it constructs from config + discovered rotors. There is no
    # public getter for those construction params (and adding one only for tests
    # is disallowed), so this asserts on the collaborator's construction kwargs.
    _install_fake_isaac(monkeypatch)
    calls = []
    _install_fake_px4_bridge(monkeypatch, calls)
    module = _import_bridge_host(monkeypatch)

    host = _make_host(module, _vehicle(startup_settle_time_s=0.0))

    host.start()

    port_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "port")
    thrust_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "thrust")
    bridge_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "bridge")

    assert port_call[1]["num_rotors"] == 2
    assert port_call[1]["sensor_noise_enabled"] is True
    assert port_call[1]["barometer_noise_std_pa"] == 1.0
    assert port_call[1]["magnetometer_noise_scale"] == 1.0
    assert port_call[1]["lockstep"].physics_rate_hz == 250.0
    assert port_call[1]["lockstep"].lockstep_timeout_s == 1.0
    assert thrust_call[1]["num_rotors"] == 2
    assert thrust_call[1]["rot_dir"] == [-1.0, 1.0]
    assert bridge_call[2] == [
        "/World/Drone/body/propulsion/motor_4/prop",
        "/World/Drone/body/propulsion/motor_2/prop",
    ]
    assert bridge_call[5] == [-1.0, 1.0]


def test_two_bridge_hosts_have_independent_endpoints(monkeypatch):
    _install_fake_isaac(monkeypatch)
    calls = []
    _install_fake_px4_bridge(monkeypatch, calls)
    module = _import_bridge_host(monkeypatch)

    host0 = _make_host(module, _vehicle(vehicle_id=0, startup_settle_time_s=0.0))
    host1 = _make_host(
        module,
        _vehicle(vehicle_id=1, startup_settle_time_s=0.0),
        spawned_path="/World/Drone1",
    )

    host0.start()
    host1.start()

    # Each host constructs its own port with the resolved per-vehicle
    # mavlink_port (baseport + id), so the simulator endpoints differ.
    port_calls = [call for call in calls if isinstance(call, tuple) and call[0] == "port"]
    assert len(port_calls) == 2
    endpoints = [c[1]["port"] for c in port_calls]
    assert endpoints == [4560, 4561]

    # Each host targets its own articulation prim.
    bridge_calls = [call for call in calls if isinstance(call, tuple) and call[0] == "bridge"]
    assert [c[1] for c in bridge_calls] == ["/World/Drone", "/World/Drone1"]

    assert module.read_bridge_host_status(host=host0).bridge_status is not None
    assert module.read_bridge_host_status(host=host1).bridge_status is not None
    host0.stop()
    assert module.read_bridge_host_status(host=host0).bridge_status is None
    assert module.read_bridge_host_status(host=host1).bridge_status is not None
    assert module.read_bridge_host_status(host=host0).sim_time_us == 0


def test_bridge_host_group_runs_all_sends_before_all_polls(monkeypatch):
    physx = _install_fake_isaac(monkeypatch)
    calls = []
    _install_fake_px4_bridge(monkeypatch, calls)
    module = _import_bridge_host(monkeypatch)

    order: list[tuple[str, int]] = []

    class _FakeHost:
        def __init__(self, name):
            self.name = name

        def start(self):
            order.append(("start", self.name))

        def stop(self):
            order.append(("stop", self.name))

        def send_phase(self, dt):
            order.append(("send", self.name))

        def recv_phase(self):
            order.append(("recv", self.name))

    group = module.BridgeHostGroup([_FakeHost(0), _FakeHost(1)])
    group.start()

    # The group owns exactly one physics-step subscription for the whole fleet.
    assert physx.subscribe_count == 1
    assert physx.callback is not None
    assert order == [("start", 0), ("start", 1)]

    order.clear()
    physx.callback(0.004)

    # Every vehicle's send leg runs before any vehicle's blocking poll leg, so a
    # single stalled PX4 cannot starve the others' sensor sends.
    assert order == [("send", 0), ("send", 1), ("recv", 0), ("recv", 1)]

    group.stop()
    assert order[-2:] == [("stop", 0), ("stop", 1)]

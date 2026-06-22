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
    # SimClock is pure Python with no Isaac deps and the host's sim-time
    # behavior is exercised here, so the fake reuses the real implementation.
    # Import it before the module is replaced in sys.modules below.
    from robotics_experiments.px4_bridge._logging import vehicle_log_context
    from robotics_experiments.px4_bridge.sim_clock import SimClock

    fake_px4_bridge = types.ModuleType("robotics_experiments.px4_bridge")

    class Px4MavlinkPort:
        def __init__(self, **kwargs):
            calls.append(("port", kwargs))

    class PegasusQuadraticThrustModel:
        def __init__(self, **kwargs):
            calls.append(("thrust", kwargs))

    class Px4Bridge:
        def __init__(
            self,
            *,
            articulation_path,
            rotor_link_paths,
            thrust_model,
            port,
            max_tilt_angle_deg=180.0,
            rotor_spin_visual_scale=0.03,
        ):
            self.is_connected = True
            self.step_times = []
            calls.append(
                (
                    "bridge",
                    articulation_path,
                    rotor_link_paths,
                    max_tilt_angle_deg,
                    rotor_spin_visual_scale,
                )
            )

        def start(self):
            calls.append("bridge_start")

        def stop(self):
            calls.append("bridge_stop")

        def send_sensors(self, dt, sim_time_us):
            self.step_times.append(sim_time_us)
            calls.append(("send", dt, sim_time_us))

        def recv_and_apply(self):
            calls.append("recv")

    def discover_px4_rotor_mapping(stage, articulation_path):
        calls.append(("discover", stage, articulation_path))
        return SimpleNamespace(
            rotor_link_paths=[
                "/World/Drone/body/propulsion/motor_4/prop",
                "/World/Drone/body/propulsion/motor_2/prop",
            ],
            rot_dir=[-1.0, 1.0],
            num_rotors=2,
        )

    class LockstepSettings:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_px4_bridge.Px4MavlinkPort = Px4MavlinkPort
    fake_px4_bridge.PegasusQuadraticThrustModel = PegasusQuadraticThrustModel
    fake_px4_bridge.Px4Bridge = Px4Bridge
    fake_px4_bridge.LockstepSettings = LockstepSettings
    fake_px4_bridge.discover_px4_rotor_mapping = discover_px4_rotor_mapping
    fake_px4_bridge.SimClock = SimClock
    fake_px4_bridge.vehicle_log_context = vehicle_log_context
    monkeypatch.setitem(sys.modules, "robotics_experiments.px4_bridge", fake_px4_bridge)


def _import_bridge_host(monkeypatch):
    for name in list(sys.modules):
        if name == "DroneSpawner_python" or name.startswith("DroneSpawner_python."):
            sys.modules.pop(name, None)
    extension_root = Path(__file__).resolve().parents[1] / "isaac/runtime/extensions/drone_spawner"
    monkeypatch.syspath_prepend(str(extension_root))
    return importlib.import_module("DroneSpawner_python.bridge_host")


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
    """Build a ResolvedVehicle-shaped object around a PX4 config for BridgeHost.

    The BridgeHost reads only ``id`` / ``px4`` / ``mavlink_port`` /
    ``debug_log_path`` off the resolved vehicle; the rest mirror the production
    shape for completeness. Derived fields follow ``id``.
    """
    px4 = _config(**px4_overrides)
    return SimpleNamespace(
        id=vehicle_id,
        spawn_xy=spawn_xy,
        drone=SimpleNamespace(prim_name=prim_name),
        px4=px4,
        mavlink_port=px4.connection_baseport + vehicle_id,
        px4_instance_index=vehicle_id,
        debug_log_path=px4.debug_log_path.replace("{vehicle}", str(vehicle_id)),
        prim_name=prim_name,
    )


def test_bridge_host_delays_px4_connection_until_startup_settle(monkeypatch):
    physx = _install_fake_isaac(monkeypatch)
    calls = []
    _install_fake_px4_bridge(monkeypatch, calls)
    module = _import_bridge_host(monkeypatch)

    host = module.BridgeHost(articulation_path="/World/Drone", vehicle=_vehicle(), physics_hz=250.0)

    # The host no longer self-subscribes; the group owns the subscription and
    # drives the two-leg API. start() only sets up the clock here (settle > 0).
    host.start()

    assert physx.callback is None
    assert calls == []

    _step(host, 0.004)
    _step(host, 0.004)

    assert calls == []

    _step(host, 0.004)

    assert "bridge_start" in calls
    assert not any(call[0] == "send" for call in calls if isinstance(call, tuple))

    _step(host, 0.004)

    assert ("send", 0.004, 4000) in calls
    assert "recv" in calls


def test_bridge_host_starts_immediately_when_settle_time_is_zero(monkeypatch):
    _install_fake_isaac(monkeypatch)
    calls = []
    _install_fake_px4_bridge(monkeypatch, calls)
    module = _import_bridge_host(monkeypatch)

    host = module.BridgeHost(
        articulation_path="/World/Drone",
        vehicle=_vehicle(startup_settle_time_s=0.0),
        physics_hz=250.0,
    )

    host.start()
    _step(host, 0.004)

    assert "bridge_start" in calls
    assert ("send", 0.004, 4000) in calls
    assert "recv" in calls


def test_bridge_host_uses_discovered_rotor_mapping(monkeypatch):
    _install_fake_isaac(monkeypatch)
    calls = []
    _install_fake_px4_bridge(monkeypatch, calls)
    module = _import_bridge_host(monkeypatch)

    host = module.BridgeHost(
        articulation_path="/World/Drone",
        vehicle=_vehicle(startup_settle_time_s=0.0),
        physics_hz=250.0,
    )

    host.start()

    port_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "port")
    thrust_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "thrust")
    bridge_call = next(call for call in calls if isinstance(call, tuple) and call[0] == "bridge")

    assert port_call[1]["num_rotors"] == 2
    assert port_call[1]["sensor_noise_enabled"] is True
    assert port_call[1]["lockstep"].physics_rate_hz == 250.0
    assert port_call[1]["lockstep"].lockstep_timeout_s == 1.0
    assert thrust_call[1]["num_rotors"] == 2
    assert thrust_call[1]["rot_dir"] == [-1.0, 1.0]
    assert bridge_call[2] == [
        "/World/Drone/body/propulsion/motor_4/prop",
        "/World/Drone/body/propulsion/motor_2/prop",
    ]


def test_two_bridge_hosts_have_independent_endpoints(monkeypatch):
    _install_fake_isaac(monkeypatch)
    calls = []
    _install_fake_px4_bridge(monkeypatch, calls)
    module = _import_bridge_host(monkeypatch)

    host0 = module.BridgeHost(
        articulation_path="/World/Drone",
        vehicle=_vehicle(vehicle_id=0, startup_settle_time_s=0.0),
        physics_hz=250.0,
    )
    host1 = module.BridgeHost(
        articulation_path="/World/Drone1",
        vehicle=_vehicle(vehicle_id=1, startup_settle_time_s=0.0),
        physics_hz=250.0,
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

    assert host0.is_running
    assert host1.is_running
    host0.stop()
    assert not host0.is_running
    assert host1.is_running


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

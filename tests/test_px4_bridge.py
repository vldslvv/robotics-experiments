import sys
import types

import numpy as np
import pytest

from robotics_experiments.px4_bridge import LinearThrustModel, Px4Bridge, Px4MavlinkPort
from robotics_experiments.px4_bridge.vehicle_state import VehicleState


class _FakeArticulationIO:
    def __init__(self, state: VehicleState):
        self.state = state
        self.initialized = False
        self.shutdown_called = False
        self.applied_forces = []

    @property
    def is_initialized(self):
        return self.initialized

    def initialize(self):
        self.initialized = True

    def shutdown(self):
        self.shutdown_called = True

    def read_state(self, sim_time_us: int):
        self.state.sim_time_us = sim_time_us
        return self.state

    def apply_forces(self, forces_n: np.ndarray):
        self.applied_forces.append(np.asarray(forces_n, dtype=np.float64))


class _FakeFlightStack:
    def __init__(self, commands):
        self.commands = np.asarray(commands, dtype=np.float64)
        self.started = False
        self.stopped = False
        self.states = []

    @property
    def num_rotors(self):
        return self.commands.shape[0]

    @property
    def is_connected(self):
        return self.started and not self.stopped

    @property
    def is_armed(self):
        return bool(np.any(self.commands))

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def step(self, state: VehicleState):
        self.states.append(state)
        return self.commands


def test_px4_bridge_tick_reads_state_and_applies_forces():
    state = VehicleState.hover_at(sim_time_us=0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    assert port.started is True
    assert port.stopped is True
    assert io.initialized is True
    assert io.shutdown_called is True
    assert port.states[0].sim_time_us == 4_000
    assert np.allclose(io.applied_forces[0], [0.0, 2.0, 4.0, 8.0])


def test_px4_bridge_validates_rotor_count_against_port():
    with pytest.raises(ValueError, match="rotor_link_paths has 3 entries"):
        Px4Bridge(
            articulation_path="/World/Drone",
            rotor_link_paths=["m1", "m2", "m3"],
            thrust_model=LinearThrustModel(num_rotors=4, max_force_n=8.0),
            port=_FakeFlightStack([0.0, 0.0, 0.0, 0.0]),
            articulation_io=_FakeArticulationIO(VehicleState.hover_at()),
        )


class _FakeConnection:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.recv_calls = []
        self.mav = types.SimpleNamespace(heartbeat_send=lambda *args: None)

    def recv_match(self, *, blocking=False, timeout=None):
        self.recv_calls.append((blocking, timeout))
        if self.messages:
            return self.messages.pop(0)
        return None


class _FakePegasusBackend:
    def __init__(self, config):
        self.config = config
        self._connection = _FakeConnection()
        self._received_first_hearbeat = True
        self._received_first_actuator = False
        self._enable_lockstep = True
        self._received_actuator = False
        self._input_reference = [0.0] * config.config["num_rotors"]
        self.sensor_times = []
        self.gps_times = []
        self.imu_updates = []
        self.mag_updates = []
        self.bar_updates = []
        self.gps_updates = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        self._connection = None

    def send_heartbeat(self):
        pass

    def wait_for_first_hearbeat(self):
        self._received_first_hearbeat = True

    def update_imu_data(self, data):
        self.imu_updates.append(data)

    def update_mag_data(self, data):
        self.mag_updates.append(data)

    def update_bar_data(self, data):
        self.bar_updates.append(data)

    def update_gps_data(self, data):
        self.gps_updates.append(data)

    def send_sensor_msgs(self, sim_time_us):
        self.sensor_times.append(sim_time_us)

    def send_gps_msgs(self, sim_time_us):
        self.gps_times.append(sim_time_us)

    def input_reference(self):
        return self._input_reference

    def handle_control(self, time_usec, controls, mode, flags):
        self._input_reference = list(controls[: self.config.config["num_rotors"]])


class _FakePegasusConfig:
    def __init__(self, config):
        self.config = config


def _install_fake_pegasus(monkeypatch):
    module_name = "pegasus.simulator.logic.backends.px4_mavlink_backend"
    fake_module = types.ModuleType(module_name)
    fake_module.PX4MavlinkBackend = _FakePegasusBackend
    fake_module.PX4MavlinkBackendConfig = _FakePegasusConfig

    for package in [
        "pegasus",
        "pegasus.simulator",
        "pegasus.simulator.logic",
        "pegasus.simulator.logic.backends",
    ]:
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    pymavlink = types.ModuleType("pymavlink")
    mavutil = types.ModuleType("pymavlink.mavutil")
    mavutil.mavlink = types.SimpleNamespace(MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS=93)
    pymavlink.mavutil = mavutil
    monkeypatch.setitem(sys.modules, "pymavlink", pymavlink)
    monkeypatch.setitem(sys.modules, "pymavlink.mavutil", mavutil)


def test_mavlink_port_uses_sim_time_and_gates_gps(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_send_rate_hz=5.0)
    backend = port._backend
    backend._connection = _FakeConnection()

    port.start()
    port.step(VehicleState.hover_at(sim_time_us=100_000))
    port.step(VehicleState.hover_at(sim_time_us=150_000))
    port.step(VehicleState.hover_at(sim_time_us=300_000))

    assert backend.started is True
    assert backend.sensor_times == [100_000, 150_000, 300_000]
    assert backend.gps_times == [100_000, 300_000]
    assert len(backend.imu_updates) == 3
    assert len(backend.mag_updates) == 3
    assert len(backend.bar_updates) == 3
    assert len(backend.gps_updates) == 3


def test_mavlink_port_timeout_marks_disconnected(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=0.0)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._received_first_actuator = True

    port.start()
    commands = port.step(VehicleState.hover_at(sim_time_us=100_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.is_connected is False
    assert backend._connection.recv_calls

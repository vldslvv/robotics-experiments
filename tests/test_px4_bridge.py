import itertools
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
        self.hil_sensor_sends = []
        self.mav = types.SimpleNamespace(
            heartbeat_send=lambda *args: None,
            hil_sensor_send=lambda *args: self.hil_sensor_sends.append(args),
        )

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
        self.first_heartbeat_available = True
        self._received_first_actuator = False
        self._enable_lockstep = True
        self._received_actuator = False
        self._input_reference = [0.0] * config.config["num_rotors"]
        self._sensor_data = types.SimpleNamespace(
            new_imu_data=False,
            received_first_imu=False,
            xacc=0.0,
            yacc=0.0,
            zacc=0.0,
            xgyro=0.0,
            ygyro=0.0,
            zgyro=0.0,
            new_mag_data=False,
            xmag=0.0,
            ymag=0.0,
            zmag=0.0,
            new_bar_data=False,
            abs_pressure=0.0,
            diff_pressure=0.0,
            pressure_alt=0.0,
            temperature=0.0,
            new_press_data=False,
        )
        self.sensor_times = []
        self.gps_times = []
        self.imu_updates = []
        self.mag_updates = []
        self.bar_updates = []
        self.gps_updates = []
        self.heartbeat_sends = 0
        self.first_heartbeat_waits = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        self._connection = None

    def send_heartbeat(self):
        self.heartbeat_sends += 1

    def wait_for_first_hearbeat(self):
        self.first_heartbeat_waits += 1
        if self.first_heartbeat_available:
            self._received_first_hearbeat = True

    def update_imu_data(self, data):
        self.imu_updates.append(data)
        self._sensor_data.xacc = data["linear_acceleration"][0]
        self._sensor_data.yacc = data["linear_acceleration"][1]
        self._sensor_data.zacc = data["linear_acceleration"][2]
        self._sensor_data.xgyro = data["angular_velocity"][0]
        self._sensor_data.ygyro = data["angular_velocity"][1]
        self._sensor_data.zgyro = data["angular_velocity"][2]
        self._sensor_data.new_imu_data = True
        self._sensor_data.received_first_imu = True

    def update_mag_data(self, data):
        self.mag_updates.append(data)
        self._sensor_data.xmag = data["magnetic_field"][0]
        self._sensor_data.ymag = data["magnetic_field"][1]
        self._sensor_data.zmag = data["magnetic_field"][2]
        self._sensor_data.new_mag_data = True

    def update_bar_data(self, data):
        self.bar_updates.append(data)
        self._sensor_data.temperature = data["temperature"]
        self._sensor_data.abs_pressure = data["absolute_pressure"]
        self._sensor_data.pressure_alt = data["pressure_altitude"]
        self._sensor_data.new_bar_data = True

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
    assert [args[0] for args in backend._connection.hil_sensor_sends] == [
        100_000,
        150_000,
        300_000,
    ]
    assert backend.gps_times == [100_000, 300_000]
    assert len(backend.imu_updates) == 3
    assert len(backend.mag_updates) == 3
    assert len(backend.bar_updates) == 3
    assert len(backend.gps_updates) == 3
    first_hil_sensor = backend._connection.hil_sensor_sends[0]
    assert first_hil_sensor[13] == 25.0
    assert first_hil_sensor[14] == 8191


def test_mavlink_port_can_disable_gps(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_enabled=False)
    backend = port._backend
    backend._connection = _FakeConnection()

    port.start()
    port.step(VehicleState.hover_at(sim_time_us=100_000))

    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert backend.gps_times == []
    assert backend.gps_updates == []


def test_mavlink_port_skips_duplicate_sim_time_in_lockstep(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4)
    backend = port._backend
    actuator_msg = types.SimpleNamespace(
        id=93,
        time_usec=100_000,
        controls=[0.1, 0.2, 0.3, 0.4],
        mode=0,
        flags=0,
    )
    backend._connection = _FakeConnection([actuator_msg])
    backend._received_first_actuator = True

    port.start()
    port.step(VehicleState.hover_at(sim_time_us=100_000))
    commands = port.step(VehicleState.hover_at(sim_time_us=100_000))

    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert len(backend._connection.recv_calls) == 1
    assert np.allclose(commands, [0.1, 0.2, 0.3, 0.4])
    assert port.is_connected is True
    assert port._last_disconnect_reason is None
    assert port._sensor_send_count == 1
    assert port._actuator_receive_count == 1


def test_mavlink_port_uses_configured_heartbeat_rate(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    import robotics_experiments.px4_bridge.mavlink as mavlink_module

    times = itertools.chain(
        [100.0, 100.0, 100.4, 100.4, 100.6, 100.6],
        itertools.repeat(100.6),
    )
    monkeypatch.setattr(mavlink_module.time, "time", lambda: next(times))

    port = Px4MavlinkPort(num_rotors=4, heartbeat_rate_hz=2.0)
    backend = port._backend
    backend._connection = _FakeConnection()

    port.start()
    port.step(VehicleState.hover_at(sim_time_us=100_000))
    port.step(VehicleState.hover_at(sim_time_us=200_000))
    port.step(VehicleState.hover_at(sim_time_us=300_000))

    assert backend.heartbeat_sends == 2


def test_mavlink_port_first_heartbeat_timeout_marks_disconnected(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, first_heartbeat_timeout_s=0.0)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._received_first_hearbeat = False
    backend.first_heartbeat_available = False

    port.start()
    commands = port.step(VehicleState.hover_at(sim_time_us=100_000))

    assert np.allclose(commands, np.zeros(4))
    assert backend.first_heartbeat_waits == 1
    assert port.is_connected is False
    assert backend.stopped is True
    assert port._last_disconnect_reason == "first heartbeat timeout"


def test_mavlink_port_blocks_for_lockstep_actuator(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=1.0)
    backend = port._backend
    actuator_msg = types.SimpleNamespace(
        id=93,
        time_usec=100_000,
        controls=[0.1, 0.2, 0.3, 0.4],
        mode=0,
        flags=0,
    )
    connection = _FakeConnection([actuator_msg])
    backend._connection = connection
    backend._received_first_actuator = True

    port.start()
    commands = port.step(VehicleState.hover_at(sim_time_us=100_000))

    assert np.allclose(commands, [0.1, 0.2, 0.3, 0.4])
    assert port.is_connected is True
    assert backend.stopped is False
    assert backend._connection is connection
    assert port._actuator_receive_count == 1
    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert port._last_disconnect_reason is None


def test_mavlink_port_lockstep_timeout_marks_disconnected(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=0.0)
    backend = port._backend
    connection = _FakeConnection()
    backend._connection = connection
    backend._received_first_actuator = True

    port.start()
    commands = port.step(VehicleState.hover_at(sim_time_us=300_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.is_connected is False
    assert backend.stopped is True
    assert backend._connection is None
    assert len(connection.recv_calls) == 1
    assert port._last_disconnect_reason == "actuator polling timed out after 0.000 seconds"

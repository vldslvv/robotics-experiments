import itertools
import importlib
import logging
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from robotics_experiments.px4_bridge import LinearThrustModel, Px4Bridge, Px4MavlinkPort
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

_MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE = 129
_PX4_BRIDGE_EXTENSION_ROOT = (
    Path(__file__).resolve().parents[1] / "isaac/runtime/extensions/px4_sitl_brigde"
)


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
            new_gps_data=False,
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
        self._sensor_data.new_gps_data = True

    def send_sensor_msgs(self, sim_time_us):
        self.sensor_times.append(sim_time_us)
        fields_updated = 0
        if self._sensor_data.new_imu_data:
            fields_updated |= 7 | 56
            self._sensor_data.new_imu_data = False
        if self._sensor_data.new_mag_data:
            fields_updated |= 448
            self._sensor_data.new_mag_data = False
        if self._sensor_data.new_bar_data:
            fields_updated |= 6656
            self._sensor_data.new_bar_data = False
        if self._sensor_data.new_press_data:
            fields_updated |= 1024
            self._sensor_data.new_press_data = False
        self._connection.mav.hil_sensor_send(
            sim_time_us,
            self._sensor_data.xacc,
            self._sensor_data.yacc,
            self._sensor_data.zacc,
            self._sensor_data.xgyro,
            self._sensor_data.ygyro,
            self._sensor_data.zgyro,
            self._sensor_data.xmag,
            self._sensor_data.ymag,
            self._sensor_data.zmag,
            self._sensor_data.abs_pressure,
            self._sensor_data.diff_pressure,
            self._sensor_data.pressure_alt,
            self._sensor_data.temperature,
            fields_updated,
        )

    def send_gps_msgs(self, sim_time_us):
        if not self._sensor_data.new_gps_data:
            return
        self._sensor_data.new_gps_data = False
        self.gps_times.append(sim_time_us)

    def input_reference(self):
        return self._input_reference

    def handle_control(self, time_usec, controls, mode, flags):
        num_rotors = self.config.config["num_rotors"]
        if mode == _MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE:
            offsets = self.config.config["input_offset"]
            scaling = self.config.config["input_scaling"]
            zero_armed = self.config.config["zero_position_armed"]
            self._input_reference = [
                (controls[i] + offsets[i]) * scaling[i] + zero_armed[i]
                for i in range(num_rotors)
            ]
        else:
            self._input_reference = [0.0] * num_rotors


class _FakePegasusConfig:
    def __init__(self, config):
        self.config = config


def _install_fake_pegasus(monkeypatch):
    monkeypatch.syspath_prepend(str(_PX4_BRIDGE_EXTENSION_ROOT))
    module_name = "pegasus.simulator.logic.backends.px4_mavlink_backend"
    fake_module = types.ModuleType(module_name)
    fake_module.PX4MavlinkBackend = _FakePegasusBackend
    fake_module.PX4MavlinkBackendConfig = _FakePegasusConfig

    monkeypatch.setitem(sys.modules, module_name, fake_module)

    pymavlink = types.ModuleType("pymavlink")
    mavutil = types.ModuleType("pymavlink.mavutil")
    mavutil.mavlink = types.SimpleNamespace(MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS=93)
    pymavlink.mavutil = mavutil
    monkeypatch.setitem(sys.modules, "pymavlink", pymavlink)
    monkeypatch.setitem(sys.modules, "pymavlink.mavutil", mavutil)


def test_vendored_pegasus_sensor_utils_are_importable(monkeypatch):
    monkeypatch.syspath_prepend(str(_PX4_BRIDGE_EXTENSION_ROOT))

    geo_mag_utils = importlib.import_module("pegasus.simulator.logic.sensors.geo_mag_utils")

    lat, lon = geo_mag_utils.reprojection(
        np.array([0.0, 0.0, 0.0]),
        np.radians(47.397742),
        np.radians(8.545594),
    )
    assert np.degrees(lat) == pytest.approx(47.397742)
    assert np.degrees(lon) == pytest.approx(8.545594)
    assert geo_mag_utils.get_mag_strength(47.397742, 8.545594) > 0.0


def test_vendored_pegasus_backend_sends_barometer_temperature(monkeypatch):
    monkeypatch.syspath_prepend(str(_PX4_BRIDGE_EXTENSION_ROOT))
    sys.modules.pop("pegasus.simulator.logic.backends.px4_mavlink_backend", None)

    carb = types.ModuleType("carb")
    carb.log_info = lambda message: None
    carb.log_warn = lambda message: None
    monkeypatch.setitem(sys.modules, "carb", carb)

    pymavlink = types.ModuleType("pymavlink")
    mavutil = types.ModuleType("pymavlink.mavutil")
    mavutil.mavlink = types.SimpleNamespace(
        MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS=93,
        MAV_TYPE_GENERIC=0,
        MAV_AUTOPILOT_INVALID=8,
        MAV_MODE_FLAG_SAFETY_ARMED=128,
    )
    mavutil.mavlink_connection = lambda *args, **kwargs: None
    pymavlink.mavutil = mavutil
    monkeypatch.setitem(sys.modules, "pymavlink", pymavlink)
    monkeypatch.setitem(sys.modules, "pymavlink.mavutil", mavutil)

    module = importlib.import_module("pegasus.simulator.logic.backends.px4_mavlink_backend")
    backend = module.PX4MavlinkBackend(
        module.PX4MavlinkBackendConfig({"num_rotors": 4, "px4_autolaunch": False})
    )
    backend._connection = _FakeConnection()
    backend.update_bar_data(
        {
            "absolute_pressure": 954.609,
            "pressure_altitude": 488.0,
            "temperature": 11.827,
        }
    )

    backend.send_sensor_msgs(123_000)

    hil_sensor = backend._connection.hil_sensor_sends[0]
    assert hil_sensor[13] == pytest.approx(11.827)
    assert hil_sensor[13] != backend._sensor_data.altitude
    assert hil_sensor[14] == 6656


def test_mavlink_port_uses_sim_time_and_gates_gps(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_send_rate_hz=5.0)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

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
    assert first_hil_sensor[12] == pytest.approx(488.0, abs=1.0)
    assert first_hil_sensor[13] == pytest.approx(11.827, abs=0.01)
    assert first_hil_sensor[14] == 7167


def test_mavlink_port_pegasus_mag_and_baro_are_not_bit_stale(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_send_rate_hz=5.0)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    port.step(VehicleState.hover_at(sim_time_us=100_000))
    port.step(VehicleState.hover_at(sim_time_us=104_000))

    assert len(backend.mag_updates) == 2
    assert len(backend.bar_updates) == 2
    first_mag = np.asarray(backend.mag_updates[0]["magnetic_field"], dtype=np.float64)
    second_mag = np.asarray(backend.mag_updates[1]["magnetic_field"], dtype=np.float64)
    first_baro = backend.bar_updates[0]
    second_baro = backend.bar_updates[1]

    assert not np.array_equal(first_mag, second_mag)
    assert first_baro["absolute_pressure"] != second_baro["absolute_pressure"]


def test_mavlink_port_writes_sensor_diagnostics_to_file(monkeypatch, tmp_path):
    _install_fake_pegasus(monkeypatch)
    log_path = tmp_path / "px4_bridge.log"
    port = Px4MavlinkPort(
        num_rotors=4,
        debug_log_path=str(log_path),
        debug_log_level="DEBUG",
        sensor_debug_log_interval=1,
    )
    port._backend._connection = _FakeConnection()
    port._backend._enable_lockstep = False

    port.start()
    port.step(VehicleState.hover_at(sim_time_us=100_000))
    port.stop()

    log_text = log_path.read_text()
    assert "PX4 HIL_SENSOR pre-send" in log_text
    assert "mask=7167" in log_text
    assert "fresh={imu:True mag:True baro:True" in log_text
    assert "PX4 HIL_SENSOR post-send" in log_text


def test_bridge_log_path_uses_px4_style_timestamp_and_fresh_handler(tmp_path):
    import robotics_experiments.px4_bridge.mavlink as mavlink_module

    logger = logging.getLogger("robotics_experiments.px4_bridge")
    first_path = tmp_path / "old_bridge.log"
    second_template = str(tmp_path / "{date}" / "{time}_bridge.log")

    try:
        resolved_first = mavlink_module._configure_file_logging(
            str(first_path),
            "DEBUG",
            now=datetime(2026, 6, 10, 21, 9, 59),
        )
        logger.debug("old run marker")

        resolved_second = mavlink_module._configure_file_logging(
            second_template,
            "DEBUG",
            now=datetime(2026, 6, 10, 21, 10, 5),
        )
        logger.debug("new run marker")

        assert resolved_first == first_path.resolve()
        assert resolved_second == (tmp_path / "2026-06-10" / "21_10_05_bridge.log").resolve()
        assert "old run marker" in first_path.read_text()
        assert "new run marker" not in first_path.read_text()
        assert "new run marker" in resolved_second.read_text()
    finally:
        for handler in list(logger.handlers):
            if getattr(handler, mavlink_module._FILE_LOG_HANDLER_ATTR, None) is not None:
                logger.removeHandler(handler)
                handler.close()


def test_mavlink_port_can_disable_gps(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_enabled=False)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

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
        mode=_MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
        flags=0,
    )
    backend._connection = _FakeConnection([actuator_msg])
    backend._received_first_actuator = True

    port.start()
    port.step(VehicleState.hover_at(sim_time_us=100_000))
    commands = port.step(VehicleState.hover_at(sim_time_us=100_000))

    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert len(backend._connection.recv_calls) == 2
    assert np.allclose(commands, [200.0, 300.0, 400.0, 500.0])
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
    monkeypatch.setattr(mavlink_module, "time", types.SimpleNamespace(time=lambda: next(times)))

    port = Px4MavlinkPort(num_rotors=4, heartbeat_rate_hz=2.0)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

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
        mode=_MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
        flags=0,
    )
    connection = _FakeConnection([actuator_msg])
    backend._connection = connection
    backend._received_first_actuator = True

    port.start()
    commands = port.step(VehicleState.hover_at(sim_time_us=100_000))

    assert np.allclose(commands, [200.0, 300.0, 400.0, 500.0])
    assert port.is_connected is True
    assert backend.stopped is False
    assert backend._connection is connection
    assert port._actuator_receive_count == 1
    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert port._last_disconnect_reason is None


def test_mavlink_port_does_not_wait_before_first_lockstep_actuator(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=0.0)
    backend = port._backend
    connection = _FakeConnection()
    backend._connection = connection
    backend._received_first_actuator = False

    port.start()
    commands = port.step(VehicleState.hover_at(sim_time_us=100_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.is_connected is True
    assert backend.stopped is False
    assert port._actuator_receive_count == 0
    assert port._last_actuator_receive_us is None
    assert connection.recv_calls == [(False, 0.0)]
    assert port._last_disconnect_reason is None


def test_mavlink_port_drains_actuator_backlog_to_newest(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=1.0)
    backend = port._backend
    old_msg = types.SimpleNamespace(
        id=93,
        time_usec=100_000,
        controls=[0.1, 0.1, 0.1, 0.1],
        mode=_MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
        flags=0,
    )
    newest_msg = types.SimpleNamespace(
        id=93,
        time_usec=112_000,
        controls=[0.2, 0.3, 0.4, 0.5],
        mode=_MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
        flags=0,
    )
    connection = _FakeConnection([old_msg, newest_msg])
    backend._connection = connection
    backend._received_first_actuator = True

    port.start()
    commands = port.step(VehicleState.hover_at(sim_time_us=112_000))

    assert np.allclose(commands, [300.0, 400.0, 500.0, 600.0])
    assert port._actuator_receive_count == 2
    assert port._last_actuator_receive_us == 112_000
    assert len(connection.recv_calls) == 3
    assert connection.recv_calls[0][0] is True
    assert connection.recv_calls[1:] == [(False, 0.0), (False, 0.0)]
    assert port.is_connected is True


def test_mavlink_port_disconnects_when_drained_actuator_is_stale(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=1.0)
    backend = port._backend
    stale_msg = types.SimpleNamespace(
        id=93,
        time_usec=100_000,
        controls=[0.5, 0.5, 0.5, 0.5],
        mode=_MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
        flags=0,
    )
    connection = _FakeConnection([stale_msg])
    backend._connection = connection
    backend._received_first_actuator = True
    port._actuator_lag_baseline_us = 0
    port._last_sensor_period_us = 4_000

    port.start()
    port._actuator_lag_baseline_us = 0
    port._last_sensor_period_us = 4_000
    commands = port.step(VehicleState.hover_at(sim_time_us=140_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.is_connected is False
    assert backend.stopped is True
    assert backend._connection is None
    assert port._last_disconnect_reason.startswith("actuator timestamp stale:")
    assert "lag_us=40000" in port._last_disconnect_reason
    assert "limit_us=20000" in port._last_disconnect_reason


def test_mavlink_port_accepts_bounded_bootstrap_actuator_lag(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=1.0)
    backend = port._backend
    bootstrap_msg = types.SimpleNamespace(
        id=93,
        time_usec=52_000,
        controls=[0.0, 0.0, 0.0, 0.0],
        mode=1,
        flags=1,
    )
    connection = _FakeConnection([bootstrap_msg])
    backend._connection = connection
    backend._received_first_actuator = True
    port._last_sensor_period_us = 4_000

    port.start()
    port._last_sensor_period_us = 4_000
    commands = port.step(VehicleState.hover_at(sim_time_us=148_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.is_connected is True
    assert backend.stopped is False
    assert port._last_actuator_receive_us == 52_000
    assert port._actuator_lag_baseline_us == 96_000
    assert port._last_disconnect_reason is None


def test_mavlink_port_accepts_slow_first_bootstrap_actuator(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=1.0)
    backend = port._backend
    bootstrap_msg = types.SimpleNamespace(
        id=93,
        time_usec=68_000,
        controls=[0.0, 0.0, 0.0, 0.0],
        mode=1,
        flags=1,
    )
    connection = _FakeConnection([bootstrap_msg])
    backend._connection = connection
    backend._received_first_actuator = True
    port._last_sensor_period_us = 4_000

    port.start()
    port._last_sensor_period_us = 4_000
    commands = port.step(VehicleState.hover_at(sim_time_us=972_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.is_connected is True
    assert backend.stopped is False
    assert port._last_actuator_receive_us == 68_000
    assert port._actuator_lag_baseline_us == 904_000
    assert port._last_disconnect_reason is None


def test_mavlink_port_accepts_actuator_within_freshness_limit(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep_timeout_s=1.0)
    backend = port._backend
    fresh_msg = types.SimpleNamespace(
        id=93,
        time_usec=124_000,
        controls=[0.0, 0.25, 0.5, 1.0],
        mode=_MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
        flags=0,
    )
    connection = _FakeConnection([fresh_msg])
    backend._connection = connection
    backend._received_first_actuator = True
    port._last_sensor_period_us = 4_000

    port.start()
    port._last_sensor_period_us = 4_000
    commands = port.step(VehicleState.hover_at(sim_time_us=140_000))

    assert np.allclose(commands, [100.0, 350.0, 600.0, 1100.0])
    assert port.is_connected is True
    assert backend.stopped is False
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


def _quat_pitch_deg(angle_deg: float) -> np.ndarray:
    """Return a [w, x, y, z] quaternion for a pitch rotation of angle_deg."""
    import math

    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)


def _quat_yaw_deg(angle_deg: float) -> np.ndarray:
    """Return a [w, x, y, z] quaternion for a yaw rotation of angle_deg."""
    import math

    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def test_tilt_angle_deg_level():
    from robotics_experiments.px4_bridge.frames import tilt_angle_deg

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert tilt_angle_deg(q) == pytest.approx(0.0, abs=1e-10)


def test_tilt_angle_deg_45_pitch():
    from robotics_experiments.px4_bridge.frames import tilt_angle_deg

    q = _quat_pitch_deg(45.0)
    assert tilt_angle_deg(q) == pytest.approx(45.0, abs=1e-10)


def test_tilt_angle_deg_inverted():
    from robotics_experiments.px4_bridge.frames import tilt_angle_deg

    q = _quat_pitch_deg(180.0)
    assert tilt_angle_deg(q) == pytest.approx(180.0, abs=1e-10)


def test_tilt_angle_deg_pure_yaw_is_zero():
    from robotics_experiments.px4_bridge.frames import tilt_angle_deg

    q = _quat_yaw_deg(90.0)
    assert tilt_angle_deg(q) == pytest.approx(0.0, abs=1e-10)


def test_bridge_tilt_cutoff_zeros_forces():
    state = VehicleState.hover_at(sim_time_us=0)
    state.orientation_flu_enu_wxyz = _quat_pitch_deg(80.0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.5, 0.5, 0.5, 0.5])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    assert len(io.applied_forces) == 1
    assert np.allclose(io.applied_forces[0], [0.0, 0.0, 0.0, 0.0])
    assert bridge._tilt_cutoff_count == 1


def test_bridge_tilt_cutoff_passes_through_within_limit():
    state = VehicleState.hover_at(sim_time_us=0)
    state.orientation_flu_enu_wxyz = _quat_pitch_deg(30.0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    assert len(io.applied_forces) == 1
    assert np.allclose(io.applied_forces[0], [0.0, 2.0, 4.0, 8.0])
    assert bridge._tilt_cutoff_count == 0


def test_bridge_tilt_cutoff_disabled_when_none():
    state = VehicleState.hover_at(sim_time_us=0)
    state.orientation_flu_enu_wxyz = _quat_pitch_deg(170.0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=None,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    assert len(io.applied_forces) == 1
    assert bridge._tilt_cutoff_count == 0

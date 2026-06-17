import itertools
import logging
import types
from datetime import datetime

import numpy as np
from _fakes import (
    _MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
    _FakeConnection,
    _install_fake_pegasus,
    lockstep_exchange,
)

from robotics_experiments.px4_bridge import LockstepSettings, Px4MavlinkPort
from robotics_experiments.px4_bridge.vehicle_state import VehicleState


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
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert len(backend._connection.recv_calls) == 2
    assert np.allclose(commands, [200.0, 300.0, 400.0, 500.0])
    assert port.is_connected is True
    assert port._last_disconnect_reason is None
    assert port._sensor_send_count == 1
    assert port._actuator_receive_count == 1


def test_mavlink_port_status_snapshots_lockstep_state(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4)
    backend = port._backend
    actuator_msg = types.SimpleNamespace(
        id=93,
        time_usec=96_000,
        controls=[0.1, 0.2, 0.3, 0.4],
        mode=_MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
        flags=0,
    )
    backend._connection = _FakeConnection([actuator_msg])
    backend._received_first_actuator = True

    port.start()
    assert port.status.sensor_send_count == 0
    assert port.status.actuator_lag_us is None

    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    status = port.status
    assert status.is_connected is True
    assert status.is_armed is True
    assert status.first_heartbeat_received is True
    assert status.sensor_send_count == 1
    assert status.actuator_receive_count == 1
    assert status.last_sensor_send_us == 100_000
    assert status.last_actuator_receive_us == 96_000
    assert status.actuator_lag_us == 4_000
    assert status.last_disconnect_reason is None


def test_mavlink_port_uses_configured_heartbeat_rate(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    import robotics_experiments.px4_bridge.mavlink as mavlink_module

    times = itertools.chain(
        [100.0, 100.0, 100.4, 100.4, 100.6, 100.6],
        itertools.repeat(100.6),
    )
    monkeypatch.setattr(mavlink_module, "time", types.SimpleNamespace(time=lambda: next(times)))

    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(heartbeat_rate_hz=2.0))
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=200_000))
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=300_000))

    assert backend.heartbeat_sends == 2


def test_mavlink_port_first_heartbeat_timeout_marks_disconnected(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(first_heartbeat_timeout_s=0.0))
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._received_first_hearbeat = False
    backend.first_heartbeat_available = False

    port.start()
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    assert np.allclose(commands, np.zeros(4))
    assert backend.first_heartbeat_waits == 1
    assert port.is_connected is False
    assert backend.stopped is True
    assert port._last_disconnect_reason == "first heartbeat timeout"


def test_mavlink_port_blocks_for_lockstep_actuator(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(lockstep_timeout_s=1.0))
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
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    assert np.allclose(commands, [200.0, 300.0, 400.0, 500.0])
    assert port.is_connected is True
    assert backend.stopped is False
    assert backend._connection is connection
    assert port._actuator_receive_count == 1
    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert port._last_disconnect_reason is None


def test_mavlink_port_lockstep_sends_current_sensor_before_wait(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(lockstep_timeout_s=1.0))
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
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    assert connection.events[0] == ("hil_sensor_send", 100_000)
    assert connection.events[1][0:2] == ("recv_match", True)
    assert connection.events[1][2] <= 1.0
    assert connection.events[1][2] > 0.0
    assert port._last_actuator_receive_us == 100_000
    assert np.allclose(commands, [200.0, 300.0, 400.0, 500.0])
    assert port.is_connected is True


def test_mavlink_port_does_not_wait_before_first_lockstep_actuator(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(lockstep_timeout_s=0.0))
    backend = port._backend
    connection = _FakeConnection()
    backend._connection = connection
    backend._received_first_actuator = False

    port.start()
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.is_connected is True
    assert backend.stopped is False
    assert port._actuator_receive_count == 0
    assert port._last_actuator_receive_us is None
    assert connection.recv_calls == [(False, 0.0)]
    assert port._last_disconnect_reason is None


def test_mavlink_port_lockstep_timeout_marks_disconnected(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(lockstep_timeout_s=0.0))
    backend = port._backend
    connection = _FakeConnection()
    backend._connection = connection
    backend._received_first_actuator = True

    port.start()
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=300_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.is_connected is False
    assert backend.stopped is True
    assert backend._connection is None
    assert len(connection.recv_calls) == 1
    assert port._last_disconnect_reason == "actuator polling timed out after 0.000 seconds"

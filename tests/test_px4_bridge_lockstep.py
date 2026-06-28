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

from robotics_experiments.px4_bridge import (
    LockstepSettings,
    Px4MavlinkPort,
)
from robotics_experiments.px4_bridge._logging import vehicle_log_context
from robotics_experiments.px4_bridge.vehicle_state import VehicleState


def test_bridge_log_path_renders_px4_style_timestamp_and_vehicle_token(tmp_path):
    import robotics_experiments.px4_bridge.mavlink as mavlink_module

    logger = logging.getLogger("robotics_experiments.px4_bridge")
    first_path = tmp_path / "old_bridge.log"
    second_template = str(tmp_path / "{date}" / "{time}_v{vehicle}_bridge.log")

    try:
        resolved_first = mavlink_module._configure_file_logging(
            str(first_path),
            "DEBUG",
            0,
            now=datetime(2026, 6, 10, 21, 9, 59),
        )
        resolved_second = mavlink_module._configure_file_logging(
            second_template,
            "DEBUG",
            1,
            now=datetime(2026, 6, 10, 21, 10, 5),
        )

        assert resolved_first == first_path.resolve()
        # The {vehicle} token renders the vehicle id into the filename.
        assert resolved_second == (tmp_path / "2026-06-10" / "21_10_05_v1_bridge.log").resolve()
    finally:
        for handler in list(logger.handlers):
            if getattr(handler, mavlink_module._FILE_LOG_HANDLER_ATTR, None) is not None:
                logger.removeHandler(handler)
                handler.close()


def test_per_vehicle_handlers_coexist_and_isolate_by_log_context(tmp_path):
    """A multi-vehicle process keeps each vehicle's records in its own file.

    The two per-vehicle handlers must coexist (the second vehicle must not steal
    the first's handler), and ``vehicle_log_context`` routes each record to only
    the matching vehicle's file. Untagged records (no context) go to every file.
    """
    import robotics_experiments.px4_bridge.mavlink as mavlink_module

    logger = logging.getLogger("robotics_experiments.px4_bridge")
    v0_path = tmp_path / "v0_bridge.log"
    v1_path = tmp_path / "v1_bridge.log"

    try:
        mavlink_module._configure_file_logging(str(v0_path), "DEBUG", 0)
        mavlink_module._configure_file_logging(str(v1_path), "DEBUG", 1)

        with vehicle_log_context(0):
            logger.debug("vehicle zero marker")
        with vehicle_log_context(1):
            logger.debug("vehicle one marker")
        logger.debug("untagged marker")

        v0_text = v0_path.read_text()
        v1_text = v1_path.read_text()

        assert "vehicle zero marker" in v0_text
        assert "vehicle zero marker" not in v1_text
        assert "vehicle one marker" in v1_text
        assert "vehicle one marker" not in v0_text
        # Untagged records are not lost: they reach every per-vehicle handler.
        assert "untagged marker" in v0_text
        assert "untagged marker" in v1_text
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
    assert port.status.last_disconnect_reason is None
    assert port.status.sensor_send_count == 1
    assert port.status.actuator_receive_count == 1


def test_mavlink_port_send_pushes_sensors_without_blocking_then_poll_returns_actuator(monkeypatch):
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
    # The send leg publishes HIL_SENSOR and returns None (a poll is owed) without
    # performing the blocking actuator receive.
    result = port.send(VehicleState.hover_at(sim_time_us=100_000))

    assert result is None
    assert [args[0] for args in connection.hil_sensor_sends] == [100_000]
    assert port.status.sensor_send_count == 1
    assert connection.recv_calls == []
    assert port.status.actuator_receive_count == 0

    # The poll leg performs the blocking recv and returns the paired actuator.
    commands = port.poll().controls

    assert np.allclose(commands, [200.0, 300.0, 400.0, 500.0])
    assert port.status.actuator_receive_count == 1
    assert port.is_connected is True
    assert port.status.last_disconnect_reason is None


def test_mavlink_port_send_returns_zeros_without_poll_when_disconnected(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4)

    # Never started -> not connected: send takes the no-block path and returns
    # concrete zero commands so the caller skips the poll.
    result = port.send(VehicleState.hover_at(sim_time_us=100_000))

    assert result is not None
    assert np.allclose(result.controls, np.zeros(4))
    assert port.status.sensor_send_count == 0


def test_mavlink_port_send_returns_references_without_poll_on_non_increasing_time(monkeypatch):
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
    backend._connection = _FakeConnection([actuator_msg])
    backend._received_first_actuator = True

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    # A second send at the same (non-increasing) timestamp neither sends nor
    # blocks; it returns the current references directly, so no poll is owed.
    result = port.send(VehicleState.hover_at(sim_time_us=100_000))

    assert result is not None
    assert np.allclose(result.controls, [200.0, 300.0, 400.0, 500.0])
    assert port.status.sensor_send_count == 1


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
    assert port.status.last_disconnect_reason == "first heartbeat timeout"


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
    assert port.status.actuator_receive_count == 1
    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert port.status.last_disconnect_reason is None


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
    assert port.status.last_actuator_receive_us == 100_000
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
    assert port.status.actuator_receive_count == 0
    assert port.status.last_actuator_receive_us is None
    assert connection.recv_calls == [(False, 0.0)]
    assert port.status.last_disconnect_reason is None


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
    assert port.status.last_disconnect_reason == "actuator polling timed out after 0.000 seconds"

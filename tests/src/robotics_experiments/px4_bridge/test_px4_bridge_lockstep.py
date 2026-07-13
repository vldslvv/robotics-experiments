import logging
import types

import numpy as np

from robotics_experiments.px4_bridge import (
    LockstepSettings,
    Px4MavlinkPort,
    VehicleState,
    vehicle_log_context,
)
from tests.support.bridge_fakes import (
    _MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
    _FakeConnection,
    _install_fake_pegasus,
    lockstep_exchange,
)


def _handler_belongs_to_tmp_path(handler, tmp_path) -> bool:
    base_filename = getattr(handler, "baseFilename", None)
    return base_filename is not None and str(base_filename).startswith(str(tmp_path))


def test_bridge_log_path_renders_px4_style_timestamp_and_vehicle_token(tmp_path, monkeypatch):
    _install_fake_pegasus(monkeypatch)
    logger = logging.getLogger("robotics_experiments.px4_bridge")
    first_path = tmp_path / "v0_bridge.log"
    second_template = str(tmp_path / "{date}" / "{time}_v{vehicle}_bridge.log")

    try:
        Px4MavlinkPort(num_rotors=4, vehicle_id=0, debug_log_path=str(first_path))
        Px4MavlinkPort(num_rotors=4, vehicle_id=1, debug_log_path=second_template)

        assert first_path.exists()
        # The {date}, {time}, and {vehicle} tokens render into the filename.
        rendered = list(tmp_path.glob("*/*_v1_bridge.log"))
        assert len(rendered) == 1
    finally:
        for handler in list(logger.handlers):
            if _handler_belongs_to_tmp_path(handler, tmp_path):
                logger.removeHandler(handler)
                handler.close()


def test_per_vehicle_handlers_coexist_and_isolate_by_log_context(tmp_path, monkeypatch):
    """A multi-vehicle process keeps each vehicle's records in its own file.

    The two per-vehicle handlers must coexist (the second vehicle must not steal
    the first's handler), and ``vehicle_log_context`` routes each record to only
    the matching vehicle's file. Untagged records (no context) go to every file.
    """
    _install_fake_pegasus(monkeypatch)
    logger = logging.getLogger("robotics_experiments.px4_bridge")
    v0_path = tmp_path / "v0_bridge.log"
    v1_path = tmp_path / "v1_bridge.log"

    try:
        Px4MavlinkPort(num_rotors=4, vehicle_id=0, debug_log_path=str(v0_path))
        Px4MavlinkPort(num_rotors=4, vehicle_id=1, debug_log_path=str(v1_path))

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
            if _handler_belongs_to_tmp_path(handler, tmp_path):
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
    assert port.read_status().is_connected is True
    assert port.read_status().last_disconnect_reason is None
    assert port.read_status().sensor_send_count == 1
    assert port.read_status().actuator_receive_count == 1


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
    assert port.read_status().sensor_send_count == 1
    assert connection.recv_calls == []
    assert port.read_status().actuator_receive_count == 0

    # The poll leg performs the blocking recv and returns the paired actuator.
    commands = port.poll().controls

    assert np.allclose(commands, [200.0, 300.0, 400.0, 500.0])
    assert port.read_status().actuator_receive_count == 1
    assert port.read_status().is_connected is True
    assert port.read_status().last_disconnect_reason is None


def test_mavlink_port_send_returns_zeros_without_poll_when_disconnected(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4)

    # Never started -> not connected: send takes the no-block path and returns
    # concrete zero commands so the caller skips the poll.
    result = port.send(VehicleState.hover_at(sim_time_us=100_000))

    assert result is not None
    assert np.allclose(result.controls, np.zeros(4))
    assert port.read_status().sensor_send_count == 0


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
    assert port.read_status().sensor_send_count == 1


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
    assert port.read_status().sensor_send_count == 0
    assert port.read_status().actuator_lag_us is None

    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    status = port.read_status()
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
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(heartbeat_rate_hz=1e-9))
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=200_000))
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=300_000))

    assert backend.heartbeat_sends == 1


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
    assert port.read_status().is_connected is False
    assert backend.stopped is True
    assert port.read_status().last_disconnect_reason == "first heartbeat timeout"


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
    assert port.read_status().is_connected is True
    assert backend.stopped is False
    assert backend._connection is connection
    assert port.read_status().actuator_receive_count == 1
    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert port.read_status().last_disconnect_reason is None


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
    assert port.read_status().last_actuator_receive_us == 100_000
    assert np.allclose(commands, [200.0, 300.0, 400.0, 500.0])
    assert port.read_status().is_connected is True


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
    assert port.read_status().is_connected is True
    assert backend.stopped is False
    assert port.read_status().actuator_receive_count == 0
    assert port.read_status().last_actuator_receive_us is None
    assert connection.recv_calls == [(False, 0.0)]
    assert port.read_status().last_disconnect_reason is None


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
    assert port.read_status().is_connected is False
    assert backend.stopped is True
    assert backend._connection is None
    assert len(connection.recv_calls) == 1
    assert (
        port.read_status().last_disconnect_reason
        == "actuator polling timed out after 0.000 seconds"
    )

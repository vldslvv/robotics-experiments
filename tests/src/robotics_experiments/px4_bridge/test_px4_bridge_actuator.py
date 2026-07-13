import types

import numpy as np

from robotics_experiments.px4_bridge import (
    LockstepSettings,
    Px4MavlinkPort,
    VehicleState,
)
from tests.support.bridge_fakes import (
    _MAV_MODE_FLAG_SAFETY_ARMED_PLUS_ONE,
    _FakeConnection,
    _install_fake_pegasus,
    lockstep_exchange,
)


def test_mavlink_port_drains_actuator_backlog_to_newest(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(lockstep_timeout_s=1.0))
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
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=112_000))

    # Backlog drained to the newest packet (never the stale queued one): the
    # public commands plus status are the observable, and the recv history (a
    # documented seam) confirms one blocking poll then non-blocking drains.
    assert np.allclose(commands, [300.0, 400.0, 500.0, 600.0])
    assert port.read_status().actuator_receive_count == 2
    assert port.read_status().last_actuator_receive_us == 112_000
    assert len(connection.recv_calls) == 3
    assert connection.recv_calls[0][0] is True
    assert connection.recv_calls[1:] == [(False, 0.0), (False, 0.0)]
    assert port.read_status().is_connected is True


def test_mavlink_port_disconnects_when_actuator_lag_exceeds_sustained_limit(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    # 0.02s = 20_000us absolute ceiling on steady-state actuator lag.
    port = Px4MavlinkPort(
        num_rotors=4,
        lockstep=LockstepSettings(lockstep_timeout_s=1.0, max_sustained_actuator_lag_s=0.02),
    )
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

    port.start()
    # lag = 140_000 - 100_000 = 40_000us > 20_000us ceiling.
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=140_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.read_status().is_connected is False
    assert backend.stopped is True
    assert backend._connection is None
    reason = port.read_status().last_disconnect_reason
    assert reason.startswith("actuator lag exceeded sustained limit")
    assert "lag_us=40000" in reason
    assert "max_us=20000" in reason


def test_mavlink_port_primes_then_blocks_for_first_actuator(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    # Allow 3 priming sends, then block; first_actuator_timeout_s=0.0 makes the
    # post-prime blocking wait time out immediately when no actuator arrives.
    port = Px4MavlinkPort(
        num_rotors=4,
        lockstep=LockstepSettings(
            lockstep_timeout_s=1.0,
            bootstrap_prime_ticks=3,
            first_actuator_timeout_s=0.0,
        ),
    )
    backend = port._backend
    connection = _FakeConnection()
    backend._connection = connection
    backend._received_first_actuator = False

    port.start()

    # The first 3 sends are priming: non-blocking, no backlog grows past them.
    for tick in range(1, 4):
        commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=tick * 4_000))
        assert np.allclose(commands, np.zeros(4))
        assert port.read_status().is_connected is True
        assert connection.recv_calls[-1] == (False, 0.0)

    assert port.read_status().actuator_receive_count == 0

    # The 4th send exceeds the prime budget: the bridge now blocks for the first
    # actuator instead of free-running, and times out (no actuator) -> disconnect.
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=4 * 4_000))

    assert np.allclose(commands, np.zeros(4))
    assert connection.recv_calls[-1][0] is True
    assert port.read_status().is_connected is False
    assert backend.stopped is True
    assert (
        port.read_status().last_disconnect_reason
        == "actuator polling timed out after 0.000 seconds"
    )


def test_mavlink_port_first_actuator_lag_is_bounded_by_prime_budget(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(
        num_rotors=4,
        lockstep=LockstepSettings(
            lockstep_timeout_s=1.0,
            bootstrap_prime_ticks=3,
            first_actuator_timeout_s=1.0,
        ),
    )
    backend = port._backend
    connection = _FakeConnection()
    backend._connection = connection
    backend._received_first_actuator = False

    port.start()

    # Prime 3 ticks with no actuator available (non-blocking).
    for tick in range(1, 4):
        lockstep_exchange(port, VehicleState.hover_at(sim_time_us=tick * 4_000))
    assert port.read_status().actuator_receive_count == 0

    # On the first blocking tick, PX4's first actuator is delivered, stamped at
    # the current sim time: the lag is ~0, not multi-second, because the bridge
    # never raced ahead.
    connection.messages.append(
        types.SimpleNamespace(
            id=93,
            time_usec=4 * 4_000,
            controls=[0.0, 0.0, 0.0, 0.0],
            mode=1,
            flags=1,
        )
    )
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=4 * 4_000))

    assert port.read_status().actuator_receive_count == 1
    assert port.read_status().last_actuator_receive_us == 16_000
    assert port.read_status().is_connected is True
    assert port.read_status().last_disconnect_reason is None


def test_mavlink_port_steady_state_lag_stays_one_tick(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(lockstep_timeout_s=1.0))
    backend = port._backend
    connection = _FakeConnection()
    backend._connection = connection
    backend._received_first_actuator = True

    port.start()

    # Each tick delivers exactly one actuator stamped one tick (4_000us) behind
    # the current sim time; the lag must stay at one tick and the port connected.
    for tick in range(2, 6):
        sim_time_us = tick * 4_000
        connection.messages.append(
            types.SimpleNamespace(
                id=93,
                time_usec=sim_time_us - 4_000,
                controls=[0.0, 0.0, 0.0, 0.0],
                mode=1,
                flags=1,
            )
        )
        lockstep_exchange(port, VehicleState.hover_at(sim_time_us=sim_time_us))
        assert sim_time_us - port.read_status().last_actuator_receive_us == 4_000
        assert port.read_status().is_connected is True

    assert port.read_status().last_disconnect_reason is None


def test_mavlink_port_accepts_actuator_within_freshness_limit(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, lockstep=LockstepSettings(lockstep_timeout_s=1.0))
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

    port.start()
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=140_000))

    assert np.allclose(commands, [100.0, 350.0, 600.0, 1100.0])
    assert port.read_status().is_connected is True
    assert backend.stopped is False
    assert port.read_status().last_disconnect_reason is None

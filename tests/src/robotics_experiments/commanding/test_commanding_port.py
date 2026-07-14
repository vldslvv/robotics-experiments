"""Unit tests for ``MavlinkCommandPort`` feedback parsing (no pymavlink needed).

These guard the MAVLink ``poll_feedback`` parsing that the standalone scenario
relies on: the failsafe ``system_status`` mapping (the off-by-one that treats
``MAV_STATE_ACTIVE`` as a failsafe would falsely abort every takeoff), the
``COMMAND_ACK`` parsing, the arm-statustext acceptance, and global-position
telemetry. They drive ``poll_feedback`` over a fake connection, so no pymavlink
connection is opened.
"""

import math
import types

import pytest

from robotics_experiments.commanding import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    CommandPortStatus,
    MavlinkCommandPort,
    PositionSetpoint,
)
from robotics_experiments.commanding.port import _fold_feedback

_MAV_FRAME_GLOBAL_INT = 5
_POSITION_TARGET_TYPEMASK_POS_YAW = 0b1001_1111_1000
_POSITION_TARGET_TYPEMASK_YAW_IGNORE = 0b0100_0000_0000

_SETPOINT_ARG_NAMES = (
    "time_boot_ms",
    "target_system",
    "target_component",
    "frame",
    "type_mask",
    "lat",
    "lon",
    "alt",
    "vx",
    "vy",
    "vz",
    "afx",
    "afy",
    "afz",
    "yaw",
    "yaw_rate",
)


def _msg(msg_type, *, src_system=1, src_component=1, **fields):
    ns = types.SimpleNamespace(**fields)
    ns.get_type = lambda: msg_type
    ns.get_srcSystem = lambda: src_system
    ns.get_srcComponent = lambda: src_component
    return ns


class _RecordingMav:
    """Records the ``set_position_target_global_int_send`` calls a port emits."""

    def __init__(self):
        self.setpoints = []

    def set_position_target_global_int_send(self, *args):
        self.setpoints.append(dict(zip(_SETPOINT_ARG_NAMES, args, strict=True)))


class _FakeConn:
    def __init__(self, messages):
        self._messages = list(messages)
        self.mav = _RecordingMav()

    def recv_match(self, *, blocking=False):
        return self._messages.pop(0) if self._messages else None


def _port_with(messages):
    return MavlinkCommandPort(
        gcs_endpoint="udp:127.0.0.1:14550",
        heartbeat_period_s=1.0,
        connection=_FakeConn(messages),
    )


def _feedback_state():
    return CommandPortStatus(
        is_handshaked=False,
        failsafe=False,
        failsafe_system_status=0,
        latest_global_position=None,
    )


def test_feedback_fold_is_pure_across_handshake_and_failsafe_transitions():
    initial = _feedback_state()

    active, ack = _fold_feedback(initial, _msg("HEARTBEAT", system_status=4))
    critical, ack = _fold_feedback(active, _msg("HEARTBEAT", system_status=5))
    recovered, ack = _fold_feedback(critical, _msg("HEARTBEAT", system_status=3))

    assert initial == _feedback_state()
    assert active.is_handshaked and not active.failsafe
    assert critical.failsafe and critical.failsafe_system_status == 5
    assert not recovered.failsafe and recovered.failsafe_system_status == 3
    assert ack is None


def test_feedback_fold_returns_command_and_arm_statustext_acks():
    state = _feedback_state()

    unchanged, rejected = _fold_feedback(
        state,
        _msg("COMMAND_ACK", command=MAV_CMD_COMPONENT_ARM_DISARM, result=1),
    )
    unchanged_again, accepted = _fold_feedback(
        state, _msg("STATUSTEXT", text="WARN Armed by external command")
    )

    assert unchanged is state and unchanged_again is state
    assert rejected is not None and not rejected.accepted
    assert accepted is not None and accepted.accepted


def test_feedback_fold_ignores_zero_gps_and_accepts_valid_position():
    state = _feedback_state()

    unchanged, _ = _fold_feedback(state, _msg("GPS_RAW_INT", lat=0, lon=0, alt=0))
    positioned, _ = _fold_feedback(
        unchanged,
        _msg("GLOBAL_POSITION_INT", lat=473977000, lon=85456000, alt=488000),
    )

    assert unchanged is state
    assert positioned.latest_global_position == pytest.approx((47.3977, 8.5456, 488.0))


def _handshaked_port():
    # A HEARTBEAT teaches the port its target system/component (sets is_handshaked).
    # Returns the port and the recording MAV so a test reads the emitted setpoints
    # off the connection seam (no private port state).
    conn = _FakeConn([_msg("HEARTBEAT", system_status=4, src_system=1, src_component=1)])
    port = MavlinkCommandPort(
        gcs_endpoint="udp:127.0.0.1:14550", heartbeat_period_s=1.0, connection=conn
    )
    port.poll_feedback()
    return port, conn.mav


def test_active_heartbeat_is_not_failsafe_and_handshakes():
    # MAV_STATE_ACTIVE (4) is the normal armed/flying state, not a failsafe.
    port = _port_with([_msg("HEARTBEAT", system_status=4, src_system=1, src_component=1)])
    port.poll_feedback()
    assert port.read_status().failsafe is False
    assert port.read_status().is_handshaked is True


def test_standby_heartbeat_is_not_failsafe():
    port = _port_with([_msg("HEARTBEAT", system_status=3)])
    port.poll_feedback()
    assert port.read_status().failsafe is False


def test_critical_emergency_termination_heartbeats_are_failsafe():
    # MAV_STATE_CRITICAL=5, MAV_STATE_EMERGENCY=6, MAV_STATE_FLIGHT_TERMINATION=8.
    for status in (5, 6, 8):
        port = _port_with([_msg("HEARTBEAT", system_status=status)])
        port.poll_feedback()
        port_status = port.read_status()
        assert port_status.failsafe is True, f"system_status={status} should be failsafe"
        assert port_status.failsafe_system_status == status


def test_command_ack_is_parsed():
    port = _port_with([_msg("COMMAND_ACK", command=MAV_CMD_COMPONENT_ARM_DISARM, result=0)])
    ack = port.poll_feedback()
    assert ack is not None
    assert ack.command_id == MAV_CMD_COMPONENT_ARM_DISARM
    assert ack.accepted is True


def test_rejected_command_ack_is_not_accepted():
    port = _port_with([_msg("COMMAND_ACK", command=MAV_CMD_COMPONENT_ARM_DISARM, result=2)])
    ack = port.poll_feedback()
    assert ack is not None
    assert ack.accepted is False


def test_arm_statustext_synthesizes_accepted_ack():
    port = _port_with([_msg("STATUSTEXT", text="WARN Armed by external command")])
    ack = port.poll_feedback()
    assert ack is not None
    assert ack.command_id == MAV_CMD_COMPONENT_ARM_DISARM
    assert ack.accepted is True


def test_global_position_is_surfaced():
    port = _port_with([_msg("GLOBAL_POSITION_INT", lat=473977000, lon=85456000, alt=488000)])
    port.poll_feedback()
    position = port.read_status().latest_global_position
    assert position is not None
    latitude_deg, longitude_deg, altitude_m = position
    assert abs(latitude_deg - 47.3977) < 1e-4
    assert abs(longitude_deg - 8.5456) < 1e-4
    assert abs(altitude_m - 488.0) < 1e-6


def test_zero_global_position_is_ignored_until_fix():
    port = _port_with([_msg("GLOBAL_POSITION_INT", lat=0, lon=0, alt=0)])
    port.poll_feedback()
    assert port.read_status().latest_global_position is None


def test_send_setpoint_emits_global_position_target_with_yaw():
    port, mav = _handshaked_port()
    port.send_setpoint(
        PositionSetpoint(
            latitude_deg=47.0, longitude_deg=8.0, altitude_m=490.0, yaw_rad=math.pi / 2.0
        )
    )
    call = mav.setpoints[-1]
    assert call["frame"] == _MAV_FRAME_GLOBAL_INT
    assert call["lat"] == 470_000_000  # 47.0 deg in 1e7 units
    assert call["lon"] == 80_000_000
    assert call["alt"] == pytest.approx(490.0)
    assert call["yaw"] == pytest.approx(math.pi / 2.0)
    # Position + yaw: ignore velocity, acceleration, and yaw-rate only.
    assert call["type_mask"] == _POSITION_TARGET_TYPEMASK_POS_YAW


def test_send_setpoint_nan_yaw_sets_yaw_ignore():
    port, mav = _handshaked_port()
    port.send_setpoint(
        PositionSetpoint(latitude_deg=47.0, longitude_deg=8.0, altitude_m=490.0, yaw_rad=math.nan)
    )
    call = mav.setpoints[-1]
    assert call["type_mask"] == (
        _POSITION_TARGET_TYPEMASK_POS_YAW | _POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )
    assert call["yaw"] == pytest.approx(0.0)


def test_send_setpoint_before_handshake_raises():
    port = _port_with([])  # no heartbeat -> target system unknown
    with pytest.raises(RuntimeError, match="learned the PX4 target system"):
        port.send_setpoint(
            PositionSetpoint(latitude_deg=47.0, longitude_deg=8.0, altitude_m=490.0, yaw_rad=0.0)
        )

"""Unit tests for the deploy-time OFFBOARD setpoint serializers (no pymavlink).

Guard that ``MavlinkCommandPort`` serializes the velocity / attitude-rate setpoints
to the correct MAVLink messages, frames, and type masks -- the wire contract the
F1 action levels deploy onto. Driven over a fake connection, so no socket opens.
"""

import types

import pytest

from robotics_experiments.commanding.commands import (
    AttitudeRateSetpoint,
    VelocitySetpoint,
)
from robotics_experiments.commanding.port import (
    _ATTITUDE_TARGET_TYPEMASK_RATES,
    _MAV_FRAME_BODY_NED,
    _VELOCITY_TARGET_TYPEMASK,
    MavlinkCommandPort,
)

_LOCAL_NED_ARG_NAMES = (
    "time_boot_ms",
    "target_system",
    "target_component",
    "frame",
    "type_mask",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "afx",
    "afy",
    "afz",
    "yaw",
    "yaw_rate",
)

_ATTITUDE_ARG_NAMES = (
    "time_boot_ms",
    "target_system",
    "target_component",
    "type_mask",
    "q",
    "body_roll_rate",
    "body_pitch_rate",
    "body_yaw_rate",
    "thrust",
)


class _RecordingMav:
    def __init__(self):
        self.velocity_setpoints = []
        self.attitude_targets = []

    def set_position_target_local_ned_send(self, *args):
        self.velocity_setpoints.append(dict(zip(_LOCAL_NED_ARG_NAMES, args, strict=True)))

    def set_attitude_target_send(self, *args):
        self.attitude_targets.append(dict(zip(_ATTITUDE_ARG_NAMES, args, strict=True)))


class _FakeConn:
    def __init__(self, messages):
        self._messages = list(messages)
        self.mav = _RecordingMav()

    def recv_match(self, *, blocking=False):
        return self._messages.pop(0) if self._messages else None


def _msg(msg_type, *, src_system=1, src_component=1, **fields):
    ns = types.SimpleNamespace(**fields)
    ns.get_type = lambda: msg_type
    ns.get_srcSystem = lambda: src_system
    ns.get_srcComponent = lambda: src_component
    return ns


def _handshaked_port():
    conn = _FakeConn([_msg("HEARTBEAT", system_status=4)])
    port = MavlinkCommandPort(
        gcs_endpoint="udp:127.0.0.1:14550", heartbeat_period_s=1.0, connection=conn
    )
    port.poll_feedback()
    return port, conn.mav


def test_send_velocity_setpoint_emits_body_ned_velocity():
    port, mav = _handshaked_port()
    port.send_velocity_setpoint(
        VelocitySetpoint(forward_mps=1.5, right_mps=-0.5, down_mps=-0.25, yaw_rate_radps=0.3)
    )

    (emitted,) = mav.velocity_setpoints
    assert emitted["frame"] == _MAV_FRAME_BODY_NED
    assert emitted["type_mask"] == _VELOCITY_TARGET_TYPEMASK
    assert (emitted["vx"], emitted["vy"], emitted["vz"]) == (1.5, -0.5, -0.25)
    assert emitted["yaw_rate"] == 0.3
    # Position and acceleration fields are zeroed (ignored by the type mask).
    assert (emitted["x"], emitted["y"], emitted["z"]) == (0.0, 0.0, 0.0)
    assert (emitted["afx"], emitted["afy"], emitted["afz"]) == (0.0, 0.0, 0.0)


def test_send_attitude_target_emits_thrust_and_body_rates():
    port, mav = _handshaked_port()
    port.send_attitude_target(
        AttitudeRateSetpoint(
            thrust=0.55, roll_rate_radps=0.1, pitch_rate_radps=-0.2, yaw_rate_radps=0.05
        )
    )

    (emitted,) = mav.attitude_targets
    assert emitted["type_mask"] == _ATTITUDE_TARGET_TYPEMASK_RATES
    assert emitted["thrust"] == 0.55
    assert emitted["body_roll_rate"] == 0.1
    assert emitted["body_pitch_rate"] == -0.2
    assert emitted["body_yaw_rate"] == 0.05


def test_deploy_setpoints_before_handshake_raise():
    conn = _FakeConn([])
    port = MavlinkCommandPort(
        gcs_endpoint="udp:127.0.0.1:14550", heartbeat_period_s=1.0, connection=conn
    )
    with pytest.raises(RuntimeError, match="velocity setpoint"):
        port.send_velocity_setpoint(VelocitySetpoint(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(RuntimeError, match="attitude setpoint"):
        port.send_attitude_target(AttitudeRateSetpoint(0.0, 0.0, 0.0, 0.0))

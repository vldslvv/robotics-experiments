"""Commanding-layer fakes shared by routine and driver unit tests."""

import numpy as np

from robotics_experiments.commanding import MavlinkCommandPort
from robotics_experiments.px4_bridge import BridgeStatus, VehicleState
from tests.support.fakes import _quat_pitch_deg, _quat_yaw_deg

__all__ = ["FakeCommandPort", "FakeDriverEnvironment", "make_routine_observation"]


class FakeCommandPort(MavlinkCommandPort):
    """An inert :class:`MavlinkCommandPort` for command-layer / driver tests.

    Captures the ``MavlinkCommand``s emitted on ``SEND`` and lets a test script
    the control feedback the driver reads through the seam: queue acks via
    :attr:`feedback`, then use :meth:`set_handshaked`, :meth:`set_failsafe`, and
    :meth:`set_global_position`. Heartbeat/connect calls are counted but inert.
    """

    def __init__(self):
        self.sent = []
        self.setpoints = []  # _PositionSetpoints emitted on STREAM
        self.velocity_setpoints = []  # _VelocitySetpoints (velocity deploy path)
        self.attitude_targets = []  # _AttitudeRateSetpoints (ctbr deploy path)
        self.effects = []
        self.connected = False
        self.heartbeats = 0
        self.feedback_polls = 0
        # Scriptable feedback for the driver tests.
        self.feedback = []  # acks popped (FIFO) by poll_feedback
        self._handshaked = False
        self._failsafe_active = False
        self._system_status = 0
        self._global_position = None

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def pump_heartbeat(self, *, now_s, force=False):
        self.heartbeats += 1

    def send(self, command):
        self.sent.append(command)
        self.effects.append(("command", command))

    def send_setpoint(self, setpoint):
        self.setpoints.append(setpoint)
        self.effects.append(("setpoint", setpoint))

    def send_velocity_setpoint(self, setpoint):
        self.velocity_setpoints.append(setpoint)
        self.effects.append(("setpoint", setpoint))

    def send_attitude_target(self, setpoint):
        self.attitude_targets.append(setpoint)
        self.effects.append(("setpoint", setpoint))

    def poll_feedback(self):
        self.feedback_polls += 1
        return self.feedback.pop(0) if self.feedback else None

    def set_handshaked(self, value=True):
        self._handshaked = value

    def set_failsafe(self, *, active, system_status=0):
        self._failsafe_active = active
        self._system_status = system_status

    def set_global_position(self, value):
        self._global_position = value

    def read_status(self):
        from robotics_experiments.commanding import CommandPortStatus

        return CommandPortStatus(
            is_handshaked=self._handshaked,
            failsafe=self._failsafe_active,
            failsafe_system_status=self._system_status,
            latest_global_position=self._global_position,
        )


def make_routine_observation(
    *,
    sim_time_us=0,
    position_enu=(0.0, 0.0, 0.0),
    linear_velocity_enu=(0.0, 0.0, 0.0),
    angular_velocity_flu=(0.0, 0.0, 0.0),
    tilt_deg=0.0,
    yaw_deg=0.0,
    is_armed=False,
    last_ack=None,
    failsafe=False,
    task_observation=None,
):
    """Build a synthetic ``RoutineObservation`` for the pure command-layer tests.

    ``tilt_deg`` tilts the body via a pitch rotation (for the tilt gate);
    ``yaw_deg`` yaws it about world-up (for the heading-alignment predicate) --
    pass one or the other. ``linear_velocity_enu`` and ``angular_velocity_flu``
    drive the flight-envelope gates and the hover-stable completion predicate.
    """
    from robotics_experiments.commanding import RoutineObservation

    orientation = _quat_yaw_deg(yaw_deg) if yaw_deg else _quat_pitch_deg(tilt_deg)
    vehicle_state = VehicleState(
        position_enu=np.asarray(position_enu, dtype=np.float64),
        orientation_flu_enu_wxyz=orientation,
        linear_velocity_enu=np.asarray(linear_velocity_enu, dtype=np.float64),
        angular_velocity_flu=np.asarray(angular_velocity_flu, dtype=np.float64),
        sim_time_us=int(sim_time_us),
    )
    bridge_status = BridgeStatus(
        is_connected=True,
        is_armed=is_armed,
        first_heartbeat_received=True,
        sensor_send_count=0,
        actuator_receive_count=0,
        last_sensor_send_us=None,
        last_actuator_receive_us=None,
        actuator_lag_us=None,
        last_disconnect_reason=None,
    )
    return RoutineObservation(
        vehicle_state=vehicle_state,
        bridge_status=bridge_status,
        last_ack=last_ack,
        failsafe=failsafe,
        sim_time_us=int(sim_time_us),
        task_observation=task_observation,
    )


class FakeDriverEnvironment:
    """A scriptable :class:`DriverEnvironment` for per-drone driver-state tests.

    The test sets :attr:`connected`, :attr:`sim_us`, and :attr:`observation`
    (the kinematic snapshot the driver reads each tick); ``read_observation``
    re-stamps it with the current sim time and the drained ack. Failures and
    completed-step labels and non-throwing failure notifications are captured.
    """

    def __init__(self):
        self.connected = False
        self.sim_us = 0
        self.observation = make_routine_observation()
        self.logs = []
        self.failures = []
        self.advanced_steps = []

    def is_bridge_connected(self) -> bool:
        return self.connected

    def sim_time_us(self) -> int:
        return self.sim_us

    def read_observation(self, *, sim_time_us, last_ack):
        import dataclasses

        return dataclasses.replace(self.observation, sim_time_us=sim_time_us, last_ack=last_ack)

    def log(self, message: str) -> None:
        self.logs.append(message)

    def on_failure(self, kind, message: str) -> None:
        self.failures.append((kind, message))

    def on_step_advanced(self, step_label: str) -> None:
        self.advanced_steps.append(step_label)

"""Shared fakes and helpers for the PX4 bridge test suite.

These fakes back the split `test_px4_bridge_*.py` modules. pytest puts the
`tests/` directory on `sys.path`, so the split test files import them with
`from _fakes import ...`.
"""

import math
import sys
import types
from pathlib import Path

import numpy as np

from robotics_experiments.px4_bridge.contract import ActuatorCommands, BridgeStatus
from robotics_experiments.px4_bridge.thrust_model import RotorDynamics
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
        self.applied_forces: list[np.ndarray] = []
        self.applied_dynamics: list[RotorDynamics] = []
        # (rotor_omega_rad_s, dt) pairs recorded from the cosmetic spin seam.
        self.spin_updates: list[tuple[np.ndarray, float]] = []

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

    def apply_dynamics(self, dynamics: RotorDynamics):
        self.applied_dynamics.append(dynamics)
        self.applied_forces.append(np.asarray(dynamics.forces_n, dtype=np.float64))

    def update_rotor_spin(self, rotor_omega_rad_s, dt):
        self.spin_updates.append((np.asarray(rotor_omega_rad_s, dtype=np.float64), float(dt)))


class _FakeFlightStack:
    def __init__(self, commands):
        self.commands = np.asarray(commands, dtype=np.float64)
        self.started = False
        self.stopped = False
        self.states = []
        # Record sim_time_us at call time; the fake articulation reuses one
        # VehicleState object, so reading state.sim_time_us later would only
        # show the last value.
        self.sim_times = []

    @property
    def num_rotors(self):
        return self.commands.shape[0]

    @property
    def is_connected(self):
        return self.started and not self.stopped

    @property
    def is_armed(self):
        return bool(np.any(self.commands))

    @property
    def status(self):
        return BridgeStatus(
            is_connected=self.is_connected,
            is_armed=self.is_armed,
            first_heartbeat_received=self.started,
            sensor_send_count=len(self.sim_times),
            actuator_receive_count=len(self.sim_times),
            last_sensor_send_us=self.sim_times[-1] if self.sim_times else None,
            last_actuator_receive_us=self.sim_times[-1] if self.sim_times else None,
            actuator_lag_us=0 if self.sim_times else None,
            last_disconnect_reason=None,
        )

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def send(self, state: VehicleState):
        self.states.append(state)
        self.sim_times.append(int(state.sim_time_us))
        return None

    def poll(self) -> ActuatorCommands:
        return ActuatorCommands(self.commands)

    def exchange(self, state: VehicleState) -> ActuatorCommands:
        result = self.send(state)
        return self.poll() if result is None else result


class _CapturingThrustModel:
    def __init__(self, num_rotors=4):
        self._num_rotors = num_rotors
        self.captured_body_velocity_flu = None

    @property
    def num_rotors(self):
        return self._num_rotors

    def commands_to_forces(self, commands: ActuatorCommands) -> np.ndarray:
        return np.zeros(self._num_rotors, dtype=np.float64)

    def commands_to_dynamics(self, commands, *, linear_body_velocity_flu=None):
        self.captured_body_velocity_flu = np.asarray(
            linear_body_velocity_flu,
            dtype=np.float64,
        )
        return RotorDynamics(
            forces_n=np.zeros(self._num_rotors, dtype=np.float64),
            body_torque_n_m=np.zeros(3, dtype=np.float64),
            body_drag_n=np.zeros(3, dtype=np.float64),
            rotor_omega_rad_s=np.zeros(self._num_rotors, dtype=np.float64),
        )


class _FakeConnection:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.recv_calls = []
        self.hil_sensor_sends = []
        self.events = []
        self.mav = types.SimpleNamespace(
            heartbeat_send=lambda *args: None,
            hil_sensor_send=lambda *args: self._record_hil_sensor_send(args),
        )

    def _record_hil_sensor_send(self, args):
        self.hil_sensor_sends.append(args)
        self.events.append(("hil_sensor_send", args[0]))

    def recv_match(self, *, blocking=False, timeout=None):
        self.recv_calls.append((blocking, timeout))
        self.events.append(("recv_match", blocking, timeout))
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
        self.gps_sensor_data = []
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
        self._sensor_data.fix_type = int(data["fix_type"])
        self._sensor_data.latitude_deg = int(data["latitude"] * 10000000)
        self._sensor_data.longitude_deg = int(data["longitude"] * 10000000)
        self._sensor_data.altitude = int(data["altitude"] * 1000)
        self._sensor_data.velocity_north = int(data["velocity_north"] * 100)
        self._sensor_data.velocity_east = int(data["velocity_east"] * 100)
        self._sensor_data.velocity_down = int(data["velocity_down"] * 100)
        self._sensor_data.sim_lat = int(data["latitude_gt"] * 10000000)
        self._sensor_data.sim_lon = int(data["longitude_gt"] * 10000000)
        self._sensor_data.sim_alt = int(data["altitude_gt"] * 1000)
        self.gps_sensor_data.append(types.SimpleNamespace(**vars(self._sensor_data)))
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
                (controls[i] + offsets[i]) * scaling[i] + zero_armed[i] for i in range(num_rotors)
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


def lockstep_exchange(port, state: VehicleState) -> np.ndarray:
    """Run one lockstep tick and return the raw rotor-command array.

    Mirrors how ``Px4Bridge`` drives the port each physics step (send sensors
    out, block for the paired actuator). Unwraps the ``ActuatorCommands`` to its
    ``controls`` array so the port-test assertions can compare directly; kept in
    one place for the port test suite.
    """
    return port.exchange(state).controls


class FakeCommandPort:
    """An :class:`_AbstractCommandPort` for the pure command-layer / driver tests.

    Captures the ``_MavlinkCommand``s emitted on ``SEND`` and lets a test script
    the control-feedback the driver reads through the seam: queue acks via
    :attr:`feedback`, and set :attr:`handshaked` / :attr:`failsafe_flag` /
    :attr:`global_position`. Heartbeat/connect calls are counted but inert.
    """

    def __init__(self):
        self.sent = []
        self.connected = False
        self.heartbeats = 0
        self.feedback_polls = 0
        # Scriptable feedback for the driver tests.
        self.feedback = []  # acks popped (FIFO) by poll_feedback
        self.handshaked = False
        self.failsafe_flag = False
        self.system_status = 0
        self.global_position = None

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def pump_heartbeat(self, *, force=False):
        self.heartbeats += 1

    def send(self, command):
        self.sent.append(command)

    def poll_feedback(self):
        self.feedback_polls += 1
        return self.feedback.pop(0) if self.feedback else None

    @property
    def is_handshaked(self):
        return self.handshaked

    @property
    def failsafe(self):
        return self.failsafe_flag

    @property
    def failsafe_system_status(self):
        return self.system_status

    @property
    def latest_global_position(self):
        return self.global_position


def make_routine_observation(
    *,
    sim_time_us=0,
    position_enu=(0.0, 0.0, 0.0),
    linear_velocity_enu=(0.0, 0.0, 0.0),
    angular_velocity_flu=(0.0, 0.0, 0.0),
    tilt_deg=0.0,
    is_armed=False,
    last_ack=None,
    failsafe=False,
):
    """Build a synthetic ``RoutineObservation`` for the pure command-layer tests.

    ``tilt_deg`` tilts the body via a pitch rotation; ``linear_velocity_enu`` and
    ``angular_velocity_flu`` drive the flight-envelope gates and the hover-stable
    completion predicate.
    """
    from robotics_experiments.commanding._observation import RoutineObservation

    vehicle_state = VehicleState(
        position_enu=np.asarray(position_enu, dtype=np.float64),
        orientation_flu_enu_wxyz=_quat_pitch_deg(tilt_deg),
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
    )


class FakeDriverEnvironment:
    """A scriptable :class:`DriverEnvironment` for the per-drone RoutineDriver tests.

    The test sets :attr:`connected`, :attr:`sim_us`, and :attr:`observation`
    (the kinematic snapshot the driver reads each tick); ``read_observation``
    re-stamps it with the current sim time and the drained ack. Failures and
    completed-step labels are captured; ``raise_on_failure`` makes ``on_failure``
    raise (the strict reaction) so a test can assert the breach propagates.
    """

    def __init__(self, *, raise_on_failure=False):
        self.connected = False
        self.sim_us = 0
        self.observation = make_routine_observation()
        self.logs = []
        self.failures = []
        self.advanced_steps = []
        self.raise_on_failure = raise_on_failure

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
        if self.raise_on_failure:
            raise AssertionError(f"{kind}: {message}")

    def on_step_advanced(self, step_label: str) -> None:
        self.advanced_steps.append(step_label)


def _quat_pitch_deg(angle_deg: float) -> np.ndarray:
    """Return a [w, x, y, z] quaternion for a pitch rotation of angle_deg."""
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)


def _quat_yaw_deg(angle_deg: float) -> np.ndarray:
    """Return a [w, x, y, z] quaternion for a yaw rotation of angle_deg."""
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)

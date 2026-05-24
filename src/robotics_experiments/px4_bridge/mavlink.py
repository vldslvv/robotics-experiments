"""Flight-stack port interface and the PX4 MAVLink adapter.

The port is the seam between the bridge and whatever flight stack is on the
other side. Components above the port (``Px4Bridge``) speak only
``VehicleState`` in / ``np.ndarray`` of normalized commands out, so they can
be paired with a fake port for staged testing without PX4 running.

The concrete ``Px4MavlinkPort`` wraps Pegasus' ``PX4MavlinkBackend`` for the
heavy lifting (socket, MAVLink packing, PX4 launch) but takes over the
sensor-frame conversions, lockstep poll, and timestamp handling so the
contract requirements (timeout, sim-time passthrough) are satisfied.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np

from . import frames
from .flight_stack import AbstractFlightStackBridge
from .vehicle_state import VehicleState

__all__ = ["Px4MavlinkPort"]

_LOGGER = logging.getLogger(__name__)

_HIL_SENSOR_ACCEL_UPDATED = 0b0000000000111
_HIL_SENSOR_GYRO_UPDATED = 0b0000000111000
_HIL_SENSOR_MAG_UPDATED = 0b0000111000000
_HIL_SENSOR_BARO_UPDATED = 0b1101000000000
_HIL_SENSOR_DIFF_PRESS_UPDATED = 0b0010000000000
_HIL_SENSOR_ALL_UPDATED = 0x1FFF
_MAG_DITHER_GAUSS = 0.0025
_BARO_DITHER_HPA = 0.012


def _flu_to_frd(vec_flu: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec_flu, dtype=np.float64)
    return np.array([vec[0], -vec[1], -vec[2]], dtype=np.float64)


# TODO get rid of px4_autolaunch and similar settings
class Px4MavlinkPort(AbstractFlightStackBridge):
    """PX4 SITL adapter over Pegasus' ``PX4MavlinkBackend``.

    Differences from a raw ``PX4MavlinkBackend.update(dt)`` call:

    - Lockstep ``recv_match`` carries a hard timeout (default 100 ms). On
      timeout the port marks itself disconnected and returns zero commands;
      Isaac keeps ticking instead of freezing.
    - ``HIL_SENSOR.time_usec`` is taken from the caller's ``sim_time_us``,
      not a wallclock-derived counter. This is what makes lockstep
      sim-speed-agnostic.
    - Sensor synthesis (specific force from gravity + attitude, magnetometer
      from a fixed NED reference, GPS from local ENU offset, barometer from
      ISA) is performed inside the port. Callers only supply ``VehicleState``.

    The Pegasus backend is bypassed for ``update`` / ``poll_mavlink_messages``;
    its ``_connection``, ``send_*`` helpers, ``handle_control``, and rotor
    bookkeeping are reused.
    """

    def __init__(
        self,
        *,
        num_rotors: int = 4,
        connection_type: str = "tcpin",
        connection_ip: str = "localhost",
        connection_baseport: int = 4560,
        vehicle_id: int = 0,
        px4_autolaunch: bool = False,
        px4_dir: str = "",
        px4_vehicle_model: str = "gazebo-classic_iris",
        lockstep_timeout_s: float = 0.1,
        first_heartbeat_timeout_s: float = 5.0,
        heartbeat_rate_hz: float = 1.0,
        gps_enabled: bool = True,
        gps_send_rate_hz: float = 5.0,
        home_lat_deg: float = frames.DEFAULT_HOME_LAT_DEG,
        home_lon_deg: float = frames.DEFAULT_HOME_LON_DEG,
        home_alt_m: float = frames.DEFAULT_HOME_ALT_M,
    ):
        """Initialize the port. No socket is opened until :meth:`start`."""
        self._num_rotors = int(num_rotors)
        self._lockstep_timeout_s = float(lockstep_timeout_s)
        self._first_heartbeat_timeout_s = float(first_heartbeat_timeout_s)
        self._heartbeat_period_s = (
            1.0 / float(heartbeat_rate_hz) if heartbeat_rate_hz > 0 else None
        )
        self._gps_enabled = bool(gps_enabled)
        self._gps_period_s = 1.0 / float(gps_send_rate_hz) if gps_send_rate_hz > 0 else None
        self._home_lat_deg = float(home_lat_deg)
        self._home_lon_deg = float(home_lon_deg)
        self._home_alt_m = float(home_alt_m)

        # Lazy-construct the Pegasus backend so importing this module does
        # not require pegasus/scipy until something actually runs.
        from pegasus.simulator.logic.backends.px4_mavlink_backend import (
            PX4MavlinkBackend,
            PX4MavlinkBackendConfig,
        )

        self._backend = PX4MavlinkBackend(
            PX4MavlinkBackendConfig(
                {
                    "num_rotors": self._num_rotors,
                    "connection_type": connection_type,
                    "connection_ip": connection_ip,
                    "connection_baseport": connection_baseport,
                    "vehicle_id": vehicle_id,
                    "px4_autolaunch": px4_autolaunch,
                    "px4_dir": px4_dir,
                    "px4_vehicle_model": px4_vehicle_model,
                    "enable_lockstep": True,
                    # Force the linear-scaling stage in the backend into a
                    # passthrough: thrust_model handles command->force.
                    "input_offset": [0.0] * self._num_rotors,
                    "input_scaling": [1.0] * self._num_rotors,
                    "zero_position_armed": [0.0] * self._num_rotors,
                }
            )
        )

        self._is_connected: bool = False
        self._last_gps_send_us: Optional[int] = None
        self._last_sensor_send_us: Optional[int] = None
        self._last_heartbeat_send_s: float = 0.0
        self._first_heartbeat_wait_start_s: Optional[float] = None
        self._first_heartbeat_received_s: Optional[float] = None
        self._last_disconnect_reason: Optional[str] = None
        self._last_poll_timeout_s: Optional[float] = None
        self._sensor_send_count: int = 0
        self._actuator_receive_count: int = 0
        self._last_actuator_receive_us: Optional[int] = None
        self._sensor_payload_debug_count: int = 0

    # ---- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._backend.start()
        self._last_gps_send_us = None
        self._last_sensor_send_us = None
        self._last_heartbeat_send_s = 0.0
        self._first_heartbeat_wait_start_s = None
        self._first_heartbeat_received_s = None
        self._last_disconnect_reason = None
        self._last_poll_timeout_s = None
        self._sensor_send_count = 0
        self._actuator_receive_count = 0
        self._last_actuator_receive_us = None
        self._sensor_payload_debug_count = 0
        self._is_connected = True
        _LOGGER.info("PX4 MAVLink port started")

    def stop(self) -> None:
        try:
            self._backend.stop()
        except Exception as exc:
            _LOGGER.warning("PX4 MAVLink backend stop failed: %s", exc)
        self._is_connected = False
        _LOGGER.info(
            "PX4 MAVLink port stopped: sensors=%d actuators=%d last_sensor_us=%s "
            "last_actuator_us=%s last_disconnect=%s",
            self._sensor_send_count,
            self._actuator_receive_count,
            self._last_sensor_send_us,
            self._last_actuator_receive_us,
            self._last_disconnect_reason,
        )

    # ---- Properties --------------------------------------------------------

    @property
    def num_rotors(self) -> int:
        return self._num_rotors

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._backend._connection is not None

    @property
    def is_armed(self) -> bool:
        # The Pegasus backend zeroes input_reference when disarmed; expose
        # the most reliable signal we have: nonzero rotor reference.
        return any(v != 0.0 for v in self._backend.input_reference())

    # ---- Per-tick ----------------------------------------------------------

    def step(self, state: VehicleState) -> np.ndarray:
        if not self.is_connected:
            return np.zeros(self._num_rotors, dtype=np.float64)

        sim_time_us = int(state.sim_time_us)
        if self._last_sensor_send_us is not None and sim_time_us <= self._last_sensor_send_us:
            _LOGGER.debug(
                "Skipping non-increasing PX4 HIL_SENSOR timestamp: current=%d last=%d",
                sim_time_us,
                self._last_sensor_send_us,
            )
            return np.asarray(self._backend.input_reference(), dtype=np.float64)

        # 1. Convert state into the sensor data the Pegasus backend expects.
        self._fill_sensor_data(state)

        # 2. Send heartbeat at configured rate (wallclock is fine - pacing only).
        now = time.time()
        if self._should_send_heartbeat(now):
            try:
                self._backend.send_heartbeat()
            except Exception as exc:
                self._mark_disconnected(f"heartbeat send failed: {exc}", close_backend=True)
                return np.zeros(self._num_rotors, dtype=np.float64)
            self._last_heartbeat_send_s = now

        # 3. Wait for the first heartbeat from PX4 before doing anything else.
        if not self._backend._received_first_hearbeat:
            if not self._wait_for_first_heartbeat(now):
                reason = self._last_disconnect_reason or "first heartbeat timeout"
                self._mark_disconnected(reason, close_backend=True)
            return np.zeros(self._num_rotors, dtype=np.float64)

        # 4. Send sensor + GPS messages with the caller's sim time.
        try:
            self._send_sensor_msgs(sim_time_us)
            self._last_sensor_send_us = sim_time_us
            self._sensor_send_count += 1
            if self._sensor_send_count == 1:
                _LOGGER.info("PX4 first HIL_SENSOR sent at sim_time_us=%d", sim_time_us)
            elif self._sensor_send_count % 250 == 0:
                _LOGGER.debug(
                    "PX4 sensor stream: sensors=%d actuators=%d sim_time_us=%d "
                    "last_actuator_us=%s connected=%s",
                    self._sensor_send_count,
                    self._actuator_receive_count,
                    sim_time_us,
                    self._last_actuator_receive_us,
                    self.is_connected,
                )
            if self._should_send_gps(sim_time_us):
                self._backend.send_gps_msgs(sim_time_us)
                self._last_gps_send_us = sim_time_us
        except Exception as exc:
            self._mark_disconnected(f"sensor/GPS send failed: {exc}", close_backend=True)
            return np.zeros(self._num_rotors, dtype=np.float64)

        # 5. Poll for HIL_ACTUATOR_CONTROLS with a hard timeout.
        if not self._poll_for_actuator(self._lockstep_timeout_s):
            return np.zeros(self._num_rotors, dtype=np.float64)

        # 6. Return the up-to-date rotor reference.
        return np.asarray(self._backend.input_reference(), dtype=np.float64)

    # ---- Internals ---------------------------------------------------------

    def _fill_sensor_data(self, state: VehicleState) -> None:
        """Populate ``self._backend._sensor_data`` from a VehicleState.

        IMU values are in body-FRD (PX4 convention); we go ENU/FLU -> FRD via
        the Pegasus rotation helpers.
        """
        # Specific force in body FLU, then to FRD for PX4.
        sf_flu = frames.specific_force_body_flu(
            state.linear_acceleration_enu, state.orientation_flu_enu_wxyz
        )
        sf_frd = _flu_to_frd(sf_flu)

        # Gyro is in body. Convert FLU -> FRD.
        gyro_frd = _flu_to_frd(np.asarray(state.angular_velocity_flu, dtype=np.float64))

        # Magnetometer in body FLU then to FRD.
        mag_flu = frames.magnetometer_body_flu(state.orientation_flu_enu_wxyz)
        mag_frd = _flu_to_frd(mag_flu)

        self._backend.update_imu_data(
            {
                "linear_acceleration": [
                    float(sf_frd[0]),
                    float(sf_frd[1]),
                    float(sf_frd[2]),
                ],
                "angular_velocity": [
                    float(gyro_frd[0]),
                    float(gyro_frd[1]),
                    float(gyro_frd[2]),
                ],
            }
        )
        self._backend.update_mag_data(
            {
                "magnetic_field": [
                    float(mag_frd[0]),
                    float(mag_frd[1]),
                    float(mag_frd[2]),
                ]
            }
        )
        self._backend.update_bar_data(frames.synth_barometer(float(state.position_enu[2])))

        if self._gps_enabled:
            # GPS - synthesized from local ENU around home.
            gps = frames.synth_gps_from_enu(
                position_enu=state.position_enu,
                velocity_enu=state.linear_velocity_enu,
                home_lat_deg=self._home_lat_deg,
                home_lon_deg=self._home_lon_deg,
                home_alt_m=self._home_alt_m,
            )
            self._backend.update_gps_data(gps)

    def _should_send_heartbeat(self, now_s: float) -> bool:
        if self._heartbeat_period_s is None:
            return False
        return (now_s - self._last_heartbeat_send_s) >= self._heartbeat_period_s

    def _wait_for_first_heartbeat(self, now_s: float) -> bool:
        if self._first_heartbeat_wait_start_s is None:
            self._first_heartbeat_wait_start_s = now_s
        try:
            self._backend.wait_for_first_hearbeat()
        except Exception as exc:
            self._last_disconnect_reason = f"first heartbeat wait failed: {exc}"
            _LOGGER.warning("PX4 first heartbeat wait failed: %s", exc)
            return False
        if self._backend._received_first_hearbeat:
            if self._first_heartbeat_received_s is None:
                self._first_heartbeat_received_s = now_s
                _LOGGER.info("PX4 first heartbeat received")
            return True
        elapsed_s = now_s - self._first_heartbeat_wait_start_s
        if elapsed_s >= self._first_heartbeat_timeout_s:
            _LOGGER.warning(
                "PX4 first heartbeat timed out after %.3f seconds",
                self._first_heartbeat_timeout_s,
            )
            return False
        return True

    def _should_send_gps(self, sim_time_us: int) -> bool:
        if not self._gps_enabled or self._gps_period_s is None:
            return False
        if self._last_gps_send_us is None:
            return True
        return (sim_time_us - self._last_gps_send_us) >= self._gps_period_s * 1_000_000

    def _send_sensor_msgs(self, sim_time_us: int) -> None:
        """Send HIL_SENSOR with a project-owned field mask and baro payload."""
        sensor = self._backend._sensor_data
        fields_updated = 0

        if sensor.new_imu_data:
            fields_updated |= _HIL_SENSOR_ACCEL_UPDATED | _HIL_SENSOR_GYRO_UPDATED
            sensor.new_imu_data = False
        if sensor.new_mag_data:
            fields_updated |= _HIL_SENSOR_MAG_UPDATED
            sensor.new_mag_data = False
        if sensor.new_bar_data:
            fields_updated |= _HIL_SENSOR_BARO_UPDATED
            sensor.new_bar_data = False
        if sensor.new_press_data:
            fields_updated |= _HIL_SENSOR_DIFF_PRESS_UPDATED
            sensor.new_press_data = False

        # PX4's SITL simulator path expects all HIL_SENSOR fields to be marked
        # valid. We synthesize zero differential pressure when no airspeed
        # model exists, which is still a valid value for this multirotor path.
        fields_updated = _HIL_SENSOR_ALL_UPDATED

        # PX4's simulator driver republishes IMU samples continuously, but the
        # mag/baro path can leave vehicle_magnetometer/vehicle_air_data stale
        # when a perfectly static HIL_SENSOR stream quantizes to identical raw
        # samples. Add deterministic sub-sensor-noise variation so the sensor
        # voter sees fresh samples without creating meaningful vehicle motion.
        t_s = sim_time_us * 1e-6
        mag_x = sensor.xmag + _MAG_DITHER_GAUSS * math.sin(2.0 * math.pi * 0.73 * t_s)
        mag_y = sensor.ymag + _MAG_DITHER_GAUSS * math.sin(2.0 * math.pi * 0.91 * t_s + 1.7)
        mag_z = sensor.zmag + _MAG_DITHER_GAUSS * math.sin(2.0 * math.pi * 0.61 * t_s + 0.4)
        abs_pressure = sensor.abs_pressure + _BARO_DITHER_HPA * math.sin(
            2.0 * math.pi * 0.37 * t_s + 0.2
        )

        if self._sensor_payload_debug_count < 5 or self._sensor_payload_debug_count % 250 == 0:
            _LOGGER.warning(
                "PX4 HIL_SENSOR payload: n=%d time_us=%d mask=%d acc=[%.5f, %.5f, %.5f] "
                "gyro=[%.5f, %.5f, %.5f] mag=[%.5f, %.5f, %.5f] "
                "abs_pressure=%.5f diff_pressure=%.5f pressure_alt=%.5f temp=%.2f",
                self._sensor_payload_debug_count + 1,
                sim_time_us,
                fields_updated,
                sensor.xacc,
                sensor.yacc,
                sensor.zacc,
                sensor.xgyro,
                sensor.ygyro,
                sensor.zgyro,
                mag_x,
                mag_y,
                mag_z,
                abs_pressure,
                sensor.diff_pressure,
                sensor.pressure_alt,
                sensor.temperature,
            )
        self._sensor_payload_debug_count += 1

        self._backend._connection.mav.hil_sensor_send(
            int(sim_time_us),
            sensor.xacc,
            sensor.yacc,
            sensor.zacc,
            sensor.xgyro,
            sensor.ygyro,
            sensor.zgyro,
            mag_x,
            mag_y,
            mag_z,
            abs_pressure,
            sensor.diff_pressure,
            sensor.pressure_alt,
            sensor.temperature,
            fields_updated,
        )

    def _mark_disconnected(self, reason: str, *, close_backend: bool = False) -> None:
        """Record a fatal MAVLink state transition and optionally close the socket."""
        detail = (
            f"sensors={self._sensor_send_count} "
            f"actuators={self._actuator_receive_count} "
            f"last_sensor_us={self._last_sensor_send_us} "
            f"last_actuator_us={self._last_actuator_receive_us}"
        )
        if self._last_disconnect_reason is None:
            _LOGGER.warning("PX4 MAVLink port disconnected: %s (%s)", reason, detail)
        else:
            _LOGGER.warning(
                "PX4 MAVLink port remains disconnected: %s (%s); previous reason: %s",
                reason,
                detail,
                self._last_disconnect_reason,
            )
        self._last_disconnect_reason = reason
        self._is_connected = False
        if close_backend and self._backend._connection is not None:
            try:
                self._backend.stop()
            except Exception as exc:
                _LOGGER.warning("PX4 MAVLink backend close after disconnect failed: %s", exc)

    def _poll_for_actuator(self, timeout_s: float) -> bool:
        """Block (with timeout) for a HIL_ACTUATOR_CONTROLS message.

        With Isaac physics configured to the PX4 actuator cadence, lockstep is
        strict: after the first actuator packet, each new sensor tick must be
        paired with a new actuator response before the next tick continues.

        Returns:
            ``True`` if an actuator message was processed, or no blocking wait
            was needed before the first actuator; ``False`` on timeout.
        """
        from pymavlink import mavutil

        backend = self._backend
        if backend._connection is None:
            self._mark_disconnected("actuator polling skipped: connection is closed")
            return False
        needs_to_wait = backend._received_first_actuator and backend._enable_lockstep
        backend._received_actuator = False

        deadline = time.time() + timeout_s

        while True:
            remaining = max(0.0, deadline - time.time()) if needs_to_wait else 0.0
            try:
                msg = backend._connection.recv_match(
                    blocking=needs_to_wait,
                    timeout=remaining,
                )
            except Exception as exc:
                self._mark_disconnected(f"actuator polling failed: {exc}", close_backend=True)
                return False

            if msg is None and not needs_to_wait:
                return True

            if msg is not None and msg.id == mavutil.mavlink.MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS:
                backend._received_first_actuator = True
                backend._received_actuator = True
                backend.handle_control(msg.time_usec, msg.controls, msg.mode, msg.flags)
                self._actuator_receive_count += 1
                self._last_actuator_receive_us = int(msg.time_usec)
                if self._actuator_receive_count == 1:
                    _LOGGER.info(
                        "PX4 first HIL_ACTUATOR_CONTROLS received at time_usec=%d",
                        self._last_actuator_receive_us,
                    )

            if not needs_to_wait or backend._received_actuator:
                return True

            if time.time() >= deadline:
                self._last_poll_timeout_s = time.time()
                self._mark_disconnected(
                    f"actuator polling timed out after {timeout_s:.3f} seconds",
                    close_backend=True,
                )
                return False

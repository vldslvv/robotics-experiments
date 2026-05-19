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

import time
from typing import Optional

import numpy as np

from . import frames
from .flight_stack import AbstractFlightStackBridge
from .vehicle_state import VehicleState

__all__ = ["Px4MavlinkPort"]


def _flu_to_frd(vec_flu: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec_flu, dtype=np.float64)
    return np.array([vec[0], -vec[1], -vec[2]], dtype=np.float64)


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
        gps_send_rate_hz: float = 5.0,
        home_lat_deg: float = frames.DEFAULT_HOME_LAT_DEG,
        home_lon_deg: float = frames.DEFAULT_HOME_LON_DEG,
        home_alt_m: float = frames.DEFAULT_HOME_ALT_M,
    ):
        """Initialize the port. No socket is opened until :meth:`start`."""
        self._num_rotors = int(num_rotors)
        self._lockstep_timeout_s = float(lockstep_timeout_s)
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
        self._last_heartbeat_send_s: float = 0.0

    # ---- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._backend.start()
        self._last_gps_send_us = None
        self._last_heartbeat_send_s = 0.0
        self._is_connected = True

    def stop(self) -> None:
        try:
            self._backend.stop()
        except Exception:
            pass
        self._is_connected = False

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

        # 1. Convert state into the sensor data the Pegasus backend expects.
        self._fill_sensor_data(state)

        # 2. Send heartbeat at 1 Hz (wallclock is fine - it's pacing only).
        now = time.time()
        if now - self._last_heartbeat_send_s > 1.0:
            try:
                self._backend.send_heartbeat()
            except Exception:
                self._is_connected = False
                return np.zeros(self._num_rotors, dtype=np.float64)
            self._last_heartbeat_send_s = now

        # 3. Wait for the first heartbeat from PX4 before doing anything else.
        if not self._backend._received_first_hearbeat:
            self._backend.wait_for_first_hearbeat()
            return np.zeros(self._num_rotors, dtype=np.float64)

        # 4. Send sensor + GPS messages with the caller's sim time.
        try:
            self._backend.send_sensor_msgs(sim_time_us)
            if self._should_send_gps(sim_time_us):
                self._backend.send_gps_msgs(sim_time_us)
                self._last_gps_send_us = sim_time_us
        except Exception:
            self._is_connected = False
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

        # GPS - synthesized from local ENU around home.
        gps = frames.synth_gps_from_enu(
            position_enu=state.position_enu,
            velocity_enu=state.linear_velocity_enu,
            home_lat_deg=self._home_lat_deg,
            home_lon_deg=self._home_lon_deg,
            home_alt_m=self._home_alt_m,
        )
        self._backend.update_gps_data(gps)

    def _should_send_gps(self, sim_time_us: int) -> bool:
        if self._gps_period_s is None:
            return False
        if self._last_gps_send_us is None:
            return True
        return (sim_time_us - self._last_gps_send_us) >= self._gps_period_s * 1_000_000

    def _poll_for_actuator(self, timeout_s: float) -> bool:
        """Block (with timeout) for a HIL_ACTUATOR_CONTROLS message.

        Replaces ``PX4MavlinkBackend.poll_mavlink_messages`` so we can pass a
        hard timeout to ``recv_match``. The contract requires that a stalled
        flight stack stalls the bridge but never the simulator - on timeout
        we mark the port disconnected and return zero commands upstream.

        Returns:
            ``True`` if an actuator message was processed (or no wait was
            needed because we have not yet seen a first actuator), ``False``
            on timeout.
        """
        from pymavlink import mavutil

        backend = self._backend
        if backend._connection is None:
            self._is_connected = False
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
            except Exception:
                self._is_connected = False
                return False

            if msg is not None and msg.id == mavutil.mavlink.MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS:
                backend._received_first_actuator = True
                backend._received_actuator = True
                backend.handle_control(msg.time_usec, msg.controls, msg.mode, msg.flags)

            if not needs_to_wait or backend._received_actuator:
                return True

            if time.time() >= deadline:
                # Hard timeout - let upstream return zeros; do not hang.
                self._is_connected = False
                return False

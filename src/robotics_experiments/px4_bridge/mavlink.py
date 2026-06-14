"""Flight-stack port interface and the PX4 MAVLink adapter.

The port is the seam between the bridge and whatever flight stack is on the
other side. Components above the port (``Px4Bridge``) speak only
``VehicleState`` in / rotor references out, so they can be paired with a fake
port for staged testing without PX4 running.

The concrete ``Px4MavlinkPort`` wraps Pegasus' ``PX4MavlinkBackend`` for the
heavy lifting (socket, MAVLink packing, PX4 launch, actuator scaling) while
preserving custom timeout and sim-time handling.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path
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
_DEFAULT_SENSOR_RATE_HZ = 250.0
_DEFAULT_DEBUG_LOG_PATH = "/tmp/robotics_experiments_px4_bridge.log"
_FILE_LOG_HANDLER_ATTR = "_robotics_experiments_px4_bridge_file_path"


def _resolve_log_path_template(log_path: str, now: datetime) -> Path:
    return Path(
        log_path.format(
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H_%M_%S"),
            timestamp=now.strftime("%Y-%m-%d/%H_%M_%S"),
        )
    ).expanduser()


def _configure_file_logging(
    log_path: str | None,
    level_name: str,
    now: datetime | None = None,
) -> Path | None:
    if not log_path:
        return None

    now = now or datetime.now()
    path = _resolve_log_path_template(log_path, now)
    path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(level_name).upper(), logging.DEBUG)
    package_logger = logging.getLogger("robotics_experiments.px4_bridge")
    package_logger.setLevel(min(package_logger.level or level, level))

    resolved = path.resolve()
    for handler in list(package_logger.handlers):
        if getattr(handler, _FILE_LOG_HANDLER_ATTR, None) is not None:
            package_logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(resolved, mode="w", encoding="utf-8")
    setattr(handler, _FILE_LOG_HANDLER_ATTR, str(resolved))
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s:%(lineno)d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    package_logger.addHandler(handler)
    return resolved


class _PegasusSensorSuite:
    """Stateful Pegasus-style sensor update pipeline."""

    def __init__(
        self,
        *,
        home_lat_deg: float,
        home_lon_deg: float,
        home_alt_m: float,
        gps_enabled: bool,
        sensor_noise_enabled: bool,
        update_rate_hz: float = _DEFAULT_SENSOR_RATE_HZ,
    ):
        self._home_lat_deg = float(home_lat_deg)
        self._home_lon_deg = float(home_lon_deg)
        self._home_alt_m = float(home_alt_m)
        self._gps_enabled = bool(gps_enabled)
        self._sensor_noise_enabled = bool(sensor_noise_enabled)
        self._last_sim_time_us: int | None = None

        from pegasus.simulator.logic.sensors.barometer import Barometer
        from pegasus.simulator.logic.sensors.gps import GPS
        from pegasus.simulator.logic.sensors.imu import IMU
        from pegasus.simulator.logic.sensors.magnetometer import Magnetometer

        # Runtime intentionally uses Pegasus sensor classes for all PX4-bound
        # sensor signals. With noise enabled their internal distributions are:
        # IMU gyro/accel white noise plus bias random walks, GPS position and
        # altitude noise plus bias walk, barometer pressure noise/drift, and
        # magnetometer white noise plus bias walk.
        self._imu = IMU(_pegasus_imu_config(update_rate_hz, self._sensor_noise_enabled))
        self._gps = GPS(_pegasus_gps_config(update_rate_hz, self._sensor_noise_enabled))
        self._barometer = Barometer(
            _pegasus_barometer_config(update_rate_hz, self._sensor_noise_enabled)
        )
        self._magnetometer = Magnetometer(
            _pegasus_magnetometer_config(update_rate_hz, self._sensor_noise_enabled)
        )
        for sensor in (self._imu, self._gps, self._barometer, self._magnetometer):
            sensor.initialize(
                None,
                self._home_lat_deg,
                self._home_lon_deg,
                self._home_alt_m,
            )

    def update(self, state: VehicleState) -> dict[str, dict | None]:
        sim_time_us = int(state.sim_time_us)
        if self._last_sim_time_us is None:
            dt_s = 1.0 / _DEFAULT_SENSOR_RATE_HZ
        else:
            dt_s = max(0.0, (sim_time_us - self._last_sim_time_us) * 1e-6)
        self._last_sim_time_us = sim_time_us

        pegasus_state = self._to_pegasus_state(state)
        imu = self._update_imu(pegasus_state, dt_s)
        mag = self._update_mag(pegasus_state, dt_s)
        baro = self._update_baro(pegasus_state, dt_s)
        gps = self._update_gps(pegasus_state, dt_s) if self._gps_enabled else None
        return {"imu": imu, "mag": mag, "baro": baro, "gps": gps}

    def _to_pegasus_state(self, state: VehicleState):
        from pegasus.simulator.logic.state import State

        pegasus_state = State()
        q = np.asarray(state.orientation_flu_enu_wxyz, dtype=np.float64)
        pegasus_state.position = np.asarray(state.position_enu, dtype=np.float64)
        pegasus_state.attitude = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
        pegasus_state.linear_velocity = np.asarray(state.linear_velocity_enu, dtype=np.float64)
        pegasus_state.angular_velocity = np.asarray(state.angular_velocity_flu, dtype=np.float64)
        pegasus_state.linear_acceleration = np.asarray(
            state.linear_acceleration_enu,
            dtype=np.float64,
        )
        pegasus_state.linear_body_velocity = frames.world_enu_to_body_flu(
            state.orientation_flu_enu_wxyz,
            state.linear_velocity_enu,
        )
        return pegasus_state

    def _update_imu(self, state, dt_s: float) -> dict | None:
        return self._imu.update(state, dt_s)

    def _update_mag(self, state, dt_s: float) -> dict | None:
        return self._magnetometer.update(state, dt_s)

    def _update_baro(self, state, dt_s: float) -> dict | None:
        return self._barometer.update(state, dt_s)

    def _update_gps(self, state, dt_s: float) -> dict | None:
        gps = self._gps.update(state, dt_s)
        if gps is None:
            return None
        return _normalize_pegasus_gps_for_backend(gps)


def _pegasus_imu_config(update_rate_hz: float, noise_enabled: bool) -> dict:
    config = {"update_rate": update_rate_hz}
    if not noise_enabled:
        # Zero both white-noise densities and random-walk terms so precision
        # tests still exercise Pegasus frame conversion without stochastic
        # gyro/accelerometer perturbations.
        config["gyroscope"] = {
            "noise_density": 0.0,
            "random_walk": 0.0,
            "bias_correlation_time": 1.0e3,
            "turn_on_bias_sigma": 0.0,
        }
        config["accelerometer"] = {
            "noise_density": 0.0,
            "random_walk": 0.0,
            "bias_correlation_time": 300.0,
            "turn_on_bias_sigma": 0.0,
        }
    return config


def _pegasus_gps_config(update_rate_hz: float, noise_enabled: bool) -> dict:
    config = {"update_rate": update_rate_hz}
    if not noise_enabled:
        # Pegasus GPS applies randomness to position/altitude through
        # white-noise and bias-walk terms. Its velocity-noise samples are
        # currently computed but not applied by the vendored model.
        config.update(
            {
                "gps_xy_random_walk": 0.0,
                "gps_z_random_walk": 0.0,
                "gps_xy_noise_density": 0.0,
                "gps_z_noise_density": 0.0,
                "gps_vxy_noise_density": 0.0,
                "gps_vz_noise_density": 0.0,
            }
        )
    return config


def _pegasus_barometer_config(update_rate_hz: float, noise_enabled: bool) -> dict:
    config = {"update_rate": update_rate_hz}
    if not noise_enabled:
        # Barometer randomness is pressure-domain Gaussian noise plus drift.
        # The vendored sensor now exposes noise_std_pa so it can be made exact.
        config["noise_std_pa"] = 0.0
        config["drift_pa_per_sec"] = 0.0
    return config


def _pegasus_magnetometer_config(update_rate_hz: float, noise_enabled: bool) -> dict:
    config = {"update_rate": update_rate_hz}
    if not noise_enabled:
        # Magnetometer randomness is Gaussian field noise plus random-walk
        # bias. Zero both terms for exact frame/field tests.
        config.update(
            {
                "noise_density": 0.0,
                "random_walk": 0.0,
                "bias_correlation_time": 6.0e2,
            }
        )
    return config


def _normalize_pegasus_gps_for_backend(gps: dict) -> dict:
    normalized = dict(gps)
    for key in ("latitude_gt", "longitude_gt"):
        value = float(normalized[key])
        if math.isfinite(value) and abs(value) <= 2.0 * math.pi:
            normalized[key] = math.degrees(value)
    return normalized


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
    - Sensor synthesis follows Pegasus' sensor-rate and payload conventions.
      Callers only supply ``VehicleState``.

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
        input_offset: list[float] | None = None,
        input_scaling: list[float] | None = None,
        zero_position_armed: list[float] | None = None,
        lockstep_timeout_s: float = 0.1,
        first_heartbeat_timeout_s: float = 5.0,
        heartbeat_rate_hz: float = 1.0,
        gps_enabled: bool = True,
        gps_send_rate_hz: float = _DEFAULT_SENSOR_RATE_HZ,
        sensor_noise_enabled: bool = True,
        home_lat_deg: float = frames.DEFAULT_HOME_LAT_DEG,
        home_lon_deg: float = frames.DEFAULT_HOME_LON_DEG,
        home_alt_m: float = frames.DEFAULT_HOME_ALT_M,
        debug_log_path: str | None = _DEFAULT_DEBUG_LOG_PATH,
        debug_log_level: str = "DEBUG",
        sensor_debug_log_interval: int = 50,
    ):
        """Initialize the port. No socket is opened until :meth:`start`."""
        self._debug_log_path = _configure_file_logging(debug_log_path, debug_log_level)
        self._sensor_debug_log_interval = max(1, int(sensor_debug_log_interval))
        self._num_rotors = int(num_rotors)
        self._lockstep_timeout_s = float(lockstep_timeout_s)
        self._first_heartbeat_timeout_s = float(first_heartbeat_timeout_s)
        self._heartbeat_period_s = 1.0 / float(heartbeat_rate_hz) if heartbeat_rate_hz > 0 else None
        self._gps_enabled = bool(gps_enabled)
        self._gps_period_s = 1.0 / float(gps_send_rate_hz) if gps_send_rate_hz > 0 else None
        self._sensor_noise_enabled = bool(sensor_noise_enabled)
        self._home_lat_deg = float(home_lat_deg)
        self._home_lon_deg = float(home_lon_deg)
        self._home_alt_m = float(home_alt_m)
        self._sensor_suite = _PegasusSensorSuite(
            home_lat_deg=self._home_lat_deg,
            home_lon_deg=self._home_lon_deg,
            home_alt_m=self._home_alt_m,
            gps_enabled=self._gps_enabled,
            sensor_noise_enabled=self._sensor_noise_enabled,
        )
        input_offset = [0.0] * self._num_rotors if input_offset is None else input_offset
        input_scaling = [1000.0] * self._num_rotors if input_scaling is None else input_scaling
        zero_position_armed = (
            [100.0] * self._num_rotors if zero_position_armed is None else zero_position_armed
        )

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
                    "input_offset": input_offset,
                    "input_scaling": input_scaling,
                    "zero_position_armed": zero_position_armed,
                }
            )
        )

        self._is_connected: bool = False
        self._last_gps_send_us: Optional[int] = None
        self._last_sensor_send_us: Optional[int] = None
        self._last_sensor_period_us: Optional[int] = None
        self._last_heartbeat_send_s: float = 0.0
        self._first_heartbeat_wait_start_s: Optional[float] = None
        self._first_heartbeat_received_s: Optional[float] = None
        self._last_disconnect_reason: Optional[str] = None
        self._last_poll_timeout_s: Optional[float] = None
        self._sensor_send_count: int = 0
        self._actuator_receive_count: int = 0
        self._last_actuator_receive_us: Optional[int] = None
        self._actuator_lag_baseline_us: Optional[int] = None
        self._sensor_payload_debug_count: int = 0
        _LOGGER.info(
            "PX4 MAVLink port constructed: endpoint=%s:%s:%d vehicle_id=%d "
            "num_rotors=%d lockstep_timeout_s=%.3f first_heartbeat_timeout_s=%.3f "
            "gps_enabled=%s gps_rate_hz=%s sensor_noise_enabled=%s "
            "debug_log=%s sensor_debug_interval=%d",
            connection_type,
            connection_ip,
            connection_baseport,
            vehicle_id,
            self._num_rotors,
            self._lockstep_timeout_s,
            self._first_heartbeat_timeout_s,
            self._gps_enabled,
            gps_send_rate_hz,
            self._sensor_noise_enabled,
            self._debug_log_path,
            self._sensor_debug_log_interval,
        )

    # ---- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._backend.start()
        self._sensor_suite = _PegasusSensorSuite(
            home_lat_deg=self._home_lat_deg,
            home_lon_deg=self._home_lon_deg,
            home_alt_m=self._home_alt_m,
            gps_enabled=self._gps_enabled,
            sensor_noise_enabled=self._sensor_noise_enabled,
        )
        self._last_gps_send_us = None
        self._last_sensor_send_us = None
        self._last_sensor_period_us = None
        self._last_heartbeat_send_s = 0.0
        self._first_heartbeat_wait_start_s = None
        self._first_heartbeat_received_s = None
        self._last_disconnect_reason = None
        self._last_poll_timeout_s = None
        self._sensor_send_count = 0
        self._actuator_receive_count = 0
        self._last_actuator_receive_us = None
        self._actuator_lag_baseline_us = None
        self._sensor_payload_debug_count = 0
        self._is_connected = True
        _LOGGER.info("PX4 MAVLink port started; debug_log=%s", self._debug_log_path)

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
        # This is an intentional Pegasus phase-order deviation: each port
        # instance treats a tick as state(t) -> sensors(t) -> actuator(t),
        # keeping lockstep causal per drone while retaining a hard timeout.
        try:
            previous_sensor_send_us = self._last_sensor_send_us
            self._send_sensor_msgs(sim_time_us)
            self._last_sensor_send_us = sim_time_us
            if previous_sensor_send_us is not None:
                self._last_sensor_period_us = sim_time_us - previous_sensor_send_us
            self._sensor_send_count += 1
            if self._sensor_send_count == 1:
                _LOGGER.info("PX4 first HIL_SENSOR sent at sim_time_us=%d", sim_time_us)
            elif self._sensor_send_count % 250 == 0:
                actuator_lag_us = (
                    None
                    if self._last_actuator_receive_us is None
                    else sim_time_us - self._last_actuator_receive_us
                )
                _LOGGER.debug(
                    "PX4 sensor stream: sensors=%d actuators=%d sim_time_us=%d "
                    "last_actuator_us=%s actuator_lag_us=%s connected=%s",
                    self._sensor_send_count,
                    self._actuator_receive_count,
                    sim_time_us,
                    self._last_actuator_receive_us,
                    actuator_lag_us,
                    self.is_connected,
                )
            if self._should_send_gps(sim_time_us):
                self._backend.send_gps_msgs(sim_time_us)
                self._last_gps_send_us = sim_time_us
                _LOGGER.debug(
                    "PX4 HIL_GPS sent: time_us=%d sensor_send_count=%d",
                    sim_time_us,
                    self._sensor_send_count,
                )
        except Exception as exc:
            self._mark_disconnected(f"sensor/GPS send failed: {exc}", close_backend=True)
            return np.zeros(self._num_rotors, dtype=np.float64)

        # 5. Poll for HIL_ACTUATOR_CONTROLS with a hard timeout.
        if not self._poll_for_actuator(self._lockstep_timeout_s, sim_time_us):
            return np.zeros(self._num_rotors, dtype=np.float64)

        # 6. Return the up-to-date rotor reference.
        return np.asarray(self._backend.input_reference(), dtype=np.float64)

    # ---- Internals ---------------------------------------------------------

    def _fill_sensor_data(self, state: VehicleState) -> None:
        """Populate ``self._backend._sensor_data`` from a VehicleState.

        Sensor values are generated through a Pegasus-style stateful sensor
        suite. Only sensors that refreshed on this tick update the Pegasus
        backend, which preserves the backend's ``fields_updated`` mask
        semantics.
        """
        updates = self._sensor_suite.update(state)
        if updates["imu"] is not None:
            self._backend.update_imu_data(updates["imu"])
        if updates["mag"] is not None:
            self._backend.update_mag_data(updates["mag"])
        if updates["baro"] is not None:
            self._backend.update_bar_data(updates["baro"])
        if updates["gps"] is not None:
            self._backend.update_gps_data(updates["gps"])

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
        """Send HIL_SENSOR through Pegasus' backend implementation."""
        sensor = self._backend._sensor_data
        fields_updated = 0
        if sensor.new_imu_data:
            fields_updated |= _HIL_SENSOR_ACCEL_UPDATED | _HIL_SENSOR_GYRO_UPDATED
        if sensor.new_mag_data:
            fields_updated |= _HIL_SENSOR_MAG_UPDATED
        if sensor.new_bar_data:
            fields_updated |= _HIL_SENSOR_BARO_UPDATED
        if sensor.new_press_data:
            fields_updated |= _HIL_SENSOR_DIFF_PRESS_UPDATED

        should_log_payload = (
            self._sensor_payload_debug_count < 10
            or self._sensor_payload_debug_count % self._sensor_debug_log_interval == 0
        )
        if should_log_payload:
            _LOGGER.debug(
                "PX4 HIL_SENSOR pre-send: n=%d time_us=%d mask=%d "
                "fresh={imu:%s mag:%s baro:%s diff_press:%s gps:%s} "
                "acc=[%.5f, %.5f, %.5f] "
                "gyro=[%.5f, %.5f, %.5f] mag=[%.5f, %.5f, %.5f] "
                "abs_pressure=%.5f diff_pressure=%.5f pressure_alt=%.5f temp=%.2f",
                self._sensor_payload_debug_count + 1,
                sim_time_us,
                fields_updated,
                sensor.new_imu_data,
                sensor.new_mag_data,
                sensor.new_bar_data,
                sensor.new_press_data,
                sensor.new_gps_data,
                sensor.xacc,
                sensor.yacc,
                sensor.zacc,
                sensor.xgyro,
                sensor.ygyro,
                sensor.zgyro,
                sensor.xmag,
                sensor.ymag,
                sensor.zmag,
                sensor.abs_pressure,
                sensor.diff_pressure,
                sensor.pressure_alt,
                sensor.temperature,
            )
        self._sensor_payload_debug_count += 1
        self._backend.send_sensor_msgs(int(sim_time_us))
        if should_log_payload:
            _LOGGER.debug(
                "PX4 HIL_SENSOR post-send: n=%d time_us=%d mask=%d "
                "fresh_after={imu:%s mag:%s baro:%s diff_press:%s gps:%s}",
                self._sensor_payload_debug_count,
                sim_time_us,
                fields_updated,
                sensor.new_imu_data,
                sensor.new_mag_data,
                sensor.new_bar_data,
                sensor.new_press_data,
                sensor.new_gps_data,
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

    def _actuator_stale_limit_us(self) -> int:
        observed_period_us = max(0, int(self._last_sensor_period_us or 0))
        return max(5 * observed_period_us, 20_000)

    def _handle_actuator_msg(self, msg) -> None:
        backend = self._backend
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
        if self._actuator_receive_count <= 10 or self._actuator_receive_count % 50 == 0:
            _LOGGER.debug(
                "PX4 HIL_ACTUATOR_CONTROLS received: n=%d time_us=%d "
                "mode=%s flags=%s input_reference=%s",
                self._actuator_receive_count,
                self._last_actuator_receive_us,
                msg.mode,
                msg.flags,
                np.asarray(backend.input_reference(), dtype=np.float64).tolist(),
            )

    def _actuator_is_fresh(self, sim_time_us: int) -> bool:
        if self._last_actuator_receive_us is None:
            return True
        actuator_lag_us = max(0, sim_time_us - self._last_actuator_receive_us)
        stale_limit_us = self._actuator_stale_limit_us()

        if self._actuator_lag_baseline_us is None:
            # PX4 can emit its first HIL_ACTUATOR_CONTROLS packet with an old
            # simulation timestamp while HIL output starts. Accept that first
            # lag as a startup baseline, then fail closed if later packets fall
            # materially farther behind instead of applying stale motor output.
            self._actuator_lag_baseline_us = actuator_lag_us
            if actuator_lag_us > stale_limit_us:
                _LOGGER.info(
                    "PX4 actuator bootstrap lag accepted: "
                    "sim_time_us=%d last_actuator_us=%d lag_us=%d steady_limit_us=%d",
                    sim_time_us,
                    self._last_actuator_receive_us,
                    actuator_lag_us,
                    stale_limit_us,
                )
            return True

        if actuator_lag_us <= stale_limit_us:
            self._actuator_lag_baseline_us = min(
                self._actuator_lag_baseline_us,
                actuator_lag_us,
            )
            return True

        allowed_lag_us = self._actuator_lag_baseline_us + stale_limit_us
        if actuator_lag_us <= allowed_lag_us:
            if actuator_lag_us < self._actuator_lag_baseline_us:
                self._actuator_lag_baseline_us = actuator_lag_us
            return True

        self._mark_disconnected(
            "actuator timestamp stale: "
            f"sim_time_us={sim_time_us} "
            f"last_actuator_us={self._last_actuator_receive_us} "
            f"lag_us={actuator_lag_us} limit_us={allowed_lag_us}",
            close_backend=True,
        )
        return False

    def _poll_for_actuator(self, timeout_s: float, sim_time_us: int) -> bool:
        """Block (with timeout) for a HIL_ACTUATOR_CONTROLS message.

        With Isaac physics configured to the PX4 actuator cadence, lockstep is
        strict after PX4 has produced its first actuator packet: each new sensor
        tick must then be paired with a fresh actuator response before the next
        tick continues. Any queued actuator backlog is drained so the bridge
        uses PX4's newest command.

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

        # Do not block before PX4 has ever produced an actuator packet. During
        # startup the controller may need several sensor ticks before it begins
        # HIL output; strict lockstep starts only after that first response.
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
                self._handle_actuator_msg(msg)

            if backend._received_actuator:
                break

            if time.time() >= deadline:
                self._last_poll_timeout_s = time.time()
                self._mark_disconnected(
                    f"actuator polling timed out after {timeout_s:.3f} seconds",
                    close_backend=True,
                )
                return False

        while True:
            try:
                msg = backend._connection.recv_match(blocking=False, timeout=0.0)
            except Exception as exc:
                self._mark_disconnected(f"actuator polling failed: {exc}", close_backend=True)
                return False
            if msg is None:
                break
            if msg.id == mavutil.mavlink.MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS:
                self._handle_actuator_msg(msg)

        return self._actuator_is_fresh(sim_time_us)

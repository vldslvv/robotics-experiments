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
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import frames
from ._flight_stack import AbstractFlightStackBridge
from ._pegasus_sensors import _DEFAULT_SENSOR_RATE_HZ, _PegasusSensorSuite
from .contract import ActuatorCommands, BridgeStatus, LockstepSettings
from .logging import VehicleLogFilter, vehicle_log_context
from .sim_clock import physics_tick_us
from .vehicle_state import VehicleState

__all__ = ["Px4MavlinkPort"]

_LOGGER = logging.getLogger(__name__)

_HIL_SENSOR_ACCEL_UPDATED = 0b0000000000111
_HIL_SENSOR_GYRO_UPDATED = 0b0000000111000
_HIL_SENSOR_MAG_UPDATED = 0b0000111000000
_HIL_SENSOR_BARO_UPDATED = 0b1101000000000
_HIL_SENSOR_DIFF_PRESS_UPDATED = 0b0010000000000
_DEFAULT_DEBUG_LOG_PATH = "/tmp/robotics_experiments_px4_bridge.log"
_FILE_LOG_HANDLER_VEHICLE_ATTR = "_robotics_experiments_px4_bridge_vehicle_id"
# Immutable shared default for the lockstep settings argument (a module-level
# singleton keeps it out of the mutable-default lint while staying a plain
# concrete default; the project value comes from the Hydra px4.lockstep group).
_DEFAULT_LOCKSTEP = LockstepSettings()


def _resolve_log_path_template(log_path: str, now: datetime, vehicle_id: int) -> Path:
    return Path(
        log_path.format(
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H_%M_%S"),
            timestamp=now.strftime("%Y-%m-%d/%H_%M_%S"),
            vehicle=vehicle_id,
        )
    ).expanduser()


def _configure_file_logging(
    log_path: str | None,
    level_name: str,
    vehicle_id: int,
    now: datetime | None = None,
) -> Path | None:
    if not log_path:
        return None

    now = now or datetime.now()
    path = _resolve_log_path_template(log_path, now, vehicle_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(level_name).upper(), logging.DEBUG)
    package_logger = logging.getLogger("robotics_experiments.px4_bridge")
    package_logger.setLevel(min(package_logger.level or level, level))

    resolved = path.resolve()
    # Replace only this vehicle's prior handler (a re-Play of the same vehicle),
    # not every file handler. Distinct vehicles must keep their own handlers so a
    # multi-vehicle process writes each vehicle's records to its own file instead
    # of the last-constructed vehicle stealing the single handler.
    for handler in list(package_logger.handlers):
        if getattr(handler, _FILE_LOG_HANDLER_VEHICLE_ATTR, None) == int(vehicle_id):
            package_logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(resolved, mode="w", encoding="utf-8")
    setattr(handler, _FILE_LOG_HANDLER_VEHICLE_ATTR, int(vehicle_id))
    handler.addFilter(VehicleLogFilter(vehicle_id))
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s:%(lineno)d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    package_logger.addHandler(handler)
    return resolved


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
        port: int = 4560,
        vehicle_id: int = 0,
        input_offset: list[float] | None = None,
        input_scaling: list[float] | None = None,
        zero_position_armed: list[float] | None = None,
        lockstep: LockstepSettings = _DEFAULT_LOCKSTEP,
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
        """Initialize the port. No socket is opened until :meth:`start`.

        ``port`` is the fully resolved simulator MAVLink TCP port (the caller
        passes ``ResolvedVehicle.mavlink_port``); it is handed to the vendored
        Pegasus backend as its ``connection_baseport`` with a zero ``vehicle_id``
        offset, so no ``baseport + id`` arithmetic happens here. ``vehicle_id``
        is the resolved vehicle ``id`` used only as the per-vehicle log identity.
        """
        self._debug_log_path = _configure_file_logging(debug_log_path, debug_log_level, vehicle_id)
        self._sensor_debug_log_interval = max(1, int(sensor_debug_log_interval))
        self._num_rotors = int(num_rotors)
        self._lockstep_timeout_s = float(lockstep.lockstep_timeout_s)
        self._first_heartbeat_timeout_s = float(lockstep.first_heartbeat_timeout_s)
        # Bound the pre-first-actuator free-run: allow this many priming
        # HIL_SENSOR sends while PX4 boots, then block for the first actuator so
        # Isaac sim time cannot race ahead and build a startup backlog. Must
        # exceed PX4's boot sensor need (~15) or the first-actuator wait times
        # out.
        self._bootstrap_prime_ticks = max(0, int(lockstep.bootstrap_prime_ticks))
        # Generous blocking deadline for PX4's first actuator (boot is several
        # seconds of wallclock); steady-state ticks use lockstep_timeout_s.
        self._first_actuator_timeout_s = float(lockstep.first_actuator_timeout_s)
        # Safety tripwire on steady-state actuator lag, stored as integer
        # microseconds. In strict lockstep the lag is structurally ~1 tick and a
        # PX4 stall is caught by lockstep_timeout_s, so this only fires if a
        # regression lets a stale-but-still-flowing backlog form (PX4 keeps
        # answering, but with persistently old timestamps). See _actuator_is_fresh.
        self._max_sustained_actuator_lag_us = int(lockstep.max_sustained_actuator_lag_s * 1_000_000)
        self._heartbeat_period_s = 1.0 / float(lockstep.heartbeat_rate_hz)
        self._gps_enabled = bool(gps_enabled)
        self._gps_period_s = 1.0 / float(gps_send_rate_hz)
        self._sensor_noise_enabled = bool(sensor_noise_enabled)
        self._home_lat_deg = float(home_lat_deg)
        self._home_lon_deg = float(home_lon_deg)
        self._home_alt_m = float(home_alt_m)
        # The PX4 sensor cadence is the Isaac physics cadence (single source of
        # truth = sim.physics_hz, plumbed in as physics_rate_hz). Pin the sensor
        # period to the integer-microsecond physics tick so the Pegasus
        # update_at_rate gate fires on every physics step regardless of rate:
        # deriving the rate as 1e6 / tick_us makes the suite's per-tick dt_s
        # (computed from integer-us sim_time deltas) exactly equal the sensor
        # period, instead of relying on 1/physics_rate_hz matching it (which
        # fails by a float ULP at rates such as 333 Hz and silently drops the
        # accel/gyro bits on half the HIL_SENSOR packets).
        physics_rate_hz = lockstep.physics_rate_hz
        # physics_tick_us is the single source of the integer tick (shared with
        # SimClock); it raises on a non-positive rate.
        self._physics_tick_us = physics_tick_us(physics_rate_hz)
        self._sensor_rate_hz = 1_000_000 / self._physics_tick_us
        self._sensor_suite = self._make_sensor_suite()
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
                    # The resolved per-vehicle port is passed straight through as
                    # the backend's base with a zero offset (the bridge already
                    # derived baseport + id into `port`).
                    "connection_baseport": port,
                    "vehicle_id": 0,
                    # The bridge never spawns PX4; PX4 SITL runs as a separate
                    # Docker process selected by the resolved airframe config.
                    # Pin the vendored backend's autolaunch off (its own default
                    # is True) so it only connects over MAVLink.
                    "px4_autolaunch": False,
                    "enable_lockstep": True,
                    "input_offset": input_offset,
                    "input_scaling": input_scaling,
                    "zero_position_armed": zero_position_armed,
                }
            )
        )

        self._is_connected: bool = False
        # sim_time stashed by send() for the paired poll() blocking receive.
        self._pending_poll_sim_time_us: int = 0
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
        # Tag this record so it lands only in this vehicle's debug log. The
        # endpoint port is the resolved per-vehicle socket port, so multi-vehicle
        # logs show distinct 4560 / 4561 endpoints.
        with vehicle_log_context(vehicle_id):
            _LOGGER.info(
                "PX4 MAVLink port constructed: endpoint=%s:%s:%d vehicle_id=%d "
                "num_rotors=%d lockstep_timeout_s=%.3f first_heartbeat_timeout_s=%.3f "
                "bootstrap_prime_ticks=%d first_actuator_timeout_s=%.3f "
                "max_sustained_actuator_lag_us=%s "
                "physics_rate_hz=%s physics_tick_us=%d sensor_rate_hz=%.3f "
                "gps_enabled=%s gps_rate_hz=%s sensor_noise_enabled=%s "
                "debug_log=%s sensor_debug_interval=%d",
                connection_type,
                connection_ip,
                port,
                vehicle_id,
                self._num_rotors,
                self._lockstep_timeout_s,
                self._first_heartbeat_timeout_s,
                self._bootstrap_prime_ticks,
                self._first_actuator_timeout_s,
                self._max_sustained_actuator_lag_us,
                physics_rate_hz,
                self._physics_tick_us,
                self._sensor_rate_hz,
                self._gps_enabled,
                gps_send_rate_hz,
                self._sensor_noise_enabled,
                self._debug_log_path,
                self._sensor_debug_log_interval,
            )

    # ---- Lifecycle ---------------------------------------------------------

    def _make_sensor_suite(self) -> _PegasusSensorSuite:
        return _PegasusSensorSuite(
            home_lat_deg=self._home_lat_deg,
            home_lon_deg=self._home_lon_deg,
            home_alt_m=self._home_alt_m,
            gps_enabled=self._gps_enabled,
            sensor_noise_enabled=self._sensor_noise_enabled,
            update_rate_hz=self._sensor_rate_hz,
        )

    def start(self) -> None:
        self._backend.start()
        # Fresh sensor state on every (re)start.
        self._sensor_suite = self._make_sensor_suite()
        self._pending_poll_sim_time_us = 0
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

    @property
    def status(self) -> BridgeStatus:
        actuator_lag_us = None
        if self._last_sensor_send_us is not None and self._last_actuator_receive_us is not None:
            actuator_lag_us = self._last_sensor_send_us - self._last_actuator_receive_us
        return BridgeStatus(
            is_connected=self.is_connected,
            is_armed=self.is_armed,
            first_heartbeat_received=bool(self._backend._received_first_hearbeat),
            sensor_send_count=self._sensor_send_count,
            actuator_receive_count=self._actuator_receive_count,
            last_sensor_send_us=self._last_sensor_send_us,
            last_actuator_receive_us=self._last_actuator_receive_us,
            actuator_lag_us=actuator_lag_us,
            last_disconnect_reason=self._last_disconnect_reason,
        )

    # ---- Per-tick ----------------------------------------------------------

    def send(self, state: VehicleState) -> ActuatorCommands | None:
        """Send leg of one tick: HIL_SENSOR (+GPS) out, no blocking receive.

        Fills sensors from ``state``, runs the heartbeat handshake, and sends
        sensors/GPS. Returns ``None`` once sensors were sent and a paired
        actuator must be polled (the caller then calls :meth:`poll`). Returns a
        concrete :class:`ActuatorCommands` for the no-block cases so ``poll`` is
        skipped this tick: the last references on a non-increasing timestamp
        (no send), or zeros when disconnected, still waiting on PX4's first
        heartbeat, or on a sensor-cadence failure.

        ``send`` strictly precedes its paired ``poll`` within a vehicle; see
        :meth:`AbstractFlightStackBridge.exchange` for the single-vehicle
        send-then-poll composition.
        """
        if not self.is_connected:
            return self._zeros()

        sim_time_us = int(state.sim_time_us)
        if self._last_sensor_send_us is not None and sim_time_us <= self._last_sensor_send_us:
            _LOGGER.debug(
                "Skipping non-increasing PX4 HIL_SENSOR timestamp: current=%d last=%d",
                sim_time_us,
                self._last_sensor_send_us,
            )
            return self._references()

        imu_refreshed = self._fill_sensor_data(state)
        now = time.time()
        if not self._heartbeat_phase(now):
            return self._zeros()
        if not self._sensor_phase(sim_time_us, imu_refreshed):
            return self._zeros()
        self._pending_poll_sim_time_us = sim_time_us
        return None

    def poll(self) -> ActuatorCommands:
        """Poll leg of one tick: block (with a hard timeout) for the actuator.

        Only valid after a :meth:`send` returned ``None`` this tick (sensors
        were sent and a paired actuator is owed). Returns the current rotor
        references, or zeros on an actuator timeout.
        """
        if not self._actuator_phase(self._pending_poll_sim_time_us):
            return self._zeros()
        return self._references()

    # ---- Internals ---------------------------------------------------------

    def _references(self) -> ActuatorCommands:
        return ActuatorCommands(np.asarray(self._backend.input_reference(), dtype=np.float64))

    def _zeros(self) -> ActuatorCommands:
        return ActuatorCommands(np.zeros(self._num_rotors, dtype=np.float64))

    def _heartbeat_phase(self, now: float) -> bool:
        """Send the paced heartbeat and gate on PX4's first heartbeat.

        Returns ``True`` once PX4's first heartbeat is in and the tick may
        proceed; ``False`` while still waiting (or after a disconnect), meaning
        ``exchange()`` returns zeros this tick.
        """
        # Send heartbeat at configured rate (wallclock is fine - pacing only).
        if (now - self._last_heartbeat_send_s) >= self._heartbeat_period_s:
            try:
                self._backend.send_heartbeat()
            except Exception as exc:
                self._mark_disconnected(f"heartbeat send failed: {exc}", close_backend=True)
                return False
            self._last_heartbeat_send_s = now

        # Wait for the first heartbeat from PX4 before doing anything else.
        if not self._backend._received_first_hearbeat:
            if not self._wait_for_first_heartbeat(now):
                reason = self._last_disconnect_reason or "first heartbeat timeout"
                self._mark_disconnected(reason, close_backend=True)
            return False
        return True

    def _sensor_phase(self, sim_time_us: int, imu_refreshed: bool) -> bool:
        """Validate sensor cadence and send HIL_SENSOR (+GPS) for this tick.

        Returns ``False`` (after a fatal disconnect) on a cadence mismatch or a
        send failure, otherwise ``True``.
        """
        # Fail fast if the IMU did not refresh on a tick we are about to
        # send. With the sensor period pinned to the physics tick this only
        # happens when Isaac physics steps faster than the configured sensor
        # rate, which would otherwise ship HIL_SENSOR packets with stale
        # accel/gyro bits and starve the EKF. This is a fatal misconfiguration,
        # not a recoverable runtime state, so disconnect with a distinct reason
        # (mirroring the lockstep-timeout / actuator-lag fatal paths) rather
        # than silently degrading the sensor stream.
        if not imu_refreshed:
            self._mark_disconnected(
                "sensor cadence mismatch: IMU did not refresh this tick; "
                "Isaac physics is stepping faster than the configured sensor rate "
                f"(sensor_rate_hz={self._sensor_rate_hz:.3f} "
                f"physics_tick_us={self._physics_tick_us} "
                f"sim_time_us={sim_time_us} last_sensor_us={self._last_sensor_send_us})",
                close_backend=True,
            )
            return False

        # Send sensor + GPS messages with the caller's sim time.
        # This is an intentional Pegasus phase-order deviation: each port
        # instance treats a tick as state(t) -> sensors(t) -> actuator(t),
        # keeping lockstep causal per drone while retaining a hard timeout.
        try:
            self._send_sensor_msgs(sim_time_us)
            self._last_sensor_send_us = sim_time_us
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
            gps_due = (
                self._last_gps_send_us is None
                or (sim_time_us - self._last_gps_send_us) >= self._gps_period_s * 1_000_000
            )
            if self._gps_enabled and gps_due:
                self._backend.send_gps_msgs(sim_time_us)
                self._last_gps_send_us = sim_time_us
                _LOGGER.debug(
                    "PX4 HIL_GPS sent: time_us=%d sensor_send_count=%d",
                    sim_time_us,
                    self._sensor_send_count,
                )
        except Exception as exc:
            self._mark_disconnected(f"sensor/GPS send failed: {exc}", close_backend=True)
            return False
        return True

    def _actuator_phase(self, sim_time_us: int) -> bool:
        """Poll for the paired HIL_ACTUATOR_CONTROLS with a hard timeout.

        The first actuator may take several seconds of wallclock while PX4 boots,
        so the pre-first-actuator wait uses a generous deadline; steady-state
        ticks use the tight lockstep timeout. Returns ``False`` on timeout.
        """
        poll_timeout = (
            self._first_actuator_timeout_s
            if not self._backend._received_first_actuator
            else self._lockstep_timeout_s
        )
        return self._poll_for_actuator(poll_timeout, sim_time_us)

    def _fill_sensor_data(self, state: VehicleState) -> bool:
        """Populate ``self._backend._sensor_data`` from a VehicleState.

        Sensor values are generated through a Pegasus-style stateful sensor
        suite. Only sensors that refreshed on this tick update the Pegasus
        backend, which preserves the backend's ``fields_updated`` mask
        semantics.

        Returns:
            ``True`` if the IMU refreshed on this tick (so the next
            ``HIL_SENSOR`` will carry fresh accel/gyro bits). ``False`` means
            the Pegasus update-rate gate skipped the IMU because Isaac physics
            is stepping faster than the configured sensor rate.
        """
        updates = self._sensor_suite.update(state)
        imu_refreshed = updates["imu"] is not None
        if imu_refreshed:
            self._backend.update_imu_data(updates["imu"])
        if updates["mag"] is not None:
            self._backend.update_mag_data(updates["mag"])
        if updates["baro"] is not None:
            self._backend.update_bar_data(updates["baro"])
        if updates["gps"] is not None:
            self._backend.update_gps_data(updates["gps"])
        return imu_refreshed

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

        # Safety tripwire (not a happy-path mechanism). Strict blocking lockstep
        # keeps the lag at ~1 tick because the bridge waits for an actuator
        # before advancing, prime-then-block prevents the startup backlog, and a
        # PX4 stall is caught by the lockstep timeout. The remaining way the lag
        # could grow is a regression that reintroduces a stale-but-still-flowing
        # backlog; that would otherwise silently fly stale motor commands and
        # surface as a confusing attitude divergence. Fail closed loudly instead.
        if actuator_lag_us > self._max_sustained_actuator_lag_us:
            self._mark_disconnected(
                "actuator lag exceeded sustained limit: "
                f"sim_time_us={sim_time_us} "
                f"last_actuator_us={self._last_actuator_receive_us} "
                f"lag_us={actuator_lag_us} "
                f"max_us={self._max_sustained_actuator_lag_us}",
                close_backend=True,
            )
            return False
        return True

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
        needs_to_wait = self._needs_to_block_for_actuator()
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

        if not self._drain_actuator_backlog():
            return False
        return self._actuator_is_fresh(sim_time_us)

    def _needs_to_block_for_actuator(self) -> bool:
        """Decide whether this tick must block for an actuator (prime-then-block).

        Block once PX4 has produced its first actuator (strict lockstep), OR once
        the bootstrap priming budget is spent. PX4 needs a few priming HIL_SENSOR
        packets before it emits its first actuator, so the first
        ``_bootstrap_prime_ticks`` sends stay non-blocking; after that the bridge
        blocks for the first actuator instead of free-running, which bounds the
        startup backlog (and therefore the steady-state actuator lag).
        ``_sensor_send_count`` is incremented before this call, so on the tick
        after the budget is spent the count exceeds the budget and we block.
        """
        backend = self._backend
        return backend._enable_lockstep and (
            backend._received_first_actuator
            or self._sensor_send_count > self._bootstrap_prime_ticks
        )

    def _drain_actuator_backlog(self) -> bool:
        """Apply the newest queued HIL_ACTUATOR_CONTROLS so the bridge flies fresh.

        After the blocking wait has paired this tick with an actuator, any further
        queued actuator packets are drained non-blocking and applied newest-last,
        so a backlog never makes the bridge fly on stale motor commands. Returns
        ``False`` (after a disconnect) on a recv failure, otherwise ``True``.
        """
        from pymavlink import mavutil

        backend = self._backend
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
        return True

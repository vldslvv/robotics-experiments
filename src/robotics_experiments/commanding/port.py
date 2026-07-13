"""The side-effecting command seam (mirrors the ``Px4MavlinkPort`` lockstep seam).

:class:`MavlinkCommandPort` is the PX4-specific boundary the orchestrators talk
across: the private commander sends MAVLink commands and OFFBOARD setpoints, and
the driver pumps GCS heartbeats and drains feedback. It uses the resolved
``vehicle.gcs_port``; its ``pymavlink`` import is function-scoped so importing
``commanding`` never pulls in the MAVLink runtime, exactly as the standalone
scenario did inline.

The command port is a separate *control* layer by design: it talks the GCS link,
never the bridge's simulator MAVLink port. The bridge contract forbids the
bridge touching ``gcs_port``, and symmetrically this port never touches the
bridge's sensor/actuator lockstep.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

from robotics_experiments.commanding.commands import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    AttitudeRateSetpoint,
    MavlinkCommand,
    PositionSetpoint,
    VelocitySetpoint,
)
from robotics_experiments.commanding.observation import MAV_RESULT_ACCEPTED, CommandAck

__all__ = ["MavlinkCommandPort"]

_LOGGER = logging.getLogger(__name__)

# SET_POSITION_TARGET type mask: use position + yaw, ignore velocity (bits 3-5),
# acceleration (bits 6-8), and yaw-rate (bit 11). A NaN setpoint yaw adds the
# yaw-ignore bit (bit 10) so PX4 holds the current heading.
_POSITION_TARGET_TYPEMASK_POS_YAW = 0b1001_1111_1000
_POSITION_TARGET_TYPEMASK_YAW_IGNORE = 0b0100_0000_0000

# MAV_FRAME_GLOBAL_INT (geodetic lat/lon e7 + AMSL altitude); a MAVLink protocol
# constant named here so streaming a setpoint does not need a pymavlink import for
# the frame id (the same pattern as the MAV_CMD ids in the command payload layer).
_MAV_FRAME_GLOBAL_INT = 5

# MAV_FRAME_BODY_NED (body-forward/right/down) for the deploy-time velocity
# setpoint. SET_POSITION_TARGET_LOCAL_NED velocity type mask: use velocity (bits
# 3-5) + yaw-rate (bit 11 clear); ignore position (bits 0-2), acceleration (bits
# 6-8), and yaw (bit 10).
_MAV_FRAME_BODY_NED = 8
_VELOCITY_TARGET_TYPEMASK = 0b0101_1100_0111

# SET_ATTITUDE_TARGET type mask for the deploy-time CTBR setpoint: ignore the
# attitude quaternion (bit 7) and use the body rates (bits 0-2 clear) + thrust
# (bit 6 clear). ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE = 128.
_ATTITUDE_TARGET_TYPEMASK_RATES = 0b1000_0000

# PX4 emits this statustext when an external force-arm succeeds without a
# COMMAND_ACK; treat it as an acceptance of the arm command. PX4-specific
# knowledge, localized to the PX4 command port.
_ARM_ACCEPTED_STATUSTEXT = "Armed by external command"

# MAVLink system_status (MAV_STATE) values that mean PX4 is in a failsafe state:
# MAV_STATE_CRITICAL=5, MAV_STATE_EMERGENCY=6, MAV_STATE_FLIGHT_TERMINATION=8.
# Note MAV_STATE_ACTIVE=4 is the normal armed/flying state, NOT a failsafe.
_FAILSAFE_SYSTEM_STATUS = frozenset({5, 6, 8})


class MavlinkCommandPort:
    """PX4 command port over a GCS MAVLink (UDP) endpoint.

    Absorbs the GCS-link concerns previously inline in the standalone takeoff
    scenario: connecting, throttled GCS heartbeats, ``command_long`` sends, and
    draining ``COMMAND_ACK`` / ``HEARTBEAT`` / ``STATUSTEXT`` /
    ``GLOBAL_POSITION_INT`` / ``GPS_RAW_INT`` to learn the target system, detect
    failsafe, accept the arm statustext, and surface the latest global position.
    """

    def __init__(
        self,
        *,
        gcs_endpoint: str,
        heartbeat_period_s: float,
        label: str = "",
        connection: Any = None,
    ) -> None:
        """Build the port. No socket is opened until :meth:`connect`.

        Args:
            gcs_endpoint: pymavlink connection string, e.g. ``udp:127.0.0.1:14550``.
            heartbeat_period_s: Minimum wall-clock spacing between GCS heartbeats.
            label: Short vehicle tag for log messages.
            connection: Optional pre-built MAVLink connection (a ``mavutil``
                connection or test double). When provided, :meth:`connect` adopts
                it instead of dialing pymavlink -- an injectable seam so feedback
                parsing can be exercised without opening a socket.
        """
        self._gcs_endpoint = gcs_endpoint
        self._heartbeat_period_s = float(heartbeat_period_s)
        self._label = label
        # Typed Any (not Optional): None before connect(), a live mavutil
        # connection after; keeping it Any avoids Optional-member churn on every
        # access of the function-scoped pymavlink object.
        self._mav: Any = connection
        self._last_heartbeat_sent_s = 0.0
        self._target_known = False
        self._failsafe = False
        self._failsafe_system_status = 0
        self._latest_global_position: Optional[tuple[float, float, float]] = None

    @property
    def is_handshaked(self) -> bool:
        """Whether the target system/component has been learned from a heartbeat."""
        return self._target_known

    @property
    def failsafe(self) -> bool:
        """Whether the last drained heartbeat reported a failsafe system status."""
        return self._failsafe

    @property
    def failsafe_system_status(self) -> int:
        """The last heartbeat ``system_status`` value (for failsafe diagnostics)."""
        return self._failsafe_system_status

    @property
    def latest_global_position(self) -> Optional[tuple[float, float, float]]:
        """Latest ``(latitude_deg, longitude_deg, altitude_m)`` telemetry, or ``None``."""
        return self._latest_global_position

    def connect(self) -> None:
        if self._mav is None:
            from pymavlink import mavutil

            _LOGGER.info("[%s] opening GCS MAVLink endpoint %s", self._label, self._gcs_endpoint)
            self._mav = mavutil.mavlink_connection(self._gcs_endpoint, source_system=255)
        self._last_heartbeat_sent_s = 0.0

    def disconnect(self) -> None:
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:  # noqa: BLE001 - best-effort close on teardown
                pass
            self._mav = None

    def pump_heartbeat(self, *, force: bool = False) -> None:
        from pymavlink import mavutil

        now_s = time.monotonic()
        if not force and now_s - self._last_heartbeat_sent_s < self._heartbeat_period_s:
            return
        self._mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )
        self._last_heartbeat_sent_s = now_s

    def _require_target(self, action: str) -> None:
        """Guard: every send needs the PX4 target learned from a heartbeat first."""
        if not self._target_known:
            raise RuntimeError(
                f"[{self._label}] cannot {action}: GCS link has not "
                "learned the PX4 target system yet (no heartbeat handshake)"
            )

    def send(self, command: MavlinkCommand) -> None:
        self._require_target(f"send {command.label!r}")
        _LOGGER.info("[%s] sending MAVLink command %s", self._label, command.label)
        self._mav.mav.command_long_send(
            self._mav.target_system,
            self._mav.target_component,
            command.command_id,
            0,
            *command.params,
        )

    def send_setpoint(self, setpoint: PositionSetpoint) -> None:
        self._require_target("stream a setpoint")
        type_mask = _POSITION_TARGET_TYPEMASK_POS_YAW
        yaw = setpoint.yaw_rad
        if not math.isfinite(yaw):
            type_mask |= _POSITION_TARGET_TYPEMASK_YAW_IGNORE
            yaw = 0.0
        self._mav.mav.set_position_target_global_int_send(
            0,  # time_boot_ms (unused by PX4)
            self._mav.target_system,
            self._mav.target_component,
            _MAV_FRAME_GLOBAL_INT,
            type_mask,
            int(round(setpoint.latitude_deg * 1e7)),
            int(round(setpoint.longitude_deg * 1e7)),
            float(setpoint.altitude_m),
            0.0,
            0.0,
            0.0,  # velocity (ignored)
            0.0,
            0.0,
            0.0,  # acceleration (ignored)
            float(yaw),
            0.0,  # yaw_rate (ignored)
        )

    def send_velocity_setpoint(self, setpoint: VelocitySetpoint) -> None:
        self._require_target("stream a velocity setpoint")
        self._mav.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (unused by PX4)
            self._mav.target_system,
            self._mav.target_component,
            _MAV_FRAME_BODY_NED,
            _VELOCITY_TARGET_TYPEMASK,
            0.0,
            0.0,
            0.0,  # position (ignored)
            float(setpoint.forward_mps),
            float(setpoint.right_mps),
            float(setpoint.down_mps),
            0.0,
            0.0,
            0.0,  # acceleration (ignored)
            0.0,  # yaw (ignored)
            float(setpoint.yaw_rate_radps),
        )

    def send_attitude_target(self, setpoint: AttitudeRateSetpoint) -> None:
        self._require_target("stream an attitude setpoint")
        self._mav.mav.set_attitude_target_send(
            0,  # time_boot_ms (unused by PX4)
            self._mav.target_system,
            self._mav.target_component,
            _ATTITUDE_TARGET_TYPEMASK_RATES,
            [1.0, 0.0, 0.0, 0.0],  # attitude quaternion (ignored)
            float(setpoint.roll_rate_radps),
            float(setpoint.pitch_rate_radps),
            float(setpoint.yaw_rate_radps),
            float(setpoint.thrust),
        )

    def poll_feedback(self) -> Optional[CommandAck]:
        latest_ack: Optional[CommandAck] = None
        while True:
            msg = self._mav.recv_match(blocking=False)
            if msg is None:
                return latest_ack
            ack = self._handle_message(msg)
            if ack is not None:
                latest_ack = ack

    def _handle_message(self, msg: Any) -> Optional[CommandAck]:
        msg_type = msg.get_type()
        if msg_type == "HEARTBEAT":
            if not self._target_known:
                self._mav.target_system = msg.get_srcSystem()
                self._mav.target_component = msg.get_srcComponent()
                self._target_known = True
                _LOGGER.info(
                    "[%s] GCS learned target system=%s component=%s",
                    self._label,
                    msg.get_srcSystem(),
                    msg.get_srcComponent(),
                )
            self._failsafe_system_status = int(getattr(msg, "system_status", 0))
            self._failsafe = self._failsafe_system_status in _FAILSAFE_SYSTEM_STATUS
            return None
        if msg_type == "COMMAND_ACK":
            return CommandAck.from_result(int(msg.command), int(msg.result))
        if msg_type == "STATUSTEXT":
            text = str(getattr(msg, "text", ""))
            _LOGGER.info("[%s] PX4 STATUSTEXT: %s", self._label, text)
            if _ARM_ACCEPTED_STATUSTEXT in text:
                return CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, MAV_RESULT_ACCEPTED)
            return None
        if msg_type in {"GLOBAL_POSITION_INT", "GPS_RAW_INT"}:
            self._update_global_position(msg)
            return None
        return None

    def _update_global_position(self, msg: Any) -> None:
        lat_e7 = getattr(msg, "lat", None)
        lon_e7 = getattr(msg, "lon", None)
        alt_mm = getattr(msg, "alt", None)
        if lat_e7 is None or lon_e7 is None or alt_mm is None:
            return
        latitude_deg = float(lat_e7) / 10_000_000.0
        longitude_deg = float(lon_e7) / 10_000_000.0
        altitude_m = float(alt_mm) / 1000.0
        if not all(math.isfinite(v) for v in (latitude_deg, longitude_deg, altitude_m)):
            return
        if abs(latitude_deg) < 1e-7 and abs(longitude_deg) < 1e-7:
            # PX4 publishes (0, 0) before it has a fix; ignore until valid.
            return
        self._latest_global_position = (latitude_deg, longitude_deg, altitude_m)

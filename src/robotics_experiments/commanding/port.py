"""The side-effecting command seam (mirrors ``AbstractFlightStackBridge``).

:class:`AbstractCommandPort` is the swappable boundary the orchestrators talk
across: the pure :class:`VehicleCommander` only ever calls :meth:`send`, and the
driver pumps heartbeats and drains feedback. :class:`MavlinkCommandPort` is the
PX4 implementation over a GCS MAVLink link (the resolved ``vehicle.gcs_port``);
its ``pymavlink`` import is function-scoped so importing ``commanding`` never
pulls in the MAVLink runtime, exactly as the standalone scenario did inline.

The command port is a separate *control* layer by design: it talks the GCS link,
never the bridge's simulator MAVLink port. The bridge contract forbids the
bridge touching ``gcs_port``, and symmetrically this port never touches the
bridge's sensor/actuator lockstep.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from .commands import MavlinkCommand
from .observation import CommandAck

__all__ = ["AbstractCommandPort", "MavlinkCommandPort"]

_LOGGER = logging.getLogger(__name__)

# PX4 emits this statustext when an external force-arm succeeds without a
# COMMAND_ACK; treat it as an acceptance of the arm command. PX4-specific
# knowledge, localized to the PX4 command port.
_ARM_ACCEPTED_STATUSTEXT = "Armed by external command"

# MAVLink system_status (MAV_STATE) values that mean PX4 is in a failsafe state:
# MAV_STATE_CRITICAL=5, MAV_STATE_EMERGENCY=6, MAV_STATE_FLIGHT_TERMINATION=8.
# Note MAV_STATE_ACTIVE=4 is the normal armed/flying state, NOT a failsafe.
_FAILSAFE_SYSTEM_STATUS = frozenset({5, 6, 8})


class AbstractCommandPort(ABC):
    """Stage-able seam between the command orchestrators and a flight stack link.

    Lifecycle: :meth:`connect` opens the link, :meth:`disconnect` closes it. Per
    tick the driver calls :meth:`pump_heartbeat` (throttled GCS heartbeat) and
    :meth:`poll_feedback` (drain inbound MAVLink, return the latest ack), while
    the commander calls :meth:`send` when a routine step begins.
    """

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def pump_heartbeat(self, *, force: bool = False) -> None: ...

    @abstractmethod
    def send(self, command: MavlinkCommand) -> None: ...

    @abstractmethod
    def poll_feedback(self) -> Optional[CommandAck]:
        """Drain inbound MAVLink and return the most recent ack this tick."""

    @property
    @abstractmethod
    def is_handshaked(self) -> bool:
        """Whether the target system/component has been learned from a heartbeat."""

    @property
    @abstractmethod
    def failsafe(self) -> bool:
        """Whether the last drained heartbeat reported a failsafe system status."""

    @property
    @abstractmethod
    def failsafe_system_status(self) -> int:
        """The last heartbeat ``system_status`` value (for failsafe diagnostics)."""

    @property
    @abstractmethod
    def latest_global_position(self) -> Optional[tuple[float, float, float]]:
        """Latest ``(latitude_deg, longitude_deg, altitude_m)`` telemetry, or ``None``."""


class MavlinkCommandPort(AbstractCommandPort):
    """PX4 command port over a GCS MAVLink (UDP) endpoint.

    Absorbs the GCS-link concerns previously inline in the standalone takeoff
    scenario: connecting, throttled GCS heartbeats, ``command_long`` sends, and
    draining ``COMMAND_ACK`` / ``HEARTBEAT`` / ``STATUSTEXT`` /
    ``GLOBAL_POSITION_INT`` / ``GPS_RAW_INT`` to learn the target system, detect
    failsafe, accept the arm statustext, and surface the latest global position.
    """

    def __init__(self, *, gcs_endpoint: str, heartbeat_period_s: float, label: str = "") -> None:
        """Build the port. No socket is opened until :meth:`connect`.

        Args:
            gcs_endpoint: pymavlink connection string, e.g. ``udp:127.0.0.1:14550``.
            heartbeat_period_s: Minimum wall-clock spacing between GCS heartbeats.
            label: Short vehicle tag for log messages.
        """
        self._gcs_endpoint = gcs_endpoint
        self._heartbeat_period_s = float(heartbeat_period_s)
        self._label = label
        # Typed Any (not Optional): None before connect(), a live mavutil
        # connection after; keeping it Any avoids Optional-member churn on every
        # access of the function-scoped pymavlink object.
        self._mav: Any = None
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

    def send(self, command: MavlinkCommand) -> None:
        if not self._target_known:
            raise RuntimeError(
                f"[{self._label}] cannot send {command.label!r}: GCS link has not "
                "learned the PX4 target system yet (no heartbeat handshake)"
            )
        _LOGGER.info("[%s] sending MAVLink command %s", self._label, command.label)
        self._mav.mav.command_long_send(
            self._mav.target_system,
            self._mav.target_component,
            command.command_id,
            0,
            *command.params,
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
                from .commands import MAV_CMD_COMPONENT_ARM_DISARM
                from .observation import MAV_RESULT_ACCEPTED

                return CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, MAV_RESULT_ACCEPTED)
            return None
        if msg_type in {"GLOBAL_POSITION_INT", "GPS_RAW_INT"}:
            self._update_global_position(msg)
            return None
        return None

    def _update_global_position(self, msg: Any) -> None:
        import math

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

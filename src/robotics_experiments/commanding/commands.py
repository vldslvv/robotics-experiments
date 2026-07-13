"""Thin, pure MAVLink command wrappers.

These are dependency-free data objects: a :class:`MavlinkCommand` is one PX4
``command_long`` (a ``MAV_CMD_*`` id plus its seven float params) plus a human
label. The module deliberately imports neither ``pymavlink`` nor Isaac so it
stays on the core import path; the numeric ``MAV_CMD_*`` ids are named int
constants here rather than read from ``pymavlink.mavutil.mavlink`` so the pure
command layer never drags the MAVLink runtime onto the import path.

The side-effecting :class:`MavlinkCommandPort` is the only layer that turns a
:class:`MavlinkCommand` into an actual ``command_long_send``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MAV_CMD_COMPONENT_ARM_DISARM",
    "MAV_CMD_DO_REPOSITION",
    "MAV_CMD_DO_SET_MODE",
    "MAV_CMD_NAV_LAND",
    "MAV_CMD_NAV_TAKEOFF",
    "MAV_DO_REPOSITION_FLAGS_CHANGE_MODE",
    "AttitudeRateSetpoint",
    "MavlinkCommand",
    "PositionSetpoint",
    "VelocitySetpoint",
    "arm",
    "land",
    "reposition",
    "set_offboard_mode",
    "takeoff",
]

# MAVLink command ids (MAV_CMD_*). Named here as plain ints so the pure command
# layer does not import pymavlink. Values match the MAVLink common message set.
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_REPOSITION = 192
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_COMPONENT_ARM_DISARM = 400

# MAV_DO_REPOSITION_FLAGS_CHANGE_MODE asks PX4 to enter the mode needed to
# execute MAV_CMD_DO_REPOSITION instead of denying it while in the wrong mode.
MAV_DO_REPOSITION_FLAGS_CHANGE_MODE = 1

# MAV_CMD_DO_SET_MODE base-mode flag: the custom (autopilot-specific) mode in
# param2/param3 is active. PX4's main mode for offboard control is 6; both are
# PX4/MAVLink protocol constants, not tunables.
_MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
_PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

# PX4 force-arm magic value, passed as the ``MAV_CMD_COMPONENT_ARM_DISARM``
# param2 float to bypass pre-arm checks. A PX4 protocol constant, not a tunable.
_PX4_FORCE_ARM_MAGIC = 21196


@dataclass(frozen=True)
class MavlinkCommand:
    """One PX4 ``command_long``: a ``MAV_CMD_*`` id plus its seven params.

    Attributes:
        command_id: The ``MAV_CMD_*`` id (see the module constants).
        params: Exactly seven float params (param1..param7) for ``command_long``.
        label: Short human-readable name used in logs and routine steps.
    """

    command_id: int
    params: tuple[float, float, float, float, float, float, float]
    label: str

    def __post_init__(self) -> None:
        if len(self.params) != 7:
            raise ValueError(f"MavlinkCommand.params must have 7 entries, got {len(self.params)}")


def arm() -> MavlinkCommand:
    """Force-arm the vehicle (``MAV_CMD_COMPONENT_ARM_DISARM``).

    param1=1 arms; param2 carries the PX4 force-arm magic to bypass pre-arm
    checks, matching the standalone takeoff scenario's behavior.
    """
    return MavlinkCommand(
        command_id=MAV_CMD_COMPONENT_ARM_DISARM,
        params=(1.0, float(_PX4_FORCE_ARM_MAGIC), 0.0, 0.0, 0.0, 0.0, 0.0),
        label="arm",
    )


def takeoff(
    latitude_deg: float, longitude_deg: float, absolute_altitude_m: float
) -> MavlinkCommand:
    """Take off to an absolute altitude at a geographic point.

    Mirrors ``MAV_CMD_NAV_TAKEOFF`` as the scenario sends it: yaw (param4) is
    ``NaN`` (hold current heading) and param5/6/7 are the target lat/lon and
    absolute altitude. The caller computes ``absolute_altitude_m`` as the current
    reported altitude plus the desired climb.
    """
    return MavlinkCommand(
        command_id=MAV_CMD_NAV_TAKEOFF,
        params=(
            0.0,
            0.0,
            0.0,
            math.nan,
            float(latitude_deg),
            float(longitude_deg),
            float(absolute_altitude_m),
        ),
        label="takeoff",
    )


def reposition(latitude_deg: float, longitude_deg: float, altitude_m: float) -> MavlinkCommand:
    """Reposition to a geographic point and altitude (``MAV_CMD_DO_REPOSITION``).

    Forward-looking scaffolding for navigate-to-point / set-altitude routines:
    param1=-1 keeps the default ground speed, param2 asks PX4 to switch into the
    mode needed to execute the reposition, param4 yaw is ``NaN`` (unchanged), and
    param5/6/7 are the target lat/lon/altitude.
    """
    return MavlinkCommand(
        command_id=MAV_CMD_DO_REPOSITION,
        params=(
            -1.0,
            float(MAV_DO_REPOSITION_FLAGS_CHANGE_MODE),
            0.0,
            math.nan,
            float(latitude_deg),
            float(longitude_deg),
            float(altitude_m),
        ),
        label="reposition",
    )


def land() -> MavlinkCommand:
    """Land at the current position (``MAV_CMD_NAV_LAND``).

    Forward-looking scaffolding: all params left at their neutral defaults
    (yaw param4 ``NaN``), so PX4 lands at the current location.
    """
    return MavlinkCommand(
        command_id=MAV_CMD_NAV_LAND,
        params=(0.0, 0.0, 0.0, math.nan, 0.0, 0.0, 0.0),
        label="land",
    )


def set_offboard_mode() -> MavlinkCommand:
    """Switch PX4 into OFFBOARD flight mode (``MAV_CMD_DO_SET_MODE``).

    param1 is the base-mode flag marking the custom mode active; param2 is PX4's
    custom main mode (OFFBOARD); param3 (custom sub mode) is unused. PX4 only
    accepts OFFBOARD once a setpoint stream is already flowing, so a ``goto`` step
    primes setpoints before this is sent (see ``decide``).
    """
    return MavlinkCommand(
        command_id=MAV_CMD_DO_SET_MODE,
        params=(
            float(_MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(_PX4_CUSTOM_MAIN_MODE_OFFBOARD),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        label="set_offboard_mode",
    )


@dataclass(frozen=True)
class PositionSetpoint:
    """One streamed OFFBOARD position-plus-yaw setpoint (a wire payload).

    A geodetic target the side-effecting port serializes to a
    ``SET_POSITION_TARGET_GLOBAL_INT``. Carried here (next to
    :class:`MavlinkCommand`) as a pure, dependency-free data object so both the
    pure stepper and the port reference it without importing each other.

    Attributes:
        latitude_deg, longitude_deg: Target geographic position.
        altitude_m: Target AMSL altitude (the ``MAV_FRAME_GLOBAL_INT`` frame).
        yaw_rad: Target heading in NED (0 = north, +clockwise); ``NaN`` means
            hold the current heading (the port sets the yaw-ignore type mask).
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    yaw_rad: float


@dataclass(frozen=True)
class VelocitySetpoint:
    """One streamed OFFBOARD body-velocity setpoint (a wire payload).

    The deploy-time analogue of :class:`PositionSetpoint` for the ``velocity``
    RL action level: a body-frame velocity + yaw rate the port serializes to a
    ``SET_POSITION_TARGET_LOCAL_NED`` in ``MAV_FRAME_BODY_NED`` with the velocity
    fields enabled and position/acceleration ignored. Components are in the PX4
    body-FRD convention (forward / right / down), so the pure action->setpoint
    mapper (``robotics_experiments.rl.press.press_action_to_offboard``) converts the body-FLU RL
    action into these.

    Attributes:
        forward_mps: Body-forward (FRD +X) velocity.
        right_mps: Body-right (FRD +Y) velocity.
        down_mps: Body-down (FRD +Z) velocity.
        yaw_rate_radps: Yaw rate in NED (+clockwise seen from above).
    """

    forward_mps: float
    right_mps: float
    down_mps: float
    yaw_rate_radps: float


@dataclass(frozen=True)
class AttitudeRateSetpoint:
    """One streamed OFFBOARD collective-thrust + body-rate setpoint (CTBR).

    The deploy-time analogue of :class:`PositionSetpoint` for the ``ctbr`` RL
    action level: the port serializes it to a ``SET_ATTITUDE_TARGET`` whose type
    mask ignores the attitude quaternion and uses the body rates + thrust. Body
    rates are in the PX4 body-FRD convention; ``thrust`` is the PX4 normalized
    collective in ``[0, 1]``.

    Attributes:
        thrust: Normalized collective thrust in ``[0, 1]``.
        roll_rate_radps: Body-roll rate (FRD +X).
        pitch_rate_radps: Body-pitch rate (FRD +Y).
        yaw_rate_radps: Body-yaw rate (FRD +Z).
    """

    thrust: float
    roll_rate_radps: float
    pitch_rate_radps: float
    yaw_rate_radps: float

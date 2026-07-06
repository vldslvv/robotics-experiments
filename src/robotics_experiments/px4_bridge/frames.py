"""Canonical frame and convention conversions for bridge-owned code.

This module is the single source of truth for non-vendored Isaac<->PX4 frame
math used by ``robotics_experiments.px4_bridge``:

- Quaternion ordering (``[w, x, y, z]`` everywhere internally; only flipped at
  the Pegasus/MAVLink boundary).
- ENU world-frame and FLU body-frame vector conversions used by bridge-side
  runtime code.
- Reference-only GPS lat/lon synthesis from a local ENU offset around a fixed
  home.
- Reference-only magnetometer body-frame synthesis from a supplied fixed NED
  reference field.

Bridge code should route frame-convention math through this module instead of
open-coding sign flips or quaternion reorders. The vendored Pegasus
``rotations.py`` module remains internal to the Pegasus sensor/backend path.

The runtime PX4 MAVLink path intentionally obtains sensor payloads from the
vendored Pegasus sensors through ``_PegasusSensorSuite``. The private
magnetometer/barometer synthesis helpers below are deterministic reference
oracles for the equivalence tests, which keep this canonical layer aligned with
the Pegasus path and fail if the two drift beyond tolerance;
``synth_gps_from_enu`` additionally serves the commanding layer's world-ENU ->
geodetic setpoint conversion.
"""

from __future__ import annotations

import math

import numpy as np

# ---- Standing constants -----------------------------------------------------

GRAVITY_M_S2: float = 9.81

# WGS84 mean radius (m). Sufficient for indoor-scale GPS synthesis.
_EARTH_RADIUS_M: float = 6_371_000.0

# Default magnetic field reference (Zurich-ish), expressed in NED, Gauss.
# Magnitude ~0.48 G; inclination ~63 degrees. Precision is irrelevant for HIL.
# PX4 only needs a consistent vector that rotates with attitude.
DEFAULT_MAG_NED_GAUSS: np.ndarray = np.array([0.215, 0.0, 0.43], dtype=np.float64)

# Default home location: PX4's built-in default (Zurich).
DEFAULT_HOME_LAT_DEG: float = 47.3977
DEFAULT_HOME_LON_DEG: float = 8.5456
DEFAULT_HOME_ALT_M: float = 488.0

__all__ = [
    "GRAVITY_M_S2",
    "DEFAULT_MAG_NED_GAUSS",
    "DEFAULT_HOME_LAT_DEG",
    "DEFAULT_HOME_LON_DEG",
    "DEFAULT_HOME_ALT_M",
    "normalize_quat_wxyz",
    "quat_conjugate_wxyz",
    "quat_multiply_wxyz",
    "quat_wxyz_to_xyzw",
    "rotate_vector_wxyz",
    "world_enu_to_body_flu",
    "body_velocity_flu_from_state",
    "tilt_angle_deg",
    "synth_barometer",
    "synth_gps_from_enu",
    "synth_magnetometer_body_flu",
]


# ---- Quaternion helpers -----------------------------------------------------


def normalize_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Return the unit quaternion; raise ``ValueError`` on zero norm."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    return q / norm


def quat_conjugate_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Return the quaternion conjugate (inverse rotation for a unit quat)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply_wxyz(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of two wxyz quaternions."""
    aw, ax, ay, az = (float(v) for v in a_wxyz)
    bw, bx, by, bz = (float(v) for v in b_wxyz)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Reorder a quaternion from ``[w, x, y, z]`` to ``[x, y, z, w]``."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def rotate_vector_wxyz(q_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by quaternion ``q_wxyz``."""
    q = normalize_quat_wxyz(q_wxyz)
    v = np.array([0.0, *np.asarray(vec, dtype=np.float64)], dtype=np.float64)
    return quat_multiply_wxyz(quat_multiply_wxyz(q, v), quat_conjugate_wxyz(q))[1:]


# ---- World<->body conversions ----------------------------------------------


def world_enu_to_body_flu(
    q_flu_enu_wxyz: np.ndarray,
    vec_enu: np.ndarray,
) -> np.ndarray:
    """Rotate a world-ENU vector into body-FLU using the body attitude."""
    return rotate_vector_wxyz(
        quat_conjugate_wxyz(normalize_quat_wxyz(q_flu_enu_wxyz)),
        np.asarray(vec_enu, dtype=np.float64),
    )


def body_velocity_flu_from_state(state) -> np.ndarray:
    """Return true body-FLU linear velocity from a ``VehicleState``.

    This is computed directly from the world-ENU velocity and the body
    attitude. Pegasus ``State.get_linear_body_velocity_ned_frd`` (fixed upstream
    in ``bef3c57``) returns body velocity in the NED-FRD convention, whereas the
    bridge needs body-FLU here, so we compute it rather than route through it.
    """
    return world_enu_to_body_flu(
        state.orientation_flu_enu_wxyz,
        state.linear_velocity_enu,
    )


def tilt_angle_deg(q_flu_enu_wxyz: np.ndarray) -> float:
    """Angle between body-Z (up) and world-Z in degrees.

    Returns 0 when the body is level and 180 when fully inverted. Pure yaw
    rotations do not change the tilt angle.
    """
    body_z_world = rotate_vector_wxyz(
        q_flu_enu_wxyz,
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    cos_angle = float(np.clip(body_z_world[2], -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def synth_magnetometer_body_flu(
    q_flu_enu_wxyz: np.ndarray,
    mag_ned_gauss: np.ndarray = DEFAULT_MAG_NED_GAUSS,
) -> np.ndarray:
    """Synthesize a reference body-FLU reading from a supplied NED field.

    The reference field is given in NED so it matches the values typically
    quoted from geomagnetic models. This helper does not compute a geomagnetic
    model; callers that need Pegasus-equivalent values should pass the field
    derived from the vendored geomagnetic utilities for the test location.
    Output units match the input (Gauss).
    """
    mag_ned = np.asarray(mag_ned_gauss, dtype=np.float64)
    mag_enu = np.array([mag_ned[1], mag_ned[0], -mag_ned[2]], dtype=np.float64)
    return world_enu_to_body_flu(q_flu_enu_wxyz, mag_enu)


# ---- GPS synthesis ----------------------------------------------------------


def synth_gps_from_enu(
    position_enu: np.ndarray,
    velocity_enu: np.ndarray,
    home_lat_deg: float = DEFAULT_HOME_LAT_DEG,
    home_lon_deg: float = DEFAULT_HOME_LON_DEG,
    home_alt_m: float = DEFAULT_HOME_ALT_M,
) -> dict:
    """Synthesize reference ``HIL_GPS`` payload fields from a local ENU pose.

    Treats the local ENU frame as a flat tangent plane around the home
    coordinate. Sufficient for indoor-scale operation where GPS realism is
    not the goal - the only requirement is internal consistency so PX4's EKF
    accepts the readings.

    Args:
        position_enu: ``[east, north, up]`` meters relative to home.
        velocity_enu: ENU velocity in m/s.
        home_lat_deg: Home latitude, degrees.
        home_lon_deg: Home longitude, degrees.
        home_alt_m: Home altitude AMSL, meters.

    Returns:
        Dict with the keys consumed by ``PX4MavlinkBackend.update_gps_data``.
    """
    east, north, up = (float(c) for c in np.asarray(position_enu, dtype=np.float64))
    ve, vn, vu = (float(c) for c in np.asarray(velocity_enu, dtype=np.float64))

    home_lat_rad = math.radians(home_lat_deg)
    dlat_deg = (north / _EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon_deg = (east / (_EARTH_RADIUS_M * math.cos(home_lat_rad))) * (180.0 / math.pi)

    speed_2d = math.hypot(vn, ve)
    cog_deg = (math.degrees(math.atan2(ve, vn)) + 360.0) % 360.0

    return {
        "fix_type": 3,  # 3D fix
        "latitude": home_lat_deg + dlat_deg,
        "longitude": home_lon_deg + dlon_deg,
        "altitude": home_alt_m + up,
        "eph": 1.0,
        "epv": 1.0,
        "speed": speed_2d,
        "velocity_north": vn,
        "velocity_east": ve,
        "velocity_down": -vu,
        "cog": cog_deg,
        "sattelites_visible": 10,  # spelling matches Pegasus' key
        "latitude_gt": home_lat_deg + dlat_deg,
        "longitude_gt": home_lon_deg + dlon_deg,
        "altitude_gt": home_alt_m + up,
    }


# ---- Barometer synthesis ----------------------------------------------------


def synth_barometer(altitude_m: float, sea_level_pressure_hpa: float = 1013.25) -> dict:
    """Synthesize a reference barometer payload (ISA approximation).

    Returns:
        Dict with keys ``temperature``, ``absolute_pressure``,
        ``pressure_altitude`` matching ``PX4MavlinkBackend.update_bar_data``.
    """
    # Standard atmosphere lapse: p(h) = p0 * (1 - 0.0065 h / 288.15)^5.255
    p_hpa = sea_level_pressure_hpa * (1.0 - 0.0065 * altitude_m / 288.15) ** 5.255
    return {
        "temperature": 25.0,
        "absolute_pressure": p_hpa,
        "pressure_altitude": altitude_m,
    }

"""Frame and convention conversions used at the Isaac<->PX4 boundary.

Single source of truth for:

- Quaternion ordering (``[w, x, y, z]`` everywhere internally; only flipped at
  the MAVLink wire boundary, which Pegasus' ``state.py`` already handles).
- ENU<->NED inertial frame conversion.
- FLU<->FRD body frame conversion.
- Specific-force synthesis (gravity-inclusive body acceleration the
  accelerometer would report).
- GPS lat/lon synthesis from a local ENU offset around a fixed home.
- Magnetometer body-frame synthesis from a fixed NED reference field.

Keeping these conversions out of every other layer ensures each conversion is
applied exactly once.
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
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "rotate_vector_wxyz",
    "world_enu_to_body_flu",
    "specific_force_body_flu",
    "magnetometer_body_flu",
    "synth_gps_from_enu",
    "synth_barometer",
]


# ---- Quaternion helpers -----------------------------------------------------


def _normalize_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    return q / norm


def _quat_conjugate_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply_wxyz(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
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


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Reorder a quaternion from ``[x, y, z, w]`` to ``[w, x, y, z]``."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def rotate_vector_wxyz(q_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by quaternion ``q_wxyz``."""
    q = _normalize_quat_wxyz(q_wxyz)
    v = np.array([0.0, *np.asarray(vec, dtype=np.float64)], dtype=np.float64)
    return _quat_multiply_wxyz(_quat_multiply_wxyz(q, v), _quat_conjugate_wxyz(q))[1:]


# ---- World<->body conversions ----------------------------------------------


def world_enu_to_body_flu(
    q_flu_enu_wxyz: np.ndarray,
    vec_enu: np.ndarray,
) -> np.ndarray:
    """Rotate a world-ENU vector into body-FLU using the body attitude."""
    return rotate_vector_wxyz(
        _quat_conjugate_wxyz(_normalize_quat_wxyz(q_flu_enu_wxyz)),
        np.asarray(vec_enu, dtype=np.float64),
    )


def specific_force_body_flu(
    linear_acceleration_enu: np.ndarray,
    q_flu_enu_wxyz: np.ndarray,
    gravity: float = GRAVITY_M_S2,
) -> np.ndarray:
    """Compute the body-FLU specific force an accelerometer would report.

    ``f_body = R_world->body @ (a_inertial - g_world_enu)`` where
    ``g_world_enu = [0, 0, -gravity]`` (gravity points down in ENU). At rest
    and level this returns ``[0, 0, +gravity]`` in FLU - i.e. the IMU reports
    +9.81 along body up, matching real-world behavior.
    """
    a_inertial = np.asarray(linear_acceleration_enu, dtype=np.float64)
    g_world_enu = np.array([0.0, 0.0, -gravity], dtype=np.float64)
    return world_enu_to_body_flu(q_flu_enu_wxyz, a_inertial - g_world_enu)


def magnetometer_body_flu(
    q_flu_enu_wxyz: np.ndarray,
    mag_ned_gauss: np.ndarray = DEFAULT_MAG_NED_GAUSS,
) -> np.ndarray:
    """Synthesize a body-FLU magnetometer reading from a NED reference field.

    The reference field is given in NED so it matches the values typically
    quoted from geomagnetic models. This routine converts to ENU, rotates into
    body using the attitude quaternion, and converts FRD-style readings back
    into FLU. Output units match the input (Gauss).
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
    """Synthesize PX4 ``HIL_GPS`` payload fields from a local ENU pose.

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
    """Synthesize a minimal barometer payload (ISA approximation).

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

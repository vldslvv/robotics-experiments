from pathlib import Path

import numpy as np
import pytest

from robotics_experiments.px4_bridge import frames
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

_PX4_BRIDGE_EXTENSION_ROOT = (
    Path(__file__).resolve().parents[1] / "isaac/runtime/extensions/px4_sitl_brigde"
)


def _pegasus_state_from_vehicle(state: VehicleState):
    from pegasus.simulator.logic.state import State

    pegasus_state = State()
    q_wxyz = np.asarray(state.orientation_flu_enu_wxyz, dtype=np.float64)
    pegasus_state.position = np.asarray(state.position_enu, dtype=np.float64)
    pegasus_state.attitude = np.array(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]],
        dtype=np.float64,
    )
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


def _flu_to_frd(vector_flu: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector_flu, dtype=np.float64)
    return np.array([vector[0], -vector[1], -vector[2]], dtype=np.float64)


def _wmm_field_ned_gauss(
    position_enu: np.ndarray,
    home_lat_deg: float,
    home_lon_deg: float,
) -> np.ndarray:
    from pegasus.simulator.logic.sensors.geo_mag_utils import (
        get_mag_declination,
        get_mag_inclination,
        get_mag_strength,
        reprojection,
    )

    latitude_rad, longitude_rad = reprojection(
        np.asarray(position_enu, dtype=np.float64),
        np.radians(home_lat_deg),
        np.radians(home_lon_deg),
    )
    latitude_deg = np.degrees(latitude_rad)
    longitude_deg = np.degrees(longitude_rad)
    declination_rad = np.radians(get_mag_declination(latitude_deg, longitude_deg))
    inclination_rad = np.radians(get_mag_inclination(latitude_deg, longitude_deg))
    strength_gauss = 0.01 * get_mag_strength(latitude_deg, longitude_deg)

    horizontal_gauss = strength_gauss * np.cos(inclination_rad)
    return np.array(
        [
            horizontal_gauss * np.cos(declination_rad),
            horizontal_gauss * np.sin(declination_rad),
            np.tan(inclination_rad) * horizontal_gauss,
        ],
        dtype=np.float64,
    )


@pytest.fixture
def pegasus_path(monkeypatch):
    monkeypatch.syspath_prepend(str(_PX4_BRIDGE_EXTENSION_ROOT))


def test_reference_specific_force_matches_pegasus_imu_without_noise(pegasus_path):
    from pegasus.simulator.logic.sensors.geo_mag_utils import GRAVITY_VECTOR
    from pegasus.simulator.logic.sensors.imu import IMU

    dt_s = 1.0
    orientation_wxyz = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float64)
    initial_velocity = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    acceleration_enu = np.array([0.25, -0.5, 1.25], dtype=np.float64)

    first = VehicleState.hover_at(sim_time_us=0)
    first.orientation_flu_enu_wxyz = orientation_wxyz
    first.linear_velocity_enu = initial_velocity
    first.linear_acceleration_enu = acceleration_enu

    second = VehicleState.hover_at(sim_time_us=1_000_000)
    second.orientation_flu_enu_wxyz = orientation_wxyz
    second.linear_velocity_enu = initial_velocity + acceleration_enu * dt_s
    second.linear_acceleration_enu = acceleration_enu

    imu = IMU(
        {
            "update_rate": 1.0,
            "gyroscope": {
                "noise_density": 0.0,
                "random_walk": 0.0,
                "bias_correlation_time": 1.0e3,
                "turn_on_bias_sigma": 0.0,
            },
            "accelerometer": {
                "noise_density": 0.0,
                "random_walk": 0.0,
                "bias_correlation_time": 300.0,
                "turn_on_bias_sigma": 0.0,
            },
        }
    )
    imu.initialize(
        None,
        frames.DEFAULT_HOME_LAT_DEG,
        frames.DEFAULT_HOME_LON_DEG,
        frames.DEFAULT_HOME_ALT_M,
    )
    imu.update(_pegasus_state_from_vehicle(first), dt_s)
    pegasus_imu = imu.update(_pegasus_state_from_vehicle(second), dt_s)

    reference_frd = _flu_to_frd(
        frames.specific_force_body_flu(
            acceleration_enu,
            orientation_wxyz,
            gravity=float(np.linalg.norm(GRAVITY_VECTOR)),
        )
    )
    np.testing.assert_allclose(pegasus_imu["linear_acceleration"], reference_frd, atol=1e-12)


def test_reference_magnetometer_matches_pegasus_with_same_wmm_field(pegasus_path):
    from pegasus.simulator.logic.sensors.magnetometer import Magnetometer

    state = VehicleState.hover_at(position_enu=(12.0, -7.5, 2.0), sim_time_us=0)
    state.orientation_flu_enu_wxyz = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])

    magnetometer = Magnetometer(
        {
            "update_rate": 1.0,
            "noise_density": 0.0,
            "random_walk": 0.0,
            "bias_correlation_time": 6.0e2,
        }
    )
    magnetometer.initialize(
        None,
        frames.DEFAULT_HOME_LAT_DEG,
        frames.DEFAULT_HOME_LON_DEG,
        frames.DEFAULT_HOME_ALT_M,
    )

    pegasus_mag = magnetometer.update(_pegasus_state_from_vehicle(state), 1.0)
    reference_frd = _flu_to_frd(
        frames.magnetometer_body_flu(
            state.orientation_flu_enu_wxyz,
            _wmm_field_ned_gauss(
                state.position_enu,
                frames.DEFAULT_HOME_LAT_DEG,
                frames.DEFAULT_HOME_LON_DEG,
            ),
        )
    )

    np.testing.assert_allclose(pegasus_mag["magnetic_field"], reference_frd, atol=1e-12)


def test_reference_gps_matches_pegasus_frame_and_unit_conventions(pegasus_path):
    from pegasus.simulator.logic.sensors.gps import GPS

    state = VehicleState.hover_at(position_enu=(18.0, -11.0, 3.5), sim_time_us=0)
    state.linear_velocity_enu = np.array([4.0, -3.0, 1.5], dtype=np.float64)

    gps = GPS(
        {
            "update_rate": 1.0,
            "gps_xy_random_walk": 0.0,
            "gps_z_random_walk": 0.0,
            "gps_xy_noise_density": 0.0,
            "gps_z_noise_density": 0.0,
            "gps_vxy_noise_density": 0.0,
            "gps_vz_noise_density": 0.0,
        }
    )
    gps.initialize(
        None,
        frames.DEFAULT_HOME_LAT_DEG,
        frames.DEFAULT_HOME_LON_DEG,
        frames.DEFAULT_HOME_ALT_M,
    )

    pegasus_gps = gps.update(_pegasus_state_from_vehicle(state), 1.0)
    reference_gps = frames.synth_gps_from_enu(
        state.position_enu,
        state.linear_velocity_enu,
    )

    assert pegasus_gps["fix_type"] == reference_gps["fix_type"]
    assert pegasus_gps["altitude"] == pytest.approx(reference_gps["altitude"])
    assert pegasus_gps["altitude_gt"] == pytest.approx(reference_gps["altitude_gt"])
    assert pegasus_gps["speed"] == pytest.approx(reference_gps["speed"])
    assert pegasus_gps["velocity_north"] == pytest.approx(reference_gps["velocity_north"])
    assert pegasus_gps["velocity_east"] == pytest.approx(reference_gps["velocity_east"])
    assert pegasus_gps["velocity_down"] == pytest.approx(reference_gps["velocity_down"])
    assert pegasus_gps["latitude"] == pytest.approx(reference_gps["latitude"], abs=1.0e-6)
    assert pegasus_gps["longitude"] == pytest.approx(reference_gps["longitude"], abs=1.0e-6)
    assert np.degrees(pegasus_gps["latitude_gt"]) == pytest.approx(
        reference_gps["latitude_gt"],
        abs=1.0e-6,
    )
    assert np.degrees(pegasus_gps["longitude_gt"]) == pytest.approx(
        reference_gps["longitude_gt"],
        abs=1.0e-6,
    )


def test_reference_barometer_matches_pegasus_altitude_sign_and_pressure(pegasus_path):
    from pegasus.simulator.logic.sensors.barometer import Barometer

    baseline = VehicleState.hover_at(position_enu=(0.0, 0.0, 0.0), sim_time_us=0)
    raised = VehicleState.hover_at(position_enu=(0.0, 0.0, 2.0), sim_time_us=1_000_000)

    barometer = Barometer({"update_rate": 1.0, "noise_std_pa": 0.0, "drift_pa_per_sec": 0.0})
    barometer.initialize(
        None,
        frames.DEFAULT_HOME_LAT_DEG,
        frames.DEFAULT_HOME_LON_DEG,
        frames.DEFAULT_HOME_ALT_M,
    )
    barometer.update(_pegasus_state_from_vehicle(baseline), 1.0)
    pegasus_baro = barometer.update(_pegasus_state_from_vehicle(raised), 1.0)

    reference_baro = frames.synth_barometer(frames.DEFAULT_HOME_ALT_M + raised.position_enu[2])

    assert pegasus_baro["pressure_altitude"] == pytest.approx(
        reference_baro["pressure_altitude"],
        abs=1.0e-9,
    )
    assert pegasus_baro["absolute_pressure"] == pytest.approx(
        reference_baro["absolute_pressure"],
        rel=2.0e-4,
    )

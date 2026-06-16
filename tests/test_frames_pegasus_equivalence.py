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
    assert pegasus_gps["latitude_gt"] == pytest.approx(
        reference_gps["latitude_gt"],
        abs=1.0e-6,
    )
    assert pegasus_gps["longitude_gt"] == pytest.approx(
        reference_gps["longitude_gt"],
        abs=1.0e-6,
    )


@pytest.mark.parametrize(
    "home_lat_deg, home_lon_deg",
    [
        (frames.DEFAULT_HOME_LAT_DEG, frames.DEFAULT_HOME_LON_DEG),
        # Near-equator home: |lat|,|lon| < 2*pi would be misclassified as radians
        # by the old magnitude heuristic. Ground truth must stay in degrees.
        (1.0, 2.0),
    ],
)
def test_pegasus_gps_ground_truth_is_degrees_for_any_home(
    pegasus_path,
    home_lat_deg,
    home_lon_deg,
):
    from pegasus.simulator.logic.sensors.gps import GPS

    state = VehicleState.hover_at(position_enu=(12.0, -7.0, 2.0), sim_time_us=0)

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
    gps.initialize(None, home_lat_deg, home_lon_deg, frames.DEFAULT_HOME_ALT_M)

    sensor = gps.update(_pegasus_state_from_vehicle(state), 1.0)

    # Ground truth and primary coordinates share units (degrees) regardless of
    # home magnitude: near home they are within a fraction of a degree.
    assert sensor["latitude_gt"] == pytest.approx(sensor["latitude"], abs=1.0e-3)
    assert sensor["longitude_gt"] == pytest.approx(sensor["longitude"], abs=1.0e-3)
    assert sensor["latitude_gt"] == pytest.approx(home_lat_deg, abs=1.0e-3)
    assert sensor["longitude_gt"] == pytest.approx(home_lon_deg, abs=1.0e-3)


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

from dataclasses import replace

import numpy as np
import pytest

from robotics_experiments.px4_bridge import (
    DEFAULT_HOME_ALT_M,
    DEFAULT_HOME_LAT_DEG,
    DEFAULT_HOME_LON_DEG,
    VehicleState,
    body_velocity_flu_from_state,
    quat_wxyz_to_xyzw,
    rotate_vector_wxyz,
    synth_barometer,
    synth_gps_from_enu,
    synth_magnetometer_body_flu,
    world_enu_to_body_flu,
)
from tests.support.paths import REPO_ROOT

_PX4_BRIDGE_EXTENSION_ROOT = REPO_ROOT / "isaac/runtime/extensions/px4_sitl_brigde"


def _pegasus_state_from_vehicle(state: VehicleState):
    from pegasus.simulator.logic.state import State

    pegasus_state = State()
    pegasus_state.position = np.array(state.position_enu, dtype=np.float64, copy=True)
    pegasus_state.attitude = quat_wxyz_to_xyzw(state.orientation_flu_enu_wxyz)
    pegasus_state.linear_velocity = np.array(state.linear_velocity_enu, dtype=np.float64, copy=True)
    pegasus_state.angular_velocity = np.array(
        state.angular_velocity_flu, dtype=np.float64, copy=True
    )
    pegasus_state.linear_body_velocity = world_enu_to_body_flu(
        state.orientation_flu_enu_wxyz,
        state.linear_velocity_enu,
    )
    return pegasus_state


def _flu_to_frd(vector_flu: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector_flu, dtype=np.float64)
    return np.array([vector[0], -vector[1], -vector[2]], dtype=np.float64)


def _enu_to_ned(vector_enu: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector_enu, dtype=np.float64)
    return np.array([vector[1], vector[0], -vector[2]], dtype=np.float64)


def _quat_wxyz_from_roll_pitch_yaw_deg(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


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


def test_pegasus_state_adapter_uses_canonical_quaternion_order(pegasus_path):
    state = replace(
        VehicleState.hover_at(position_enu=(1.0, 2.0, 3.0), sim_time_us=0),
        orientation_flu_enu_wxyz=np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64),
    )

    pegasus_state = _pegasus_state_from_vehicle(state)

    np.testing.assert_allclose(
        pegasus_state.attitude,
        quat_wxyz_to_xyzw(state.orientation_flu_enu_wxyz),
        atol=1e-12,
    )


def test_pegasus_state_position_velocity_and_gyro_match_frame_helpers(pegasus_path):
    state = replace(
        VehicleState.hover_at(position_enu=(3.0, 4.0, 5.0), sim_time_us=0),
        linear_velocity_enu=np.array([6.0, 7.0, 8.0], dtype=np.float64),
        angular_velocity_flu=np.array([0.1, -0.2, 0.3], dtype=np.float64),
    )

    pegasus_state = _pegasus_state_from_vehicle(state)

    np.testing.assert_allclose(pegasus_state.get_position_ned(), _enu_to_ned(state.position_enu))
    np.testing.assert_allclose(
        pegasus_state.get_linear_velocity_ned(),
        _enu_to_ned(state.linear_velocity_enu),
    )
    np.testing.assert_allclose(
        pegasus_state.get_angular_velocity_frd(),
        _flu_to_frd(state.angular_velocity_flu),
    )


def test_pegasus_attitude_matches_frames_rotation_convention(pegasus_path):
    from scipy.spatial.transform import Rotation

    state = replace(
        VehicleState.hover_at(sim_time_us=0),
        orientation_flu_enu_wxyz=_quat_wxyz_from_roll_pitch_yaw_deg(
            roll_deg=17.0,
            pitch_deg=-23.0,
            yaw_deg=41.0,
        ),
    )
    pegasus_state = _pegasus_state_from_vehicle(state)
    pegasus_attitude = Rotation.from_quat(pegasus_state.get_attitude_ned_frd())

    for vector_frd in (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.25, -0.5, 0.75], dtype=np.float64),
    ):
        vector_flu = _flu_to_frd(vector_frd)
        expected_ned = _enu_to_ned(rotate_vector_wxyz(state.orientation_flu_enu_wxyz, vector_flu))
        np.testing.assert_allclose(
            pegasus_attitude.apply(vector_frd),
            expected_ned,
            atol=1e-5,
        )


def test_body_velocity_helper_matches_pegasus_rotation_convention(pegasus_path):
    state = replace(
        VehicleState.hover_at(sim_time_us=0),
        orientation_flu_enu_wxyz=_quat_wxyz_from_roll_pitch_yaw_deg(
            roll_deg=-11.0,
            pitch_deg=8.0,
            yaw_deg=73.0,
        ),
        linear_velocity_enu=np.array([2.0, -5.0, 1.5], dtype=np.float64),
    )

    pegasus_state = _pegasus_state_from_vehicle(state)

    np.testing.assert_allclose(
        pegasus_state.linear_body_velocity,
        body_velocity_flu_from_state(state),
        atol=1e-12,
    )


def test_reference_magnetometer_matches_pegasus_with_same_wmm_field(pegasus_path):
    from pegasus.simulator.logic.sensors.magnetometer import Magnetometer

    state = replace(
        VehicleState.hover_at(position_enu=(12.0, -7.5, 2.0), sim_time_us=0),
        orientation_flu_enu_wxyz=np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
    )

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
        DEFAULT_HOME_LAT_DEG,
        DEFAULT_HOME_LON_DEG,
        DEFAULT_HOME_ALT_M,
    )

    pegasus_mag = magnetometer.update(_pegasus_state_from_vehicle(state), 1.0)
    reference_frd = _flu_to_frd(
        synth_magnetometer_body_flu(
            state.orientation_flu_enu_wxyz,
            _wmm_field_ned_gauss(
                state.position_enu,
                DEFAULT_HOME_LAT_DEG,
                DEFAULT_HOME_LON_DEG,
            ),
        )
    )

    np.testing.assert_allclose(pegasus_mag["magnetic_field"], reference_frd, atol=1e-12)


def test_reference_gps_matches_pegasus_frame_and_unit_conventions(pegasus_path):
    from pegasus.simulator.logic.sensors.gps import GPS

    state = replace(
        VehicleState.hover_at(position_enu=(18.0, -11.0, 3.5), sim_time_us=0),
        linear_velocity_enu=np.array([4.0, -3.0, 1.5], dtype=np.float64),
    )

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
        DEFAULT_HOME_LAT_DEG,
        DEFAULT_HOME_LON_DEG,
        DEFAULT_HOME_ALT_M,
    )

    pegasus_gps = gps.update(_pegasus_state_from_vehicle(state), 1.0)
    reference_gps = synth_gps_from_enu(
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
        (DEFAULT_HOME_LAT_DEG, DEFAULT_HOME_LON_DEG),
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
    gps.initialize(None, home_lat_deg, home_lon_deg, DEFAULT_HOME_ALT_M)

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
        DEFAULT_HOME_LAT_DEG,
        DEFAULT_HOME_LON_DEG,
        DEFAULT_HOME_ALT_M,
    )
    barometer.update(_pegasus_state_from_vehicle(baseline), 1.0)
    pegasus_baro = barometer.update(_pegasus_state_from_vehicle(raised), 1.0)

    reference_baro = synth_barometer(DEFAULT_HOME_ALT_M + raised.position_enu[2])

    assert pegasus_baro["pressure_altitude"] == pytest.approx(
        reference_baro["pressure_altitude"],
        abs=1.0e-9,
    )
    assert pegasus_baro["absolute_pressure"] == pytest.approx(
        reference_baro["absolute_pressure"],
        rel=2.0e-4,
    )

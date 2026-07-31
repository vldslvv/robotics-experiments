import importlib
import sys
import types
from dataclasses import replace

import numpy as np
import pytest

from robotics_experiments.px4_bridge import (
    LockstepSettings,
    Px4MavlinkPort,
    VehicleState,
)
from tests.support.bridge_fakes import (
    _PX4_BRIDGE_EXTENSION_ROOT,
    _FakeConnection,
    _install_fake_pegasus,
    lockstep_exchange,
)


def test_vendored_pegasus_sensor_utils_are_importable(monkeypatch):
    monkeypatch.syspath_prepend(str(_PX4_BRIDGE_EXTENSION_ROOT))

    geo_mag_utils = importlib.import_module("pegasus.simulator.logic.sensors.geo_mag_utils")

    lat, lon = geo_mag_utils.reprojection(
        np.array([0.0, 0.0, 0.0]),
        np.radians(47.397742),
        np.radians(8.545594),
    )
    assert np.degrees(lat) == pytest.approx(47.397742)
    assert np.degrees(lon) == pytest.approx(8.545594)
    assert geo_mag_utils.get_mag_strength(47.397742, 8.545594) > 0.0


def test_vendored_pegasus_backend_sends_barometer_temperature(monkeypatch):
    monkeypatch.syspath_prepend(str(_PX4_BRIDGE_EXTENSION_ROOT))
    sys.modules.pop("pegasus.simulator.logic.backends.px4_mavlink_backend", None)

    carb = types.ModuleType("carb")
    carb.log_info = lambda message: None
    carb.log_warn = lambda message: None
    monkeypatch.setitem(sys.modules, "carb", carb)

    pymavlink = types.ModuleType("pymavlink")
    mavutil = types.ModuleType("pymavlink.mavutil")
    mavutil.mavlink = types.SimpleNamespace(
        MAVLINK_MSG_ID_HIL_ACTUATOR_CONTROLS=93,
        MAV_TYPE_GENERIC=0,
        MAV_AUTOPILOT_INVALID=8,
        MAV_MODE_FLAG_SAFETY_ARMED=128,
    )
    mavutil.mavlink_connection = lambda *args, **kwargs: None
    pymavlink.mavutil = mavutil
    monkeypatch.setitem(sys.modules, "pymavlink", pymavlink)
    monkeypatch.setitem(sys.modules, "pymavlink.mavutil", mavutil)

    module = importlib.import_module("pegasus.simulator.logic.backends.px4_mavlink_backend")
    backend = module.PX4MavlinkBackend(
        module.PX4MavlinkBackendConfig({"num_rotors": 4, "px4_autolaunch": False})
    )
    backend._connection = _FakeConnection()
    backend.update_bar_data(
        {
            "absolute_pressure": 954.609,
            "pressure_altitude": 488.0,
            "temperature": 11.827,
        }
    )

    backend.send_sensor_msgs(123_000)

    hil_sensor = backend._connection.hil_sensor_sends[0]
    assert hil_sensor[13] == pytest.approx(11.827)
    assert hil_sensor[13] != backend._sensor_data.altitude
    assert hil_sensor[14] == 6656


def test_vendored_pegasus_barometer_applies_accumulated_drift(monkeypatch):
    monkeypatch.syspath_prepend(str(_PX4_BRIDGE_EXTENSION_ROOT))

    from pegasus.simulator.logic.sensors.barometer import Barometer
    from pegasus.simulator.logic.sensors.geo_mag_utils import GRAVITY_VECTOR
    from pegasus.simulator.logic.state import State

    state = State()
    state.position = np.array([0.0, 0.0, 0.0])

    baseline = Barometer({"noise_std_pa": 0.0, "drift_pa_per_sec": 0.0, "update_rate": 1.0})
    drifted = Barometer({"noise_std_pa": 0.0, "drift_pa_per_sec": 2.0, "update_rate": 1.0})

    baseline_first = baseline.update(state, 1.0)
    drifted_first = drifted.update(state, 1.0)
    baseline_second = baseline.update(state, 1.0)
    drifted_second = drifted.update(state, 1.0)

    first_abs_drift = drifted_first["absolute_pressure"] - baseline_first["absolute_pressure"]
    second_abs_drift = drifted_second["absolute_pressure"] - baseline_second["absolute_pressure"]
    assert first_abs_drift == pytest.approx(0.02)
    assert second_abs_drift == pytest.approx(0.04)

    air_density = drifted._AIR_DENSITY_MSL / np.power(
        drifted._TEMPERATURE_MSL
        / (drifted._TEMPERATURE_MSL - drifted._LAPSE_RATE * drifted._origin_alt),
        4.256,
    )
    first_altitude_shift = -2.0 / (np.linalg.norm(GRAVITY_VECTOR) * air_density)
    second_altitude_shift = -4.0 / (np.linalg.norm(GRAVITY_VECTOR) * air_density)
    first_alt_drift = drifted_first["pressure_altitude"] - baseline_first["pressure_altitude"]
    second_alt_drift = drifted_second["pressure_altitude"] - baseline_second["pressure_altitude"]
    assert first_alt_drift == pytest.approx(first_altitude_shift)
    assert second_alt_drift == pytest.approx(second_altitude_shift)


def test_mavlink_port_uses_sim_time_and_gates_gps(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_send_rate_hz=5.0, sensor_noise_enabled=False)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=150_000))
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=300_000))

    assert backend.started is True
    assert [args[0] for args in backend._connection.hil_sensor_sends] == [
        100_000,
        150_000,
        300_000,
    ]
    # The public status agrees with the HIL_SENSOR wiring seam on the send count.
    assert port.read_status().sensor_send_count == 3
    assert backend.gps_times == [100_000, 300_000]
    assert len(backend.imu_updates) == 3
    assert len(backend.mag_updates) == 3
    assert len(backend.bar_updates) == 3
    assert len(backend.gps_updates) == 3
    first_hil_sensor = backend._connection.hil_sensor_sends[0]
    assert first_hil_sensor[12] == pytest.approx(488.0, abs=1.0)
    assert first_hil_sensor[13] == pytest.approx(11.827, abs=0.01)
    assert first_hil_sensor[14] == 7167


def test_mavlink_port_pegasus_sensors_can_disable_noise_for_precision(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_send_rate_hz=250.0, sensor_noise_enabled=False)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    state = replace(
        VehicleState.hover_at(position_enu=(3.0, 4.0, 1.0), sim_time_us=100_000),
        linear_velocity_enu=np.array([5.0, 6.0, 7.0], dtype=np.float64),
        angular_velocity_flu=np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )

    port.start()
    lockstep_exchange(port, state)

    imu = backend.imu_updates[0]
    np.testing.assert_allclose(imu["angular_velocity"], [1.0, -2.0, -3.0], atol=1e-12)
    np.testing.assert_allclose(
        imu["linear_acceleration"],
        [1250.0, -1500.0, -1759.80665],
        atol=1e-5,
    )

    gps = backend.gps_updates[0]
    assert gps["velocity_east"] == pytest.approx(5.0)
    assert gps["velocity_north"] == pytest.approx(6.0)
    assert gps["velocity_down"] == pytest.approx(-7.0)
    assert gps["latitude"] == pytest.approx(gps["latitude_gt"])
    assert gps["longitude"] == pytest.approx(gps["longitude_gt"])
    assert abs(gps["latitude_gt"]) > 10.0
    assert abs(gps["longitude_gt"]) > 1.0


def test_mavlink_port_can_retain_only_baro_and_mag_noise(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    np.random.seed(42)
    port = Px4MavlinkPort(
        num_rotors=4,
        gps_send_rate_hz=250.0,
        sensor_noise_enabled=False,
        barometer_noise_std_pa=1.0,
        magnetometer_noise_scale=1.0,
    )
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=104_000))

    for imu in backend.imu_updates:
        np.testing.assert_allclose(
            imu["linear_acceleration"],
            [0.0, 0.0, -9.80665],
            atol=1e-8,
        )
    assert not np.array_equal(
        backend.mag_updates[0]["magnetic_field"],
        backend.mag_updates[1]["magnetic_field"],
    )
    assert backend.gps_updates[0]["latitude"] == pytest.approx(
        backend.gps_updates[0]["latitude_gt"]
    )
    assert (
        backend.bar_updates[0]["absolute_pressure"] != backend.bar_updates[1]["absolute_pressure"]
    )


def test_mavlink_port_deterministic_sensor_excitation_is_repeatable(monkeypatch):
    _install_fake_pegasus(monkeypatch)

    def samples_for_seed(*, seed):
        np.random.seed(seed)
        port = Px4MavlinkPort(
            num_rotors=4,
            gps_send_rate_hz=250.0,
            sensor_noise_enabled=False,
            sensor_excitation_mode="deterministic",
            barometer_noise_std_pa=0.1,
            magnetometer_noise_scale=0.1,
        )
        backend = port._backend
        backend._connection = _FakeConnection()
        backend._enable_lockstep = False
        port.start()
        lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
        lockstep_exchange(port, VehicleState.hover_at(sim_time_us=104_000))
        lockstep_exchange(port, VehicleState.hover_at(sim_time_us=120_000))
        return backend

    first = samples_for_seed(seed=1)
    second = samples_for_seed(seed=999)

    for first_sample, second_sample in zip(
        first.mag_updates,
        second.mag_updates,
        strict=True,
    ):
        np.testing.assert_array_equal(
            first_sample["magnetic_field"],
            second_sample["magnetic_field"],
        )
    assert first.bar_updates == second.bar_updates
    assert not np.array_equal(
        first.mag_updates[0]["magnetic_field"],
        first.mag_updates[1]["magnetic_field"],
    )
    assert first.bar_updates[0]["absolute_pressure"] != first.bar_updates[1]["absolute_pressure"]
    np.testing.assert_array_equal(
        first.mag_updates[0]["magnetic_field"],
        first.mag_updates[2]["magnetic_field"],
    )
    assert first.bar_updates[0] == first.bar_updates[2]


def test_mavlink_port_rejects_unknown_sensor_excitation_mode(monkeypatch):
    _install_fake_pegasus(monkeypatch)

    with pytest.raises(
        ValueError,
        match="sensor_excitation_mode must be 'stochastic' or 'deterministic'",
    ):
        Px4MavlinkPort(num_rotors=4, sensor_excitation_mode="random")


def test_mavlink_port_pegasus_imu_rest_accel_is_frd_when_noise_disabled(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, sensor_noise_enabled=False)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    imu = backend.imu_updates[0]
    np.testing.assert_allclose(
        imu["linear_acceleration"],
        [0.0, 0.0, -9.80665],
        atol=1e-8,
    )


def test_mavlink_port_barometer_altitude_sign_with_noise_disabled(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, sensor_noise_enabled=False)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(
        port, VehicleState.hover_at(position_enu=(0.0, 0.0, 1.0), sim_time_us=100_000)
    )
    lockstep_exchange(
        port, VehicleState.hover_at(position_enu=(0.0, 0.0, 2.0), sim_time_us=110_000)
    )

    assert backend.bar_updates[1]["pressure_altitude"] > backend.bar_updates[0]["pressure_altitude"]


def test_mavlink_port_defaults_to_pegasus_sensor_noise(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    # Default sensor noise is on: the at-rest IMU acceleration is perturbed off
    # the exact noiseless gravity vector that the noise-disabled tests assert.
    imu = backend.imu_updates[0]
    assert not np.allclose(imu["linear_acceleration"], [0.0, 0.0, -9.80665], atol=1e-9)


def test_mavlink_port_pegasus_mag_and_baro_are_not_bit_stale(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_send_rate_hz=5.0)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=104_000))

    assert len(backend.mag_updates) == 2
    assert len(backend.bar_updates) == 2
    first_mag = np.asarray(backend.mag_updates[0]["magnetic_field"], dtype=np.float64)
    second_mag = np.asarray(backend.mag_updates[1]["magnetic_field"], dtype=np.float64)
    first_baro = backend.bar_updates[0]
    second_baro = backend.bar_updates[1]

    assert not np.array_equal(first_mag, second_mag)
    assert first_baro["absolute_pressure"] != second_baro["absolute_pressure"]


@pytest.mark.parametrize(
    ("physics_rate_hz", "timestamps", "expected_tick_us"),
    [
        pytest.param(250.0, (100_000, 104_000, 108_000), 4000, id="matched_250hz"),
        pytest.param(
            333.0,
            (100_000, 103_003, 106_006, 109_009),
            3003,
            id="333hz_float_boundary",
        ),
    ],
)
def test_mavlink_port_imu_fresh_every_tick_at_physics_rate(
    monkeypatch, physics_rate_hz, timestamps, expected_tick_us
):
    # The 333 Hz case is a regression for a float-boundary skip: the naive
    # 1/rate period is a ULP above the integer-us tick, so the IMU would refresh
    # only every other tick unless the period is pinned to the integer-us tick.
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(
        num_rotors=4,
        lockstep=LockstepSettings(physics_rate_hz=physics_rate_hz),
        sensor_noise_enabled=False,
    )
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    for sim_time_us in timestamps:
        lockstep_exchange(port, VehicleState.hover_at(sim_time_us=sim_time_us))

    # The integer-us tick pinning is observable as cadence: the IMU refreshes on
    # EVERY tick (one update per timestamp, fresh bits set), with no every-other
    # skip at the 333 Hz float boundary, and the link stays connected.
    assert len(backend.imu_updates) == len(timestamps)
    assert all(send[14] & (7 | 56) for send in backend._connection.hil_sensor_sends)
    assert port.read_status().sensor_send_count == len(timestamps)
    assert port.read_status().is_connected is True
    assert port.read_status().last_disconnect_reason is None


def test_mavlink_port_disconnects_when_physics_faster_than_sensor_rate(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(
        num_rotors=4, lockstep=LockstepSettings(physics_rate_hz=250.0), sensor_noise_enabled=False
    )
    backend = port._backend
    connection = _FakeConnection()
    backend._connection = connection
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
    # 1000 us step is faster than the 4000 us (250 Hz) sensor period, so the
    # IMU gate skips and the bridge must fail closed.
    commands = lockstep_exchange(port, VehicleState.hover_at(sim_time_us=101_000))

    assert np.allclose(commands, np.zeros(4))
    assert port.read_status().is_connected is False
    assert backend.stopped is True
    assert "sensor cadence mismatch" in port.read_status().last_disconnect_reason
    # Only the first (fresh) tick was sent before the guard tripped.
    assert [args[0] for args in connection.hil_sensor_sends] == [100_000]
    assert port.read_status().sensor_send_count == 1


def test_mavlink_port_requires_positive_physics_rate(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    with pytest.raises(ValueError, match="physics_hz must be positive"):
        Px4MavlinkPort(
            num_rotors=4, lockstep=LockstepSettings(physics_rate_hz=0.0), debug_log_path=None
        )


def test_mavlink_port_writes_sensor_diagnostics_to_file(monkeypatch, tmp_path):
    _install_fake_pegasus(monkeypatch)
    log_path = tmp_path / "px4_bridge.log"
    port = Px4MavlinkPort(
        num_rotors=4,
        debug_log_path=str(log_path),
        debug_log_level="DEBUG",
        sensor_debug_log_interval=1,
    )
    port._backend._connection = _FakeConnection()
    port._backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))
    port.stop()

    log_text = log_path.read_text()
    assert "PX4 HIL_SENSOR pre-send" in log_text
    assert "mask=7167" in log_text
    assert "fresh={imu:True mag:True baro:True" in log_text
    assert "PX4 HIL_SENSOR post-send" in log_text


def test_mavlink_port_can_disable_gps(monkeypatch):
    _install_fake_pegasus(monkeypatch)
    port = Px4MavlinkPort(num_rotors=4, gps_enabled=False)
    backend = port._backend
    backend._connection = _FakeConnection()
    backend._enable_lockstep = False

    port.start()
    lockstep_exchange(port, VehicleState.hover_at(sim_time_us=100_000))

    assert [args[0] for args in backend._connection.hil_sensor_sends] == [100_000]
    assert backend.gps_times == []
    assert backend.gps_updates == []


def test_hover_state_arrays_are_read_only():
    state = VehicleState.hover_at()

    for array in (
        state.position_enu,
        state.orientation_flu_enu_wxyz,
        state.linear_velocity_enu,
        state.angular_velocity_flu,
    ):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 1.0

"""Pegasus sensor wrapping for the PX4 MAVLink bridge.

This module isolates the stateful Pegasus sensor pipeline (IMU, GPS, barometer,
magnetometer) and its config factories from the lockstep MAVLink transport in
``mavlink.py``. The Pegasus imports stay lazy (function-scoped) so a lightweight
``import robotics_experiments`` does not pull in the simulator dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

import robotics_experiments.px4_bridge.frames as frames
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

_DEFAULT_SENSOR_RATE_HZ = 250.0
_STANDARD_GRAVITY_MPS2 = 9.80665
_DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
_MAGNETOMETER_NOISE_DENSITY_GA_SQRT_HZ = 0.4e-3
_DETERMINISTIC_EXCITATION_PERIOD_US = 20_000


class _PegasusSensorSuite:
    """Stateful Pegasus-style sensor update pipeline."""

    def __init__(
        self,
        *,
        home_lat_deg: float,
        home_lon_deg: float,
        home_alt_m: float,
        gps_enabled: bool,
        sensor_noise_enabled: bool,
        sensor_excitation_mode: str,
        barometer_noise_std_pa: float,
        magnetometer_noise_scale: float,
        update_rate_hz: float = _DEFAULT_SENSOR_RATE_HZ,
    ):
        self._home_lat_deg = float(home_lat_deg)
        self._home_lon_deg = float(home_lon_deg)
        self._home_alt_m = float(home_alt_m)
        self._gps_enabled = bool(gps_enabled)
        self._sensor_noise_enabled = bool(sensor_noise_enabled)
        self._sensor_excitation_mode = sensor_excitation_mode
        self._barometer_noise_std_pa = float(barometer_noise_std_pa)
        self._magnetometer_noise_scale = float(magnetometer_noise_scale)
        self._update_rate_hz = float(update_rate_hz)
        self._last_sim_time_us: int | None = None

        from pegasus.simulator.logic.sensors.barometer import Barometer
        from pegasus.simulator.logic.sensors.gps import GPS
        from pegasus.simulator.logic.sensors.imu import IMU
        from pegasus.simulator.logic.sensors.magnetometer import Magnetometer

        # Runtime intentionally uses Pegasus sensor classes for all PX4-bound
        # sensor signals. With noise enabled their internal distributions are:
        # IMU gyro/accel white noise plus bias random walks, GPS position and
        # altitude noise plus bias walk, barometer pressure noise/drift, and
        # magnetometer white noise plus bias walk.
        self._imu = IMU(_pegasus_imu_config(update_rate_hz, self._sensor_noise_enabled))
        self._gps = GPS(_pegasus_gps_config(update_rate_hz, self._sensor_noise_enabled))
        self._barometer = Barometer(
            _pegasus_barometer_config(
                update_rate_hz,
                (
                    self._barometer_noise_std_pa
                    if self._sensor_excitation_mode == "stochastic"
                    else 0.0
                ),
            )
        )
        self._magnetometer = Magnetometer(
            _pegasus_magnetometer_config(
                update_rate_hz,
                (
                    self._magnetometer_noise_scale
                    if self._sensor_excitation_mode == "stochastic"
                    else 0.0
                ),
            )
        )
        for sensor in (self._imu, self._gps, self._barometer, self._magnetometer):
            sensor.initialize(
                None,
                self._home_lat_deg,
                self._home_lon_deg,
                self._home_alt_m,
            )

    def update(self, state: VehicleState) -> dict[str, dict | None]:
        sim_time_us = int(state.sim_time_us)
        if self._last_sim_time_us is None:
            dt_s = 1.0 / self._update_rate_hz
        else:
            dt_s = max(0.0, (sim_time_us - self._last_sim_time_us) * 1e-6)
        self._last_sim_time_us = sim_time_us

        pegasus_state = self._to_pegasus_state(state)
        magnetometer = self._magnetometer.update(pegasus_state, dt_s)
        barometer = self._barometer.update(pegasus_state, dt_s)
        if self._sensor_excitation_mode == "deterministic":
            magnetometer = self._deterministic_magnetometer(
                sample=magnetometer,
                sim_time_us=sim_time_us,
            )
            barometer = self._deterministic_barometer(
                sample=barometer,
                sim_time_us=sim_time_us,
            )
        return {
            "imu": self._imu.update(pegasus_state, dt_s),
            "mag": magnetometer,
            "baro": barometer,
            "gps": self._gps.update(pegasus_state, dt_s) if self._gps_enabled else None,
        }

    def _deterministic_magnetometer(
        self,
        *,
        sample: dict,
        sim_time_us: int,
    ) -> dict:
        amplitude_ga = (
            _MAGNETOMETER_NOISE_DENSITY_GA_SQRT_HZ
            * math.sqrt(self._update_rate_hz)
            * self._magnetometer_noise_scale
        )
        phase_rad = (
            2.0
            * math.pi
            * (sim_time_us % _DETERMINISTIC_EXCITATION_PERIOD_US)
            / _DETERMINISTIC_EXCITATION_PERIOD_US
        )
        phases_rad = phase_rad + np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
        magnetic_field = np.asarray(sample["magnetic_field"], dtype=np.float64)
        return {"magnetic_field": magnetic_field + amplitude_ga * np.sin(phases_rad)}

    def _deterministic_barometer(
        self,
        *,
        sample: dict,
        sim_time_us: int,
    ) -> dict:
        phase_rad = (
            2.0
            * math.pi
            * (sim_time_us % _DETERMINISTIC_EXCITATION_PERIOD_US)
            / _DETERMINISTIC_EXCITATION_PERIOD_US
        )
        pressure_delta_pa = self._barometer_noise_std_pa * math.sin(phase_rad)
        temperature_k = float(sample["temperature"]) + 273.15
        pressure_pa = float(sample["absolute_pressure"]) * 100.0
        air_density_kg_m3 = pressure_pa / (_DRY_AIR_GAS_CONSTANT_J_KG_K * temperature_k)
        return {
            "absolute_pressure": float(sample["absolute_pressure"]) + pressure_delta_pa * 0.01,
            "pressure_altitude": float(sample["pressure_altitude"])
            - pressure_delta_pa / (_STANDARD_GRAVITY_MPS2 * air_density_kg_m3),
            "temperature": float(sample["temperature"]),
        }

    def _to_pegasus_state(self, state: VehicleState):
        from pegasus.simulator.logic.state import State

        pegasus_state = State()
        pegasus_state.position = np.asarray(state.position_enu, dtype=np.float64)
        pegasus_state.attitude = frames.quat_wxyz_to_xyzw(state.orientation_flu_enu_wxyz)
        pegasus_state.linear_velocity = np.asarray(state.linear_velocity_enu, dtype=np.float64)
        pegasus_state.angular_velocity = np.asarray(state.angular_velocity_flu, dtype=np.float64)
        pegasus_state.linear_body_velocity = frames.world_enu_to_body_flu(
            state.orientation_flu_enu_wxyz,
            state.linear_velocity_enu,
        )
        return pegasus_state


def _zero_noise(cfg: dict, keys: Iterable[str]) -> None:
    """Set each of ``keys`` in ``cfg`` to ``0.0`` in place.

    Collapses the repeated "zero the white-noise / random-walk terms when noise
    is disabled" step shared by the flat-keyed sensor configs. Structural
    constants (such as a non-zero bias correlation time) are set by the caller
    and left untouched.
    """
    for key in keys:
        cfg[key] = 0.0


def _pegasus_imu_config(update_rate_hz: float, noise_enabled: bool) -> dict:
    # Holds both the scalar update_rate and the nested gyro/accel sub-dicts.
    config: dict[str, object] = {"update_rate": update_rate_hz}
    if not noise_enabled:
        # Zero both white-noise densities and random-walk terms so precision
        # tests still exercise Pegasus frame conversion without stochastic
        # gyro/accelerometer perturbations.
        config["gyroscope"] = {
            "noise_density": 0.0,
            "random_walk": 0.0,
            "bias_correlation_time": 1.0e3,
            "turn_on_bias_sigma": 0.0,
        }
        config["accelerometer"] = {
            "noise_density": 0.0,
            "random_walk": 0.0,
            "bias_correlation_time": 300.0,
            "turn_on_bias_sigma": 0.0,
        }
    return config


def _pegasus_gps_config(update_rate_hz: float, noise_enabled: bool) -> dict:
    config = {"update_rate": update_rate_hz}
    if not noise_enabled:
        # Pegasus GPS applies randomness to position/altitude through
        # white-noise and bias-walk terms. Its velocity-noise samples are
        # currently computed but not applied by the vendored model.
        _zero_noise(
            config,
            (
                "gps_xy_random_walk",
                "gps_z_random_walk",
                "gps_xy_noise_density",
                "gps_z_noise_density",
                "gps_vxy_noise_density",
                "gps_vz_noise_density",
            ),
        )
    return config


def _pegasus_barometer_config(update_rate_hz: float, noise_std_pa: float) -> dict:
    # Pressure noise is independent from the aggregate sensor-noise switch.
    # PX4 HIL needs a changing pressure stream even when the other sensor
    # models are exact. Drift remains disabled unless explicitly supported by
    # a future project configuration field.
    config = {
        "update_rate": update_rate_hz,
        "noise_std_pa": noise_std_pa,
        "drift_pa_per_sec": 0.0,
    }
    return config


def _pegasus_magnetometer_config(update_rate_hz: float, noise_scale: float) -> dict:
    # Keep the vendored defaults explicit so an alignment experiment can retain
    # the excitation PX4 needs for yaw initialization while other sensor noise
    # is disabled.
    return {
        "update_rate": update_rate_hz,
        "noise_density": 0.4e-3 * noise_scale,
        "random_walk": 6.4e-6 * noise_scale,
        "bias_correlation_time": 6.0e2,
    }

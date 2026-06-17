"""Pegasus sensor wrapping for the PX4 MAVLink bridge.

This module isolates the stateful Pegasus sensor pipeline (IMU, GPS, barometer,
magnetometer) and its config factories from the lockstep MAVLink transport in
``mavlink.py``. The Pegasus imports stay lazy (function-scoped) so a lightweight
``import robotics_experiments`` does not pull in the simulator dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from . import frames
from .vehicle_state import VehicleState

_DEFAULT_SENSOR_RATE_HZ = 250.0


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
        update_rate_hz: float = _DEFAULT_SENSOR_RATE_HZ,
    ):
        self._home_lat_deg = float(home_lat_deg)
        self._home_lon_deg = float(home_lon_deg)
        self._home_alt_m = float(home_alt_m)
        self._gps_enabled = bool(gps_enabled)
        self._sensor_noise_enabled = bool(sensor_noise_enabled)
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
            _pegasus_barometer_config(update_rate_hz, self._sensor_noise_enabled)
        )
        self._magnetometer = Magnetometer(
            _pegasus_magnetometer_config(update_rate_hz, self._sensor_noise_enabled)
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
        imu = self._update_imu(pegasus_state, dt_s)
        mag = self._update_mag(pegasus_state, dt_s)
        baro = self._update_baro(pegasus_state, dt_s)
        gps = self._update_gps(pegasus_state, dt_s) if self._gps_enabled else None
        return {"imu": imu, "mag": mag, "baro": baro, "gps": gps}

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

    def _update_imu(self, state, dt_s: float) -> dict | None:
        return self._imu.update(state, dt_s)

    def _update_mag(self, state, dt_s: float) -> dict | None:
        return self._magnetometer.update(state, dt_s)

    def _update_baro(self, state, dt_s: float) -> dict | None:
        return self._barometer.update(state, dt_s)

    def _update_gps(self, state, dt_s: float) -> dict | None:
        return self._gps.update(state, dt_s)


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


def _pegasus_barometer_config(update_rate_hz: float, noise_enabled: bool) -> dict:
    config = {"update_rate": update_rate_hz}
    if not noise_enabled:
        # Barometer randomness is pressure-domain Gaussian noise plus drift.
        # The vendored sensor now exposes noise_std_pa so it can be made exact.
        _zero_noise(config, ("noise_std_pa", "drift_pa_per_sec"))
    return config


def _pegasus_magnetometer_config(update_rate_hz: float, noise_enabled: bool) -> dict:
    config = {"update_rate": update_rate_hz}
    if not noise_enabled:
        # Magnetometer randomness is Gaussian field noise plus random-walk
        # bias. Zero both terms for exact frame/field tests; the bias
        # correlation time is a structural constant, not a noise term.
        config["bias_correlation_time"] = 6.0e2
        _zero_noise(config, ("noise_density", "random_walk"))
    return config

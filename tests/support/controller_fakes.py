"""Controller settings shared by policy artifact and evaluation tests."""

from robotics_experiments.controller import ControllerSettings

__all__: list[str] = []


def _controller_settings() -> ControllerSettings:
    return ControllerSettings(
        command_delay_s=0.2,
        velocity_controller_rate_hz=125.0,
        attitude_controller_rate_hz=187.5,
        kp_vel=(2.0, 2.0, 3.0),
        horizontal_velocity_error_gain_scale=1.0,
        horizontal_velocity_error_gain_start_mps=0.15,
        horizontal_velocity_error_gain_full_mps=0.30,
        horizontal_velocity_error_gain_end_mps=0.80,
        horizontal_velocity_braking_gain_scale=1.0,
        horizontal_velocity_braking_gain_start_mps=0.05,
        horizontal_velocity_braking_gain_full_mps=0.07,
        horizontal_velocity_braking_gain_end_mps=0.10,
        ki_vel=(0.4, 0.4, 2.0),
        kd_vel=(1.5, 1.5, 1.5),
        velocity_observation_scale=(1.0, 1.0, 1.0),
        velocity_observation_hold_scale=(1.0, 1.0, 1.0),
        velocity_observation_bias_mps=(0.0, 0.0, 0.0),
        velocity_observation_bias_command_start_mps=0.0,
        velocity_observation_bias_command_full_mps=0.01,
        velocity_estimator_bias_decay_hz=0.0,
        velocity_estimator_bias_hold_decay_hz=(0.0, 0.0, 0.0),
        velocity_filter_hz=(10.0, 10.0, 10.0),
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 4.0),
        hover_thrust_normalized=0.57,
        ctbr_effective_collective_scale=1.0,
        kp_att=(6.0, 6.0, 2.0),
        attitude_yaw_weight=0.4,
        kp_rate=(1.2, 1.2, 0.4),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.003, 0.003, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        angular_velocity_filter_hz=0.0,
        rate_derivative_filter_hz=30.0,
        yaw_torque_filter_hz=2.0,
        rate_integral_limit=(0.3, 0.3, 0.2),
        max_tilt_cmd_deg=30.0,
        min_collective_thrust_fraction=0.05,
        max_collective_thrust_fraction=1.0,
        horizontal_thrust_margin_fraction=0.3,
        thruster_tau_inc_s=0.0,
        thruster_tau_dec_s=0.0,
        thruster_max_thrust_rate_nps=100000.0,
    )

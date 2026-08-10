"""Immutable host-safe controller contracts."""

from __future__ import annotations

from dataclasses import dataclass

__all__: list[str] = []

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class ControllerSettings:
    """Complete normalized PX4-like controller and actuator settings."""

    command_delay_s: float
    velocity_controller_rate_hz: float
    attitude_controller_rate_hz: float
    kp_vel: Vector3
    horizontal_velocity_error_gain_scale: float
    horizontal_velocity_error_gain_start_mps: float
    horizontal_velocity_error_gain_full_mps: float
    horizontal_velocity_error_gain_end_mps: float
    ki_vel: Vector3
    kd_vel: Vector3
    velocity_observation_scale: Vector3
    velocity_observation_bias_mps: Vector3
    velocity_estimator_bias_decay_hz: float
    velocity_filter_hz: Vector3
    velocity_derivative_filter_hz: float
    velocity_integral_limit_mps2: Vector3
    hover_thrust_normalized: float
    ctbr_effective_collective_scale: float
    kp_att: Vector3
    attitude_yaw_weight: float
    kp_rate: Vector3
    ki_rate: Vector3
    kd_rate: Vector3
    kff_rate: Vector3
    angular_velocity_filter_hz: float
    rate_derivative_filter_hz: float
    yaw_torque_filter_hz: float
    rate_integral_limit: Vector3
    max_tilt_cmd_deg: float
    min_collective_thrust_fraction: float
    max_collective_thrust_fraction: float
    horizontal_thrust_margin_fraction: float
    thruster_tau_inc_s: float
    thruster_tau_dec_s: float
    thruster_max_thrust_rate_nps: float


@dataclass(frozen=True)
class ControllerArtifact:
    """Strict v1 controller contract consumed by training and runtime."""

    artifact_kind: str
    schema_version: int
    alignment_id: str
    run_id: str
    target_controller_id: str
    reference_controller_id: str
    target_backend: str
    reference_backend: str
    target_definition_fingerprint: str
    reference_definition_fingerprint: str
    comparison_fingerprint: str
    experiment_config: str
    experiment_fingerprint: str
    vehicle: str
    airframe: str
    command_rate_hz: float
    certified_control_modes: tuple[str, ...]
    probe_suite_fingerprint: str
    selected_settings: ControllerSettings
    settings_fingerprint: str
    advisory_quality_passed: bool
    advisory_quality_reasons: tuple[str, ...]
    source_commit: str
    source_dirty: bool
    target_image: str
    reference_image: str

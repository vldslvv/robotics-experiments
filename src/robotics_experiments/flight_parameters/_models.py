"""Immutable, host-safe flight-parameter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__: list[str] = []

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class HoverFlightConfig:
    """One repeated task-neutral hover measurement."""

    kind: str
    repetitions: int
    mode_transition_timeout_s: float
    mode_report_max_age_s: float
    settle_s: float
    measure_s: float
    sample_rate_hz: float
    target_position_enu_m: Vector3
    target_yaw_rad: float


@dataclass(frozen=True)
class FlightExecutionConfig:
    """Campaign scheduling settings."""

    worker_count: int


@dataclass(frozen=True)
class FlightConfig:
    """Strict v1 input contract for one flight-parameter campaign."""

    schema_version: int
    flight_id: str
    source_backend: str
    state_source: str
    experiment_config: str
    vehicle: str
    scenario: HoverFlightConfig
    position_margin_m: float
    attitude_margin_rad: float
    execution: FlightExecutionConfig
    config_fingerprint: str


@dataclass(frozen=True)
class FlightTrace:
    """Backend-neutral pose samples from one complete flight trial."""

    artifact_kind: str
    schema_version: int
    flight_id: str
    campaign_run_id: str
    trial_index: int
    source_backend: str
    state_source: str
    config_fingerprint: str
    experiment_fingerprint: str
    vehicle_fingerprint: str
    source_image: str
    verification_fingerprint: str
    measurement_start_us: int
    measurement_end_us: int
    target_position_enu_m: Vector3
    target_attitude_flu_enu_wxyz: Quaternion
    time_s: tuple[float, ...]
    position_enu_m: tuple[Vector3, ...]
    attitude_flu_enu_wxyz: tuple[Quaternion, ...]
    completed: bool
    failure_reason: str


@dataclass(frozen=True)
class RawFlightTrace:
    """Timestamped simulator pose evidence captured before PX4 verification."""

    artifact_kind: str
    schema_version: int
    flight_id: str
    campaign_run_id: str
    trial_index: int
    source_backend: str
    state_source: str
    config_fingerprint: str
    experiment_fingerprint: str
    vehicle_fingerprint: str
    source_image: str
    requested_target_position_enu_m: Vector3
    requested_target_attitude_flu_enu_wxyz: Quaternion
    px4_home_lat_lon_alt: Vector3
    time_us: tuple[int, ...]
    position_enu_m: tuple[Vector3, ...]
    attitude_flu_enu_wxyz: tuple[Quaternion, ...]
    completed: bool
    failure_reason: str


@dataclass(frozen=True)
class Px4CommandEvent:
    """One accepted external PX4 command from a trial ULog."""

    timestamp_us: int
    command_id: int
    ack_timestamp_us: int
    ack_result: int


@dataclass(frozen=True)
class Px4ModeEvent:
    """One PX4 navigation-state transition from a trial ULog."""

    timestamp_us: int
    nav_state: int
    mode: str


@dataclass(frozen=True)
class Px4PositionGoalEvent:
    """One valid PX4 global position-goal update from a trial ULog."""

    timestamp_us: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    yaw_valid: bool
    yaw_rad: float


@dataclass(frozen=True)
class HoverVerification:
    """PX4 control-state proof for one selected hover measurement window."""

    artifact_kind: str
    schema_version: int
    flight_id: str
    campaign_run_id: str
    trial_index: int
    config_fingerprint: str
    px4_software_version: str
    raw_trace_sha256: str
    ulog_sha256: str
    command_events: tuple[Px4CommandEvent, ...]
    mode_events: tuple[Px4ModeEvent, ...]
    position_goal_events: tuple[Px4PositionGoalEvent, ...]
    loiter_enter_us: int
    measurement_start_us: int
    measurement_end_us: int
    mode_report_max_age_us: int
    control_mode_report_count: int
    land_state_report_count: int
    requested_target_position_enu_m: Vector3
    effective_target_position_enu_m: Vector3
    effective_target_attitude_flu_enu_wxyz: Quaternion
    px4_yaw_target_ned_rad: float
    verification_fingerprint: str


@dataclass(frozen=True)
class VectorBounds:
    """Observed component extrema and their expanded envelope bounds."""

    observed_min: Vector3
    observed_max: Vector3
    lower: Vector3
    upper: Vector3


@dataclass(frozen=True)
class UpperBound:
    """Observed maximum and its expanded one-sided envelope bound."""

    observed_max: float
    upper: float


@dataclass(frozen=True)
class HoverEnvelope:
    """Task-neutral pose-hold bounds around a commanded pose."""

    position_error_enu_m: VectorBounds
    horizontal_position_error_m: UpperBound
    position_error_m: UpperBound
    attitude_error_rpy_rad: VectorBounds
    tilt_error_rad: UpperBound
    orientation_error_rad: UpperBound


@dataclass(frozen=True)
class FlightEnvelope:
    """Strict v1 flight-envelope artifact consumed by alignment and RL."""

    artifact_kind: str
    schema_version: int
    flight_id: str
    run_id: str
    scenario_kind: str
    source_backend: str
    state_source: str
    experiment_config: str
    vehicle: str
    config_fingerprint: str
    experiment_fingerprint: str
    vehicle_fingerprint: str
    source_image: str
    source_commit: str
    source_dirty: bool
    evidence_fingerprint: str
    trial_count: int
    sample_count: int
    settle_s: float
    measure_s: float
    sample_rate_hz: float
    position_margin_m: float
    attitude_margin_rad: float
    hover: HoverEnvelope
    artifact_fingerprint: str


@dataclass(frozen=True)
class FlightEnvelopeBundle:
    """One validated immutable measurement resolved from a publication selector."""

    run_dir: Path
    envelope: FlightEnvelope
    envelope_digest: str
    manifest_digest: str


@dataclass(frozen=True)
class HoverEnvelopeEvaluation:
    """Pose error and normalized envelope utilization for one sample."""

    position_error_enu_m: Vector3
    attitude_error_rpy_rad: Vector3
    horizontal_position_error_m: float
    position_error_m: float
    tilt_error_rad: float
    orientation_error_rad: float
    position_utilization: float
    attitude_utilization: float
    inside: bool

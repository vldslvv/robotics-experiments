"""Immutable, host-safe controller-alignment contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

__all__: list[str] = []

Scalar = float
Vector3 = tuple[float, float, float]
SettingValue = Scalar | Vector3


@dataclass(frozen=True)
class ControllerSettings:
    """Complete controller and actuator settings owned by an alignment artifact."""

    command_delay_s: float
    kp_vel: Vector3
    ki_vel: Vector3
    kd_vel: Vector3
    velocity_filter_hz: float
    velocity_derivative_filter_hz: float
    velocity_integral_limit_mps2: Vector3
    kp_att: Vector3
    attitude_yaw_weight: float
    kp_rate: Vector3
    ki_rate: Vector3
    kd_rate: Vector3
    kff_rate: Vector3
    rate_output_scale_nm: Vector3
    rate_derivative_filter_hz: float
    rate_integral_limit_nm: Vector3
    max_tilt_cmd_deg: float
    min_collective_thrust_fraction: float
    max_collective_thrust_fraction: float
    horizontal_thrust_margin_fraction: float
    thruster_tau_inc_s: float
    thruster_tau_dec_s: float
    thruster_max_thrust_rate_nps: float


@dataclass(frozen=True)
class CommandInterface:
    """Policy-facing velocity command contract used by both controllers."""

    kind: str
    dimension: int
    frame: str
    linear_scale_mps: float
    yaw_rate_scale_radps: float


@dataclass(frozen=True)
class ProbeCommand:
    """One held body-FLU velocity/yaw-rate command and its exact duration."""

    velocity_body_flu_mps: Vector3
    yaw_rate_radps: float
    duration_s: float


@dataclass(frozen=True)
class ProbeProfile:
    """One named fit or validation excitation profile."""

    name: str
    split: str
    axis: str
    commands: tuple[ProbeCommand, ...]


@dataclass(frozen=True)
class ParameterSpec:
    """Declarative candidates for one existing :class:`ControllerSettings` field."""

    field: str
    stage: int
    candidates: tuple[SettingValue, ...]


@dataclass(frozen=True)
class SearchSpec:
    """Deterministic staged beam-search contract."""

    strategy: str
    beam_width: int
    passes_per_stage: int


@dataclass(frozen=True)
class ProbeTrace:
    """One backend response sampled at the exact command-update times."""

    profile: str
    split: str
    axis: str
    time_s: tuple[float, ...]
    command_velocity_body_flu_mps: tuple[Vector3, ...]
    command_yaw_rate_radps: tuple[float, ...]
    position_enu_m: tuple[Vector3, ...]
    attitude_flu_enu_wxyz: tuple[tuple[float, float, float, float], ...]
    velocity_body_flu_mps: tuple[Vector3, ...]
    velocity_enu_mps: tuple[Vector3, ...]
    angular_velocity_body_flu_radps: tuple[Vector3, ...]
    initial_position_enu_m: Vector3
    initial_attitude_flu_enu_wxyz: tuple[float, float, float, float]
    initial_linear_velocity_enu_mps: Vector3
    initial_angular_velocity_body_flu_radps: Vector3
    completed: bool
    failure_reason: str | None = None


class BackendEvaluator(Protocol):
    """Host-side adapter for one concrete controller backend."""

    def evaluate(
        self,
        *,
        definition: ControllerDefinition,
        settings: ControllerSettings,
        reference_traces: tuple[ProbeTrace, ...] | None,
    ) -> tuple[ProbeTrace, ...]:
        """Evaluate all configured profiles and return exact-cadence traces."""


@dataclass(frozen=True)
class ControllerDefinition:
    """Executable controller definition composed from YAML."""

    alignment_id: str
    controller_id: str
    backend: str
    experiment_config: str
    vehicle: str
    airframe: str
    command_rate_hz: float
    command_interface: CommandInterface
    base_settings: ControllerSettings
    probes: tuple[ProbeProfile, ...]
    parameters: tuple[ParameterSpec, ...]
    search: SearchSpec
    selection_metric: str
    publication_metrics: tuple[str, ...]
    evaluator: BackendEvaluator = field(compare=False, repr=False)
    definition_fingerprint: str
    image: str


@dataclass(frozen=True)
class CandidateScore:
    """One evaluated settings candidate and its registered metric values."""

    index: int
    settings: ControllerSettings
    usable: bool
    metrics: Mapping[str, float]
    ranking_values: tuple[float, ...]
    failure_reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@dataclass(frozen=True)
class AlignmentArtifact:
    """Immutable selected controller settings plus complete comparison provenance."""

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
    command_interface: CommandInterface
    probe_suite_fingerprint: str
    selection_metric: str
    selected_settings: ControllerSettings
    declared_parameters: tuple[ParameterSpec, ...]
    candidates: tuple[CandidateScore, ...]
    selected_candidate_index: int
    selected_ranking_values: tuple[float, ...]
    fit_metrics: Mapping[str, float]
    validation_metrics: Mapping[str, float]
    advisory_quality_passed: bool
    advisory_quality_reasons: tuple[str, ...]
    source_commit: str
    source_dirty: bool
    target_image: str
    reference_image: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fit_metrics", MappingProxyType(dict(self.fit_metrics)))
        object.__setattr__(
            self,
            "validation_metrics",
            MappingProxyType(dict(self.validation_metrics)),
        )


@dataclass(frozen=True)
class MetricResult:
    """Registered metric output used for ranking and artifact diagnostics."""

    metrics: Mapping[str, float]
    ranking_values: tuple[float, ...]
    fit_metrics: Mapping[str, float]
    validation_metrics: Mapping[str, float]
    beam_values: tuple[float, ...] = ()


MetricFunction = Callable[
    [tuple[ProbeTrace, ...], tuple[ProbeTrace, ...]],
    MetricResult,
]

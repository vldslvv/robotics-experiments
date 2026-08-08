"""Immutable, host-safe controller-alignment contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

from robotics_experiments.controller import (
    CommandInterface,
    ControllerArtifact,
    ControllerSettings,
)

__all__: list[str] = []

Scalar = float
Vector3 = tuple[float, float, float]
SettingValue = Scalar | Vector3


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
    selection_regularization_scale: float | None = None


@dataclass(frozen=True)
class SearchSpec:
    """Deterministic staged beam-search contract."""

    strategy: str
    beam_width: int
    passes_per_stage: int
    refinement_levels: int = 0
    min_refinement_survivors: int = 2
    initial_settings: tuple[ControllerSettings, ...] = ()


@dataclass(frozen=True)
class ControllerStateSourceAge:
    """Age of each lower-rate PX4 topic used to seed one probe boundary."""

    vehicle_local_position_setpoint: int
    vehicle_local_position: int
    vehicle_angular_velocity: int
    rate_ctrl_status: int
    hover_thrust_estimate: int


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
    initial_estimated_linear_velocity_enu_mps: Vector3
    initial_angular_velocity_body_flu_radps: Vector3
    initial_estimated_angular_velocity_body_flu_radps: Vector3
    completed: bool
    initial_controller_state_valid: bool = False
    initial_controller_state_source_age_us: ControllerStateSourceAge = ControllerStateSourceAge(
        0, 0, 0, 0, 0
    )
    initial_hover_thrust_normalized: float = 0.0
    initial_acceleration_setpoint_enu_mps2: Vector3 = (0.0, 0.0, 0.0)
    initial_linear_acceleration_enu_mps2: Vector3 = (0.0, 0.0, 0.0)
    initial_angular_acceleration_body_flu_radps2: Vector3 = (0.0, 0.0, 0.0)
    initial_rate_integral_body_flu: Vector3 = (0.0, 0.0, 0.0)
    start_sim_time_us: int = 0
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
    controller_schema_version: int
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
class AlignmentResult:
    """Selected controller contract plus the in-memory search decision."""

    controller: ControllerArtifact
    selection_metric: str
    base_settings: ControllerSettings
    declared_parameters: tuple[ParameterSpec, ...]
    candidates: tuple[CandidateScore, ...]
    selected_candidate_index: int


@dataclass(frozen=True)
class MetricResult:
    """Registered metric output used for ranking and artifact diagnostics."""

    metrics: Mapping[str, float]
    ranking_values: tuple[float, ...]
    fit_metrics: Mapping[str, float]
    validation_metrics: Mapping[str, float]
    beam_values: tuple[float, ...] = ()
    quality_passed: bool = True


MetricFunction = Callable[
    [tuple[ProbeTrace, ...], tuple[ProbeTrace, ...]],
    MetricResult,
]

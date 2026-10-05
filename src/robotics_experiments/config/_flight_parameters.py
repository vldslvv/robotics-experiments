"""Authored flight capture settings within a complete experiment."""

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from robotics_experiments.config._immutability import _FreezableConfig

__all__ = ["FlightExecutionConfig", "FlightPreparation", "FlightStabilityConfig"]


@dataclass
class FlightExecutionConfig(_FreezableConfig):
    """Campaign scheduling, compression and storage settings."""

    worker_count: int
    gzip_level: int
    minimum_free_bytes: int
    raw_flight_budget_bytes: int
    estimated_archive_bytes_per_flight: int


@dataclass
class FlightPreparation(_FreezableConfig):
    """Fixed response preparation schedule and readiness requirements."""

    pusher_deployed: bool
    minimum_neutral_s: float
    stable_hold_s: float
    timeout_s: float
    recenter_gain_hz: float
    recenter_max_speed_mps: float
    max_linear_speed_mps: float
    max_angular_speed_radps: float
    max_tilt_deg: float


@dataclass
class FlightStabilityConfig(_FreezableConfig):
    """Settings consumed only by final, read-only stability analysis."""

    excursion_horizontal_m: float
    drift_window_s: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    confidence: float
    relative_change_limit: float
    minimum_flights: int


class _FlightStageConfig(TypedDict):
    """Authored stage descriptor; the flight module validates command modes."""

    name: str
    kind: str
    control_mode: str
    axis: str
    hold_position: bool
    commands: list[dict[str, Any]]
    hover: NotRequired[dict[str, Any]]


@dataclass
class _FlightParametersInput(_FreezableConfig):
    """Flight-specific blocks composed alongside simulation and vehicle settings."""

    schema_version: int
    flight_parameters_id: str
    source_backend: str
    state_source: str
    vehicle: str
    start_seed: int
    flight_count: int
    takeoff_min_climb_m: float
    stages: list[_FlightStageConfig]
    preparation: FlightPreparation
    execution: FlightExecutionConfig
    stability: FlightStabilityConfig

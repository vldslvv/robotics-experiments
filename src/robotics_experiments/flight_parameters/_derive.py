"""Deterministic hover-envelope derivation and pose evaluation."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from robotics_experiments.flight_parameters._artifact import (
    _artifact_fingerprint,
    _evidence_fingerprint,
)
from robotics_experiments.flight_parameters._math import (
    _relative_attitude_errors,
)
from robotics_experiments.flight_parameters._models import (
    FlightConfig,
    FlightEnvelope,
    FlightTrace,
    HoverEnvelope,
    HoverEnvelopeEvaluation,
    Quaternion,
    UpperBound,
    Vector3,
    VectorBounds,
)

__all__: list[str] = []


def _vector3(value: np.ndarray, /) -> Vector3:
    return float(value[0]), float(value[1]), float(value[2])


def _vector_bounds(*, samples: np.ndarray, margin: float) -> VectorBounds:
    observed_min = np.min(samples, axis=0)
    observed_max = np.max(samples, axis=0)
    return VectorBounds(
        observed_min=_vector3(observed_min),
        observed_max=_vector3(observed_max),
        lower=_vector3(np.minimum(observed_min - margin, 0.0)),
        upper=_vector3(np.maximum(observed_max + margin, 0.0)),
    )


def _upper(*, values: np.ndarray, margin: float) -> UpperBound:
    observed = float(np.max(values))
    return UpperBound(observed_max=observed, upper=observed + margin)


def _validate_traces(*, config: FlightConfig, run_id: str, traces: tuple[FlightTrace, ...]) -> None:
    scenario = config.scenario
    if len(traces) != scenario.repetitions:
        raise ValueError(
            f"hover campaign requires {scenario.repetitions} traces, got {len(traces)}"
        )
    expected_samples = round(scenario.measure_s * scenario.sample_rate_hz)
    if expected_samples <= 1:
        raise ValueError("hover campaign measurement must contain at least two samples")
    expected_indices = tuple(range(1, scenario.repetitions + 1))
    if tuple(trace.trial_index for trace in traces) != expected_indices:
        raise ValueError("hover traces must use every ordered trial index exactly once")
    first = traces[0]
    for trace in traces:
        if not trace.completed or trace.failure_reason:
            raise ValueError(f"hover trial {trace.trial_index} did not complete")
        if (
            trace.flight_id != config.flight_id
            or trace.campaign_run_id != run_id
            or trace.source_backend != config.source_backend
            or trace.state_source != config.state_source
            or trace.config_fingerprint != config.config_fingerprint
        ):
            raise ValueError(f"hover trial {trace.trial_index} provenance mismatch")
        if (
            trace.experiment_fingerprint != first.experiment_fingerprint
            or trace.vehicle_fingerprint != first.vehicle_fingerprint
            or trace.source_image != first.source_image
        ):
            raise ValueError("hover traces do not share one source configuration")
        if len(trace.verification_fingerprint) != 64:
            raise ValueError(f"hover trial {trace.trial_index} has no verified PX4 evidence")
        expected_duration_us = round(scenario.measure_s * 1_000_000)
        if trace.measurement_end_us - trace.measurement_start_us != expected_duration_us:
            raise ValueError(f"hover trial {trace.trial_index} measurement duration mismatch")
        if len(trace.time_s) != expected_samples:
            raise ValueError(
                f"hover trial {trace.trial_index} has {len(trace.time_s)} samples; "
                f"expected {expected_samples}"
            )
        expected_time = np.arange(expected_samples, dtype=np.float64) / scenario.sample_rate_hz
        if not np.allclose(trace.time_s, expected_time, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"hover trial {trace.trial_index} cadence mismatch")


def derive_flight_envelope(
    *,
    config: FlightConfig,
    run_id: str,
    traces: tuple[FlightTrace, ...],
    source_commit: str,
    source_dirty: bool,
) -> FlightEnvelope:
    """Derive one strict hover envelope from all configured complete trials."""

    if not run_id or not source_commit:
        raise ValueError("flight envelope run and source commit must be non-empty")
    if type(source_dirty) is not bool:
        raise ValueError("source_dirty must be boolean")
    _validate_traces(config=config, run_id=run_id, traces=traces)
    positions: list[np.ndarray] = []
    attitudes: list[np.ndarray] = []
    tilts: list[float] = []
    orientations: list[float] = []
    for trace in traces:
        target_position = np.asarray(trace.target_position_enu_m, dtype=np.float64)
        target_attitude = trace.target_attitude_flu_enu_wxyz
        positions.extend(
            np.asarray(position, dtype=np.float64) - target_position
            for position in trace.position_enu_m
        )
        for attitude in trace.attitude_flu_enu_wxyz:
            rpy, tilt, orientation = _relative_attitude_errors(target_attitude, attitude)
            attitudes.append(rpy)
            tilts.append(tilt)
            orientations.append(orientation)
    position_array = np.asarray(positions, dtype=np.float64)
    attitude_array = np.asarray(attitudes, dtype=np.float64)
    horizontal = np.linalg.norm(position_array[:, :2], axis=1)
    position_norm = np.linalg.norm(position_array, axis=1)
    hover = HoverEnvelope(
        position_error_enu_m=_vector_bounds(
            samples=position_array, margin=config.position_margin_m
        ),
        horizontal_position_error_m=_upper(values=horizontal, margin=config.position_margin_m),
        position_error_m=_upper(values=position_norm, margin=config.position_margin_m),
        attitude_error_rpy_rad=_vector_bounds(
            samples=attitude_array, margin=config.attitude_margin_rad
        ),
        tilt_error_rad=_upper(values=np.asarray(tilts), margin=config.attitude_margin_rad),
        orientation_error_rad=_upper(
            values=np.asarray(orientations), margin=config.attitude_margin_rad
        ),
    )
    first = traces[0]
    evidence_fingerprint = _evidence_fingerprint(
        tuple(trace.verification_fingerprint for trace in traces)
    )
    envelope = FlightEnvelope(
        artifact_kind="robotics-experiments-flight-envelope",
        schema_version=1,
        flight_id=config.flight_id,
        run_id=run_id,
        scenario_kind="hover",
        source_backend=config.source_backend,
        state_source=config.state_source,
        experiment_config=config.experiment_config,
        vehicle=config.vehicle,
        config_fingerprint=config.config_fingerprint,
        experiment_fingerprint=first.experiment_fingerprint,
        vehicle_fingerprint=first.vehicle_fingerprint,
        source_image=first.source_image,
        source_commit=source_commit,
        source_dirty=source_dirty,
        evidence_fingerprint=evidence_fingerprint,
        trial_count=len(traces),
        sample_count=sum(len(trace.time_s) for trace in traces),
        settle_s=config.scenario.settle_s,
        measure_s=config.scenario.measure_s,
        sample_rate_hz=config.scenario.sample_rate_hz,
        position_margin_m=config.position_margin_m,
        attitude_margin_rad=config.attitude_margin_rad,
        hover=hover,
        artifact_fingerprint="",
    )
    payload = asdict(envelope)
    return FlightEnvelope(
        **{
            **payload,
            "hover": hover,
            "artifact_fingerprint": _artifact_fingerprint(payload),
        }
    )


def _component_utilization(*, value: float, lower: float, upper: float) -> float:
    scale = upper if value >= 0.0 else -lower
    if scale <= 0.0:
        return math.inf
    return abs(value) / scale


def evaluate_hover_pose(
    *,
    envelope: FlightEnvelope,
    target_position_enu_m: Vector3,
    target_attitude_flu_enu_wxyz: Quaternion,
    position_enu_m: Vector3,
    attitude_flu_enu_wxyz: Quaternion,
) -> HoverEnvelopeEvaluation:
    """Evaluate one pose against an immutable task-neutral hover envelope."""

    position_error = np.asarray(position_enu_m) - np.asarray(target_position_enu_m)
    rpy, tilt, orientation = _relative_attitude_errors(
        target_attitude_flu_enu_wxyz, attitude_flu_enu_wxyz
    )
    position_bounds = envelope.hover.position_error_enu_m
    attitude_bounds = envelope.hover.attitude_error_rpy_rad
    horizontal = float(np.linalg.norm(position_error[:2]))
    position_norm = float(np.linalg.norm(position_error))
    position_utilization = max(
        *(
            _component_utilization(value=float(value), lower=lower, upper=upper)
            for value, lower, upper in zip(
                position_error, position_bounds.lower, position_bounds.upper, strict=True
            )
        ),
        horizontal / envelope.hover.horizontal_position_error_m.upper,
        position_norm / envelope.hover.position_error_m.upper,
    )
    attitude_utilization = max(
        *(
            _component_utilization(value=float(value), lower=lower, upper=upper)
            for value, lower, upper in zip(
                rpy, attitude_bounds.lower, attitude_bounds.upper, strict=True
            )
        ),
        tilt / envelope.hover.tilt_error_rad.upper,
        orientation / envelope.hover.orientation_error_rad.upper,
    )
    return HoverEnvelopeEvaluation(
        position_error_enu_m=_vector3(position_error),
        attitude_error_rpy_rad=_vector3(rpy),
        horizontal_position_error_m=horizontal,
        position_error_m=position_norm,
        tilt_error_rad=tilt,
        orientation_error_rad=orientation,
        position_utilization=position_utilization,
        attitude_utilization=attitude_utilization,
        inside=position_utilization <= 1.0 and attitude_utilization <= 1.0,
    )

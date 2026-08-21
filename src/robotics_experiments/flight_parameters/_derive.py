"""Deterministic flight-parameter derivation and hover evaluation."""

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
    FlightParameters,
    FlightParametersConfig,
    FlightTrace,
    HoverEvaluation,
    HoverParameters,
    HoverTraceStatistics,
    Quaternion,
    ResponseParameters,
    ResponseTraceEvidence,
    SignalStatistics,
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


def _statistics(*, name: str, unit: str, values: np.ndarray) -> SignalStatistics:
    absolute = np.abs(values)
    return SignalStatistics(
        name=name,
        unit=unit,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        maximum_absolute=float(np.max(absolute)),
        rmse=float(np.sqrt(np.mean(np.square(values)))),
        p95_absolute=float(np.percentile(absolute, 95.0)),
        p99_absolute=float(np.percentile(absolute, 99.0)),
    )


def summarize_signal(*, name: str, unit: str, values: np.ndarray) -> SignalStatistics:
    """Summarize one non-empty finite scalar trace using the schema-1 statistics."""

    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("flight-parameter signal samples must be a non-empty finite vector")
    return _statistics(name=name, unit=unit, values=samples)


def _trace_signal_values(*, trace: FlightTrace) -> tuple[tuple[str, str, np.ndarray], ...]:
    target_position = np.asarray(trace.target_position_enu_m, dtype=np.float64)
    positions = np.asarray(trace.position_enu_m, dtype=np.float64) - target_position
    attitude_rows = tuple(
        _relative_attitude_errors(trace.target_attitude_flu_enu_wxyz, attitude)
        for attitude in trace.attitude_flu_enu_wxyz
    )
    attitudes = np.asarray(tuple(row[0] for row in attitude_rows), dtype=np.float64)
    return (
        ("position_error_enu_x", "m", positions[:, 0]),
        ("position_error_enu_y", "m", positions[:, 1]),
        ("position_error_enu_z", "m", positions[:, 2]),
        ("horizontal_position_error", "m", np.linalg.norm(positions[:, :2], axis=1)),
        ("position_error", "m", np.linalg.norm(positions, axis=1)),
        ("attitude_error_roll", "rad", attitudes[:, 0]),
        ("attitude_error_pitch", "rad", attitudes[:, 1]),
        ("attitude_error_yaw", "rad", attitudes[:, 2]),
        ("tilt_error", "rad", np.asarray(tuple(row[1] for row in attitude_rows))),
        ("orientation_error", "rad", np.asarray(tuple(row[2] for row in attitude_rows))),
    )


def _trace_statistics(
    *, config: FlightParametersConfig, trace: FlightTrace
) -> HoverTraceStatistics:
    warmup_samples = round(config.policy_warmup_s * config.hover.sample_rate_hz)
    signals = _trace_signal_values(trace=trace)
    return HoverTraceStatistics(
        trial_index=trace.trial_index,
        full_window_sample_count=len(trace.time_s),
        policy_window_start_s=config.policy_warmup_s,
        policy_window_sample_count=len(trace.time_s) - warmup_samples,
        full_window=tuple(
            _statistics(name=name, unit=unit, values=values) for name, unit, values in signals
        ),
        policy_window=tuple(
            _statistics(name=name, unit=unit, values=values[warmup_samples:])
            for name, unit, values in signals
        ),
    )


def _acceptance_limits(
    *, config: FlightParametersConfig, traces: tuple[HoverTraceStatistics, ...]
) -> tuple[SignalStatistics, ...]:
    limits: list[SignalStatistics] = []
    for signal_index, first in enumerate(traces[0].policy_window):
        statistics = tuple(trace.policy_window[signal_index] for trace in traces)
        if any(item.name != first.name or item.unit != first.unit for item in statistics):
            raise ValueError("hover trace statistics do not share one signal order")
        margin = config.position_margin_m if first.unit == "m" else config.attitude_margin_rad
        limits.append(
            SignalStatistics(
                name=first.name,
                unit=first.unit,
                minimum=min(item.minimum for item in statistics) - margin,
                maximum=max(item.maximum for item in statistics) + margin,
                maximum_absolute=max(item.maximum_absolute for item in statistics) + margin,
                rmse=max(item.rmse for item in statistics) + margin,
                p95_absolute=max(item.p95_absolute for item in statistics) + margin,
                p99_absolute=max(item.p99_absolute for item in statistics) + margin,
            )
        )
    return tuple(limits)


def _validate_traces(
    *, config: FlightParametersConfig, run_id: str, traces: tuple[FlightTrace, ...]
) -> None:
    hover = config.hover
    if len(traces) != hover.repetitions:
        raise ValueError(f"hover campaign requires {hover.repetitions} traces, got {len(traces)}")
    expected_samples = round(hover.measure_s * hover.sample_rate_hz)
    if expected_samples <= 1:
        raise ValueError("hover campaign measurement must contain at least two samples")
    expected_indices = tuple(range(1, hover.repetitions + 1))
    if tuple(trace.trial_index for trace in traces) != expected_indices:
        raise ValueError("hover traces must use every ordered trial index exactly once")
    first = traces[0]
    for trace in traces:
        if not trace.completed or trace.failure_reason:
            raise ValueError(f"hover trial {trace.trial_index} did not complete")
        if (
            trace.flight_parameters_id != config.flight_parameters_id
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
        expected_duration_us = round(hover.measure_s * 1_000_000)
        if trace.measurement_end_us - trace.measurement_start_us != expected_duration_us:
            raise ValueError(f"hover trial {trace.trial_index} measurement duration mismatch")
        if len(trace.time_s) != expected_samples:
            raise ValueError(
                f"hover trial {trace.trial_index} has {len(trace.time_s)} samples; "
                f"expected {expected_samples}"
            )
        expected_time = np.arange(expected_samples, dtype=np.float64) / hover.sample_rate_hz
        if not np.allclose(trace.time_s, expected_time, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"hover trial {trace.trial_index} cadence mismatch")


def derive_flight_parameters(
    *,
    config: FlightParametersConfig,
    run_id: str,
    traces: tuple[FlightTrace, ...],
    response_traces: tuple[ResponseTraceEvidence, ...],
    source_commit: str,
    source_dirty: bool,
    created_at_utc: str,
) -> FlightParameters:
    """Derive one strict flight-parameters artifact from complete hover trials."""

    if not run_id or not created_at_utc or not source_commit:
        raise ValueError(
            "flight parameters run, creation time, and source commit must be non-empty"
        )
    if type(source_dirty) is not bool:
        raise ValueError("source_dirty must be boolean")
    _validate_traces(config=config, run_id=run_id, traces=traces)
    if len(response_traces) != config.response.repetitions or tuple(
        trace.repetition for trace in response_traces
    ) != tuple(range(1, config.response.repetitions + 1)):
        raise ValueError("response traces must cover every configured ordered repetition")
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
    trace_statistics = tuple(_trace_statistics(config=config, trace=trace) for trace in traces)
    hover = HoverParameters(
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
        trace_statistics=trace_statistics,
        acceptance_limits=_acceptance_limits(config=config, traces=trace_statistics),
    )
    first = traces[0]
    evidence_fingerprint = _evidence_fingerprint(
        tuple(trace.verification_fingerprint for trace in traces)
        + tuple(
            digest
            for trace in response_traces
            for digest in (
                trace.reference_sha256,
                trace.ulog_sha256,
                trace.bridge_log_sha256,
            )
        )
    )
    response = ResponseParameters(
        kind=config.response.kind,
        repetitions=config.response.repetitions,
        traces=response_traces,
    )
    flight_parameters = FlightParameters(
        artifact_kind="robotics-experiments-flight-parameters",
        schema_version=1,
        flight_parameters_id=config.flight_parameters_id,
        run_id=run_id,
        created_at_utc=created_at_utc,
        behavior_kinds=("hover", "controller-response"),
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
        hover_trial_count=len(traces),
        hover_sample_count=sum(len(trace.time_s) for trace in traces),
        settle_s=config.hover.settle_s,
        measure_s=config.hover.measure_s,
        sample_rate_hz=config.hover.sample_rate_hz,
        policy_warmup_s=config.policy_warmup_s,
        position_margin_m=config.position_margin_m,
        attitude_margin_rad=config.attitude_margin_rad,
        hover=hover,
        response=response,
        artifact_fingerprint="",
    )
    payload = asdict(flight_parameters)
    return FlightParameters(
        **{
            **payload,
            "hover": hover,
            "response": response,
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
    flight_parameters: FlightParameters,
    target_position_enu_m: Vector3,
    target_attitude_flu_enu_wxyz: Quaternion,
    position_enu_m: Vector3,
    attitude_flu_enu_wxyz: Quaternion,
) -> HoverEvaluation:
    """Evaluate one pose against immutable task-neutral hover parameters."""

    position_error = np.asarray(position_enu_m) - np.asarray(target_position_enu_m)
    rpy, tilt, orientation = _relative_attitude_errors(
        target_attitude_flu_enu_wxyz, attitude_flu_enu_wxyz
    )
    position_bounds = flight_parameters.hover.position_error_enu_m
    attitude_bounds = flight_parameters.hover.attitude_error_rpy_rad
    horizontal = float(np.linalg.norm(position_error[:2]))
    position_norm = float(np.linalg.norm(position_error))
    position_utilization = max(
        *(
            _component_utilization(value=float(value), lower=lower, upper=upper)
            for value, lower, upper in zip(
                position_error, position_bounds.lower, position_bounds.upper, strict=True
            )
        ),
        horizontal / flight_parameters.hover.horizontal_position_error_m.upper,
        position_norm / flight_parameters.hover.position_error_m.upper,
    )
    attitude_utilization = max(
        *(
            _component_utilization(value=float(value), lower=lower, upper=upper)
            for value, lower, upper in zip(
                rpy, attitude_bounds.lower, attitude_bounds.upper, strict=True
            )
        ),
        tilt / flight_parameters.hover.tilt_error_rad.upper,
        orientation / flight_parameters.hover.orientation_error_rad.upper,
    )
    return HoverEvaluation(
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

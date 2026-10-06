"""Deterministic flight-parameter derivation and hover evaluation."""

from __future__ import annotations

import math

import numpy as np

from robotics_experiments.flight_parameters._math import (
    _relative_attitude_errors,
)
from robotics_experiments.flight_parameters._models import (
    FlightParameters,
    FlightTrace,
    HoverEvaluation,
    Quaternion,
    SignalStatistics,
    Vector3,
)

__all__: list[str] = []


def _vector3(value: np.ndarray, /) -> Vector3:
    return float(value[0]), float(value[1]), float(value[2])


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


def hover_pose_errors(
    *,
    target_position_enu_m: Vector3,
    target_attitude_flu_enu_wxyz: Quaternion,
    position_enu_m: Vector3,
    attitude_flu_enu_wxyz: Quaternion,
) -> dict[str, float]:
    """Task-neutral pose measurements, including each signed extent and norm."""
    position = np.asarray(position_enu_m) - np.asarray(target_position_enu_m)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("hover positions must be finite three-component vectors")
    rpy, tilt, orientation = _relative_attitude_errors(
        target_attitude_flu_enu_wxyz, attitude_flu_enu_wxyz
    )
    values = {}
    for prefix, vector, axes in (
        ("position", position, ("x", "y", "z")),
        ("attitude", rpy, ("roll", "pitch", "yaw")),
    ):
        for axis, value in zip(axes, vector, strict=True):
            values[f"hover.{prefix}.{axis}.positive"] = max(0.0, float(value))
            values[f"hover.{prefix}.{axis}.negative"] = max(0.0, -float(value))
    values.update(
        {
            "hover.horizontal_position": float(np.linalg.norm(position[:2])),
            "hover.position": float(np.linalg.norm(position)),
            "hover.tilt": tilt,
            "hover.orientation": orientation,
        }
    )
    return values

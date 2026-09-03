"""Batched NumPy/torch evaluation of a measured hover envelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import numpy as np
import torch

from robotics_experiments.flight_parameters import (
    FlightParameters,
    FlightParametersBundle,
    load_flight_trace,
)

__all__ = [
    "HoverEnvelopeEvaluation",
    "HoverResetSamples",
    "evaluate_hover_envelope",
    "load_hover_reset_samples",
]

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class HoverEnvelopeEvaluation(Generic[ArrayT]):
    """Per-sample pose errors and maximum measured-envelope utilization."""

    position_error_enu_m: ArrayT
    attitude_error_rpy_rad: ArrayT
    horizontal_position_error_m: ArrayT
    position_error_m: ArrayT
    tilt_error_rad: ArrayT
    orientation_error_rad: ArrayT
    position_utilization: ArrayT
    attitude_utilization: ArrayT
    inside: ArrayT


@dataclass(frozen=True)
class HoverResetSamples:
    """Matched pose and motion samples from the measured policy windows."""

    position_error_enu_m: tuple[tuple[float, float, float], ...]
    attitude_error_flu_enu_wxyz: tuple[tuple[float, float, float, float], ...]
    linear_velocity_enu_mps: tuple[tuple[float, float, float], ...]
    angular_velocity_enu_radps: tuple[tuple[float, float, float], ...]

    def summary(self, *, flight_parameters: FlightParameters) -> dict[str, float]:
        """Summarize the fixed reset population for metadata and monitoring."""
        position = np.asarray(self.position_error_enu_m)
        attitude = np.asarray(self.attitude_error_flu_enu_wxyz)
        linear_speed = np.linalg.norm(np.asarray(self.linear_velocity_enu_mps), axis=-1)
        angular_speed = np.linalg.norm(np.asarray(self.angular_velocity_enu_radps), axis=-1)
        evaluated = evaluate_hover_envelope(
            flight_parameters=flight_parameters,
            position_error_enu_m=position,
            attitude_error_flu_enu_wxyz=attitude,
        )
        return {
            "sample_count": float(position.shape[0]),
            "position_error_enu_x_mean_m": float(position[:, 0].mean()),
            "position_error_enu_y_mean_m": float(position[:, 1].mean()),
            "position_error_enu_z_mean_m": float(position[:, 2].mean()),
            "position_envelope_utilization_mean": float(evaluated.position_utilization.mean()),
            "position_envelope_utilization_p95": float(
                np.percentile(evaluated.position_utilization, 95.0)
            ),
            "position_envelope_utilization_max": float(evaluated.position_utilization.max()),
            "attitude_envelope_utilization_mean": float(evaluated.attitude_utilization.mean()),
            "attitude_envelope_utilization_p95": float(
                np.percentile(evaluated.attitude_utilization, 95.0)
            ),
            "attitude_envelope_utilization_max": float(evaluated.attitude_utilization.max()),
            "linear_speed_mean_mps": float(linear_speed.mean()),
            "linear_speed_p95_mps": float(np.percentile(linear_speed, 95.0)),
            "angular_speed_mean_radps": float(angular_speed.mean()),
            "angular_speed_p95_radps": float(np.percentile(angular_speed, 95.0)),
        }


def _normalize_quaternion(quaternion: np.ndarray, /) -> np.ndarray:
    return quaternion / np.linalg.norm(quaternion)


def _multiply_quaternions(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def load_hover_reset_samples(*, bundle: FlightParametersBundle) -> HoverResetSamples:
    """Load complete final-window states from each validated hover trace.

    Position, attitude, and derived velocity always come from the same sample
    index. Velocity differences stay within one trial; samples from separate
    flights are never joined.
    """

    flight_parameters = bundle.flight_parameters
    position_errors: list[tuple[float, float, float]] = []
    attitude_errors: list[tuple[float, float, float, float]] = []
    linear_velocities: list[tuple[float, float, float]] = []
    angular_velocities: list[tuple[float, float, float]] = []
    for trial_index in range(1, flight_parameters.hover_trial_count + 1):
        trace = load_flight_trace(
            bundle.run_dir / "evidence" / f"trial-{trial_index:03d}" / "trace.json"
        )
        target_position = np.asarray(trace.target_position_enu_m)
        target_attitude = _normalize_quaternion(np.asarray(trace.target_attitude_flu_enu_wxyz))
        target_attitude_inverse = target_attitude * np.asarray((1.0, -1.0, -1.0, -1.0))
        policy_sample_count = flight_parameters.hover.trace_statistics[
            trial_index - 1
        ].policy_window_sample_count
        time_s = np.asarray(trace.time_s, dtype=np.float64)
        positions = np.asarray(trace.position_enu_m, dtype=np.float64)
        attitudes = np.asarray(trace.attitude_flu_enu_wxyz, dtype=np.float64)
        attitudes /= np.linalg.norm(attitudes, axis=-1, keepdims=True)
        for sample_index in range(1, len(attitudes)):
            if np.dot(attitudes[sample_index - 1], attitudes[sample_index]) < 0.0:
                attitudes[sample_index] *= -1.0
        if len(time_s) == 1:
            trial_linear_velocities = np.zeros((1, 3), dtype=np.float64)
            trial_angular_velocities = np.zeros((1, 3), dtype=np.float64)
        else:
            trial_linear_velocities = np.gradient(positions, time_s, axis=0)
            attitude_derivatives = np.gradient(attitudes, time_s, axis=0)
            trial_angular_velocities = np.asarray(
                [
                    2.0
                    * _multiply_quaternions(
                        derivative,
                        attitude * np.asarray((1.0, -1.0, -1.0, -1.0)),
                    )[1:]
                    for derivative, attitude in zip(
                        attitude_derivatives,
                        attitudes,
                        strict=True,
                    )
                ]
            )
        for position, attitude, linear_velocity, angular_velocity in zip(
            positions[-policy_sample_count:],
            attitudes[-policy_sample_count:],
            trial_linear_velocities[-policy_sample_count:],
            trial_angular_velocities[-policy_sample_count:],
            strict=True,
        ):
            position_error = np.asarray(position) - target_position
            position_errors.append(
                (
                    float(position_error[0]),
                    float(position_error[1]),
                    float(position_error[2]),
                )
            )
            relative_attitude = _normalize_quaternion(
                _multiply_quaternions(
                    target_attitude_inverse,
                    _normalize_quaternion(np.asarray(attitude)),
                )
            )
            if relative_attitude[0] < 0.0:
                relative_attitude = -relative_attitude
            attitude_errors.append(
                (
                    float(relative_attitude[0]),
                    float(relative_attitude[1]),
                    float(relative_attitude[2]),
                    float(relative_attitude[3]),
                )
            )
            linear_velocities.append(
                (
                    float(linear_velocity[0]),
                    float(linear_velocity[1]),
                    float(linear_velocity[2]),
                )
            )
            angular_velocities.append(
                (
                    float(angular_velocity[0]),
                    float(angular_velocity[1]),
                    float(angular_velocity[2]),
                )
            )
    expected_samples = sum(
        trace.policy_window_sample_count for trace in flight_parameters.hover.trace_statistics
    )
    if len(position_errors) != expected_samples:
        raise ValueError(
            f"hover reset sample count {len(position_errors)} does not match "
            f"flight parameters {expected_samples}"
        )
    if not position_errors:
        raise ValueError("flight parameters contain no policy-window hover reset samples")
    return HoverResetSamples(
        position_error_enu_m=tuple(position_errors),
        attitude_error_flu_enu_wxyz=tuple(attitude_errors),
        linear_velocity_enu_mps=tuple(linear_velocities),
        angular_velocity_enu_radps=tuple(angular_velocities),
    )


def _as_array(*, values: tuple[float, ...], reference: ArrayT) -> ArrayT:
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    return np.asarray(values, dtype=reference.dtype)


def _as_same_array_type(*, value: object, reference: ArrayT) -> ArrayT:
    return cast(ArrayT, value)


def _norm(*, value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return torch.linalg.vector_norm(value, dim=-1)
    return np.linalg.norm(value, axis=-1)


def _maximum(*, values: tuple[ArrayT, ...]) -> ArrayT:
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values, dim=-1).amax(dim=-1)
    return np.stack(values, axis=-1).max(axis=-1)


def _component_max(*, value: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        return value.amax(dim=-1)
    return value.max(axis=-1)


def _relative_attitude(*, quaternion_wxyz: ArrayT) -> tuple[ArrayT, ArrayT, ArrayT]:
    if isinstance(quaternion_wxyz, torch.Tensor):
        quaternion = quaternion_wxyz / torch.linalg.vector_norm(quaternion_wxyz, dim=-1)[..., None]
        w, x, y, z = (quaternion[..., index] for index in range(4))
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        attitude_rpy = torch.stack((roll, pitch, yaw), dim=-1)
        tilt = torch.acos((1.0 - 2.0 * (x.square() + y.square())).clamp(-1.0, 1.0))
        orientation = 2.0 * torch.acos(w.abs().clamp(0.0, 1.0))
        return (
            _as_same_array_type(value=attitude_rpy, reference=quaternion_wxyz),
            _as_same_array_type(value=tilt, reference=quaternion_wxyz),
            _as_same_array_type(value=orientation, reference=quaternion_wxyz),
        )
    quaternion = quaternion_wxyz / np.linalg.norm(quaternion_wxyz, axis=-1)[..., None]
    w, x, y, z = (quaternion[..., index] for index in range(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    attitude_rpy_numpy = np.stack((roll, pitch, yaw), axis=-1)
    tilt_numpy = np.arccos(np.clip(1.0 - 2.0 * (x**2 + y**2), -1.0, 1.0))
    orientation_numpy = 2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0))
    return (
        _as_same_array_type(value=attitude_rpy_numpy, reference=quaternion_wxyz),
        _as_same_array_type(value=tilt_numpy, reference=quaternion_wxyz),
        _as_same_array_type(value=orientation_numpy, reference=quaternion_wxyz),
    )


def _component_utilization(*, value: ArrayT, lower: ArrayT, upper: ArrayT) -> ArrayT:
    if isinstance(value, torch.Tensor):
        assert isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor)
        utilization = abs(value) / torch.where(value >= 0.0, upper, -lower)
        return _as_same_array_type(value=utilization, reference=value)
    assert isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray)
    utilization_numpy = abs(value) / np.where(value >= 0.0, upper, -lower)
    return _as_same_array_type(value=utilization_numpy, reference=value)


def evaluate_hover_envelope(
    *,
    flight_parameters: FlightParameters,
    position_error_enu_m: ArrayT,
    attitude_error_flu_enu_wxyz: ArrayT,
) -> HoverEnvelopeEvaluation[ArrayT]:
    """Evaluate batched pose errors against the pinned hover envelope."""

    hover = flight_parameters.hover
    position_bounds = hover.position_error_enu_m
    attitude_bounds = hover.attitude_error_rpy_rad
    position_components = _component_utilization(
        value=position_error_enu_m,
        lower=_as_array(values=position_bounds.lower, reference=position_error_enu_m),
        upper=_as_array(values=position_bounds.upper, reference=position_error_enu_m),
    )
    rpy, tilt, orientation = _relative_attitude(quaternion_wxyz=attitude_error_flu_enu_wxyz)
    attitude_components = _component_utilization(
        value=rpy,
        lower=_as_array(values=attitude_bounds.lower, reference=rpy),
        upper=_as_array(values=attitude_bounds.upper, reference=rpy),
    )
    horizontal = _norm(value=position_error_enu_m[..., :2])
    position_norm = _norm(value=position_error_enu_m)
    position_utilization = _maximum(
        values=(
            _component_max(value=position_components),
            horizontal / hover.horizontal_position_error_m.upper,
            position_norm / hover.position_error_m.upper,
        )
    )
    attitude_utilization = _maximum(
        values=(
            _component_max(value=attitude_components),
            tilt / hover.tilt_error_rad.upper,
            orientation / hover.orientation_error_rad.upper,
        )
    )
    return HoverEnvelopeEvaluation(
        position_error_enu_m=position_error_enu_m,
        attitude_error_rpy_rad=rpy,
        horizontal_position_error_m=horizontal,
        position_error_m=position_norm,
        tilt_error_rad=tilt,
        orientation_error_rad=orientation,
        position_utilization=position_utilization,
        attitude_utilization=attitude_utilization,
        inside=(position_utilization <= 1.0) & (attitude_utilization <= 1.0),
    )

"""Batched NumPy/torch evaluation of a measured hover envelope."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np
import torch

from robotics_experiments.flight_parameters import (
    FlightDataset,
    FlightParameters,
    flight_parameters_from_payload,
    flight_stage_views,
    load_flight_selection,
)

__all__ = [
    "HoverEnvelopeEvaluation",
    "HoverResetSamples",
    "HoverResetBounds",
    "evaluate_hover_envelope",
    "load_hover_reset_samples",
    "load_hover_reset_bounds",
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
class HoverResetBounds:
    """Component extrema of the exact paired states used for PPO resets.

    These bounds describe the population's extent, not a uniform distribution
    or a guarantee that every combination inside it was sampled.
    Position tuples contain ENU X/Y/Z followed by horizontal and total norms;
    attitude tuples contain roll/pitch/yaw followed by tilt and rotation angle.
    """

    position_min_m: tuple[float, ...]
    position_max_m: tuple[float, ...]
    attitude_min_rad: tuple[float, ...]
    attitude_max_rad: tuple[float, ...]
    linear_velocity_min_mps: tuple[float, ...]
    linear_velocity_max_mps: tuple[float, ...]
    angular_velocity_min_radps: tuple[float, ...]
    angular_velocity_max_radps: tuple[float, ...]

    def rejection_reason(
        self,
        *,
        position_error_enu_m: np.ndarray,
        attitude_error_wxyz: np.ndarray,
        linear_velocity_enu_mps: np.ndarray,
        angular_velocity_enu_radps: np.ndarray,
    ) -> str | None:
        """Return the first state component outside the recorded reset extent."""
        rpy, tilt, orientation = _relative_attitude(quaternion_wxyz=attitude_error_wxyz)
        checks = (
            (
                np.r_[
                    position_error_enu_m,
                    np.linalg.norm(position_error_enu_m[:2]),
                    np.linalg.norm(position_error_enu_m),
                ],
                self.position_min_m,
                self.position_max_m,
                "position",
            ),
            (
                np.r_[rpy, tilt, orientation],
                self.attitude_min_rad,
                self.attitude_max_rad,
                "attitude",
            ),
            (
                linear_velocity_enu_mps,
                self.linear_velocity_min_mps,
                self.linear_velocity_max_mps,
                "linear_speed",
            ),
            (
                angular_velocity_enu_radps,
                self.angular_velocity_min_radps,
                self.angular_velocity_max_radps,
                "angular_speed",
            ),
        )
        for value, lower, upper, reason in checks:
            if not np.all(np.isfinite(value) & (value >= lower) & (value <= upper)):
                return reason
        return None


@dataclass(frozen=True)
class HoverResetSamples:
    """Matched pose and motion samples from the measured policy windows."""

    position_error_enu_m: tuple[tuple[float, float, float], ...]
    attitude_error_flu_enu_wxyz: tuple[tuple[float, float, float, float], ...]
    linear_velocity_enu_mps: tuple[tuple[float, float, float], ...]
    angular_velocity_enu_radps: tuple[tuple[float, float, float], ...]

    def bounds(self) -> HoverResetBounds:
        """Derive deployment limits without margins or held-out flight samples."""
        position = np.asarray(self.position_error_enu_m)
        rpy, tilt, orientation = _relative_attitude(
            quaternion_wxyz=np.asarray(self.attitude_error_flu_enu_wxyz)
        )
        # Include aggregate errors to exclude corners outside the observed norms.
        position = np.column_stack(
            (position, np.linalg.norm(position[:, :2], axis=-1), np.linalg.norm(position, axis=-1))
        )
        attitude = np.column_stack((rpy, tilt, orientation))
        linear = np.asarray(self.linear_velocity_enu_mps)
        angular = np.asarray(self.angular_velocity_enu_radps)
        return HoverResetBounds(
            position_min_m=tuple(position[:, :3].min(axis=0)) + (0.0, 0.0),
            position_max_m=tuple(position.max(axis=0)),
            attitude_min_rad=tuple(attitude[:, :3].min(axis=0)) + (0.0, 0.0),
            attitude_max_rad=tuple(attitude.max(axis=0)),
            linear_velocity_min_mps=tuple(linear.min(axis=0)),
            linear_velocity_max_mps=tuple(linear.max(axis=0)),
            angular_velocity_min_radps=tuple(angular.min(axis=0)),
            angular_velocity_max_radps=tuple(angular.max(axis=0)),
        )

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


def load_hover_reset_samples(
    *, dataset: FlightDataset, flight_indices: tuple[int, ...], flight_parameters: FlightParameters
) -> HoverResetSamples:
    """Read final-window states at paired original indices from training flights."""
    count = round(flight_parameters.policy_window_s * flight_parameters.sample_rate_hz)
    positions, attitudes, linear, angular = [], [], [], []
    for flight in dataset.flights:
        if flight.flight_index not in flight_indices:
            continue
        view = flight_stage_views(flight)[0]
        target = np.asarray(view["effective_target_position_enu_m"])
        inverse = _normalize_quaternion(
            np.asarray(view["effective_target_attitude_flu_enu_wxyz"])
        ) * (1.0, -1.0, -1.0, -1.0)
        for position, attitude, velocity, rate in zip(
            view["position_enu_m"][-count:],
            view["attitude_flu_enu_wxyz"][-count:],
            view["linear_velocity_enu_mps"][-count:],
            view["angular_velocity_enu_radps"][-count:],
            strict=True,
        ):
            relative = _normalize_quaternion(_multiply_quaternions(inverse, np.asarray(attitude)))
            if relative[0] < 0.0:
                relative = -relative
            positions.append(tuple(np.asarray(position) - target))
            attitudes.append(tuple(relative))
            linear.append(tuple(velocity))
            angular.append(tuple(rate))
    if not positions or len(positions) != count * len(flight_indices):
        raise ValueError("training flights do not contain the complete hover reset population")
    return HoverResetSamples(
        position_error_enu_m=tuple(positions),
        attitude_error_flu_enu_wxyz=tuple(attitudes),
        linear_velocity_enu_mps=tuple(linear),
        angular_velocity_enu_radps=tuple(angular),
    )


@lru_cache(maxsize=None)
def load_hover_reset_bounds(*, path: Path) -> HoverResetBounds:
    """Load training reset bounds before the bridge starts ticking."""
    dataset, selection = load_flight_selection(path)
    return load_hover_reset_samples(
        dataset=dataset,
        flight_indices=tuple(selection["train"]),
        flight_parameters=flight_parameters_from_payload(selection["hover"]),
    ).bounds()


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

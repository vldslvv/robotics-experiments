"""Batched NumPy/torch evaluation of a measured flight envelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch

from robotics_experiments.flight_parameters import FlightEnvelope

__all__ = ["FlightEnvelopeEvaluation", "evaluate_flight_envelope"]

ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class FlightEnvelopeEvaluation(Generic[ArrayT]):
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


def _as_array(*, values: tuple[float, ...], reference: ArrayT) -> ArrayT:
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    return np.asarray(values, dtype=reference.dtype)


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
    quaternion = quaternion_wxyz / _norm(value=quaternion_wxyz)[..., None]
    w, x, y, z = (quaternion[..., index] for index in range(4))
    if isinstance(quaternion, torch.Tensor):
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
        pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        rpy = torch.stack((roll, pitch, yaw), dim=-1)
        tilt = torch.acos((1.0 - 2.0 * (x.square() + y.square())).clamp(-1.0, 1.0))
        orientation = 2.0 * torch.acos(w.abs().clamp(0.0, 1.0))
        return rpy, tilt, orientation
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    rpy = np.stack((roll, pitch, yaw), axis=-1)
    tilt = np.arccos(np.clip(1.0 - 2.0 * (x**2 + y**2), -1.0, 1.0))
    orientation = 2.0 * np.arccos(np.clip(abs(w), 0.0, 1.0))
    return rpy, tilt, orientation


def _component_utilization(*, value: ArrayT, lower: ArrayT, upper: ArrayT) -> ArrayT:
    scale = (
        torch.where(value >= 0.0, upper, -lower)
        if isinstance(value, torch.Tensor)
        else np.where(value >= 0.0, upper, -lower)
    )
    return abs(value) / scale


def evaluate_flight_envelope(
    *,
    envelope: FlightEnvelope,
    position_error_enu_m: ArrayT,
    attitude_error_flu_enu_wxyz: ArrayT,
) -> FlightEnvelopeEvaluation[ArrayT]:
    """Evaluate batched pose errors against one immutable measured envelope."""

    hover = envelope.hover
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
    return FlightEnvelopeEvaluation(
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

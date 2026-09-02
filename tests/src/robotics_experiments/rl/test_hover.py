"""Parity tests for batched measured-envelope evaluation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from robotics_experiments.flight_parameters import (
    evaluate_hover_pose,
    load_flight_parameters,
    load_flight_parameters_bundle,
)
from robotics_experiments.rl import evaluate_hover_envelope, load_hover_reset_samples

_ENVELOPE = load_flight_parameters(
    Path(__file__).parents[3] / "fixtures/flight_parameters/flight_parameters.json"
)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_batched_evaluation_matches_flight_parameter_contract(backend: str) -> None:
    positions = np.array([[0.05, 0.0, 0.0], [0.08, 0.08, 0.0]])
    attitudes = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(0.04), np.sin(0.04), 0.0, 0.0],
        ]
    )
    expected = tuple(
        evaluate_hover_pose(
            flight_parameters=_ENVELOPE,
            target_position_enu_m=(0.0, 0.0, 0.0),
            target_attitude_flu_enu_wxyz=(1.0, 0.0, 0.0, 0.0),
            position_enu_m=tuple(position),
            attitude_flu_enu_wxyz=tuple(attitude),
        )
        for position, attitude in zip(positions, attitudes, strict=True)
    )
    position_input: np.ndarray | torch.Tensor
    attitude_input: np.ndarray | torch.Tensor
    if backend == "torch":
        position_input = torch.as_tensor(positions)
        attitude_input = torch.as_tensor(attitudes)
    else:
        position_input = positions
        attitude_input = attitudes

    actual = evaluate_hover_envelope(
        flight_parameters=_ENVELOPE,
        position_error_enu_m=position_input,
        attitude_error_flu_enu_wxyz=attitude_input,
    )
    position_utilization = np.asarray(actual.position_utilization)
    attitude_utilization = np.asarray(actual.attitude_utilization)
    inside = np.asarray(actual.inside)
    assert position_utilization == pytest.approx(
        [sample.position_utilization for sample in expected]
    )
    assert attitude_utilization == pytest.approx(
        [sample.attitude_utilization for sample in expected]
    )
    assert inside.tolist() == [sample.inside for sample in expected]


def test_hover_resets_are_exact_paired_policy_window_samples() -> None:
    bundle = load_flight_parameters_bundle(
        path=Path(__file__).parents[3] / "fixtures/flight_parameters"
    )
    samples = load_hover_reset_samples(bundle=bundle)
    assert samples.position_error_enu_m == ((0.0, 0.0, 0.0),)
    assert samples.attitude_error_flu_enu_wxyz == ((1.0, 0.0, 0.0, 0.0),)
    assert samples.summary(flight_parameters=bundle.flight_parameters) == {
        "sample_count": 1.0,
        "position_error_enu_x_mean_m": 0.0,
        "position_error_enu_y_mean_m": 0.0,
        "position_error_enu_z_mean_m": 0.0,
        "position_envelope_utilization_mean": 0.0,
        "position_envelope_utilization_p95": 0.0,
        "position_envelope_utilization_max": 0.0,
        "attitude_envelope_utilization_mean": 0.0,
        "attitude_envelope_utilization_p95": 0.0,
        "attitude_envelope_utilization_max": 0.0,
    }

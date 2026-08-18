"""Parity tests for batched measured-envelope evaluation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from robotics_experiments.flight_parameters import evaluate_hover_pose, load_flight_parameters
from robotics_experiments.rl import evaluate_flight_parameters

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
    if backend == "torch":
        position_input = torch.as_tensor(positions)
        attitude_input = torch.as_tensor(attitudes)
    else:
        position_input = positions
        attitude_input = attitudes

    actual = evaluate_flight_parameters(
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

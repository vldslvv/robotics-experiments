"""Parity tests for batched measured-envelope evaluation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from robotics_experiments.flight_parameters import (
    evaluate_hover_pose,
    flight_parameters_from_payload,
    load_flight_selection,
)
from robotics_experiments.rl import evaluate_hover_envelope, load_hover_reset_samples
from tests.support.flight_parameters import HOVER_PARAMETERS, write_hover_dataset

_ENVELOPE = HOVER_PARAMETERS


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


def test_hover_resets_and_bounds_use_paired_training_samples(tmp_path: Path) -> None:
    path = write_hover_dataset(tmp_path)
    dataset, selection = load_flight_selection(path)
    summary = flight_parameters_from_payload(selection["hover"])
    samples = load_hover_reset_samples(
        dataset=dataset,
        flight_indices=tuple(selection["train"]),
        flight_parameters=summary,
    )
    assert len(samples.position_error_enu_m) == 24
    for index, (position, attitude, linear, angular) in enumerate(
        zip(
            samples.position_error_enu_m,
            samples.attitude_error_flu_enu_wxyz,
            samples.linear_velocity_enu_mps,
            samples.angular_velocity_enu_radps,
            strict=True,
        ),
        start=6,
    ):
        assert position == (index * 0.001, 0.0, 0.0)
        assert attitude == pytest.approx((np.cos(index * 0.0005), 0.0, 0.0, np.sin(index * 0.0005)))
        assert linear == (index * 0.01, 0.0, 0.0)
        assert angular == (0.0, 0.0, index * 0.001)
        assert (
            samples.bounds().rejection_reason(
                position_error_enu_m=np.asarray(position),
                attitude_error_wxyz=np.asarray(attitude),
                linear_velocity_enu_mps=np.asarray(linear),
                angular_velocity_enu_radps=np.asarray(angular),
            )
            is None
        )
    # Held-out flights have ten- and twenty-metre offsets.
    assert summary.hover.position_error_m.upper == pytest.approx(0.034)
    assert samples.bounds().position_max_m[0] == pytest.approx(0.029)
    assert samples.summary(flight_parameters=summary)["sample_count"] == 24.0

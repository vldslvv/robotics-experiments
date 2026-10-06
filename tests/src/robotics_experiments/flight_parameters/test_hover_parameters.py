"""Shared hover derivation and its strict numeric metadata contract."""

import json
from dataclasses import asdict

import pytest

from robotics_experiments.flight_parameters import (
    derive_hover_parameters,
    flight_parameters_from_payload,
    load_flight_selection,
)
from tests.support.flight_parameters import write_hover_dataset


def test_hover_summary_round_trip_and_training_statistics(tmp_path):
    dataset, selection = load_flight_selection(write_hover_dataset(tmp_path))
    summary = flight_parameters_from_payload(selection["hover"])
    assert flight_parameters_from_payload(json.loads(json.dumps(asdict(summary)))) == summary
    assert summary.run_id == dataset.run_id
    assert summary.hover.position_error_enu_m.observed_max[0] == pytest.approx(0.029)
    x = summary.hover.acceptance_limits[0]
    assert x.maximum_absolute == pytest.approx(0.034)
    assert x.rmse == pytest.approx(
        (sum((j * 0.001) ** 2 for j in range(6, 30)) / 24) ** 0.5 + 0.005
    )
    with pytest.raises(ValueError, match="fields mismatch"):
        flight_parameters_from_payload({**selection["hover"], "unexpected": 0})


@pytest.mark.parametrize("warmup", [0.0, 7.0])
def test_hover_timing_cannot_shift_or_remove_warmup(tmp_path, warmup):
    dataset, _selection = load_flight_selection(write_hover_dataset(tmp_path))
    with pytest.raises(ValueError, match="hover"):
        derive_hover_parameters(
            dataset=dataset,
            flight_indices=(0,),
            policy_warmup_s=warmup,
            policy_window_s=24.0,
            position_margin_m=0.005,
            attitude_margin_rad=0.005,
        )

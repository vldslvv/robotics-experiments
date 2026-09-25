"""Shared mechanics, identity and scalar/array simulation-time activation."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from robotics_experiments.config import load_switch_model
from robotics_experiments.environment import SwitchState, apply_button_sample
from robotics_experiments.rl.press.contact import classify_press_contacts, contact_force_totals
from robotics_experiments.switches import ButtonLatchState, load_switch_asset, step_button_latch
from tests.support.paths import REPO_ROOT
from tests.support.switches import write_switch_asset


@pytest.fixture
def model():
    return load_switch_model(REPO_ROOT / "configs/experiment/switches/model/nominal.yaml")


def test_derived_mechanics_and_validation(*, model):
    assert model.stiffness_n_m == 260
    assert model.damping_n_s_m == pytest.approx(2.257432)
    assert model.stiffness_n_m * model.travel_m == 5.2
    for changes in (
        {"activation_displacement_m": model.travel_m},
        {"button_width_m": model.housing_width_m},
        {"button_height_m": model.housing_height_m},
        {"clearance_m": model.contact_offset_m},
        {"button_mass_kg": -1},
        {"damping_ratio": float("nan")},
        {"confirmation_s": 0},
    ):
        with pytest.raises(ValueError):
            replace(model, **changes).validate()


@pytest.mark.parametrize("policy_hz", [10, 20, 50])
@pytest.mark.parametrize("backend", [np, torch])
def test_timed_latch_matches_scalar_across_policy_rates_and_resets(*, model, policy_hz, backend):
    dt = 1 / 250
    state = ButtonLatchState(backend.zeros(2), backend.zeros(2, dtype=bool))
    scalars = [SwitchState("first"), SwitchState("second")]
    events = np.zeros(2, dtype=int)
    # A short tap, release, full-stroke hold, release and a second confirmed press.
    positions = [[0.02, 0.0]] * 25 + [[0, 0]] * 25 + [[0.02, 0.01]] * 250
    positions += [[0, 0]] * 10 + [[0.02, 0.02]] * 100
    observations = []
    for index, positions_m in enumerate(positions):
        array = np.array(positions_m) if backend is np else torch.tensor(positions_m)
        state, toggled = step_button_latch(state=state, displacement_m=array, dt_s=dt, model=model)
        for slot in range(2):
            scalars[slot], event = apply_button_sample(
                switch=scalars[slot], button_position=positions_m[slot], dt_s=dt, model=model
            )
            assert bool(toggled[slot]) == event
            events[slot] += event
        if (index + 1) % round(250 / policy_hz) == 0:
            observations.append(tuple(events))
    assert events.tolist() == [2, 2]
    assert all(not value.state for value in scalars)
    assert observations
    reset = ButtonLatchState(backend.zeros(2), backend.zeros(2, dtype=bool))
    _, toggled = step_button_latch(state=reset, displacement_m=array, dt_s=dt, model=model)
    assert not bool(toggled.any())


def test_switch_asset_detects_mutation_and_never_repairs(*, tmp_path, model):
    asset = write_switch_asset(root=tmp_path, model=model)
    assert load_switch_asset(model=model, repo_root=tmp_path, expected=asset) == asset
    asset.path.write_text("changed USD")
    with pytest.raises(ValueError, match="changed"):
        load_switch_asset(model=model, expected=asset)
    assert asset.path.read_text() == "changed USD"


def test_simultaneous_opposing_contacts_cannot_cancel():
    # Opposite normals contribute positive magnitudes to separate pairs.
    totals = contact_force_totals(
        forces=torch.tensor([2.6, -2.6, 1.0, 0.0]),
        starts=torch.tensor([[0, 1, 2, 3]]),
        counts=torch.tensor([[1, 1, 1, 1]]),
    )
    categories = classify_press_contacts(
        pad_forces_n=totals, body_force_n=torch.tensor([1.0]), threshold_n=0.25
    )
    assert categories.tolist() == [[True, True, True, False, True]]

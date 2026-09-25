from dataclasses import FrozenInstanceError, replace

import pytest

from robotics_experiments.config import load_switch_model
from robotics_experiments.environment import SwitchState, apply_button_sample
from tests.support.paths import REPO_ROOT


def _sample(
    *, switch: SwitchState, position: float, confirmation_s: float
) -> tuple[SwitchState, bool]:
    return apply_button_sample(
        switch=switch,
        button_position=position,
        dt_s=0.05,
        model=replace(
            load_switch_model(REPO_ROOT / "configs/experiment/switches/model/nominal.yaml"),
            confirmation_s=confirmation_s,
        ),
    )


def test_switch_toggles_only_on_rising_edge():
    switch = SwitchState("/Root/Switch")

    switch, toggled = _sample(switch=switch, position=0.0, confirmation_s=0.05)
    assert not toggled
    assert not switch.state

    switch, toggled = _sample(switch=switch, position=0.01, confirmation_s=0.05)
    assert toggled
    assert switch.state

    switch, toggled = _sample(switch=switch, position=0.016, confirmation_s=0.05)
    assert not toggled
    assert switch.state

    switch, toggled = _sample(switch=switch, position=0.002, confirmation_s=0.05)
    assert not toggled

    switch, toggled = _sample(switch=switch, position=0.012, confirmation_s=0.05)
    assert toggled
    assert not switch.state


def test_transition_leaves_input_state_unchanged():
    initial = SwitchState("/Root/Switch")

    successor, toggled = _sample(switch=initial, position=0.018, confirmation_s=0.05)

    assert toggled
    assert initial == SwitchState("/Root/Switch")
    assert successor.state
    with pytest.raises(FrozenInstanceError):
        initial.state = True


def test_sustained_press_toggles_after_confirmation_window():
    switch = SwitchState("/Root/Switch")

    for _ in range(4):
        switch, toggled = _sample(switch=switch, position=0.018, confirmation_s=0.25)
        assert not toggled
        assert not switch.state

    switch, toggled = _sample(switch=switch, position=0.018, confirmation_s=0.25)
    assert toggled
    assert switch.state


def test_brief_tap_does_not_toggle():
    switch = SwitchState("/Root/Switch")

    for _ in range(3):
        switch, _ = _sample(switch=switch, position=0.018, confirmation_s=0.25)
    switch, toggled = _sample(switch=switch, position=0.0, confirmation_s=0.25)

    assert not toggled
    assert not switch.state
    assert switch.press_duration_s == 0


def test_held_press_toggles_exactly_once():
    switch = SwitchState("/Root/Switch")
    toggles = []

    for _ in range(10):
        switch, toggled = _sample(switch=switch, position=0.018, confirmation_s=0.15)
        toggles.append(toggled)

    assert toggles.count(True) == 1
    assert switch.state


def test_release_then_press_toggles_again():
    switch = SwitchState("/Root/Switch")

    for _ in range(3):
        switch, _ = _sample(switch=switch, position=0.018, confirmation_s=0.15)
    assert switch.state

    switch, toggled = _sample(switch=switch, position=0.0, confirmation_s=0.15)
    assert not toggled
    for _ in range(2):
        switch, toggled = _sample(switch=switch, position=0.018, confirmation_s=0.15)
        assert not toggled
    switch, toggled = _sample(switch=switch, position=0.018, confirmation_s=0.15)
    assert toggled
    assert not switch.state

from dataclasses import FrozenInstanceError

import pytest

from robotics_experiments.environment import SwitchState, apply_button_sample


def _sample(
    switch: SwitchState, *, position: float, confirm_steps: int
) -> tuple[SwitchState, bool]:
    return apply_button_sample(
        switch=switch,
        button_position=position,
        press_threshold=0.5,
        confirm_steps=confirm_steps,
    )


def test_switch_toggles_only_on_rising_edge():
    switch = SwitchState("/Root/Switch")

    switch, toggled = _sample(switch, position=0.0, confirm_steps=1)
    assert not toggled
    assert not switch.state

    switch, toggled = _sample(switch, position=0.5, confirm_steps=1)
    assert toggled
    assert switch.state

    switch, toggled = _sample(switch, position=0.8, confirm_steps=1)
    assert not toggled
    assert switch.state

    switch, toggled = _sample(switch, position=0.1, confirm_steps=1)
    assert not toggled

    switch, toggled = _sample(switch, position=0.6, confirm_steps=1)
    assert toggled
    assert not switch.state


def test_transition_leaves_input_state_unchanged():
    initial = SwitchState("/Root/Switch")

    successor, toggled = _sample(initial, position=0.9, confirm_steps=1)

    assert toggled
    assert initial == SwitchState("/Root/Switch")
    assert successor.state
    with pytest.raises(FrozenInstanceError):
        initial.state = True


def test_sustained_press_toggles_after_confirmation_window():
    switch = SwitchState("/Root/Switch")

    for _ in range(4):
        switch, toggled = _sample(switch, position=0.9, confirm_steps=5)
        assert not toggled
        assert not switch.state

    switch, toggled = _sample(switch, position=0.9, confirm_steps=5)
    assert toggled
    assert switch.state


def test_brief_tap_does_not_toggle():
    switch = SwitchState("/Root/Switch")

    for _ in range(3):
        switch, _ = _sample(switch, position=0.9, confirm_steps=5)
    switch, toggled = _sample(switch, position=0.0, confirm_steps=5)

    assert not toggled
    assert not switch.state
    assert switch.press_steps == 0


def test_held_press_toggles_exactly_once():
    switch = SwitchState("/Root/Switch")
    toggles = []

    for _ in range(10):
        switch, toggled = _sample(switch, position=0.9, confirm_steps=3)
        toggles.append(toggled)

    assert toggles.count(True) == 1
    assert switch.state


def test_release_then_press_toggles_again():
    switch = SwitchState("/Root/Switch")

    for _ in range(3):
        switch, _ = _sample(switch, position=0.9, confirm_steps=3)
    assert switch.state

    switch, toggled = _sample(switch, position=0.0, confirm_steps=3)
    assert not toggled
    for _ in range(2):
        switch, toggled = _sample(switch, position=0.9, confirm_steps=3)
        assert not toggled
    switch, toggled = _sample(switch, position=0.9, confirm_steps=3)
    assert toggled
    assert not switch.state

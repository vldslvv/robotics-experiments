from robotics_experiments.environment import SwitchState
from robotics_experiments.environment.switches import _apply_button_sample as apply_button_sample


def test_switch_toggles_only_on_rising_edge():
    # confirm_steps=1 reduces the latch to a plain rising-edge toggle.
    switch = SwitchState("/Root/Switch")

    assert not apply_button_sample(
        switch, button_position=0.0, press_threshold=0.5, confirm_steps=1
    )
    assert not switch.state

    assert apply_button_sample(switch, button_position=0.5, press_threshold=0.5, confirm_steps=1)
    assert switch.state

    assert not apply_button_sample(
        switch, button_position=0.8, press_threshold=0.5, confirm_steps=1
    )
    assert switch.state

    assert not apply_button_sample(
        switch, button_position=0.1, press_threshold=0.5, confirm_steps=1
    )
    assert switch.state

    assert apply_button_sample(switch, button_position=0.6, press_threshold=0.5, confirm_steps=1)
    assert not switch.state


def test_sustained_press_toggles_after_confirmation_window():
    # The latch requires the press to hold for the full confirmation window.
    switch = SwitchState("/Root/Switch")

    # First four pressed samples are within the window: no toggle yet.
    for _ in range(4):
        assert not apply_button_sample(
            switch, button_position=0.9, press_threshold=0.5, confirm_steps=5
        )
        assert not switch.state

    # Fifth consecutive pressed sample confirms the toggle.
    assert apply_button_sample(switch, button_position=0.9, press_threshold=0.5, confirm_steps=5)
    assert switch.state


def test_brief_tap_does_not_toggle():
    # A tap that releases before the window completes must not count.
    switch = SwitchState("/Root/Switch")

    for _ in range(3):
        assert not apply_button_sample(
            switch, button_position=0.9, press_threshold=0.5, confirm_steps=5
        )
    # Released before the 5-step window -> no toggle, counter resets.
    assert not apply_button_sample(
        switch, button_position=0.0, press_threshold=0.5, confirm_steps=5
    )
    assert not switch.state


def test_held_press_toggles_exactly_once():
    # Holding past the window must not re-toggle every step.
    switch = SwitchState("/Root/Switch")

    toggles = [
        apply_button_sample(switch, button_position=0.9, press_threshold=0.5, confirm_steps=3)
        for _ in range(10)
    ]
    assert toggles.count(True) == 1
    assert switch.state


def test_release_then_press_toggles_again():
    # A fresh sustained press after release confirms a second toggle.
    switch = SwitchState("/Root/Switch")

    for _ in range(3):
        apply_button_sample(switch, button_position=0.9, press_threshold=0.5, confirm_steps=3)
    assert switch.state  # toggled on the first sustained press

    # Release, then press again for a full window.
    assert not apply_button_sample(
        switch, button_position=0.0, press_threshold=0.5, confirm_steps=3
    )
    for _ in range(2):
        assert not apply_button_sample(
            switch, button_position=0.9, press_threshold=0.5, confirm_steps=3
        )
    assert apply_button_sample(switch, button_position=0.9, press_threshold=0.5, confirm_steps=3)
    assert not switch.state  # toggled back off

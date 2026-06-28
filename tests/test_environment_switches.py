from robotics_experiments.environment import SwitchState
from robotics_experiments.environment._switches import _apply_button_sample as apply_button_sample


def test_switch_toggles_only_on_rising_edge():
    switch = SwitchState("/Root/Switch")

    assert not apply_button_sample(switch, button_position=0.0, press_threshold=0.5)
    assert not switch.state

    assert apply_button_sample(switch, button_position=0.5, press_threshold=0.5)
    assert switch.state

    assert not apply_button_sample(switch, button_position=0.8, press_threshold=0.5)
    assert switch.state

    assert not apply_button_sample(switch, button_position=0.1, press_threshold=0.5)
    assert switch.state

    assert apply_button_sample(switch, button_position=0.6, press_threshold=0.5)
    assert not switch.state

"""Pure tests for the backend-neutral press-task transition."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from robotics_experiments.environment import SwitchState, apply_button_sample
from robotics_experiments.rl.press import (
    PressTaskInput,
    initial_press_task_state,
    step_press_task,
)


def _reward_cfg():
    return SimpleNamespace(
        alive_weight=0.025,
        stability_weight=0.025,
        approach_progress_weight=10.0,
        gentle_contact_weight=10.0,
        compression_progress_weight=100.0,
        button_progress_weight=100.0,
        latch_progress_weight=10.0,
        toggle_weight=30.0,
        retreat_progress_weight=200.0,
        retreat_alignment_weight=50.0,
        retreat_overspeed_weight=50.0,
        success_weight=100.0,
        action_energy_weight=0.001,
        action_delta_weight=0.004,
        contact_oscillation_weight=0.2,
        scraping_weight=0.1,
        misalignment_weight=0.2,
        failure_weight=30.0,
        timeout_weight=25.0,
        gentle_contact_speed_max_mps=0.2,
        compression_target_m=0.01,
        compression_overshoot_tolerance_m=0.003,
        misalignment_tolerance_m=0.035,
    )


def _termination_cfg(**overrides):
    values = dict(
        crash_tilt_cos=0.5,
        min_altitude_m=0.25,
        max_wall_normal_distance_m=1.0,
        max_lateral_error_m=0.3,
        max_vertical_error_m=0.3,
        retreat_distance_m=0.35,
        retreat_speed_max_mps=0.2,
        retreat_alignment_tolerance_m=0.05,
        retreat_angular_speed_max_radps=0.5,
        stable_retreat_steps=1,
        first_contact_speed_max_mps=0.25,
        first_contact_angular_speed_max_radps=0.5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(
    *,
    button,
    relative=None,
    contact=None,
    linear_velocity=None,
    angular_velocity=None,
    body_z=None,
    altitude=None,
    time_out=None,
    action=None,
):
    button = np.asarray(button, dtype=np.float64)
    batch = button.shape[0]

    def vector(value, width):
        if value is None:
            return np.zeros((batch, width), dtype=np.float64)
        return np.asarray(value, dtype=np.float64).reshape(batch, width)

    def scalar(value, default, dtype=np.float64):
        if value is None:
            return np.full(batch, default, dtype=dtype)
        return np.asarray(value, dtype=dtype)

    relative_value = vector(relative, 3)
    if relative is None:
        relative_value[:, 0] = 0.3
    return PressTaskInput(
        relative_switch_pos_m=relative_value,
        body_linear_velocity_mps=vector(linear_velocity, 3),
        body_angular_velocity_radps=vector(angular_velocity, 3),
        body_z_world_z=scalar(body_z, 1.0),
        altitude_m=scalar(altitude, 1.0),
        pusher_compression_m=np.maximum(button, 0.0),
        pusher_compression_rate_mps=np.zeros(batch),
        button_position_m=button,
        button_rate_mps=np.zeros(batch),
        contact_flag=scalar(contact, False, bool),
        action=vector(action, 4),
        elapsed_s=np.arange(batch, dtype=np.float64) + 1.0,
        time_out=scalar(time_out, False, bool),
    )


def _initial(*, batch=1, switch_range_m=0.3):
    return initial_press_task_state(
        relative_switch_range_m=np.full(batch, switch_range_m),
        action=np.zeros((batch, 4)),
    )


def _step(*, state, sample, confirm_steps=2, termination_cfg=None):
    return step_press_task(
        state=state,
        sample=sample,
        reward_cfg=_reward_cfg(),
        termination_cfg=termination_cfg or _termination_cfg(),
        press_threshold_m=0.01,
        confirm_steps=confirm_steps,
    )


def test_vector_latch_matches_scalar_sequences() -> None:
    sequences = ([0.0, 0.02, 0.02, 0.02, 0.0], [0.02, 0.0, 0.02, 0.02, 0.02])
    vector_state = _initial(batch=2)
    scalar_states = [SwitchState("/switch/0"), SwitchState("/switch/1")]
    for samples in zip(*sequences, strict=True):
        result = _step(
            state=vector_state,
            sample=_sample(button=samples),
            confirm_steps=2,
        )
        vector_state = result.state
        scalar_toggles = []
        for index, position in enumerate(samples):
            scalar_states[index], toggled = apply_button_sample(
                switch=scalar_states[index],
                button_position=position,
                press_threshold=0.01,
                confirm_steps=2,
            )
            scalar_toggles.append(toggled)
        np.testing.assert_array_equal(
            vector_state.press_steps,
            [state.press_steps for state in scalar_states],
        )
        np.testing.assert_array_equal(
            vector_state.toggle_latched,
            [state.toggle_latched for state in scalar_states],
        )
        np.testing.assert_array_equal(
            vector_state.lamp_state,
            [state.state for state in scalar_states],
        )
        np.testing.assert_array_equal(result.metrics.toggled_this_step, scalar_toggles)


def test_reward_is_phase_gated_and_toggle_is_one_shot() -> None:
    state = _initial()
    approach = _step(
        state=state,
        sample=_sample(button=[0.0], relative=[[0.2, 0.0, 0.0]], contact=[False]),
    )
    assert approach.rewards.approach[0] > 0.0
    assert approach.rewards.toggle[0] == 0.0

    moved_away = _step(
        state=approach.state,
        sample=_sample(button=[0.0], relative=[[0.25, 0.0, 0.0]], contact=[False]),
    )
    assert moved_away.rewards.approach[0] < 0.0

    generic_contact = _step(
        state=moved_away.state,
        sample=_sample(button=[0.0], relative=[[0.19, 0.0, 0.0]], contact=[True]),
    )
    assert generic_contact.rewards.toggle[0] == 0.0
    assert generic_contact.rewards.gentle_contact[0] > 0.0

    unstable_contact = _step(
        state=approach.state,
        sample=_sample(
            button=[0.0],
            relative=[[0.19, 0.0, 0.0]],
            contact=[True],
            linear_velocity=[[0.5, 0.0, 0.0]],
        ),
    )
    assert unstable_contact.rewards.gentle_contact[0] < 0.0

    pressed = _step(
        state=generic_contact.state,
        sample=_sample(button=[0.02], relative=[[0.18, 0.0, 0.0]], contact=[True]),
    )
    toggled = _step(
        state=pressed.state,
        sample=_sample(button=[0.02], relative=[[0.18, 0.0, 0.0]], contact=[True]),
    )
    held = _step(
        state=toggled.state,
        sample=_sample(button=[0.02], relative=[[0.18, 0.0, 0.0]], contact=[True]),
    )
    assert pressed.rewards.toggle[0] == 0.0
    assert toggled.rewards.toggle[0] == _reward_cfg().toggle_weight
    assert held.rewards.toggle[0] == 0.0
    assert held.rewards.approach[0] == 0.0

    released = _step(
        state=held.state,
        sample=_sample(button=[0.0], relative=[[0.18, 0.0, 0.0]], contact=[False]),
    )
    repressed = _step(
        state=released.state,
        sample=_sample(button=[0.02], relative=[[0.18, 0.0, 0.0]], contact=[True]),
    )
    toggled_off = _step(
        state=repressed.state,
        sample=_sample(button=[0.02], relative=[[0.18, 0.0, 0.0]], contact=[True]),
    )
    assert not toggled_off.state.lamp_state[0]
    assert toggled_off.rewards.toggle[0] == 0.0
    assert toggled_off.metrics.toggled[0]


def test_latch_progress_rewards_hold_and_charges_full_regression() -> None:
    state = _initial()
    first = _step(
        state=state,
        sample=_sample(button=[0.02], contact=[True]),
        confirm_steps=3,
    )
    second = _step(
        state=first.state,
        sample=_sample(button=[0.02], contact=[True]),
        confirm_steps=3,
    )
    released = _step(
        state=second.state,
        sample=_sample(button=[0.0], contact=[False]),
        confirm_steps=3,
    )

    assert first.rewards.latch_progress[0] == pytest.approx(10.0 / 3.0)
    assert second.rewards.latch_progress[0] == pytest.approx(10.0 / 3.0)
    assert released.rewards.latch_progress[0] == pytest.approx(-20.0 / 3.0)
    assert first.rewards.latch_progress[0] + second.rewards.latch_progress[
        0
    ] + released.rewards.latch_progress[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("sample", "flag"),
    [
        (_sample(button=[0.0], body_z=[0.0]), "crash"),
        (_sample(button=[0.0], altitude=[0.1]), "altitude"),
        (
            _sample(button=[0.0], relative=[[0.2, 0.4, 0.0]]),
            "relative_envelope",
        ),
        (_sample(button=[0.0], time_out=[True]), "time_out"),
    ],
)
def test_termination_reasons_are_distinct(sample, flag) -> None:
    result = _step(state=_initial(), sample=sample)
    assert bool(getattr(result.termination, flag)[0])
    if flag == "time_out":
        assert not bool(result.termination.terminated[0])
    else:
        assert bool(result.termination.terminated[0])
        assert result.rewards.failure[0] < 0.0


def test_success_requires_latch_and_stable_retreat() -> None:
    state = _initial()
    first = _step(
        state=state,
        sample=_sample(button=[0.02], contact=[True]),
        confirm_steps=1,
    )
    assert first.metrics.toggled[0]
    assert not first.termination.success[0]
    contact_range = _step(
        state=first.state,
        sample=_sample(button=[0.0], relative=[[0.3, 0.0, 0.0]], contact=[False]),
        confirm_steps=1,
    )
    assert not contact_range.termination.success[0]
    retreat = _step(
        state=contact_range.state,
        sample=_sample(button=[0.0], relative=[[0.4, 0.0, 0.0]], contact=[False]),
        confirm_steps=1,
    )
    assert retreat.termination.success[0]
    assert retreat.rewards.success[0] > 0.0


def test_retreat_returns_to_initial_range_with_alignment_and_speed_control() -> None:
    state = _initial(switch_range_m=0.5)
    toggled = _step(
        state=state,
        sample=_sample(
            button=[0.02],
            relative=[[0.36, 0.0, 0.0]],
            contact=[True],
        ),
        confirm_steps=1,
    )
    partial_retreat = _step(
        state=toggled.state,
        sample=_sample(button=[0.0], relative=[[0.4, 0.0, 0.0]], contact=[False]),
        confirm_steps=1,
    )
    assert partial_retreat.rewards.retreat[0] > 0.0
    assert not partial_retreat.termination.success[0]

    misaligned = _step(
        state=partial_retreat.state,
        sample=_sample(button=[0.0], relative=[[0.5, 0.06, 0.0]], contact=[False]),
        confirm_steps=1,
    )
    assert misaligned.rewards.retreat_alignment[0] < 0.0
    assert not misaligned.termination.success[0]

    too_fast = _step(
        state=misaligned.state,
        sample=_sample(
            button=[0.0],
            relative=[[0.5, 0.0, 0.0]],
            contact=[False],
            linear_velocity=[[0.3, 0.0, 0.0]],
        ),
        confirm_steps=1,
    )
    assert too_fast.rewards.retreat_overspeed[0] < 0.0
    assert not too_fast.termination.success[0]

    stable = _step(
        state=too_fast.state,
        sample=_sample(button=[0.0], relative=[[0.5, 0.0, 0.0]], contact=[False]),
        confirm_steps=1,
    )
    assert stable.termination.success[0]
    assert stable.rewards.success[0] > 0.0

    overshot = _step(
        state=stable.state,
        sample=_sample(button=[0.0], relative=[[0.6, 0.0, 0.0]], contact=[False]),
        confirm_steps=1,
    )
    assert overshot.rewards.retreat[0] == 0.0


def test_failure_cannot_also_be_success() -> None:
    toggled = _step(
        state=_initial(),
        sample=_sample(button=[0.02], contact=[True]),
        confirm_steps=1,
    )
    crashed_retreat = _step(
        state=toggled.state,
        sample=_sample(
            button=[0.0],
            relative=[[0.4, 0.0, 0.0]],
            contact=[False],
            body_z=[0.0],
        ),
        confirm_steps=1,
    )

    assert crashed_retreat.termination.crash[0]
    assert not crashed_retreat.termination.success[0]
    assert crashed_retreat.rewards.success[0] == 0.0


def test_press_state_is_frozen() -> None:
    state = _initial()
    with pytest.raises(FrozenInstanceError):
        state.press_steps = np.ones(1)

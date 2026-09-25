"""Contact outcomes, production latch parity and independent qualification."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from robotics_experiments.config import load_config, load_curriculum_config
from robotics_experiments.environment.switches import SwitchState, apply_button_sample
from robotics_experiments.flight_parameters import load_flight_parameters
from robotics_experiments.rl.evaluation import EvaluationMetric, PolicyEvaluation
from robotics_experiments.rl.press.contact import (
    initial_press_contact_state,
    reset_press_contact_state,
    step_press_contact,
)
from robotics_experiments.rl.press.evaluation import (
    create_press_acceptance_criteria,
    evaluate_press_acceptance,
)
from tests.support.paths import REPO_ROOT


def _step(
    *,
    state,
    config,
    button=0.0,
    contact=True,
    retreat=False,
    compression=0.005,
    collision=False,
    toggles=None,
):
    flag = torch.tensor([False])
    flight = SimpleNamespace(
        linear_speed_mps=torch.tensor([0.01]),
        angular_speed_radps=torch.tensor([0.0]),
        alignment_error_m=torch.tensor([0.0]),
        attitude_error_rad=torch.tensor([0.0]),
        range_m=torch.tensor([0.48 if retreat else 0.36]),
        range_error_m=torch.tensor([0.0 if retreat else 0.12]),
        strict_goal=torch.tensor([retreat]),
        termination=SimpleNamespace(
            collision=torch.tensor([collision]), flight_failure=flag, time_out=flag
        ),
    )
    return step_press_contact(
        state=state,
        flight=flight,
        button_position_m=torch.tensor([button]),
        compression_m=torch.tensor([compression]),
        intended_contact=torch.tensor([contact]),
        contact_categories=torch.tensor([[contact, collision, False, False, False]]),
        simultaneous_contact=torch.tensor([contact and collision]),
        contact_seen=torch.tensor([True]),
        first_contact_stable=torch.tensor([True]),
        action=torch.zeros((1, 4)),
        config=config.rl.press,
        press_threshold_m=config.switches.model.activation_displacement_m,
        toggle_count=state.toggle_count if toggles is None else torch.tensor([toggles]),
    )


def test_contact_latch_matches_production_and_repeated_press_fails():
    config = load_config(REPO_ROOT / "configs/train/stage/press_easy_quad.yaml")
    state = initial_press_contact_state(ranges=torch.tensor([0.48]))
    switch = SwitchState(prim_path="switch")
    transitions = []
    for button in [0.0, 0.01, 0.01, 0.0] + [0.01] * 8 + [0.0] + [0.01] * 5:
        switch, toggled = apply_button_sample(
            switch=switch,
            button_position=button,
            dt_s=0.05,
            model=config.switches.model,
        )
        result = _step(
            state=state, config=config, button=button, toggles=sum(transitions) + toggled
        )
        state = result.state
        transitions.append(toggled)
        assert bool(state.toggle_count.remainder(2)) == switch.state
        assert torch.isfinite(result.reward).all()
    assert sum(transitions) == 2
    assert bool(result.failure)
    assert not bool(result.success)


@pytest.mark.parametrize("stroke_fraction", [0.5, 1.0])
def test_press_requires_toggle_release_and_complete_retreat_hold(*, stroke_fraction):
    config = load_config(REPO_ROOT / "configs/train/stage/press_easy_quad.yaml")
    state = initial_press_contact_state(ranges=torch.tensor([0.48]))
    for _ in range(5):
        result = _step(
            state=state,
            config=config,
            button=config.switches.model.travel_m * stroke_fraction,
            toggles=1,
        )
        state = result.state
    assert int(state.toggle_count) == 1
    assert not bool(result.success)
    for index in range(config.rl.press.goal.stable_steps):
        result = _step(state=state, config=config, contact=False, retreat=True, compression=0.0)
        state = result.state
        assert bool(result.success) == (index == config.rl.press.goal.stable_steps - 1)
    assert not bool(result.failure)


@pytest.mark.parametrize("collision,compression", [(True, 0.0), (False, 0.016)])
def test_contact_rejects_unsafe_collision_or_excess_compression(collision, compression):
    config = load_config(REPO_ROOT / "configs/train/stage/press_contact_quad.yaml")
    result = _step(
        state=initial_press_contact_state(ranges=torch.tensor([0.48])),
        config=config,
        collision=collision,
        compression=compression,
    )
    assert bool(result.failure)
    assert not bool(result.success)


def test_gentle_contact_and_reset_do_not_carry_episode_state():
    config = load_config(REPO_ROOT / "configs/train/stage/press_contact_quad.yaml")
    state = initial_press_contact_state(ranges=torch.tensor([0.48]))
    for _ in range(config.rl.press.contact.stable_steps):
        result = _step(state=state, config=config)
        state = result.state
    assert bool(result.success)
    fresh = reset_press_contact_state(
        state=state, env_ids=torch.tensor([0]), ranges=torch.tensor([0.5])
    )
    assert not bool(fresh.contact_seen)
    assert int(fresh.stable_contact_steps) == 0
    assert int(state.stable_contact_steps) == config.rl.press.contact.stable_steps
    assert bool(_step(state=state, config=config, button=0.01, toggles=1).failure)


def test_contact_acceptance_requires_real_contact_and_retreat_populations():
    config = load_config(REPO_ROOT / "configs/train/stage/press_easy_quad.yaml")
    criteria = create_press_acceptance_criteria(
        config=config.rl.press,
        flight_parameters=load_flight_parameters(REPO_ROOT / "tests/fixtures/flight_parameters"),
    )
    values = {requirement.metric: requirement.required for requirement in criteria.requirements}
    evaluation = PolicyEvaluation(
        iteration=0,
        checkpoint_path=REPO_ROOT,
        policy_digest="test",
        source="fresh-test",
        metrics=tuple(EvaluationMetric(name, value) for name, value in values.items()),
        reward_means=(),
    )
    assert evaluate_press_acceptance(criteria=criteria, evaluation=evaluation).passed
    missing = replace(
        evaluation,
        metrics=tuple(
            EvaluationMetric(
                name, None if name in {"retreat_rate", "stable_first_contact_rate"} else value
            )
            for name, value in values.items()
        ),
    )
    acceptance = evaluate_press_acceptance(criteria=criteria, evaluation=missing)
    assert not acceptance.passed
    assert set(acceptance.reasons) == {"stable first contact", "retreat among toggles"}


def test_complete_curriculum_shares_learner_contract_and_provisional_limits():
    curriculum = load_curriculum_config(REPO_ROOT / "configs/train/press_curriculum_quad.yaml")
    experiments = [load_config(REPO_ROOT / stage.config_path) for stage in curriculum.stages]
    assert curriculum.initial_learner is None
    assert [config.rl.press.task for config in experiments] == [
        "hover",
        "approach",
        "contact",
        "press",
        "press",
    ]
    for config in experiments:
        assert config.rl.press.policy.normalize_observations
        assert (
            config.rl.press.policy.actor.hidden_dims
            == experiments[0].rl.press.policy.actor.hidden_dims
        )
        assert config.rl.inputs == experiments[0].rl.inputs
    assert experiments[0].rl.training.gamma == 0.995
    assert experiments[0].rl.training.lam == 0.99
    assert experiments[0].rl.training.plateau.metric_name == "episode_longest_hover_duration_mean_s"
    assert experiments[-1].switches.model.button_width_m == 0.06
    assert all(config.switches.model.button_width_m == 0.09 for config in experiments[:-1])
    assert experiments[-1].rl.press.acceptance.success_rate_min == 0.8
    assert experiments[-1].rl.press.contact.first_contact_rate_min == 0.85
    assert experiments[-1].rl.press.contact.retreat_rate_min == 0.95

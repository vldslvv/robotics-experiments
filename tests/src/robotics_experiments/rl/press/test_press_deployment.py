"""Tests for the pure deploy-time press-policy transition."""

from dataclasses import replace

import numpy as np

from robotics_experiments.rl.press import (
    HandoffGate,
    PressObservation,
    PressObservationMask,
    PressPolicyParams,
    initial_press_policy_progress,
    step_press_policy,
)


def _observation(**overrides) -> PressObservation:
    observation = PressObservation(
        relative_switch_pos_m=np.array([0.2, 0.0, 0.0]),
        relative_switch_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pose_confidence=1.0,
        pose_age_s=0.0,
        detection_valid=True,
        body_lin_vel_mps=np.zeros(3),
        body_ang_vel_radps=np.zeros(3),
        attitude_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pusher_compression_m=0.0,
        pusher_compression_rate_mps=0.0,
        contact_valid=True,
        contact_flag=False,
        lamp_state=0,
        lamp_confidence=1.0,
        lamp_age_s=0.0,
        previous_action=np.zeros(4),
        handoff_mode=1,
        estimator_health=1.0,
        obs_mask=int(
            PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
        ),
    )
    return replace(observation, **overrides)


_GATE = HandoffGate(
    pose_confidence_min=0.9,
    pose_age_max_s=0.1,
    switch_range_max_m=0.3,
    lateral_align_max_m=0.05,
    wall_normal_speed_max_mps=0.2,
    lateral_speed_max_mps=0.2,
    angular_speed_max_radps=0.5,
    estimator_health_min=0.9,
)


_PARAMS = PressPolicyParams(
    policy=lambda vector: np.full(4, 0.5),
    action_dim=4,
    handoff_gate=_GATE,
    first_contact_speed_max_mps=0.25,
    first_contact_angular_speed_max_radps=0.5,
    retreat_distance_m=0.35,
    retreat_speed_max_mps=0.2,
    retreat_alignment_tolerance_m=0.05,
    retreat_angular_speed_max_radps=0.5,
    stable_retreat_steps=2,
)


def _step(progress, observation, policy=lambda vector: np.full(4, 0.5), /):
    return step_press_policy(
        params=replace(_PARAMS, policy=policy),
        progress=progress,
        observation=observation,
    )


def test_policy_waits_for_handoff_and_requires_initial_lamp_off() -> None:
    progress = initial_press_policy_progress(action_dim=4)
    calls: list[np.ndarray] = []

    def policy(vector: np.ndarray, /) -> np.ndarray:
        calls.append(vector)
        return np.ones(4)

    waiting = _step(
        progress,
        _observation(relative_switch_pos_m=np.array([0.5, 0.0, 0.0])),
        policy,
    )

    assert not waiting.progress.handoff_complete
    assert np.array_equal(waiting.action, np.zeros(4))
    assert calls == []

    invalid = _step(progress, _observation(lamp_state=1))
    assert invalid.failure_reason == "press policy requires the paired lamp to start off"


def test_one_toggle_then_stable_retreat_completes() -> None:
    decision = _step(initial_press_policy_progress(action_dim=4), _observation())
    assert decision.progress.handoff_complete
    assert np.array_equal(decision.action, np.full(4, 0.5))

    decision = _step(
        decision.progress,
        _observation(contact_flag=True, body_lin_vel_mps=np.array([0.1, 0.0, 0.0])),
    )
    assert decision.progress.first_contact_stable

    decision = _step(decision.progress, _observation(contact_flag=True, lamp_state=1))
    assert decision.progress.toggle_count == 1
    retreat = _observation(
        lamp_state=1,
        relative_switch_pos_m=np.array([0.4, 0.0, 0.0]),
    )
    decision = _step(decision.progress, retreat)
    assert not decision.complete
    decision = _step(decision.progress, retreat)

    assert decision.complete
    assert decision.progress.stable_retreat_steps == 2
    assert decision.progress.initial_switch_range_m == 0.2
    assert decision.progress.closest_switch_range_m == 0.2
    assert decision.progress.peak_abs_action == 0.5
    assert np.array_equal(decision.action, np.zeros(4))


def test_deployment_progress_records_approach_and_peak_compression() -> None:
    decision = _step(initial_press_policy_progress(action_dim=4), _observation())
    decision = _step(
        decision.progress,
        _observation(
            relative_switch_pos_m=np.array([0.08, 0.0, 0.0]),
            pusher_compression_m=0.004,
        ),
        lambda vector: np.array([0.25, -0.75, 0.0, 0.1]),
    )
    decision = _step(
        decision.progress,
        _observation(
            relative_switch_pos_m=np.array([0.12, 0.0, 0.0]),
            pusher_compression_m=0.002,
        ),
        lambda vector: np.zeros(4),
    )

    assert decision.progress.initial_switch_range_m == 0.2
    assert decision.progress.closest_switch_range_m == 0.08
    assert decision.progress.peak_pusher_compression_m == 0.004
    assert decision.progress.peak_abs_action == 0.75


def test_retreat_returns_to_initial_range_aligned_and_stable() -> None:
    initial = _observation(relative_switch_pos_m=np.array([0.5, 0.0, 0.0]))
    decision = _step(initial_press_policy_progress(action_dim=4), initial)
    decision = _step(
        decision.progress,
        _observation(
            lamp_state=1,
            contact_flag=True,
            relative_switch_pos_m=np.array([0.36, 0.0, 0.0]),
        ),
    )

    decision = _step(
        decision.progress,
        _observation(lamp_state=1, relative_switch_pos_m=np.array([0.4, 0.0, 0.0])),
    )
    assert not decision.complete
    assert decision.progress.stable_retreat_steps == 0

    decision = _step(
        decision.progress,
        _observation(lamp_state=1, relative_switch_pos_m=np.array([0.5, 0.06, 0.0])),
    )
    assert decision.progress.stable_retreat_steps == 0

    decision = _step(
        decision.progress,
        _observation(
            lamp_state=1,
            relative_switch_pos_m=np.array([0.5, 0.0, 0.0]),
            body_ang_vel_radps=np.array([0.0, 0.0, 0.6]),
        ),
    )
    assert decision.progress.stable_retreat_steps == 0

    stable = _observation(
        lamp_state=1,
        relative_switch_pos_m=np.array([0.5, 0.0, 0.0]),
    )
    decision = _step(decision.progress, stable)
    assert not decision.complete
    decision = _step(decision.progress, stable)
    assert decision.complete


def test_second_lamp_transition_fails_instead_of_rewarding_toggle_gaming() -> None:
    decision = _step(initial_press_policy_progress(action_dim=4), _observation())
    decision = _step(decision.progress, _observation(lamp_state=1))
    decision = _step(decision.progress, _observation(lamp_state=0))

    assert decision.progress.toggle_count == 2
    assert decision.failure_reason == "paired lamp toggled more than once during one press step"
    assert not decision.complete


def test_first_contact_stability_is_recorded_only_on_first_contact() -> None:
    decision = _step(initial_press_policy_progress(action_dim=4), _observation())
    decision = _step(
        decision.progress,
        _observation(contact_flag=True, body_lin_vel_mps=np.array([0.3, 0.0, 0.0])),
    )
    assert decision.progress.first_contact_seen
    assert not decision.progress.first_contact_stable

    decision = _step(
        decision.progress,
        _observation(contact_flag=True, body_lin_vel_mps=np.zeros(3)),
    )
    assert not decision.progress.first_contact_stable


def test_missing_sensor_mask_and_invalid_policy_output_fail_closed() -> None:
    progress = initial_press_policy_progress(action_dim=4)
    missing = _step(progress, _observation(obs_mask=int(PressObservationMask.POSE)))
    assert "missing pose, contact, or lamp" in str(missing.failure_reason)

    invalid_action = _step(progress, _observation(), lambda vector: np.zeros(3))
    assert "invalid action shape" in str(invalid_action.failure_reason)

    nonfinite = replace(_observation(), relative_switch_pos_m=np.array([np.nan, 0.0, 0.0]))
    assert "non-finite" in str(_step(progress, nonfinite).failure_reason)


def test_transition_injects_previous_action_and_handoff_mode_before_inference() -> None:
    vectors = []

    def policy(vector: np.ndarray, /) -> np.ndarray:
        vectors.append(vector.copy())
        return np.array([2.0, -2.0, 0.25, -0.25])

    decision = _step(
        initial_press_policy_progress(action_dim=4),
        _observation(previous_action=np.full(4, 99.0), handoff_mode=99),
        policy,
    )
    assert np.array_equal(vectors[-1][27:31], np.zeros(4))
    assert vectors[-1][31] == 1.0
    assert np.array_equal(decision.action, np.array([1.0, -1.0, 0.25, -0.25]))

    decision = _step(
        decision.progress,
        _observation(contact_flag=True),
        policy,
    )
    decision = _step(decision.progress, _observation(), policy)
    assert np.array_equal(vectors[-1][27:31], np.array([1.0, -1.0, 0.25, -0.25]))
    assert vectors[-1][31] == 2.0

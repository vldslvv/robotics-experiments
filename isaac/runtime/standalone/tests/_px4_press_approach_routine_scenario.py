"""Strict assertions for the trained contact-free approach mission."""

from __future__ import annotations

from _px4_takeoff_scenario import Px4TakeoffScenarioOptions, run_px4_takeoff_scenario
from _runtime import SceneTestContext

from robotics_experiments.rl.press import PressApproachExecutionState

__all__ = ["run_px4_press_approach_routine_scenario"]

_PHYSICAL_ROUTINE_KINDS = (
    "arm",
    "takeoff",
    "hover",
    "goto",
    "hover",
    "goto",
    "hover",
    "policy",
    "goto",
    "land",
)
_TEMPORAL_ROUTINE_KINDS = (
    "arm",
    "takeoff",
    "hover",
    "goto",
    "hover",
    "goto",
    "hover",
    "policy",
    "goto",
    "hover",
    "goto",
    "hover",
    "policy",
    "goto",
    "hover",
    "goto",
    "hover",
    "policy",
)


def _validate_result(context: SceneTestContext, fleet) -> None:
    if len(fleet.flying) != 1:
        raise AssertionError(f"approach routine requires one flying drone, got {len(fleet.flying)}")
    if fleet.switch_events:
        raise AssertionError(f"approach routine must not toggle the lamp: {fleet.switch_events!r}")
    policy_state = fleet.flying[0].driver.routine_state.last_policy_state
    if not isinstance(policy_state, PressApproachExecutionState):
        raise AssertionError(f"approach routine did not retain approach state: {policy_state!r}")
    required = context.config.rl.press.goal.stable_steps
    if (
        not policy_state.complete
        or policy_state.contact_failure
        or policy_state.stable_goal_steps < required
    ):
        raise AssertionError(f"approach routine did not complete stably: {policy_state!r}")
    goal = context.config.rl.press.goal
    if not goal.range_min_m <= policy_state.closest_switch_range_m <= goal.range_max_m:
        raise AssertionError(f"approach routine closest range missed the goal: {policy_state!r}")
    observer = fleet.flying[0].spawned.policy_observer
    if observer is None:
        raise AssertionError("approach routine has no press contact diagnostics")
    contacts = observer.read_contact_diagnostics()
    if contacts.body_contact or contacts.pusher_contact:
        raise AssertionError(f"approach routine made physical contact: {contacts!r}")
    context.report(
        "approach routine verified handoff, stable contact-free completion, "
        "no lamp transition, return waypoint, landing, and disarm; "
        f"action={policy_state.previous_action} "
        f"closest_range_m={policy_state.closest_switch_range_m:.4f}"
    )


def _validate_temporal_result(context: SceneTestContext, fleet) -> None:
    if len(fleet.flying) != 1:
        raise AssertionError(
            f"temporal conformance requires one flying drone, got {len(fleet.flying)}"
        )
    if context.config.rl.press.reset.physical_contact_enabled:
        raise AssertionError("temporal conformance requires physical_contact_enabled=false")
    if fleet.switch_events:
        raise AssertionError(
            f"temporal conformance toggled the virtual switch: {fleet.switch_events!r}"
        )
    policy_state = fleet.flying[0].driver.routine_state.last_policy_state
    if not isinstance(policy_state, PressApproachExecutionState) or not policy_state.complete:
        raise AssertionError(
            f"temporal conformance did not complete its final trace: {policy_state!r}"
        )
    context.report(
        "wall-free temporal conformance completed production-actor traces "
        "from nominal 0.460, 0.475, and 0.490 m entry ranges"
    )


def run_px4_press_approach_routine_scenario(*, context: SceneTestContext) -> None:
    """Validate config shape, then drive and inspect the complete mission."""
    kinds = tuple(step["kind"] for step in context.config.vehicles["x500_01"].routine.steps)
    if kinds == _PHYSICAL_ROUTINE_KINDS:
        post_routine_hook = _validate_result
        success_label = "trained press-approach mission"
    elif kinds == _TEMPORAL_ROUTINE_KINDS:
        post_routine_hook = _validate_temporal_result
        success_label = "wall-free press temporal conformance"
    else:
        raise AssertionError(
            "approach routine kinds are unsupported: "
            f"{kinds!r}; expected {_PHYSICAL_ROUTINE_KINDS!r} "
            f"or {_TEMPORAL_ROUTINE_KINDS!r}"
        )
    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name="px4_press_approach_routine",
            px4_container_log_name="test_px4_press_approach_routine_px4.log",
            post_routine_hook=post_routine_hook,
            success_label=success_label,
        ),
    )

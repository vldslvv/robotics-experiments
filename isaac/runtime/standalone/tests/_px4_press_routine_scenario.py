"""Shared strict assertions for the trained press-policy mission routine."""

from __future__ import annotations

from _px4_takeoff_scenario import Px4TakeoffScenarioOptions, run_px4_takeoff_scenario
from _runtime import SceneTestContext

from robotics_experiments.rl.press import PressPolicyExecutionState

__all__ = ["run_px4_press_routine_scenario"]


def _validate_result(context: SceneTestContext, fleet) -> None:
    if len(fleet.flying) != 1:
        raise AssertionError(f"press routine requires one flying drone, got {len(fleet.flying)}")
    events = fleet.switch_events
    if len(events) != 1 or events[0].switch_index != 0 or not events[0].state:
        raise AssertionError(
            f"press routine requires exactly one off-to-on lamp toggle: {events!r}"
        )
    policy_state = fleet.flying[0].driver.routine_state.last_policy_state
    if not isinstance(policy_state, PressPolicyExecutionState):
        raise AssertionError(f"press routine did not retain press policy state: {policy_state!r}")
    if not policy_state.first_contact_seen or not policy_state.first_contact_stable:
        raise AssertionError(
            f"press routine first contact was absent or unstable: {policy_state!r}"
        )
    if policy_state.toggle_count != 1:
        raise AssertionError(f"press routine toggle count is not one: {policy_state!r}")
    required = context.config.rl.press.termination.stable_retreat_steps
    if policy_state.stable_retreat_steps < required:
        raise AssertionError(f"press routine did not complete stable retreat: {policy_state!r}")
    context.report(
        "press routine verified handoff, stable first contact, one latch, stable retreat, "
        "return waypoint, landing, and disarm"
    )


def run_px4_press_routine_scenario(*, context: SceneTestContext) -> None:
    """Validate config shape, then drive and inspect the complete mission."""
    kinds = tuple(step["kind"] for step in context.config.vehicles["x500_01"].routine.steps)
    expected = ("arm", "takeoff", "hover", "goto", "hover", "policy", "goto", "land")
    if kinds != expected:
        raise AssertionError(f"press routine kinds are {kinds!r}, expected {expected!r}")
    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name="px4_press_policy_routine",
            px4_container_log_name="test_px4_press_policy_routine_px4.log",
            post_routine_hook=_validate_result,
            success_label="trained press-policy mission",
        ),
    )

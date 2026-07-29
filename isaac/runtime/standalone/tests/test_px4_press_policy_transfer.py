"""Wall-free downstream PX4 acceptance for a trained press policy."""

from __future__ import annotations

from _px4_takeoff_scenario import Px4TakeoffScenarioOptions, run_px4_takeoff_scenario
from _runtime import SceneTestContext, run_scene_test

from robotics_experiments.rl.press import PressApproachExecutionState

_TEST_NAME = "px4_press_policy_transfer"


def _validate(*, context: SceneTestContext, fleet) -> None:
    if len(fleet.flying) != 1:
        raise AssertionError(f"policy transfer requires one flying drone, got {len(fleet.flying)}")
    if context.config.rl.press.reset.physical_contact_enabled:
        raise AssertionError("policy transfer requires physical_contact_enabled=false")
    if fleet.switch_events:
        raise AssertionError(f"policy transfer toggled the switch: {fleet.switch_events!r}")
    state = fleet.flying[0].driver.routine_state.last_policy_state
    if not isinstance(state, PressApproachExecutionState) or not state.complete:
        raise AssertionError(f"policy transfer did not complete its final stage: {state!r}")
    context.report("wall-free press policy transfer completed at three distinct entry ranges")


def main() -> None:
    run_scene_test(
        test_name=_TEST_NAME,
        manage_stage=False,
        extra_python_paths=["isaac/runtime/extensions/px4_sitl_brigde"],
        test_body=lambda context: run_px4_takeoff_scenario(
            context,
            Px4TakeoffScenarioOptions(
                test_name=_TEST_NAME,
                px4_container_log_name="test_px4_press_policy_transfer_px4.log",
                post_routine_hook=lambda hook_context, fleet: _validate(
                    context=hook_context,
                    fleet=fleet,
                ),
                success_label="wall-free press policy transfer",
            ),
        ),
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()

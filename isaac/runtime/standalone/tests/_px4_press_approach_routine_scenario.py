"""Waypoint-to-policy handoff measurement and actor-invocation smoke gate."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from _px4_takeoff_scenario import Px4TakeoffScenarioOptions, run_px4_takeoff_scenario
from _runtime import SceneTestContext

from robotics_experiments.rl.press import (
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    PressApproachExecutionState,
)

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


@dataclass(frozen=True)
class _HandoffEvidence:
    navigation_reached_policy: bool = False
    entry_observation: tuple[float, ...] = ()
    handoff_latched: bool = False
    actor_evaluations: int = 0
    actor_policy_generation: int = 0
    actor_setpoint_transmitted: bool = False


def _observe_handoff(*, fleet, evidence: _HandoffEvidence) -> _HandoffEvidence:
    if len(fleet.flying) != 1:
        return evidence
    routine_state = fleet.flying[0].driver.routine_state
    policy_index = _PHYSICAL_ROUTINE_KINDS.index("policy")
    navigation_reached_policy = evidence.navigation_reached_policy or (
        routine_state.cursor >= policy_index
    )
    policy_state = routine_state.offboard.policy_state
    if not isinstance(policy_state, PressApproachExecutionState):
        return replace(evidence, navigation_reached_policy=navigation_reached_policy)
    actor_policy_generation = evidence.actor_policy_generation
    if policy_state.actor_evaluations > 0 and actor_policy_generation == 0:
        actor_policy_generation = routine_state.offboard.policy_generation
    return _HandoffEvidence(
        navigation_reached_policy=navigation_reached_policy,
        entry_observation=(evidence.entry_observation or policy_state.entry_observation),
        handoff_latched=evidence.handoff_latched or policy_state.handoff_latched,
        actor_evaluations=max(evidence.actor_evaluations, policy_state.actor_evaluations),
        actor_policy_generation=actor_policy_generation,
        actor_setpoint_transmitted=(
            evidence.actor_setpoint_transmitted
            or (
                actor_policy_generation > 0
                and routine_state.offboard.transmitted_policy_generation == actor_policy_generation
            )
        ),
    )


def _actor_invoked(*, evidence: _HandoffEvidence) -> bool:
    return (
        evidence.navigation_reached_policy
        and evidence.handoff_latched
        and evidence.actor_evaluations > 0
        and evidence.actor_setpoint_transmitted
    )


def _measurement_payload(
    *, context: SceneTestContext, evidence: _HandoffEvidence
) -> dict[str, object]:
    steps = context.config.vehicles["x500_01"].routine.steps
    policy_index = next(index for index, step in enumerate(steps) if step["kind"] == "policy")
    goto = next(step for step in reversed(steps[:policy_index]) if step["kind"] == "goto")
    hover = steps[policy_index - 1]
    observation = evidence.entry_observation
    if len(observation) != PRESS_OBSERVATION_DIM:
        raise AssertionError(
            "navigation handoff measurement has no exact entry observation: " f"{evidence!r}"
        )
    return {
        "artifact_kind": "press-navigation-handoff-sample",
        "schema_version": 2,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation": list(observation),
        "relative_switch_pos_m": list(observation[:3]),
        "relative_switch_quat_xyzw": list(observation[3:7]),
        "route": {
            "goto": dict(goto),
            "settle_hold_s": float(hover["hold_s"]),
        },
    }


def _validate_invocation(
    context: SceneTestContext,
    fleet,
    evidence: _HandoffEvidence,
) -> None:
    if len(fleet.flying) != 1:
        raise AssertionError(f"approach routine requires one flying drone, got {len(fleet.flying)}")
    if not _actor_invoked(evidence=evidence):
        raise AssertionError(f"approach actor invocation evidence is incomplete: {evidence!r}")
    context.report(
        "approach routine verified final waypoint, settling, handoff latch, one valid "
        "actor evaluation, and transmission of that evaluation's setpoint; "
        f"entry_position_m={evidence.entry_observation[:3]}"
    )


def run_px4_press_approach_routine_scenario(*, context: SceneTestContext) -> None:
    """Measure waypoint arrival or prove that it invokes the configured actor."""
    kinds = tuple(step["kind"] for step in context.config.vehicles["x500_01"].routine.steps)
    if kinds != _PHYSICAL_ROUTINE_KINDS:
        raise AssertionError(
            "approach routine kinds are unsupported: "
            f"{kinds!r}; expected {_PHYSICAL_ROUTINE_KINDS!r}"
        )
    measurement_output_text = os.environ.get("PRESS_NAVIGATION_MEASUREMENT_OUTPUT", "")
    measurement_output = Path(measurement_output_text) if measurement_output_text else None
    evidence = _HandoffEvidence()

    def stop(fleet) -> bool:
        nonlocal evidence
        evidence = _observe_handoff(fleet=fleet, evidence=evidence)
        return (
            bool(evidence.entry_observation)
            if measurement_output
            else _actor_invoked(evidence=evidence)
        )

    def validate(context: SceneTestContext, fleet) -> None:
        if measurement_output is None:
            _validate_invocation(context, fleet, evidence)
            return
        payload = _measurement_payload(context=context, evidence=evidence)
        measurement_output.parent.mkdir(parents=True, exist_ok=True)
        measurement_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        context.report(
            "recorded production-route press handoff measurement at "
            f"{measurement_output}: relative_switch_pos_m="
            f"{payload['relative_switch_pos_m']}"
        )

    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name="px4_press_approach_routine",
            px4_container_log_name="test_px4_press_approach_routine_px4.log",
            routine_stop_predicate=stop,
            post_routine_hook=validate,
            success_label=(
                "production waypoint handoff measurement"
                if measurement_output
                else "production waypoint actor invocation"
            ),
        ),
    )

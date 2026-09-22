"""Host tests for the production press actor-invocation predicate."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

from robotics_experiments.rl.press import PressApproachExecutionState
from tests.support.paths import REPO_ROOT


def _load_scenario(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    takeoff = ModuleType("_px4_takeoff_scenario")
    takeoff.Px4TakeoffScenarioOptions = object  # type: ignore[attr-defined]  # dynamic fake
    takeoff.run_px4_takeoff_scenario = (  # type: ignore[attr-defined]  # dynamic fake
        lambda *_args, **_kwargs: None
    )
    runtime = ModuleType("_runtime")
    runtime.SceneTestContext = object  # type: ignore[attr-defined]  # dynamic fake
    monkeypatch.setitem(sys.modules, "_px4_takeoff_scenario", takeoff)
    monkeypatch.setitem(sys.modules, "_runtime", runtime)
    path = REPO_ROOT / "isaac/runtime/standalone/tests/_px4_press_approach_routine_scenario.py"
    spec = importlib.util.spec_from_file_location("_press_approach_routine_scenario", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _fleet(
    *,
    cursor: int,
    handoff_latched: bool,
    actor_evaluations: int,
    policy_generation: int,
    transmitted_policy_generation: int,
    contact_failure: bool = False,
):
    policy_state = PressApproachExecutionState(
        previous_action=(0.0, 0.0, 0.0, 0.0),
        handoff_latched=handoff_latched,
        entry_observation=(0.0,) * 12 + (1.0, 0.0, 0.0, 0.0, 1.0, 0.0) + (0.0,) * 17,
        actor_evaluations=actor_evaluations,
        contact_failure=contact_failure,
    )
    routine_state = SimpleNamespace(
        cursor=cursor,
        offboard=SimpleNamespace(
            policy_state=policy_state,
            policy_generation=policy_generation,
            transmitted_policy_generation=transmitted_policy_generation,
        ),
    )
    return SimpleNamespace(
        flying=(SimpleNamespace(driver=SimpleNamespace(routine_state=routine_state)),),
    )


def test_handoff_wait_and_policy_entry_do_not_count_as_actor_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_scenario(monkeypatch)
    evidence = module._HandoffEvidence()  # noqa: SLF001 - focused predicate test

    evidence = module._observe_handoff(  # noqa: SLF001 - focused predicate test
        fleet=_fleet(
            cursor=7,
            handoff_latched=False,
            actor_evaluations=0,
            policy_generation=1,
            transmitted_policy_generation=1,
        ),
        evidence=evidence,
    )

    assert evidence.navigation_reached_policy
    assert evidence.entry_observation
    assert not module._actor_invoked(evidence=evidence)  # noqa: SLF001


def test_actor_result_must_be_transmitted_but_later_quality_is_irrelevant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_scenario(monkeypatch)
    evidence = module._HandoffEvidence()  # noqa: SLF001 - focused predicate test

    evidence = module._observe_handoff(  # noqa: SLF001 - focused predicate test
        fleet=_fleet(
            cursor=7,
            handoff_latched=True,
            actor_evaluations=1,
            policy_generation=2,
            transmitted_policy_generation=1,
        ),
        evidence=evidence,
    )
    assert evidence.handoff_latched
    assert evidence.actor_evaluations == 1
    assert not evidence.actor_setpoint_transmitted
    assert not module._actor_invoked(evidence=evidence)  # noqa: SLF001

    evidence = module._observe_handoff(  # noqa: SLF001 - focused predicate test
        fleet=_fleet(
            cursor=7,
            handoff_latched=True,
            actor_evaluations=1,
            policy_generation=2,
            transmitted_policy_generation=2,
            contact_failure=True,
        ),
        evidence=evidence,
    )

    assert evidence.actor_setpoint_transmitted
    assert module._actor_invoked(evidence=evidence)  # noqa: SLF001


@pytest.mark.parametrize("yaw", [0.0, 90.0, -90.0, 180.0])
def test_measurement_recovers_physical_contact_pose(monkeypatch, yaw):
    import numpy as np
    from scipy.spatial.transform import Rotation

    from robotics_experiments.config import load_config
    from robotics_experiments.rl import pack_pose_observation

    module = _load_scenario(monkeypatch)
    config = load_config(REPO_ROOT / "configs/policy_press_approach_quad.yaml")
    target_range = config.rl.press.goal.velocity_reference.range_center_m
    goal = Rotation.from_euler("z", yaw, degrees=True)
    relative = Rotation.from_euler("xyz", [0.01, -0.02, 0.03])
    body = goal * relative
    contact_position = np.array([0.49, 0.02, -0.01])
    vector = pack_pose_observation(
        goal_offset_enu_m=goal.apply(contact_position - [target_range, 0.0, 0.0]),
        quat_flu_enu_wxyz=body.as_quat()[[3, 0, 1, 2]],
        goal_quat_world_wxyz=goal.as_quat()[[3, 0, 1, 2]],
        linear_velocity_body_flu=np.zeros(3),
        angular_velocity_body_flu=np.zeros(3),
    )
    evidence = module._HandoffEvidence(  # noqa: SLF001 - measurement boundary test
        entry_observation=tuple(vector) + (0.0,) * 17
    )
    payload = module._measurement_payload(  # noqa: SLF001 - measurement boundary test
        context=SimpleNamespace(config=config), evidence=evidence
    )
    assert payload["observation_schema_version"] == 3
    np.testing.assert_allclose(payload["relative_switch_pos_m"], contact_position, atol=1e-12)
    np.testing.assert_allclose(
        payload["relative_switch_quat_xyzw"], relative.as_quat(canonical=True), atol=1e-12
    )

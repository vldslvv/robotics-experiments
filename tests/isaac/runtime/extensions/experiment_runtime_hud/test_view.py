"""Pure HUD presentation tests."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

from tests.support.paths import REPO_ROOT


def _load_view_module():
    path = (
        REPO_ROOT
        / "isaac/runtime/extensions/experiment_runtime_hud/ExperimentRuntimeHud_python/_view.py"
    )
    spec = importlib.util.spec_from_file_location("experiment_runtime_hud_view_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _vehicle(*, kind, phase="DRIVE", policy=None, press=None, navigation=None, motion=None):
    step = SimpleNamespace(
        index=5,
        total_steps=9,
        kind=kind,
        label=f"{kind}[test]",
        elapsed_s=1.25,
        timeout_s=20.0,
        hold_elapsed_s=0.5,
        hold_required_s=2.0 if kind == "hover" else 0.0,
    )
    return SimpleNamespace(
        label="x500_01",
        sim_time_us=2_000_000,
        driver_phase=phase,
        routine_step=step,
        navigation=navigation,
        press=press,
        policy=policy,
        motion=motion,
        failure=None,
        recovery_outcome=None,
    )


def _policy(*, handoff_latched=True, accepted=True, generation=7, transmitted_generation=6):
    transmitted = SimpleNamespace(
        kind="VelocitySetpoint",
        values=(
            ("forward_mps", 0.03),
            ("right_mps", 0.01),
            ("down_mps", -0.02),
            ("yaw_rate_radps", -0.04),
        ),
    )
    return SimpleNamespace(
        failure_reason=None,
        handoff_latched=handoff_latched,
        complete=False,
        offboard_accepted=accepted,
        offboard_requested=True,
        actor_evaluations=7,
        stable_goal_steps=1,
        stable_goal_required=100,
        normalized_action=(0.06, -0.02, 0.04, 0.01),
        policy_generation=generation,
        actor_sim_time_us=2_000_000,
        requested_setpoint=transmitted,
        transmitted_policy_generation=transmitted_generation,
        transmitted_sim_time_us=1_980_000,
        transmitted_setpoint=transmitted,
    )


def _press():
    return SimpleNamespace(
        switch_range_m=0.48,
        lateral_alignment_m=0.032,
        detection_valid=True,
        pose_confidence=1.0,
        pose_age_s=0.0,
        contact_valid=True,
        contact=False,
        handoff_ready=True,
        handoff_reason="ready",
        handoff_range_min_m=0.44,
        handoff_range_max_m=0.52,
        handoff_alignment_max_m=0.08,
        goal_range_min_m=0.40,
        goal_range_max_m=0.42,
        goal_alignment_max_m=0.01,
    )


def test_navigation_and_policy_are_unambiguous_command_sources():
    module = _load_view_module()
    navigation = SimpleNamespace(
        target_enu=(0.0, 17.52, 2.5),
        remaining_distance_m=0.04,
        position_tolerance_m=0.06,
        ground_speed_mps=0.08,
    )
    nav = module._build_vehicle_view(  # noqa: SLF001 - pure view contract
        vehicle=_vehicle(kind="goto", navigation=navigation, press=_press()),
        transitions=("1.00 s NAVIGATION",),
        stale=False,
    )
    policy = module._build_vehicle_view(  # noqa: SLF001 - pure view contract
        vehicle=_vehicle(kind="policy", policy=_policy(), press=_press()),
        transitions=("2.00 s POLICY",),
        stale=False,
    )

    assert nav.status == "NAVIGATION - GOTO"
    assert nav.command_indicator == ""
    assert any("Waypoint error" in line for line in nav.lines)
    assert any("Handoff: READY" in line for line in nav.lines)
    assert policy.status == "POLICY - ACTIVE"
    assert "Body-FLU transmitted command" in policy.command_indicator
    assert any("APPROACH ONLY - CONTACT NOT EXPECTED" in line for line in policy.lines)
    assert any("Stable goal: 1/100" in line for line in policy.lines)
    assert any("ACTOR RESULT PENDING" in line for line in policy.lines)


def test_handoff_wait_and_stale_state_hide_the_command_indicator():
    module = _load_view_module()
    waiting = module._build_vehicle_view(  # noqa: SLF001 - pure view contract
        vehicle=_vehicle(
            kind="policy",
            policy=_policy(handoff_latched=False, accepted=False),
            press=_press(),
        ),
        transitions=("2.00 s HANDOFF",),
        stale=False,
    )
    stale = module._build_vehicle_view(  # noqa: SLF001 - pure view contract
        vehicle=_vehicle(kind="policy", policy=_policy(), press=_press()),
        transitions=("2.00 s POLICY", "3.00 s STALE"),
        stale=True,
    )

    assert waiting.status == "POLICY - HANDOFF WAIT"
    assert waiting.command_indicator == ""
    assert stale.status == "STALE - TIMELINE STOPPED"
    assert stale.command_indicator == ""

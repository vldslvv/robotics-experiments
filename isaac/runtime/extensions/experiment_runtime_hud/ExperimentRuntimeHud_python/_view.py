"""Pure formatting of immutable runtime snapshots for the diagnostic HUD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__: list[str] = []


@dataclass(frozen=True)
class _HudVehicleView:
    label: str
    semantic_key: str
    status: str
    tone: str
    transitions: tuple[str, ...]
    lines: tuple[str, ...]
    command_indicator: str


def _semantic(*, vehicle: Any, stale: bool) -> tuple[str, str, str]:
    if stale:
        return "stale", "STALE - TIMELINE STOPPED", "idle"
    phase = vehicle.driver_phase
    if phase == "RECOVERING":
        return "recovery", "RECOVERY - LANDING", "failure"
    if phase == "FAILED":
        return "failed", "FAILED", "failure"
    if phase == "SUCCEEDED":
        return "complete", "ROUTINE COMPLETE", "complete"
    step = vehicle.routine_step
    if phase != "DRIVE" or step is None:
        return f"setup:{phase}", f"SETUP - {phase}", "idle"
    if step.kind == "hover":
        return f"settle:{step.index}", "SETTLING - HOVER", "wait"
    if step.kind == "policy":
        policy = vehicle.policy
        if policy is None:
            return f"policy-enter:{step.index}", "POLICY - ENTERING", "policy"
        if policy.failure_reason is not None:
            return f"policy-failed:{step.index}", "POLICY - FAILED", "failure"
        if not policy.handoff_latched:
            return f"handoff:{step.index}", "POLICY - HANDOFF WAIT", "wait"
        if policy.complete:
            return f"policy-complete:{step.index}", "POLICY - GOAL REACHED", "complete"
        if not policy.offboard_accepted:
            return f"policy-prime:{step.index}", "POLICY - PRIMING OFFBOARD", "policy"
        return f"policy-active:{step.index}", "POLICY - ACTIVE", "policy"
    if step.kind == "arm":
        return f"arm:{step.index}", "SETUP - ARM", "idle"
    if step.kind == "land":
        return f"land:{step.index}", "NAVIGATION - LAND", "navigation"
    return f"navigation:{step.index}", f"NAVIGATION - {step.kind.upper()}", "navigation"


def _setpoint_text(*, prefix: str, setpoint: Any) -> str:
    if setpoint is None:
        return f"{prefix}: none"
    values = "  ".join(f"{name}={value:+.3f}" for name, value in setpoint.values)
    return f"{prefix}: {setpoint.kind}  {values}"


def _command_indicator(*, vehicle: Any, tone: str) -> str:
    policy = vehicle.policy
    if tone != "policy" or policy is None or policy.transmitted_setpoint is None:
        return ""
    setpoint = policy.transmitted_setpoint
    if setpoint.kind != "VelocitySetpoint":
        return ""
    values = dict(setpoint.values)
    forward = values["forward_mps"]
    left = -values["right_mps"]
    up = -values["down_mps"]
    yaw_ccw = -values["yaw_rate_radps"]
    return (
        "Body-FLU transmitted command\n"
        f"       FORWARD {forward:+.3f} m/s\n"
        f"LEFT {left:+.3f}   [DRONE]   RIGHT {-left:+.3f}\n"
        f"       UP {up:+.3f} m/s   YAW-CCW {yaw_ccw:+.3f} rad/s"
    )


def _routine_lines(*, vehicle: Any) -> list[str]:
    step = vehicle.routine_step
    if step is None:
        return [f"Driver phase: {vehicle.driver_phase}"]
    lines = [
        f"Routine: {step.index + 1}/{step.total_steps}  {step.label}",
        f"Step time: {step.elapsed_s:.2f}/{step.timeout_s:.2f} s",
    ]
    if step.hold_required_s > 0.0:
        lines.append(f"Stable hold: {step.hold_elapsed_s:.2f}/{step.hold_required_s:.2f} s")
    return lines


def _navigation_lines(*, vehicle: Any) -> list[str]:
    navigation = vehicle.navigation
    if navigation is None:
        return []
    speed = (
        "PX4 default"
        if math.isnan(navigation.ground_speed_mps)
        else f"{navigation.ground_speed_mps:.3f} m/s"
    )
    return [
        "Waypoint ENU: " + ", ".join(f"{value:.3f}" for value in navigation.target_enu),
        "Waypoint error: "
        f"{navigation.remaining_distance_m:.3f} m / <= {navigation.position_tolerance_m:.3f} m",
        f"Authored ground speed: {speed}",
    ]


def _press_lines(*, vehicle: Any) -> list[str]:
    press = vehicle.press
    if press is None:
        return []
    policy = vehicle.policy
    handoff_status = (
        "LATCHED"
        if policy is not None and policy.handoff_latched
        else ("READY" if press.handoff_ready else f"WAIT: {press.handoff_reason}")
    )
    in_goal_range = press.goal_range_min_m <= press.switch_range_m <= press.goal_range_max_m
    in_goal_alignment = press.lateral_alignment_m <= press.goal_alignment_max_m
    lines = [
        "Press task: APPROACH ONLY - CONTACT NOT EXPECTED",
        f"Handoff: {handoff_status}",
        "Handoff range: "
        f"{press.switch_range_m:.4f} m / "
        f"[{press.handoff_range_min_m:.3f}, {press.handoff_range_max_m:.3f}] m",
        "Handoff alignment: "
        f"{press.lateral_alignment_m:.4f} m / <= {press.handoff_alignment_max_m:.3f} m",
        "Approach goal range: "
        f"{press.switch_range_m:.4f} m / "
        f"[{press.goal_range_min_m:.3f}, {press.goal_range_max_m:.3f}] m "
        f"({'IN' if in_goal_range else 'OUT'})",
        "Approach goal alignment: "
        f"{press.lateral_alignment_m:.4f} m / <= {press.goal_alignment_max_m:.3f} m "
        f"({'IN' if in_goal_alignment else 'OUT'})",
        "Perception: "
        f"valid={press.detection_valid} confidence={press.pose_confidence:.2f} "
        f"age={press.pose_age_s:.3f} s",
        f"Contact: valid={press.contact_valid} active={press.contact}",
    ]
    if policy is not None:
        lines.extend(
            (
                f"Actor evaluations: {policy.actor_evaluations}",
                f"Stable goal: {policy.stable_goal_steps}/{policy.stable_goal_required}",
                "Actor normalized FLU action: "
                + (
                    "none"
                    if not policy.normalized_action
                    else " ".join(f"{value:+.3f}" for value in policy.normalized_action)
                ),
                f"Actor generation: {policy.policy_generation} at {policy.actor_sim_time_us} us",
                _setpoint_text(prefix="Requested", setpoint=policy.requested_setpoint),
                "Transmission: "
                f"generation {policy.transmitted_policy_generation} at "
                f"{policy.transmitted_sim_time_us} us"
                + (
                    " (ACTOR RESULT PENDING)"
                    if policy.policy_generation > policy.transmitted_policy_generation
                    else ""
                ),
                _setpoint_text(prefix="Transmitted", setpoint=policy.transmitted_setpoint),
            )
        )
    return lines


def _motion_lines(*, vehicle: Any) -> list[str]:
    motion = vehicle.motion
    if motion is None:
        return []
    forward, left, up = motion.linear_velocity_flu_mps
    angular_speed = math.sqrt(sum(value * value for value in motion.angular_velocity_flu_radps))
    return [
        f"Actual body-FLU velocity: F={forward:+.3f} L={left:+.3f} U={up:+.3f} m/s",
        f"Actual angular speed: {angular_speed:.3f} rad/s",
    ]


def _build_vehicle_view(
    *,
    vehicle: Any,
    transitions: tuple[str, ...],
    stale: bool,
) -> _HudVehicleView:
    semantic_key, status, tone = _semantic(vehicle=vehicle, stale=stale)
    lines = _routine_lines(vehicle=vehicle)
    lines.extend(_navigation_lines(vehicle=vehicle))
    lines.extend(_press_lines(vehicle=vehicle))
    lines.extend(_motion_lines(vehicle=vehicle))
    if vehicle.failure is not None:
        lines.append(f"Failure: {vehicle.failure}")
    if vehicle.recovery_outcome is not None:
        lines.append(f"Recovery outcome: {vehicle.recovery_outcome}")
    return _HudVehicleView(
        label=vehicle.label,
        semantic_key=semantic_key,
        status=status,
        tone=tone,
        transitions=transitions,
        lines=tuple(lines),
        command_indicator=_command_indicator(vehicle=vehicle, tone=tone),
    )

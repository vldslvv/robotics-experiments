"""Pure projection of live routine state into immutable operator telemetry."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Optional, cast

import numpy as np

from robotics_experiments.commanding import StreamSetpoint
from robotics_experiments.px4_bridge import BridgeStatus, body_velocity_flu_from_state
from robotics_experiments.rl.press import (
    HandoffGate,
    PressApproachExecutionState,
    PressObservation,
    handoff_ready,
)

__all__ = [
    "MotionRuntimeSnapshot",
    "NavigationRuntimeSnapshot",
    "PolicyRuntimeSnapshot",
    "PressRuntimeSnapshot",
    "RoutineStepRuntimeSnapshot",
    "SetpointRuntimeSnapshot",
    "VehicleRuntimeSnapshot",
]


@dataclass(frozen=True)
class MotionRuntimeSnapshot:
    """Vehicle kinematics already consumed by the routine driver."""

    position_enu: tuple[float, float, float]
    linear_velocity_flu_mps: tuple[float, float, float]
    angular_velocity_flu_radps: tuple[float, float, float]


@dataclass(frozen=True)
class RoutineStepRuntimeSnapshot:
    """Current authored routine step and its timing progress."""

    index: int
    total_steps: int
    kind: str
    label: str
    elapsed_s: float
    timeout_s: float
    hold_elapsed_s: float
    hold_required_s: float


@dataclass(frozen=True)
class NavigationRuntimeSnapshot:
    """World-waypoint progress for the active navigation step."""

    target_enu: tuple[float, float, float]
    remaining_distance_m: float
    position_tolerance_m: float
    ground_speed_mps: float


@dataclass(frozen=True)
class PressRuntimeSnapshot:
    """Current switch-relative measurements and configured entry/goal bounds."""

    switch_range_m: float
    lateral_alignment_m: float
    detection_valid: bool
    pose_confidence: float
    pose_age_s: float
    contact_valid: bool
    contact: bool
    handoff_ready: bool
    handoff_reason: str
    handoff_range_min_m: float
    handoff_range_max_m: float
    handoff_alignment_max_m: float
    goal_range_min_m: float
    goal_range_max_m: float
    goal_alignment_max_m: float


@dataclass(frozen=True)
class SetpointRuntimeSnapshot:
    """One immutable OFFBOARD payload represented without task-specific branches."""

    kind: str
    values: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PolicyRuntimeSnapshot:
    """Current policy computation and the most recent predicted wire effect."""

    offboard_requested: bool
    offboard_accepted: bool
    policy_generation: int
    actor_sim_time_us: int
    transmitted_policy_generation: int
    transmitted_sim_time_us: int
    actor_evaluations: int
    handoff_latched: bool
    stable_goal_steps: int
    stable_goal_required: int
    normalized_action: tuple[float, ...]
    requested_setpoint: Optional[SetpointRuntimeSnapshot]
    transmitted_setpoint: Optional[SetpointRuntimeSnapshot]
    complete: bool
    failure_reason: Optional[str]


@dataclass(frozen=True)
class VehicleRuntimeSnapshot:
    """Immutable progress, motion, and policy telemetry for one vehicle."""

    label: str
    sim_time_us: int
    bridge_status: Optional[BridgeStatus]
    completed_steps: int
    driver_phase: str
    failure: Optional[str]
    recovery_outcome: Optional[str]
    motion: Optional[MotionRuntimeSnapshot]
    routine_step: Optional[RoutineStepRuntimeSnapshot]
    navigation: Optional[NavigationRuntimeSnapshot]
    press: Optional[PressRuntimeSnapshot]
    policy: Optional[PolicyRuntimeSnapshot]


def _float_tuple(*, values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _setpoint_snapshot(*, setpoint: Optional[StreamSetpoint]) -> Optional[SetpointRuntimeSnapshot]:
    if setpoint is None:
        return None
    return SetpointRuntimeSnapshot(
        kind=type(setpoint).__name__,
        values=tuple(
            (field.name, float(getattr(setpoint, field.name))) for field in fields(setpoint)
        ),
    )


def _motion_snapshot(*, observation: Any) -> MotionRuntimeSnapshot:
    vehicle = observation.vehicle_state
    return MotionRuntimeSnapshot(
        position_enu=cast(
            tuple[float, float, float],
            _float_tuple(values=vehicle.position_enu),
        ),
        linear_velocity_flu_mps=cast(
            tuple[float, float, float],
            _float_tuple(values=body_velocity_flu_from_state(vehicle)),
        ),
        angular_velocity_flu_radps=cast(
            tuple[float, float, float],
            _float_tuple(values=vehicle.angular_velocity_flu),
        ),
    )


def _step_snapshot(*, flying: Any, sim_time_us: int) -> Optional[RoutineStepRuntimeSnapshot]:
    driver = flying.driver
    routine = driver.routine
    cursor = int(driver.routine_state.cursor)
    if routine is None or cursor >= len(routine.steps):
        return None
    step = routine.steps[cursor]
    started_us = driver.routine_state.step_started_us
    elapsed_s = 0.0 if started_us is None else max(0.0, (sim_time_us - started_us) * 1e-6)
    hold_since_us = driver.routine_state.hold_since_us
    hold_elapsed_s = (
        0.0 if hold_since_us is None else max(0.0, (sim_time_us - hold_since_us) * 1e-6)
    )
    descriptor = flying.spawned.vehicle.routine.steps[cursor]
    return RoutineStepRuntimeSnapshot(
        index=cursor,
        total_steps=len(routine.steps),
        kind=str(descriptor["kind"]),
        label=step.label,
        elapsed_s=elapsed_s,
        timeout_s=float(step.timeout_s),
        hold_elapsed_s=hold_elapsed_s,
        hold_required_s=float(step.hold_s),
    )


def _navigation_snapshot(
    *,
    flying: Any,
    motion: Optional[MotionRuntimeSnapshot],
    step: Optional[RoutineStepRuntimeSnapshot],
) -> Optional[NavigationRuntimeSnapshot]:
    if motion is None or step is None or step.kind != "goto":
        return None
    descriptor = flying.spawned.vehicle.routine.steps[step.index]
    target = (float(descriptor["x"]), float(descriptor["y"]), float(descriptor["z"]))
    return NavigationRuntimeSnapshot(
        target_enu=target,
        remaining_distance_m=math.dist(motion.position_enu, target),
        position_tolerance_m=float(
            descriptor.get(
                "position_tol_m",
                flying.spawned.vehicle.routine.goto_position_tol_m,
            )
        ),
        ground_speed_mps=float(descriptor.get("ground_speed_m_s", math.nan)),
    )


def _press_snapshot(*, config: Any, observation: Any) -> Optional[PressRuntimeSnapshot]:
    task_observation = observation.task_observation
    if not isinstance(task_observation, PressObservation):
        return None
    handoff_config = config.rl.press.handoff
    goal = config.rl.press.goal
    gate = handoff_ready(task_observation, HandoffGate.from_config(handoff_config))
    relative_position = np.asarray(task_observation.relative_switch_pos_m, dtype=np.float64)
    return PressRuntimeSnapshot(
        switch_range_m=float(relative_position[0]),
        lateral_alignment_m=float(np.linalg.norm(relative_position[1:3])),
        detection_valid=bool(task_observation.detection_valid),
        pose_confidence=float(task_observation.pose_confidence),
        pose_age_s=float(task_observation.pose_age_s),
        contact_valid=bool(task_observation.contact_valid),
        contact=bool(task_observation.contact_flag),
        handoff_ready=bool(gate.ready),
        handoff_reason="ready" if gate.reason is None else gate.reason.value,
        handoff_range_min_m=float(handoff_config.switch_range_min_m),
        handoff_range_max_m=float(handoff_config.switch_range_max_m),
        handoff_alignment_max_m=float(handoff_config.lateral_align_max_m),
        goal_range_min_m=float(goal.range_min_m),
        goal_range_max_m=float(goal.range_max_m),
        goal_alignment_max_m=float(goal.alignment_error_max_m),
    )


def _policy_snapshot(
    *,
    config: Any,
    flying: Any,
    step: Optional[RoutineStepRuntimeSnapshot],
) -> Optional[PolicyRuntimeSnapshot]:
    if step is None or step.kind != "policy" or not flying.driver.routine_state.sent:
        return None
    offboard = flying.driver.routine_state.offboard
    policy_state = offboard.policy_state
    if isinstance(policy_state, PressApproachExecutionState):
        actor_evaluations = int(policy_state.actor_evaluations)
        handoff_latched = bool(policy_state.handoff_latched)
        stable_goal_steps = int(policy_state.stable_goal_steps)
        stable_goal_required = int(config.rl.press.goal.stable_steps)
        normalized_action = tuple(float(value) for value in policy_state.previous_action)
    else:
        actor_evaluations = int(offboard.policy_generation)
        handoff_latched = True
        stable_goal_steps = 0
        stable_goal_required = 0
        normalized_action = ()
    actor_time_ns = offboard.policy_cadence.previous_time_ns
    return PolicyRuntimeSnapshot(
        offboard_requested=bool(offboard.requested),
        offboard_accepted=bool(offboard.accepted),
        policy_generation=int(offboard.policy_generation),
        actor_sim_time_us=-1 if actor_time_ns is None else int(actor_time_ns // 1_000),
        transmitted_policy_generation=int(offboard.transmitted_policy_generation),
        transmitted_sim_time_us=int(offboard.transmitted_setpoint_sim_time_us),
        actor_evaluations=actor_evaluations,
        handoff_latched=handoff_latched,
        stable_goal_steps=stable_goal_steps,
        stable_goal_required=stable_goal_required,
        normalized_action=normalized_action,
        requested_setpoint=_setpoint_snapshot(setpoint=offboard.cached_setpoint),
        transmitted_setpoint=_setpoint_snapshot(setpoint=offboard.transmitted_setpoint),
        complete=bool(offboard.policy_complete),
        failure_reason=offboard.policy_failure_reason,
    )


def _vehicle_runtime_snapshot(
    *,
    config: Any,
    flying: Any,
    host_status: Any,
) -> VehicleRuntimeSnapshot:
    """Project existing runtime values without advancing or sampling resources."""
    environment_status = flying.context.env.read_status()
    observation = environment_status.observation
    sim_time_us = int(host_status.sim_time_us)
    if observation is not None:
        sim_time_us = int(observation.sim_time_us)
    motion = None if observation is None else _motion_snapshot(observation=observation)
    step = _step_snapshot(flying=flying, sim_time_us=sim_time_us)
    navigation = _navigation_snapshot(flying=flying, motion=motion, step=step)
    press = None if observation is None else _press_snapshot(config=config, observation=observation)
    policy = _policy_snapshot(config=config, flying=flying, step=step)
    failure = flying.driver.failure
    return VehicleRuntimeSnapshot(
        label=flying.spawned.label,
        sim_time_us=sim_time_us,
        bridge_status=host_status.bridge_status,
        completed_steps=flying.driver.previous_cursor,
        driver_phase=flying.driver.phase.name,
        failure=None if failure is None else failure.message,
        recovery_outcome=None if failure is None else failure.recovery_outcome.name,
        motion=motion,
        routine_step=step,
        navigation=navigation,
        press=press,
        policy=policy,
    )

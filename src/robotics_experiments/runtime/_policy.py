"""Host-safe policy preflight and runtime adapter composition."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from robotics_experiments.commanding import PolicyAdapter, PolicyInput, PolicyResult
from robotics_experiments.config import ExperimentConfig
from robotics_experiments.rl import (
    LoadedPolicy,
    PolicyInterface,
    PolicyMetadata,
    hover_observation_from_state,
    load_policy,
    policy_action_to_offboard,
    validate_policy_artifact,
)
from robotics_experiments.rl.hover import load_hover_reset_bounds
from robotics_experiments.rl.press import (
    HandoffGate,
    PressApproachExecutionParams,
    PressApproachExecutionState,
    PressObservation,
    advance_press_approach_execution,
    initial_press_approach_execution_state,
)

__all__: list[str] = []


@dataclass(frozen=True)
class _ConfiguredPolicy:
    vehicle_label: str
    switch_index: int
    artifact: Path
    metadata: PolicyMetadata


@dataclass(frozen=True)
class _PolicyPreflight:
    policies: tuple[_ConfiguredPolicy, ...]
    press_rate_hz: Optional[float]


@dataclass(frozen=True)
class _HoverPolicyExecutionState:
    started_us: int = -1


def _structured_policy_diagnostic(*, payload: dict[str, object]) -> str:
    return f"PRESS_POLICY {json.dumps(payload, separators=(',', ':'), sort_keys=True)}"


def _artifact_path(*, artifact: str, repo_root: Path) -> Path:
    path = Path(artifact)
    return path if path.is_absolute() else repo_root / path


def _preflight_policy_artifacts(
    *,
    config: ExperimentConfig,
    repo_root: Path,
) -> _PolicyPreflight:
    """Validate every configured artifact before simulator mutation."""
    policies = []
    press_rates: set[float] = set()
    for label, vehicle in config.vehicles.items():
        for descriptor in vehicle.routine.steps:
            if descriptor.get("kind") != "policy":
                continue
            artifact = _artifact_path(
                artifact=str(descriptor["artifact"]),
                repo_root=repo_root,
            )
            metadata = validate_policy_artifact(artifact_dir=artifact)
            contract = metadata.contract
            if contract.policy_rate_hz > config.driver.control_rate_hz:
                raise ValueError(
                    f"policy artifact {artifact} cadence {contract.policy_rate_hz} Hz "
                    f"exceeds driver.control_rate_hz={config.driver.control_rate_hz}"
                )
            if contract.policy_interface is PolicyInterface.PRESS:
                if contract.curriculum_stage != config.rl.press.curriculum.active_stage:
                    raise ValueError(
                        f"press artifact {artifact} stage "
                        f"{contract.curriculum_stage!r} does not match configured "
                        f"stage {config.rl.press.curriculum.active_stage!r}"
                    )
                press_rates.add(contract.policy_rate_hz)
                if contract.curriculum_stage == "hover":
                    load_hover_reset_bounds(
                        path=_artifact_path(
                            artifact=config.rl.inputs.flight_parameters_artifact,
                            repo_root=repo_root,
                        ).resolve(strict=True),
                        run_id=metadata.flight_parameters.run_id,
                        digest=metadata.flight_parameters_digest,
                    )
            policies.append(
                _ConfiguredPolicy(
                    vehicle_label=label,
                    switch_index=int(descriptor.get("switch_index", -1)),
                    artifact=artifact,
                    metadata=metadata,
                )
            )
    if len(press_rates) > 1:
        raise ValueError(
            "press policy artifacts in one fleet must share one policy_rate_hz; "
            f"found {sorted(press_rates)}"
        )
    return _PolicyPreflight(
        policies=tuple(policies),
        press_rate_hz=next(iter(press_rates), None),
    )


def _hover_adapter(
    *,
    hold_s: Optional[float],
    loaded: LoadedPolicy,
) -> PolicyAdapter:
    if hold_s is None:
        raise ValueError("hover policy step requires 'hold_s'")
    contract = loaded.contract

    def advance(*, policy_input: PolicyInput, state: object) -> PolicyResult:
        assert isinstance(state, _HoverPolicyExecutionState)
        started_us = policy_input.sim_time_us if state.started_us < 0 else state.started_us
        successor = _HoverPolicyExecutionState(started_us=started_us)
        action = loaded.inference(
            hover_observation_from_state(
                state=policy_input.vehicle_state,
                target_enu=np.asarray(policy_input.reference_position_enu, dtype=np.float64),
            )
        )
        return PolicyResult(
            state=successor,
            setpoint=policy_action_to_offboard(
                np.clip(action, -1.0, 1.0),
                contract,
            ),
            complete=policy_input.sim_time_us - started_us >= hold_s * 1_000_000,
        )

    return PolicyAdapter(
        policy_rate_hz=contract.policy_rate_hz,
        initial_state=_HoverPolicyExecutionState(),
        advance=advance,
    )


def _press_approach_adapter(
    *,
    config: ExperimentConfig,
    loaded: LoadedPolicy,
    repo_root: Path,
) -> PolicyAdapter:
    contract = loaded.contract
    if contract.curriculum_stage not in {"hover", "approach"}:
        raise ValueError("unsupported press curriculum_stage=" f"{contract.curriculum_stage!r}")
    handoff_gate = HandoffGate.from_config(config.rl.press.handoff)
    if contract.curriculum_stage == "hover":
        handoff_gate = replace(
            handoff_gate,
            hover_reset_bounds=load_hover_reset_bounds(
                path=_artifact_path(
                    artifact=config.rl.inputs.flight_parameters_artifact,
                    repo_root=repo_root,
                ).resolve(strict=True),
                run_id=loaded.metadata.flight_parameters.run_id,
                digest=loaded.metadata.flight_parameters_digest,
            ),
        )
    feedback = config.rl.press.handoff
    if (
        not all(
            np.isfinite(value) and value > 0.0
            for value in (
                feedback.feedback_position_gain_per_s,
                feedback.feedback_yaw_gain_per_s,
                feedback.feedback_max_velocity_mps,
            )
        )
        or not np.isfinite(feedback.feedback_velocity_damping)
        or feedback.feedback_velocity_damping < 0.0
    ):
        raise ValueError(
            "handoff feedback requires positive gains/velocity limit and nonnegative damping"
        )
    params = PressApproachExecutionParams(
        inference=loaded.inference,
        action_dim=contract.action_dim,
        handoff_gate=handoff_gate,
        goal=config.rl.press.goal,
        flight_parameters=loaded.metadata.flight_parameters,
        handoff_feedback=feedback,
        max_body_velocity_mps=contract.max_body_velocity_mps,
        max_yaw_rate_radps=contract.max_yaw_rate_radps,
    )

    def advance(*, policy_input: PolicyInput, state: object) -> PolicyResult:
        assert isinstance(state, PressApproachExecutionState)
        task_observation = policy_input.task_observation
        if not isinstance(task_observation, PressObservation):
            failure_reason = (
                "press policy requires a PressObservation task observation, "
                f"got {type(task_observation).__name__}"
            )
            return PolicyResult(
                state=state,
                setpoint=policy_action_to_offboard(
                    np.zeros(contract.action_dim),
                    contract,
                ),
                failure_reason=failure_reason,
                diagnostics=(
                    _structured_policy_diagnostic(
                        payload={
                            "event": "failure",
                            "failure": failure_reason,
                            "sim_time_us": policy_input.sim_time_us,
                        }
                    ),
                ),
            )
        result = advance_press_approach_execution(
            params=params,
            state=state,
            observation=task_observation,
        )
        setpoint = policy_action_to_offboard(result.action, contract)
        payload: dict[str, object] = {
            "actor_entry": (not state.handoff_latched and result.state.handoff_latched),
            "complete": result.complete,
            "contact": task_observation.contact_flag,
            "event": ("actor_evaluation" if result.actor_observation else "handoff_wait"),
            "failure": result.failure_reason,
            "handoff_reason": result.handoff_reason,
            "normalized_action": [float(value) for value in result.action],
            "setpoint": {
                "kind": type(setpoint).__name__,
                **asdict(setpoint),
            },
            "sim_time_us": policy_input.sim_time_us,
            "stable_goal_count": result.state.stable_goal_steps,
        }
        if result.actor_observation:
            payload["observation"] = list(result.actor_observation)
            payload["raw_normalized_action"] = list(result.raw_action)
        return PolicyResult(
            state=result.state,
            setpoint=setpoint,
            complete=result.complete,
            failure_reason=result.failure_reason,
            diagnostics=(_structured_policy_diagnostic(payload=payload),),
        )

    return PolicyAdapter(
        policy_rate_hz=contract.policy_rate_hz,
        initial_state=initial_press_approach_execution_state(action_dim=contract.action_dim),
        advance=advance,
    )


def _make_policy_adapter(
    *,
    config: ExperimentConfig,
    artifact: str,
    hold_s: Optional[float],
    repo_root: Path,
) -> PolicyAdapter:
    """Load one artifact and compose its interface-selected routine adapter."""
    artifact_path = _artifact_path(artifact=artifact, repo_root=repo_root)
    loaded = load_policy(artifact_dir=artifact_path)
    interface = loaded.contract.policy_interface
    if interface is PolicyInterface.HOVER:
        return _hover_adapter(hold_s=hold_s, loaded=loaded)
    if interface is PolicyInterface.PRESS:
        return _press_approach_adapter(config=config, loaded=loaded, repo_root=repo_root)
    raise ValueError(
        f"unsupported policy_interface={interface.value!r} in artifact {artifact_path}"
    )

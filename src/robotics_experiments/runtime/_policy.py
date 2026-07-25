"""Host-safe policy preflight and runtime adapter composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from robotics_experiments.commanding import PolicyAdapter, PolicyStepResult
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

__all__: list[str] = []


@dataclass(frozen=True)
class _ConfiguredPolicy:
    vehicle_label: str
    descriptor: Mapping[str, Any]
    artifact: Path
    metadata: PolicyMetadata


@dataclass(frozen=True)
class _PolicyPreflight:
    policies: tuple[_ConfiguredPolicy, ...]
    press_rate_hz: Optional[float]


@dataclass(frozen=True)
class _HoverPolicyProgress:
    started_us: Optional[int] = None


def _artifact_path(*, descriptor: Mapping[str, Any], repo_root: Path) -> Path:
    artifact = Path(str(descriptor["artifact"]))
    return artifact if artifact.is_absolute() else repo_root / artifact


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
            artifact = _artifact_path(descriptor=descriptor, repo_root=repo_root)
            metadata = validate_policy_artifact(artifact_dir=artifact)
            contract = metadata.contract
            if contract.policy_rate_hz > config.driver.control_rate_hz:
                raise ValueError(
                    f"policy artifact {artifact} cadence {contract.policy_rate_hz} Hz "
                    f"exceeds driver.control_rate_hz={config.driver.control_rate_hz}"
                )
            if contract.policy_interface is PolicyInterface.PRESS_V1:
                press_rates.add(contract.policy_rate_hz)
            policies.append(
                _ConfiguredPolicy(
                    vehicle_label=label,
                    descriptor=descriptor,
                    artifact=artifact,
                    metadata=metadata,
                )
            )
    if len(press_rates) > 1:
        raise ValueError(
            "press-v1 policy artifacts in one fleet must share one policy_rate_hz; "
            f"found {sorted(press_rates)}"
        )
    return _PolicyPreflight(
        policies=tuple(policies),
        press_rate_hz=next(iter(press_rates), None),
    )


def _hover_adapter(
    *,
    descriptor: Mapping[str, Any],
    loaded: LoadedPolicy,
) -> PolicyAdapter:
    if "hold_s" not in descriptor:
        raise ValueError("hover-v1 policy step requires 'hold_s'")
    hold_s = float(descriptor["hold_s"])
    contract = loaded.contract

    def step(observation, target_enu, progress):
        assert isinstance(progress, _HoverPolicyProgress)
        if target_enu is None:
            return PolicyStepResult(
                progress=progress,
                setpoint=policy_action_to_offboard(
                    np.zeros(contract.action_dim),
                    contract,
                ),
                failure_reason="hover-v1 policy has no target or entry baseline",
            )
        started_us = observation.sim_time_us if progress.started_us is None else progress.started_us
        successor = _HoverPolicyProgress(started_us=started_us)
        action = loaded.inference(
            hover_observation_from_state(
                observation.vehicle_state,
                target_enu,
            )
        )
        return PolicyStepResult(
            progress=successor,
            setpoint=policy_action_to_offboard(
                np.clip(action, -1.0, 1.0),
                contract,
            ),
            complete=observation.sim_time_us - started_us >= hold_s * 1_000_000,
        )

    return PolicyAdapter(
        rate_hz=contract.policy_rate_hz,
        initial_progress=_HoverPolicyProgress(),
        step=step,
    )


def _press_adapter(*, config: ExperimentConfig, loaded: LoadedPolicy) -> PolicyAdapter:
    from robotics_experiments.rl.press import (
        HandoffGate,
        PressPolicyParams,
        PressPolicyProgress,
        initial_press_policy_progress,
        step_press_policy,
    )

    contract = loaded.contract
    termination = config.rl.press.termination
    params = PressPolicyParams(
        policy=loaded.inference,
        action_dim=contract.action_dim,
        handoff_gate=HandoffGate.from_config(config.rl.press.handoff),
        first_contact_speed_max_mps=termination.first_contact_speed_max_mps,
        first_contact_angular_speed_max_radps=(termination.first_contact_angular_speed_max_radps),
        retreat_distance_m=termination.retreat_distance_m,
        retreat_speed_max_mps=termination.retreat_speed_max_mps,
        retreat_alignment_tolerance_m=termination.retreat_alignment_tolerance_m,
        retreat_angular_speed_max_radps=termination.retreat_angular_speed_max_radps,
        stable_retreat_steps=termination.stable_retreat_steps,
    )

    def step(observation, target_enu, progress):
        del target_enu
        assert isinstance(progress, PressPolicyProgress)
        task_observation = observation.policy_observation
        if task_observation is None:
            return PolicyStepResult(
                progress=progress,
                setpoint=policy_action_to_offboard(
                    np.zeros(contract.action_dim),
                    contract,
                ),
                failure_reason="press-v1 policy has no runtime task observation",
            )
        decision = step_press_policy(
            params=params,
            progress=progress,
            observation=task_observation,
        )
        return PolicyStepResult(
            progress=decision.progress,
            setpoint=policy_action_to_offboard(decision.action, contract),
            complete=decision.complete,
            failure_reason=decision.failure_reason,
        )

    return PolicyAdapter(
        rate_hz=contract.policy_rate_hz,
        initial_progress=initial_press_policy_progress(action_dim=contract.action_dim),
        step=step,
    )


def _make_policy_adapter(
    *,
    config: ExperimentConfig,
    descriptor: Mapping[str, Any],
    repo_root: Path,
) -> PolicyAdapter:
    """Load one artifact and compose its interface-selected routine adapter."""
    artifact = _artifact_path(descriptor=descriptor, repo_root=repo_root)
    loaded = load_policy(artifact_dir=artifact)
    interface = loaded.contract.policy_interface
    if interface is PolicyInterface.HOVER_V1:
        return _hover_adapter(descriptor=descriptor, loaded=loaded)
    if interface is PolicyInterface.PRESS_V1:
        return _press_adapter(config=config, loaded=loaded)
    raise ValueError(f"unsupported policy_interface={interface.value!r} in artifact {artifact}")

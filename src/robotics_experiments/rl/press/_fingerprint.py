"""Stable press-policy comparison identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from robotics_experiments.config import ExperimentConfig
from robotics_experiments.rl.policy import PolicyContract, PolicyInterface

__all__: list[str] = []


def _press_comparison_fingerprint(
    *,
    config: ExperimentConfig,
    contract: PolicyContract,
    flight_parameters_fingerprint: str,
) -> str | None:
    if contract.policy_interface is not PolicyInterface.PRESS:
        return None
    vehicle_label, vehicle = next(
        (label, item)
        for label, item in config.vehicles.items()
        if item.id == config.rl.training.vehicle_id
    )
    payload = {
        "comparison_revision": 36,
        "acceptance": asdict(config.rl.press.acceptance),
        "evaluation": {
            name: value
            for name, value in asdict(config.rl.press.evaluation).items()
            if not name.startswith("feasibility_")
        },
        "goal": asdict(config.rl.press.goal),
        "handoff": {
            name: value
            for name, value in asdict(config.rl.press.handoff).items()
            if not name.startswith("feedback_")
        },
        "reset": asdict(config.rl.press.reset),
        "termination": asdict(config.rl.press.termination),
        "scene": config.scene,
        "switches": asdict(config.switches),
        "vehicle": vehicle_label,
        "airframe": vehicle.spawn.usd_path,
        "lab_vehicle": asdict(config.rl.lab_vehicle),
        "physics_hz": config.sim.physics_hz,
        "episode_length_s": config.rl.press.selected_stage().episode_length_s,
        "task_id": config.rl.training.task_id,
        "policy_interface": contract.policy_interface.value,
        "curriculum_stage": contract.curriculum_stage,
        "observation_schema_version": contract.obs_schema_version,
        "observation_dimension": contract.obs_dim,
        "policy_rate_hz": contract.policy_rate_hz,
        "action_kind": contract.action_kind,
        "action_dimension": contract.action_dim,
        "flight_parameters_fingerprint": flight_parameters_fingerprint,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

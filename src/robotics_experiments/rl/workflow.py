"""Inputs and procedure shared by policy-training workflows.

The executable workflow is intentionally procedural::

    inputs = prepare_policy_inputs(...)
    criteria = create_press_acceptance_criteria(...)
    candidates = train_and_validate(inputs=inputs, ...)
    selected = select_candidate(candidates=candidates, ...)
    evaluation = run_final_test(inputs=inputs, policy=selected, ...)
    acceptance = evaluate_press_acceptance(criteria=criteria, evaluation=evaluation)
    if acceptance.passed:
        publish_policy(policy=selected, evaluation=evaluation, ...)

Validation only determines which checkpoint may enter the final test. The
fresh final test is the only evaluation used for acceptance, and only the exact
tested actor may be published. Isaac-specific code implements the operations
after preparation in the same order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from robotics_experiments.config import ExperimentConfig
from robotics_experiments.controller import (
    ControllerArtifact,
    load_controller_artifact,
    resolve_controller_path,
)
from robotics_experiments.flight_parameters import FlightParameters, load_flight_parameters_bundle

__all__ = ["PolicyInputs", "prepare_policy_inputs"]


@dataclass(frozen=True)
class PolicyInputs:
    """The two validated, pinned producer artifacts used by one campaign."""

    controller: ControllerArtifact
    controller_path: Path
    controller_digest: str
    flight_parameters: FlightParameters
    flight_parameters_path: Path
    flight_parameters_digest: str


def _file_sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _training_vehicle(config: ExperimentConfig, /):
    vehicle_id = config.rl.training.vehicle_id
    try:
        return next(
            (label, vehicle)
            for label, vehicle in config.vehicles.items()
            if vehicle.id == vehicle_id
        )
    except StopIteration as exc:
        raise ValueError(f"rl.training.vehicle_id={vehicle_id} is not in the experiment") from exc


def _resolve_path(*, authored: str | Path, repo_root: Path) -> Path:
    path = Path(authored)
    return path if path.is_absolute() else repo_root / path


def prepare_policy_inputs(
    *,
    config: ExperimentConfig,
    repo_root: Path,
    controller_artifact: str | Path | None = None,
    flight_parameters_artifact: str | Path | None = None,
) -> PolicyInputs:
    """Load, validate, match, and pin the two policy-training inputs."""

    controller_path = resolve_controller_path(
        _resolve_path(
            authored=controller_artifact or config.rl.inputs.controller_artifact,
            repo_root=repo_root,
        )
    ).resolve(strict=True)
    controller = load_controller_artifact(controller_path)
    vehicle_label, vehicle = _training_vehicle(config)
    action = config.rl.action
    controller_mismatches = []
    if controller.vehicle != vehicle_label:
        controller_mismatches.append(f"vehicle {controller.vehicle!r} != {vehicle_label!r}")
    if controller.airframe != Path(vehicle.spawn.usd_path).name:
        controller_mismatches.append(
            f"airframe {controller.airframe!r} != {Path(vehicle.spawn.usd_path).name!r}"
        )
    if controller.command_rate_hz != config.rl.policy_rate_hz:
        controller_mismatches.append(
            f"cadence {controller.command_rate_hz} != {config.rl.policy_rate_hz}"
        )
    if action.kind not in controller.certified_control_modes or action.action_dim != 4:
        controller_mismatches.append(
            f"action contract {(action.kind, action.action_dim)!r} is incompatible"
        )
    if controller_mismatches:
        raise ValueError("controller artifact is incompatible: " + "; ".join(controller_mismatches))
    if not controller.advisory_quality_passed:
        raise ValueError(
            "controller artifact failed its quality checks: "
            + "; ".join(controller.advisory_quality_reasons)
        )

    flight_parameters_path = _resolve_path(
        authored=flight_parameters_artifact or config.rl.inputs.flight_parameters_artifact,
        repo_root=repo_root,
    )
    bundle = load_flight_parameters_bundle(path=flight_parameters_path)
    flight_parameters = bundle.flight_parameters
    flight_mismatches = []
    if flight_parameters.vehicle != vehicle_label:
        flight_mismatches.append(f"vehicle {flight_parameters.vehicle!r} != {vehicle_label!r}")
    if flight_parameters.source_backend != "px4-sitl":
        flight_mismatches.append(
            f"source backend {flight_parameters.source_backend!r} != 'px4-sitl'"
        )
    if flight_parameters.run_id != controller.flight_parameters_run_id:
        flight_mismatches.append("run ID does not match the controller qualification")
    if flight_parameters.artifact_fingerprint != controller.flight_parameters_fingerprint:
        flight_mismatches.append("artifact fingerprint does not match the controller qualification")
    if flight_mismatches:
        raise ValueError("flight parameters are incompatible: " + "; ".join(flight_mismatches))

    return PolicyInputs(
        controller=controller,
        controller_path=controller_path,
        controller_digest=_file_sha256(controller_path),
        flight_parameters=flight_parameters,
        flight_parameters_path=bundle.run_dir,
        flight_parameters_digest=bundle.flight_parameters_digest,
    )

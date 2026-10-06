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
from robotics_experiments.flight_parameters import (
    FlightParameters,
    flight_parameters_from_payload,
    load_flight_selection,
)
from robotics_experiments.rl.hover import HoverResetSamples, load_hover_reset_samples
from robotics_experiments.switches import SwitchAsset, load_switch_asset

__all__ = ["PolicyInputs", "prepare_policy_inputs"]


@dataclass(frozen=True)
class PolicyInputs:
    """Validated, pinned controller, flight evidence and switch used by one campaign."""

    controller: ControllerArtifact
    controller_path: Path
    controller_digest: str
    flight_parameters: FlightParameters
    flight_parameters_path: Path
    flight_parameters_digest: str
    dataset_path: Path
    switch_asset: SwitchAsset
    hover_reset_samples: HoverResetSamples


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
    procedural_smoke: bool = False,
) -> PolicyInputs:
    """Load, validate, match, and pin the policy-training inputs."""

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
    if controller.command_rate_hz != config.rates.command_rate_hz:
        controller_mismatches.append(
            f"rate {controller.command_rate_hz} != {config.rates.command_rate_hz}"
        )
    if action.kind not in controller.certified_control_modes or action.action_dim != 4:
        controller_mismatches.append(
            f"action contract {(action.kind, action.action_dim)!r} is incompatible"
        )
    if controller_mismatches:
        raise ValueError("controller artifact is incompatible: " + "; ".join(controller_mismatches))
    if not procedural_smoke and not controller.advisory_quality_passed:
        raise ValueError(
            "controller artifact failed its quality checks: "
            + "; ".join(controller.advisory_quality_reasons)
        )

    flight_parameters_path = _resolve_path(
        authored=flight_parameters_artifact or config.rl.inputs.flight_parameters_artifact,
        repo_root=repo_root,
    )
    dataset_path = controller_path.parent / "dataset.json"
    dataset, selection = load_flight_selection(dataset_path)
    if flight_parameters_path.resolve(strict=True) not in (
        dataset.run_dir,
        dataset.run_dir / "flight_parameters.json",
    ):
        raise ValueError("capture source does not match the controller's dataset.json")
    if (
        dataset.definition["vehicle"] != vehicle_label
        or dataset.definition["source_backend"] != "px4-sitl"
    ):
        raise ValueError("capture vehicle or source backend is incompatible")
    parameters = flight_parameters_from_payload(selection["hover"])
    return PolicyInputs(
        switch_asset=load_switch_asset(model=config.switches.model, repo_root=repo_root),
        controller=controller,
        controller_path=controller_path,
        controller_digest=_file_sha256(controller_path),
        flight_parameters=parameters,
        flight_parameters_path=dataset.run_dir,
        flight_parameters_digest=dataset.artifact_digest,
        dataset_path=dataset_path,
        hover_reset_samples=load_hover_reset_samples(
            dataset=dataset,
            flight_indices=tuple(selection["train"]),
            flight_parameters=parameters,
        ),
    )

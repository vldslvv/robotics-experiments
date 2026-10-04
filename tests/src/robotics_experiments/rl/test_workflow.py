"""Tests for policy-workflow input preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.config import load_config
from robotics_experiments.flight_parameters import (
    load_flight_parameters,
    load_flight_parameters_bundle,
)
from robotics_experiments.rl import prepare_policy_inputs
from robotics_experiments.rl.workflow import _load_pinned_policy_inputs
from tests.support.paths import REPO_ROOT
from tests.support.switches import write_switch_asset


def _controller(*, config, flight_parameters, fingerprint: str | None = None):
    vehicle_label, vehicle = next(iter(config.vehicles.items()))
    return SimpleNamespace(
        vehicle=vehicle_label,
        airframe=Path(vehicle.spawn.usd_path).name,
        command_rate_hz=config.rates.command_rate_hz,
        certified_control_modes=(config.rl.action.kind,),
        advisory_quality_passed=True,
        advisory_quality_reasons=(),
        flight_parameters_run_id=flight_parameters.run_id,
        flight_parameters_fingerprint=(
            flight_parameters.artifact_fingerprint if fingerprint is None else fingerprint
        ),
    )


def test_prepare_policy_inputs_pins_and_matches_both_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robotics_experiments.rl.workflow as workflow

    config = load_config(REPO_ROOT / "configs/train/stage/press_hover_quad.yaml")
    flight_parameters_path = REPO_ROOT / "tests/fixtures/flight_parameters"
    flight_parameters = load_flight_parameters(flight_parameters_path)
    controller_path = tmp_path / "controller.json"
    controller_path.write_bytes(b"accepted controller")
    monkeypatch.setattr(workflow, "resolve_controller_path", lambda _path: controller_path)
    monkeypatch.setattr(
        workflow,
        "load_controller_artifact",
        lambda _path: _controller(config=config, flight_parameters=flight_parameters),
    )

    write_switch_asset(root=tmp_path, model=config.switches.model)
    inputs = prepare_policy_inputs(
        config=config,
        repo_root=tmp_path,
        controller_artifact=controller_path,
        flight_parameters_artifact=flight_parameters_path,
    )

    assert inputs.controller_path == controller_path
    assert inputs.controller_digest == hashlib.sha256(b"accepted controller").hexdigest()
    assert inputs.flight_parameters == flight_parameters
    assert inputs.flight_parameters_path == flight_parameters_path
    assert len(inputs.flight_parameters_digest) == 64


def test_prepare_policy_inputs_rejects_flight_parameters_not_used_by_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robotics_experiments.rl.workflow as workflow

    config = load_config(REPO_ROOT / "configs/train/stage/press_hover_quad.yaml")
    flight_parameters_path = REPO_ROOT / "tests/fixtures/flight_parameters"
    flight_parameters = load_flight_parameters(flight_parameters_path)
    controller_path = tmp_path / "controller.json"
    controller_path.write_bytes(b"accepted controller")
    monkeypatch.setattr(workflow, "resolve_controller_path", lambda _path: controller_path)
    monkeypatch.setattr(
        workflow,
        "load_controller_artifact",
        lambda _path: _controller(
            config=config,
            flight_parameters=flight_parameters,
            fingerprint="different-flight-parameters",
        ),
    )

    with pytest.raises(ValueError, match="artifact fingerprint does not match"):
        prepare_policy_inputs(
            config=config,
            repo_root=tmp_path,
            controller_artifact=controller_path,
            flight_parameters_artifact=flight_parameters_path,
        )


def test_load_pinned_policy_inputs_checks_identity_without_revalidating_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import robotics_experiments.rl.workflow as workflow

    controller_path = tmp_path / "controller.json"
    controller_path.write_bytes(b"pinned controller")
    controller = object()
    monkeypatch.setattr(workflow, "load_controller_artifact", lambda _path: controller)
    flight_parameters_path = REPO_ROOT / "tests/fixtures/flight_parameters"
    bundle = load_flight_parameters_bundle(path=flight_parameters_path)

    config = load_config(REPO_ROOT / "configs/train/stage/press_hover_quad.yaml")
    asset = write_switch_asset(root=tmp_path, model=config.switches.model)
    inputs = _load_pinned_policy_inputs(
        config=config,
        switch_asset=asset,
        controller_path=controller_path,
        controller_digest=hashlib.sha256(b"pinned controller").hexdigest(),
        flight_parameters_path=flight_parameters_path,
        flight_parameters_digest=bundle.flight_parameters_digest,
    )

    assert inputs.controller is controller
    assert inputs.flight_parameters == bundle.flight_parameters

    with pytest.raises(ValueError, match="controller artifact changed"):
        _load_pinned_policy_inputs(
            config=config,
            switch_asset=asset,
            controller_path=controller_path,
            controller_digest="wrong",
            flight_parameters_path=flight_parameters_path,
            flight_parameters_digest=bundle.flight_parameters_digest,
        )

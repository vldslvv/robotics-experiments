"""Tests for the validated policy artifact loader and step composer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotics_experiments.rl.press import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    HOVER_POLICY_INTERFACE,
    METADATA_FILENAME,
    POLICY_FILENAME,
    PRESS_POLICY_INTERFACE,
    build_policy_step_fn,
    load_policy_metadata,
    load_vector_policy,
    policy_metadata,
    validate_policy_artifact,
)
from robotics_experiments.workflows._policy_preflight import _preflight_policy_artifacts
from tests.support.paths import REPO_ROOT


def _action_cfg(**overrides):
    base = dict(
        kind="velocity",
        action_dim=4,
        max_body_velocity_mps=3.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        max_tilt_cmd_deg=30.0,
        hover_thrust_fraction=0.4,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _hover_metadata(action_cfg) -> dict:
    return policy_metadata(
        action_cfg=action_cfg,
        obs_schema_version=HOVER_OBSERVATION_SCHEMA_VERSION,
        obs_dim=HOVER_OBSERVATION_DIM,
        policy_rate_hz=20.0,
        policy_interface=HOVER_POLICY_INTERFACE,
    )


def _write_artifact(tmp_path: Path, metadata: dict) -> Path:
    torch = pytest.importorskip("torch")
    module = torch.jit.trace(
        torch.nn.Linear(HOVER_OBSERVATION_DIM, 4),
        torch.zeros(1, HOVER_OBSERVATION_DIM),
    )
    module.save(str(tmp_path / POLICY_FILENAME))
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(metadata))
    return tmp_path


def _build_step(artifact_dir: Path, action_cfg):
    return build_policy_step_fn(
        artifact_dir=artifact_dir,
        action_cfg=action_cfg,
        policy_rate_hz=20.0,
    )


def test_missing_artifact_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train one first"):
        _build_step(tmp_path, _action_cfg())


def test_metadata_mismatches_reject_before_torch(tmp_path: Path) -> None:
    # Validation precedes the torch import, so a fake (non-torch) module file
    # suffices to prove every mismatch is refused at load time.
    (tmp_path / POLICY_FILENAME).write_bytes(b"not a module")
    good = _hover_metadata(_action_cfg())
    for corrupt in (
        {"obs_schema_version": 999},
        {"obs_dim": 7},
        {"policy_rate_hz": 50.0},
        {"action_kind": "ctbr"},
        {"action_dim": 6},
        {"policy_interface": PRESS_POLICY_INTERFACE},
        {"action_scales": {**good["action_scales"], "max_body_velocity_mps": 9.9}},
    ):
        (tmp_path / METADATA_FILENAME).write_text(json.dumps({**good, **corrupt}))
        with pytest.raises(ValueError, match="metadata mismatch"):
            _build_step(tmp_path, _action_cfg())
    without_scales = {k: v for k, v in good.items() if k != "action_scales"}
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(without_scales))
    with pytest.raises(ValueError, match="action_scales"):
        _build_step(tmp_path, _action_cfg())


def test_build_policy_step_fn_maps_state_to_setpoint(tmp_path: Path) -> None:
    from robotics_experiments.commanding import VelocitySetpoint
    from robotics_experiments.px4_bridge import VehicleState

    action_cfg = _action_cfg()
    artifact = _write_artifact(tmp_path, _hover_metadata(action_cfg))
    step_fn = build_policy_step_fn(
        artifact_dir=artifact,
        action_cfg=action_cfg,
        policy_rate_hz=20.0,
    )
    setpoint = step_fn(VehicleState.hover_at(), np.array([0.0, 0.0, 1.0]))
    assert isinstance(setpoint, VelocitySetpoint)
    # The action is clipped to [-1, 1] before scaling, so every command obeys
    # the configured scale bounds.
    assert abs(setpoint.forward_mps) <= action_cfg.max_body_velocity_mps + 1e-9
    assert abs(setpoint.yaw_rate_radps) <= action_cfg.max_yaw_rate_radps + 1e-9


def test_load_vector_policy_accepts_batched_press_observations(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    action_cfg = _action_cfg()
    obs_dim = 34
    metadata = policy_metadata(
        action_cfg=action_cfg,
        obs_schema_version=1,
        obs_dim=obs_dim,
        policy_rate_hz=20.0,
        policy_interface=PRESS_POLICY_INTERFACE,
    )
    module = torch.jit.trace(torch.nn.Linear(obs_dim, 4), torch.zeros(1, obs_dim))
    module.save(str(tmp_path / POLICY_FILENAME))
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(metadata))

    policy = load_vector_policy(
        artifact_dir=tmp_path,
        action_cfg=action_cfg,
        obs_schema_version=1,
        obs_dim=obs_dim,
        policy_rate_hz=20.0,
        policy_interface=PRESS_POLICY_INTERFACE,
    )

    assert policy(np.zeros(obs_dim)).shape == (4,)
    assert policy(np.zeros((3, obs_dim))).shape == (3, 4)
    with pytest.raises(ValueError, match="observation shape"):
        policy(np.zeros((2, obs_dim // 2)))


def test_metadata_interface_selects_adapter_and_exact_legacy_contracts(tmp_path: Path) -> None:
    hover = _hover_metadata(_action_cfg())
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(hover))
    assert load_policy_metadata(artifact_dir=tmp_path)["policy_interface"] == (
        HOVER_POLICY_INTERFACE
    )

    legacy_hover = {key: value for key, value in hover.items() if key != "policy_interface"}
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_hover))
    assert load_policy_metadata(artifact_dir=tmp_path)["policy_interface"] == (
        HOVER_POLICY_INTERFACE
    )

    legacy_press = {**legacy_hover, "obs_dim": 34}
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_press))
    with pytest.warns(RuntimeWarning, match="inferred press-v1"):
        loaded = load_policy_metadata(artifact_dir=tmp_path)
    assert loaded["policy_interface"] == PRESS_POLICY_INTERFACE

    ambiguous = {**legacy_press, "policy_rate_hz": 19.0}
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(ambiguous))
    with pytest.raises(ValueError, match="re-export"):
        load_policy_metadata(artifact_dir=tmp_path)


def test_legacy_hover_metadata_loads_through_vector_validation(tmp_path: Path) -> None:
    action_cfg = _action_cfg()
    metadata = _hover_metadata(action_cfg)
    metadata.pop("policy_interface")
    artifact = _write_artifact(tmp_path, metadata)

    step = build_policy_step_fn(
        artifact_dir=artifact,
        action_cfg=action_cfg,
        policy_rate_hz=20.0,
    )

    assert callable(step)


def test_metadata_only_validation_does_not_import_torch(tmp_path: Path) -> None:
    action_cfg = _action_cfg()
    (tmp_path / POLICY_FILENAME).write_bytes(b"not a torch module")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(_hover_metadata(action_cfg)))

    metadata = validate_policy_artifact(
        artifact_dir=tmp_path,
        action_cfg=action_cfg,
        obs_schema_version=HOVER_OBSERVATION_SCHEMA_VERSION,
        obs_dim=HOVER_OBSERVATION_DIM,
        policy_rate_hz=20.0,
        policy_interface=HOVER_POLICY_INTERFACE,
    )

    assert metadata["policy_interface"] == HOVER_POLICY_INTERFACE


def test_policy_preflight_rejects_action_scale_before_simulator_start(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    trained_action = _action_cfg(max_body_velocity_mps=0.5)
    metadata = policy_metadata(
        action_cfg=trained_action,
        obs_schema_version=1,
        obs_dim=34,
        policy_rate_hz=20.0,
        policy_interface=PRESS_POLICY_INTERFACE,
    )
    (artifact / POLICY_FILENAME).write_bytes(b"not loaded during preflight")
    (artifact / METADATA_FILENAME).write_text(json.dumps(metadata))
    routine = SimpleNamespace(steps=[{"kind": "policy", "artifact": "artifact", "timeout_s": 20.0}])

    def config(action_cfg):
        return SimpleNamespace(
            vehicles={"x500_01": SimpleNamespace(routine=routine)},
            rl=SimpleNamespace(action=action_cfg, policy_rate_hz=20.0),
        )

    validated = _preflight_policy_artifacts(
        config=config(trained_action),
        repo_root=tmp_path,
    )
    assert validated[0][1] == artifact

    with pytest.raises(ValueError, match="max_body_velocity_mps"):
        _preflight_policy_artifacts(
            config=config(_action_cfg(max_body_velocity_mps=2.0)),
            repo_root=tmp_path,
        )


def test_press_deployment_config_matches_current_artifact_scale() -> None:
    from robotics_experiments.config import load_config

    config = load_config(REPO_ROOT / "configs/policy_press_quad.yaml")

    assert config.rl.action.max_body_velocity_mps == 0.5

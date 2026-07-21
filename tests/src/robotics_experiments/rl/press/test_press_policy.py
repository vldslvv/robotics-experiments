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
    METADATA_FILENAME,
    POLICY_FILENAME,
    build_policy_step_fn,
    policy_metadata,
)


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

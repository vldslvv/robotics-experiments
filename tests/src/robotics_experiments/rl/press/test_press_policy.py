"""Tests for artifact-owned policy contracts and vector loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robotics_experiments.rl.press import (
    HOVER_OBSERVATION_DIM,
    HOVER_POLICY_INTERFACE,
    METADATA_FILENAME,
    POLICY_FILENAME,
    PRESS_POLICY_INTERFACE,
    LoadedVectorPolicy,
    PolicyContract,
    build_policy_step_fn,
    load_policy_contract,
    load_policy_metadata,
    load_vector_policy,
    policy_metadata,
    validate_policy_artifact,
)
from robotics_experiments.workflows._policy_preflight import _preflight_policy_artifacts
from tests.support.paths import REPO_ROOT


def _action_cfg(**overrides):
    values = {
        "kind": "velocity",
        "action_dim": 4,
        "max_body_velocity_mps": 3.0,
        "max_yaw_rate_radps": 1.5,
        "max_body_rate_radps": 6.0,
        "max_tilt_cmd_deg": 30.0,
        "hover_thrust_fraction": 0.4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _metadata(
    *,
    interface: str = HOVER_POLICY_INTERFACE,
    action_cfg=None,
    rate_hz: float = 20.0,
) -> dict:
    action_cfg = _action_cfg() if action_cfg is None else action_cfg
    obs_dim = HOVER_OBSERVATION_DIM if interface == HOVER_POLICY_INTERFACE else 34
    return policy_metadata(
        action_cfg=action_cfg,
        obs_schema_version=1,
        obs_dim=obs_dim,
        policy_rate_hz=rate_hz,
        policy_interface=interface,
    )


def _write_metadata_artifact(path: Path, metadata: dict) -> Path:
    (path / POLICY_FILENAME).write_bytes(b"not loaded during metadata validation")
    (path / METADATA_FILENAME).write_text(json.dumps(metadata))
    return path


def _write_torch_artifact(path: Path, metadata: dict) -> Path:
    torch = pytest.importorskip("torch")
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    module = torch.nn.Linear(obs_dim, action_dim)
    torch.nn.init.zeros_(module.weight)
    module.bias.data.copy_(torch.tensor([1.0, 0.5, 0.25, 0.4]))
    traced = torch.jit.trace(module, torch.zeros(1, obs_dim))
    traced.save(str(path / POLICY_FILENAME))
    (path / METADATA_FILENAME).write_text(json.dumps(metadata))
    return path


def test_missing_artifact_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train and select one first"):
        build_policy_step_fn(artifact_dir=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"policy_interface": "future-v2"}, "unsupported policy_interface"),
        ({"obs_dim": 7}, "unsupported hover-v1 observation"),
        ({"policy_rate_hz": float("nan")}, "finite and positive"),
        ({"policy_rate_hz": 0.0}, "finite and positive"),
        ({"action_kind": "attitude"}, "unsupported action_kind"),
        ({"action_dim": 6}, "unsupported action_dim"),
        ({"action_scales": None}, "action_scales must be an object"),
        (
            {
                "action_scales": {
                    **_metadata()["action_scales"],
                    "max_body_velocity_mps": float("inf"),
                }
            },
            "finite and positive",
        ),
    ],
)
def test_contract_rejects_malformed_or_unsupported_metadata_before_torch(
    tmp_path: Path,
    mutation: dict,
    message: str,
) -> None:
    _write_metadata_artifact(tmp_path, {**_metadata(), **mutation})

    with pytest.raises(ValueError, match=message):
        validate_policy_artifact(
            artifact_dir=tmp_path,
            obs_schema_version=1,
            obs_dim=HOVER_OBSERVATION_DIM,
            policy_interface=HOVER_POLICY_INTERFACE,
        )


def test_contract_rejects_invalid_json_and_missing_scale(tmp_path: Path) -> None:
    (tmp_path / POLICY_FILENAME).write_bytes(b"not torch")
    (tmp_path / METADATA_FILENAME).write_text("{")
    with pytest.raises(ValueError, match="malformed"):
        load_policy_contract(artifact_dir=tmp_path)

    metadata = _metadata()
    del metadata["action_scales"]["max_yaw_rate_radps"]
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="max_yaw_rate_radps"):
        load_policy_contract(artifact_dir=tmp_path)


def test_artifact_scales_drive_hover_offboard_mapping(tmp_path: Path) -> None:
    from robotics_experiments.commanding import VelocitySetpoint
    from robotics_experiments.px4_bridge import VehicleState

    trained_action = _action_cfg(
        max_body_velocity_mps=0.5,
        max_yaw_rate_radps=0.25,
    )
    artifact = _write_torch_artifact(
        tmp_path,
        _metadata(action_cfg=trained_action),
    )

    step = build_policy_step_fn(artifact_dir=artifact)
    setpoint = step(VehicleState.hover_at(), np.array([0.0, 0.0, 1.0]))

    assert isinstance(setpoint, VelocitySetpoint)
    assert setpoint.forward_mps == pytest.approx(0.5)
    assert setpoint.right_mps == pytest.approx(-0.25)
    assert setpoint.down_mps == pytest.approx(-0.125)
    assert setpoint.yaw_rate_radps == pytest.approx(-0.1)


def test_load_vector_policy_carries_contract_and_accepts_batches(tmp_path: Path) -> None:
    artifact = _write_torch_artifact(
        tmp_path,
        _metadata(interface=PRESS_POLICY_INTERFACE),
    )

    loaded = load_vector_policy(
        artifact_dir=artifact,
        obs_schema_version=1,
        obs_dim=34,
        policy_interface=PRESS_POLICY_INTERFACE,
    )

    assert isinstance(loaded, LoadedVectorPolicy)
    assert isinstance(loaded.contract, PolicyContract)
    assert loaded.contract.policy_rate_hz == 20.0
    assert loaded.inference(np.zeros(34)).shape == (4,)
    assert loaded.inference(np.zeros((3, 34))).shape == (3, 4)
    with pytest.raises(ValueError, match="observation shape"):
        loaded.inference(np.zeros((2, 17)))


def test_metadata_interface_selects_exact_legacy_contracts(tmp_path: Path) -> None:
    hover = _metadata()
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(hover))
    assert load_policy_contract(artifact_dir=tmp_path).policy_interface == HOVER_POLICY_INTERFACE

    legacy_hover = {key: value for key, value in hover.items() if key != "policy_interface"}
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_hover))
    assert load_policy_contract(artifact_dir=tmp_path).policy_interface == HOVER_POLICY_INTERFACE

    legacy_press = _metadata(interface=PRESS_POLICY_INTERFACE)
    legacy_press.pop("policy_interface")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_press))
    with pytest.warns(RuntimeWarning, match="inferred press-v1"):
        contract = load_policy_contract(artifact_dir=tmp_path)
    assert contract.policy_interface == PRESS_POLICY_INTERFACE

    legacy_press["policy_rate_hz"] = 19.0
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_press))
    with pytest.raises(ValueError, match="re-export"):
        load_policy_contract(artifact_dir=tmp_path)

    legacy_hover.pop("action_scales")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_hover))
    with pytest.raises(ValueError, match="re-export"):
        load_policy_metadata(artifact_dir=tmp_path)


def _preflight_config(*, artifacts: list[Path], action_cfg=None, policy_rate_hz=99.0):
    vehicles = {
        f"drone{index}": SimpleNamespace(
            routine=SimpleNamespace(
                steps=[
                    {
                        "kind": "policy",
                        "artifact": str(artifact),
                        "timeout_s": 20.0,
                    }
                ]
            )
        )
        for index, artifact in enumerate(artifacts)
    }
    return SimpleNamespace(
        vehicles=vehicles,
        rl=SimpleNamespace(
            action=_action_cfg(max_body_velocity_mps=99.0) if action_cfg is None else action_cfg,
            policy_rate_hz=policy_rate_hz,
        ),
        driver=SimpleNamespace(control_rate_hz=50.0),
    )


def test_policy_preflight_ignores_deployment_action_and_rl_cadence(tmp_path: Path) -> None:
    policy_root = tmp_path / "training" / "press"
    old_run = policy_root / "old-run"
    old_run.mkdir(parents=True)
    _write_metadata_artifact(
        old_run,
        _metadata(interface=PRESS_POLICY_INTERFACE),
    )
    latest_run = policy_root / "latest-run"
    latest_run.mkdir()
    _write_metadata_artifact(
        latest_run,
        _metadata(
            interface=PRESS_POLICY_INTERFACE,
            action_cfg=_action_cfg(max_body_velocity_mps=0.5),
            rate_hz=17.0,
        ),
    )
    artifact = policy_root / "latest"
    artifact.symlink_to(latest_run.name)

    validated = _preflight_policy_artifacts(
        config=_preflight_config(artifacts=[artifact]),
        repo_root=tmp_path,
    )

    assert validated[0][1] == artifact
    assert validated[0][2]["policy_rate_hz"] == 17.0
    assert validated[0][2]["action_scales"]["max_body_velocity_mps"] == 0.5


def test_policy_preflight_rejects_conflicting_press_cadences(tmp_path: Path) -> None:
    artifacts = []
    for index, rate_hz in enumerate((10.0, 20.0)):
        artifact = tmp_path / f"artifact-{index}"
        artifact.mkdir()
        _write_metadata_artifact(
            artifact,
            _metadata(interface=PRESS_POLICY_INTERFACE, rate_hz=rate_hz),
        )
        artifacts.append(artifact)

    with pytest.raises(ValueError, match="must share one policy_rate_hz"):
        _preflight_policy_artifacts(
            config=_preflight_config(artifacts=artifacts),
            repo_root=tmp_path,
        )


def test_press_deployment_config_has_no_action_scale_override() -> None:
    from robotics_experiments.config import load_config

    config_text = (REPO_ROOT / "configs/policy_press_quad.yaml").read_text()
    config = load_config(REPO_ROOT / "configs/policy_press_quad.yaml")
    policy_steps = [
        step for step in config.vehicles["x500_01"].routine.steps if step["kind"] == "policy"
    ]

    assert "max_body_velocity_mps" not in config_text
    assert "policy_rate_hz" not in config_text
    assert len(policy_steps) == 1
    assert policy_steps[0]["artifact"] == ".build/training/press/latest"


def test_current_press_artifact_preflights_when_available() -> None:
    from robotics_experiments.config import load_config

    artifact = REPO_ROOT / ".build/training/press/latest"
    if not artifact.exists():
        pytest.skip("current local press artifact is not available")
    validated = _preflight_policy_artifacts(
        config=load_config(REPO_ROOT / "configs/policy_press_quad.yaml"),
        repo_root=REPO_ROOT,
    )

    assert validated[0][1].resolve() == artifact.resolve()

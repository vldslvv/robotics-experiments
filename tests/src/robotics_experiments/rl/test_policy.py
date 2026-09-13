"""Tests for artifact-owned policy contracts and vector loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import robotics_experiments.rl.hover as hover_module
from robotics_experiments.commanding import PolicyInput
from robotics_experiments.config import load_config
from robotics_experiments.controller import ControllerArtifact, ControllerSettings
from robotics_experiments.flight_parameters import (
    FlightParameters,
    load_flight_parameters,
    load_flight_parameters_bundle,
)
from robotics_experiments.rl import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    METADATA_FILENAME,
    POLICY_FILENAME,
    LoadedPolicy,
    PolicyContract,
    PolicyInterface,
    PolicyMetadata,
    hover_observation_from_state,
    load_policy,
    load_policy_metadata,
    policy_action_to_offboard,
    policy_metadata,
    validate_policy_artifact,
)
from robotics_experiments.rl.hover import load_hover_reset_bounds
from robotics_experiments.rl.press import (
    OBSERVATION_SCHEMA_VERSION,
    PressObservation,
    PressObservationMask,
)
from robotics_experiments.runtime._policy import (
    _hover_adapter,
    _press_approach_adapter,
)
from robotics_experiments.workflows._policy_preflight import _preflight_policy_artifacts
from tests.support.commanding_fakes import make_routine_observation
from tests.support.paths import REPO_ROOT

_HOVER_DEPLOY_CONFIG = load_config(REPO_ROOT / "configs/policy_transfer_press_hover_quad.yaml")


def _action_cfg(**overrides):
    values = {
        "kind": "velocity",
        "action_dim": 4,
        "max_body_velocity_mps": 3.0,
        "max_yaw_rate_radps": 1.5,
        "max_body_rate_radps": 6.0,
        "max_tilt_cmd_deg": 30.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _controller_settings() -> ControllerSettings:
    return ControllerSettings(
        command_delay_s=0.2,
        velocity_controller_rate_hz=125.0,
        attitude_controller_rate_hz=187.5,
        kp_vel=(2.0, 2.0, 3.0),
        horizontal_velocity_error_gain_scale=1.0,
        horizontal_velocity_error_gain_start_mps=0.15,
        horizontal_velocity_error_gain_full_mps=0.30,
        horizontal_velocity_error_gain_end_mps=0.80,
        horizontal_velocity_braking_gain_scale=1.0,
        horizontal_velocity_braking_gain_start_mps=0.05,
        horizontal_velocity_braking_gain_full_mps=0.07,
        horizontal_velocity_braking_gain_end_mps=0.10,
        ki_vel=(0.4, 0.4, 2.0),
        kd_vel=(1.5, 1.5, 1.5),
        velocity_observation_scale=(1.0, 1.0, 1.0),
        velocity_observation_hold_scale=(1.0, 1.0, 1.0),
        velocity_observation_bias_mps=(0.0, 0.0, 0.0),
        velocity_observation_bias_command_start_mps=0.0,
        velocity_observation_bias_command_full_mps=0.01,
        velocity_estimator_bias_decay_hz=0.0,
        velocity_estimator_bias_hold_decay_hz=(0.0, 0.0, 0.0),
        velocity_filter_hz=(10.0, 10.0, 10.0),
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 4.0),
        hover_thrust_normalized=0.57,
        ctbr_effective_collective_scale=1.0,
        kp_att=(6.0, 6.0, 2.0),
        attitude_yaw_weight=0.4,
        kp_rate=(1.2, 1.2, 0.4),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.003, 0.003, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        angular_velocity_filter_hz=0.0,
        rate_derivative_filter_hz=30.0,
        yaw_torque_filter_hz=2.0,
        rate_integral_limit=(0.3, 0.3, 0.2),
        max_tilt_cmd_deg=30.0,
        min_collective_thrust_fraction=0.05,
        max_collective_thrust_fraction=1.0,
        horizontal_thrust_margin_fraction=0.3,
        thruster_tau_inc_s=0.0,
        thruster_tau_dec_s=0.0,
        thruster_max_thrust_rate_nps=100000.0,
    )


def _flight_parameters() -> FlightParameters:
    return load_flight_parameters(REPO_ROOT / "tests/fixtures/flight_parameters")


def _controller_artifact() -> ControllerArtifact:
    settings = _controller_settings()
    return ControllerArtifact(
        artifact_kind="robotics-experiments-controller",
        schema_version=1,
        alignment_id="test-alignment",
        run_id="test-run",
        target_controller_id="target",
        reference_controller_id="reference",
        target_backend="isaac-lab",
        reference_backend="px4-sitl",
        comparison_fingerprint="comparison",
        target_definition_fingerprint="target",
        reference_definition_fingerprint="reference",
        experiment_config="configs/test.yaml",
        experiment_fingerprint="experiment",
        vehicle="x500_01",
        airframe="holybro_x500_v2.usda",
        command_rate_hz=20.0,
        certified_control_modes=("velocity", "ctbr"),
        probe_suite_fingerprint="probes",
        flight_parameters_run_id=_flight_parameters().run_id,
        flight_parameters_fingerprint=_flight_parameters().artifact_fingerprint,
        selected_settings=settings,
        settings_fingerprint=hashlib.sha256(
            json.dumps(
                asdict(settings),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        advisory_quality_passed=True,
        advisory_quality_reasons=(),
        source_commit="abc123",
        source_dirty=False,
        target_image="target-image",
        reference_image="reference-image",
    )


def _metadata(
    *,
    interface: PolicyInterface = PolicyInterface.HOVER,
    action_cfg=None,
    rate_hz: float = 20.0,
) -> dict:
    action_cfg = _action_cfg() if action_cfg is None else action_cfg
    obs_dim = HOVER_OBSERVATION_DIM if interface is PolicyInterface.HOVER else 34
    obs_schema_version = (
        HOVER_OBSERVATION_SCHEMA_VERSION
        if interface is PolicyInterface.HOVER
        else OBSERVATION_SCHEMA_VERSION
    )
    return policy_metadata(
        action_cfg=action_cfg,
        controller_artifact=_controller_artifact(),
        controller_digest="digest",
        flight_parameters=_flight_parameters(),
        flight_parameters_digest="flight-parameters-digest",
        obs_schema_version=obs_schema_version,
        obs_dim=obs_dim,
        policy_rate_hz=rate_hz,
        policy_interface=interface,
        curriculum_stage=("approach" if interface is PolicyInterface.PRESS else None),
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
    with pytest.raises(ValueError, match="run make train-rl first"):
        validate_policy_artifact(artifact_dir=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"policy_interface": "future-v2"}, "unsupported policy_interface"),
        ({"obs_dim": 7}, "unsupported hover observation"),
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
        validate_policy_artifact(artifact_dir=tmp_path)


def test_contract_rejects_invalid_json_and_missing_scale(tmp_path: Path) -> None:
    (tmp_path / POLICY_FILENAME).write_bytes(b"not torch")
    (tmp_path / METADATA_FILENAME).write_text("{")
    with pytest.raises(ValueError, match="malformed"):
        load_policy_metadata(artifact_dir=tmp_path)

    metadata = _metadata()
    del metadata["action_scales"]["max_yaw_rate_radps"]
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="max_yaw_rate_radps"):
        load_policy_metadata(artifact_dir=tmp_path)


def test_policy_rejects_flight_parameters_not_qualified_by_controller(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["controller"]["artifact"]["flight_parameters_fingerprint"] = "other-flight-parameters"
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="does not match its controller"):
        load_policy_metadata(artifact_dir=tmp_path)


def test_policy_rejects_flight_parameters_run_not_qualified_by_controller(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    metadata["controller"]["artifact"]["flight_parameters_run_id"] = "another-run"
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="does not match its controller"):
        load_policy_metadata(artifact_dir=tmp_path)


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

    loaded = load_policy(artifact_dir=artifact)
    observation = hover_observation_from_state(
        VehicleState.hover_at(),
        np.array([0.0, 0.0, 1.0]),
    )
    setpoint = policy_action_to_offboard(
        np.clip(loaded.inference(observation), -1.0, 1.0),
        loaded.contract,
    )

    assert isinstance(setpoint, VelocitySetpoint)
    assert setpoint.forward_mps == pytest.approx(0.5)
    assert setpoint.right_mps == pytest.approx(-0.25)
    assert setpoint.down_mps == pytest.approx(-0.125)
    assert setpoint.yaw_rate_radps == pytest.approx(-0.1)


def test_load_policy_carries_metadata_contract_and_accepts_batches(tmp_path: Path) -> None:
    artifact = _write_torch_artifact(
        tmp_path,
        _metadata(interface=PolicyInterface.PRESS),
    )

    loaded = load_policy(artifact_dir=artifact)

    assert isinstance(loaded, LoadedPolicy)
    assert isinstance(loaded.metadata, PolicyMetadata)
    assert isinstance(loaded.contract, PolicyContract)
    assert loaded.contract.policy_rate_hz == 20.0
    assert loaded.contract.curriculum_stage == "approach"
    assert loaded.inference(np.zeros(34)).shape == (4,)
    assert loaded.inference(np.zeros((3, 34))).shape == (3, 4)
    with pytest.raises(ValueError, match="observation shape"):
        loaded.inference(np.zeros((2, 17)))


def test_metadata_rejects_every_legacy_artifact(tmp_path: Path) -> None:
    hover = _metadata()
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(hover))
    assert (
        load_policy_metadata(artifact_dir=tmp_path).contract.policy_interface
        is PolicyInterface.HOVER
    )

    legacy_hover = {key: value for key, value in hover.items() if key != "controller"}
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_hover))
    with pytest.raises(ValueError, match="controller"):
        load_policy_metadata(artifact_dir=tmp_path)

    legacy_press = _metadata(interface=PolicyInterface.PRESS)
    legacy_press.pop("controller")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_press))
    with pytest.raises(ValueError, match="controller"):
        load_policy_metadata(artifact_dir=tmp_path)

    legacy_envelope = _metadata()
    legacy_envelope.pop("flight_parameters")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(legacy_envelope))
    with pytest.raises(ValueError, match="flight_parameters"):
        load_policy_metadata(artifact_dir=tmp_path)


@pytest.mark.parametrize(
    ("stage", "message"),
    [(None, "stage-less"), ("future", "unsupported curriculum_stage")],
)
def test_press_metadata_requires_known_curriculum_stage(
    tmp_path: Path,
    stage: str | None,
    message: str,
) -> None:
    metadata = _metadata(interface=PolicyInterface.PRESS)
    if stage is None:
        metadata.pop("curriculum_stage")
    else:
        metadata["curriculum_stage"] = stage
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
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
            press=SimpleNamespace(curriculum=SimpleNamespace(active_stage="approach")),
        ),
        driver=SimpleNamespace(control_rate_hz=50.0),
    )


def test_policy_preflight_ignores_deployment_action_and_rl_cadence(tmp_path: Path) -> None:
    policy_root = tmp_path / "training" / "press-approach"
    old_run = policy_root / "old-run"
    old_run.mkdir(parents=True)
    _write_metadata_artifact(
        old_run,
        _metadata(interface=PolicyInterface.PRESS),
    )
    latest_run = policy_root / "latest-run"
    latest_run.mkdir()
    _write_metadata_artifact(
        latest_run,
        _metadata(
            interface=PolicyInterface.PRESS,
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

    assert validated.policies[0].artifact == artifact
    assert validated.policies[0].metadata.contract.policy_rate_hz == 17.0
    assert validated.policies[0].metadata.contract.max_body_velocity_mps == 0.5


def test_policy_preflight_rejects_conflicting_press_cadences(tmp_path: Path) -> None:
    artifacts = []
    for index, rate_hz in enumerate((10.0, 20.0)):
        artifact = tmp_path / f"artifact-{index}"
        artifact.mkdir()
        _write_metadata_artifact(
            artifact,
            _metadata(interface=PolicyInterface.PRESS, rate_hz=rate_hz),
        )
        artifacts.append(artifact)

    with pytest.raises(ValueError, match="must share one policy_rate_hz"):
        _preflight_policy_artifacts(
            config=_preflight_config(artifacts=artifacts),
            repo_root=tmp_path,
        )


def test_press_deployment_config_has_no_action_scale_override() -> None:
    from robotics_experiments.config import load_config

    config_path = REPO_ROOT / "configs/policy_press_approach_quad.yaml"
    config_text = config_path.read_text()
    config = load_config(config_path)
    policy_steps = [
        step for step in config.vehicles["x500_01"].routine.steps if step["kind"] == "policy"
    ]

    assert "max_body_velocity_mps" not in config_text
    assert "policy_rate_hz" not in config_text
    assert len(policy_steps) == 1
    assert policy_steps[0]["artifact"] == ".build/training/press-approach/latest"


def test_hover_adapter_requires_hold_duration() -> None:
    loaded = LoadedPolicy(
        metadata=PolicyMetadata(
            contract=PolicyContract(
                policy_interface=PolicyInterface.HOVER,
                obs_schema_version=1,
                obs_dim=HOVER_OBSERVATION_DIM,
                policy_rate_hz=20.0,
                action_kind="velocity",
                action_dim=4,
                max_body_velocity_mps=3.0,
                max_yaw_rate_radps=1.5,
                max_body_rate_radps=6.0,
                hover_thrust_normalized=0.57,
            ),
            controller_settings=_controller_settings(),
            controller_run_id="test-run",
            controller_digest="digest",
            controller_comparison_fingerprint="comparison",
            flight_parameters=_flight_parameters(),
            flight_parameters_digest="flight-parameters-digest",
            values={},
        ),
        inference=lambda observation: np.zeros(4),
    )

    with pytest.raises(ValueError, match="requires 'hold_s'"):
        _hover_adapter(hold_s=None, loaded=loaded)


@pytest.mark.parametrize("task_observation", [None, (), "wrong"])
def test_press_adapter_fails_closed_on_incompatible_task_observation(
    task_observation,
) -> None:
    from robotics_experiments.config import load_config

    inference_calls: list[np.ndarray] = []
    loaded = LoadedPolicy(
        metadata=PolicyMetadata(
            contract=PolicyContract(
                policy_interface=PolicyInterface.PRESS,
                obs_schema_version=1,
                obs_dim=34,
                policy_rate_hz=20.0,
                action_kind="velocity",
                action_dim=4,
                max_body_velocity_mps=0.5,
                max_yaw_rate_radps=1.5,
                max_body_rate_radps=6.0,
                hover_thrust_normalized=0.57,
                curriculum_stage="approach",
            ),
            controller_settings=_controller_settings(),
            controller_run_id="test-run",
            controller_digest="digest",
            controller_comparison_fingerprint="comparison",
            flight_parameters=_flight_parameters(),
            flight_parameters_digest="flight-parameters-digest",
            values={},
        ),
        inference=lambda observation: inference_calls.append(observation) or np.ones(4),  # type: ignore[func-returns-value]  # concise call recording
    )
    adapter = _press_approach_adapter(
        repo_root=REPO_ROOT,
        config=load_config(REPO_ROOT / "configs/policy_press_approach_quad.yaml"),
        loaded=loaded,
    )
    observation = make_routine_observation(sim_time_us=10)

    result = adapter.advance(
        policy_input=PolicyInput(
            vehicle_state=observation.vehicle_state,
            sim_time_us=observation.sim_time_us,
            task_observation=task_observation,
            reference_position_enu=(0.0, 0.0, 0.0),
        ),
        state=adapter.initial_state,
    )

    assert "requires a PressObservation" in str(result.failure_reason)
    assert result.state is adapter.initial_state
    assert inference_calls == []
    assert (
        json.loads(result.diagnostics[0].split("PRESS_POLICY ", maxsplit=1)[1])["event"]
        == "failure"
    )


def test_press_adapter_emits_exact_actor_entry_diagnostic() -> None:
    from robotics_experiments.config import load_config

    loaded = LoadedPolicy(
        metadata=PolicyMetadata(
            contract=PolicyContract(
                policy_interface=PolicyInterface.PRESS,
                obs_schema_version=1,
                obs_dim=34,
                policy_rate_hz=20.0,
                action_kind="velocity",
                action_dim=4,
                max_body_velocity_mps=0.5,
                max_yaw_rate_radps=1.5,
                max_body_rate_radps=6.0,
                hover_thrust_normalized=0.57,
                curriculum_stage="approach",
            ),
            controller_settings=_controller_settings(),
            controller_run_id="test-run",
            controller_digest="digest",
            controller_comparison_fingerprint="comparison",
            flight_parameters=_flight_parameters(),
            flight_parameters_digest="flight-parameters-digest",
            values={},
        ),
        inference=lambda _observation: np.array([1.2, -0.5, 0.25, -0.1]),
    )
    adapter = _press_approach_adapter(
        repo_root=REPO_ROOT,
        config=load_config(REPO_ROOT / "configs/policy_press_approach_quad.yaml"),
        loaded=loaded,
    )
    routine_observation = make_routine_observation(sim_time_us=123_000)
    mask = int(PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP)
    press_observation = PressObservation(
        relative_switch_pos_m=np.array([0.48, 0.0, 0.0]),
        relative_switch_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pose_confidence=1.0,
        pose_age_s=0.0,
        detection_valid=True,
        body_lin_vel_mps=np.zeros(3),
        body_ang_vel_radps=np.zeros(3),
        attitude_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pusher_compression_m=0.0,
        pusher_compression_rate_mps=0.0,
        contact_valid=True,
        contact_flag=False,
        lamp_state=0,
        lamp_confidence=1.0,
        lamp_age_s=0.0,
        previous_action=np.full(4, 99.0),
        handoff_mode=0,
        estimator_health=1.0,
        obs_mask=mask,
    )

    result = adapter.advance(
        policy_input=PolicyInput(
            vehicle_state=routine_observation.vehicle_state,
            sim_time_us=routine_observation.sim_time_us,
            task_observation=press_observation,
            reference_position_enu=(0.0, 0.0, 0.0),
        ),
        state=adapter.initial_state,
    )

    diagnostic = json.loads(result.diagnostics[0].split("PRESS_POLICY ", maxsplit=1)[1])
    assert diagnostic["event"] == "actor_evaluation"
    assert diagnostic["actor_entry"] is True
    assert len(diagnostic["observation"]) == 34
    assert diagnostic["observation"][27:31] == [0.0, 0.0, 0.0, 0.0]
    assert diagnostic["observation"][31] == 1.0
    assert diagnostic["raw_normalized_action"] == [1.2, -0.5, 0.25, -0.1]
    assert diagnostic["normalized_action"] == [1.0, -0.5, 0.25, -0.1]
    assert diagnostic["setpoint"] == {
        "down_mps": -0.125,
        "forward_mps": 0.5,
        "kind": "VelocitySetpoint",
        "right_mps": 0.25,
        "yaw_rate_radps": 0.15000000000000002,
    }
    assert diagnostic["stable_goal_count"] == 1
    assert diagnostic["contact"] is False
    assert diagnostic["complete"] is False


@pytest.mark.parametrize("mismatch", ["run_id", "digest"])
def test_hover_handoff_rejects_mismatched_producer(mismatch: str) -> None:
    path = REPO_ROOT / "tests/fixtures/flight_parameters"
    bundle = load_flight_parameters_bundle(path=path)
    with pytest.raises(ValueError, match="pinned input"):
        load_hover_reset_bounds(
            path=path.resolve(),
            run_id="wrong" if mismatch == "run_id" else bundle.flight_parameters.run_id,
            digest="wrong" if mismatch == "digest" else bundle.flight_parameters_digest,
        )


def test_hover_reset_loading_is_cached_before_live_adapter_construction(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = load_flight_parameters_bundle(path=REPO_ROOT / "tests/fixtures/flight_parameters")
    metadata = _metadata(interface=PolicyInterface.PRESS)
    metadata["curriculum_stage"] = "hover"
    metadata["flight_parameters"]["digest"] = bundle.flight_parameters_digest
    artifact = _write_metadata_artifact(tmp_path, metadata)
    config = _HOVER_DEPLOY_CONFIG
    vehicle = config.vehicles["x500_01"]
    config = replace(
        config,
        vehicles={
            "x500_01": replace(
                vehicle,
                routine=replace(
                    vehicle.routine,
                    steps=({"kind": "policy", "artifact": str(artifact), "switch_index": 0},),
                ),
            )
        },
        rl=replace(
            config.rl,
            inputs=replace(config.rl.inputs, flight_parameters_artifact=str(bundle.run_dir)),
        ),
    )
    _preflight_policy_artifacts(config=config, repo_root=REPO_ROOT)

    def unexpected_load(**_kwargs):
        raise AssertionError("reset traces must not be read during live routine construction")

    monkeypatch.setattr(hover_module, "load_flight_parameters_bundle", unexpected_load)
    loaded = LoadedPolicy(
        metadata=load_policy_metadata(artifact_dir=artifact), inference=lambda obs: np.zeros(4)
    )
    adapter = _press_approach_adapter(config=config, loaded=loaded, repo_root=REPO_ROOT)
    assert adapter.policy_rate_hz == loaded.contract.policy_rate_hz

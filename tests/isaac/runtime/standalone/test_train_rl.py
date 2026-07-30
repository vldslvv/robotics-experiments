"""Unit tests for the RL training entrypoint's artifact path contract."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.alignment import ControllerSettings
from tests.support.paths import REPO_ROOT


def _load_train_rl_module():
    pytest.importorskip("torch")
    script = REPO_ROOT / "isaac" / "runtime" / "standalone" / "train_rl.py"
    spec = importlib.util.spec_from_file_location("train_rl", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controller_settings() -> ControllerSettings:
    return ControllerSettings(
        command_delay_s=0.2,
        kp_vel=(2.0, 2.0, 3.0),
        horizontal_velocity_error_gain_scale=1.0,
        horizontal_velocity_error_gain_start_mps=0.15,
        horizontal_velocity_error_gain_full_mps=0.30,
        ki_vel=(0.4, 0.4, 2.0),
        kd_vel=(1.5, 1.5, 1.5),
        velocity_filter_hz=10.0,
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 4.0),
        hover_thrust_fraction=0.5,
        hover_thrust_normalized=0.57,
        kp_att=(6.0, 6.0, 2.0),
        attitude_yaw_weight=0.4,
        kp_rate=(1.2, 1.2, 0.4),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.003, 0.003, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        rate_output_scale_nm=(1.0, 1.0, 1.0),
        rate_derivative_filter_hz=30.0,
        rate_integral_limit_nm=(0.3, 0.3, 0.2),
        max_tilt_cmd_deg=30.0,
        min_collective_thrust_fraction=0.05,
        max_collective_thrust_fraction=1.0,
        horizontal_thrust_margin_fraction=0.3,
        thruster_tau_inc_s=0.0,
        thruster_tau_dec_s=0.0,
        thruster_max_thrust_rate_nps=100000.0,
    )


def _controller_artifact():
    return SimpleNamespace(
        schema_version=1,
        alignment_id="test-alignment",
        run_id="alignment-run",
        comparison_fingerprint="comparison",
        target_definition_fingerprint="target",
        reference_definition_fingerprint="reference",
        selected_settings=_controller_settings(),
        advisory_quality_passed=True,
    )


def _controller_metadata() -> dict:
    settings = _controller_settings()
    return {
        "schema_version": 1,
        "alignment_id": "test-alignment",
        "run_id": "alignment-run",
        "controller_digest": "digest",
        "controller_comparison_fingerprint": "comparison",
        "target_definition_fingerprint": "target",
        "reference_definition_fingerprint": "reference",
        "settings": json.loads(json.dumps(asdict(settings))),
        "advisory_quality_passed": True,
    }


def test_train_rl_uses_the_requested_artifact_run_id(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    training_root = tmp_path / "training"
    policy_root = train_rl._policy_out_root(  # noqa: SLF001 - helper contract.
        training_root,
        "hover",
    )
    requested_run_id = tmp_path.name

    run_dir = train_rl._make_run_dir(  # noqa: SLF001 - helper contract.
        policy_root,
        requested_run_id,
    )

    assert run_dir.parent == training_root / "hover"
    assert run_dir.name == requested_run_id
    assert not (policy_root / "latest").exists()


def test_training_alignment_keeps_normalized_action_scales_task_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_rl = _load_train_rl_module()
    artifact_path = tmp_path / ".build" / "alignment" / "test" / "best"
    artifact_path.mkdir(parents=True)
    (artifact_path / "controller.json").write_text("{}")
    artifact = SimpleNamespace(
        vehicle="x500_01",
        airframe="holybro_x500_v2.usda",
        command_rate_hz=20.0,
        command_interface=SimpleNamespace(
            kind="body-flu-velocity-yaw-rate-v1",
            dimension=4,
            linear_scale_mps=3.0,
            yaw_rate_scale_radps=1.5,
        ),
        advisory_quality_passed=True,
    )
    monkeypatch.setattr(train_rl, "load_controller_artifact", lambda _path: artifact)
    config = SimpleNamespace(
        vehicles={
            "x500_01": SimpleNamespace(
                id=0,
                spawn=SimpleNamespace(usd_path="@assets/holybro_x500_v2.usda"),
            )
        },
        rl=SimpleNamespace(
            policy_rate_hz=20.0,
            action=SimpleNamespace(
                kind="velocity",
                action_dim=4,
                max_body_velocity_mps=2.0,
                max_yaw_rate_radps=1.0,
            ),
            training=SimpleNamespace(
                vehicle_id=0,
                controller_artifact=".build/alignment/test/best",
            ),
        ),
    )

    loaded, digest = train_rl._load_training_controller(  # noqa: SLF001
        args=SimpleNamespace(controller_artifact=None),
        config=config,
        repo_root=tmp_path,
    )

    assert loaded is artifact
    assert digest


@pytest.mark.parametrize("policy_name", ["", ".", "..", "/hover", "hover/latest", "hover.latest"])
def test_train_rl_rejects_invalid_policy_names(policy_name: str, tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()

    with pytest.raises(ValueError, match="policy_name"):
        train_rl._policy_out_root(  # noqa: SLF001 - helper contract.
            tmp_path,
            policy_name,
        )


def test_hover_artifact_metadata_remains_task_owned_and_unchanged() -> None:
    train_rl = _load_train_rl_module()
    action = SimpleNamespace(
        kind="velocity",
        action_dim=4,
        max_body_velocity_mps=3.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_fraction=0.4,
    )
    config = SimpleNamespace(rl=SimpleNamespace(action=action, policy_rate_hz=20.0))
    env_cfg = SimpleNamespace(
        observation_schema_version=1,
        observation_space=13,
        policy_interface="hover-v1",
    )

    metadata = train_rl._artifact_metadata(  # noqa: SLF001
        config=config,
        env_cfg=env_cfg,
        source_metadata={"source_commit": "abc123", "source_dirty": False},
        artifact_run_id="run-1",
        controller_artifact=_controller_artifact(),
        controller_digest="digest",
    )

    assert metadata["obs_schema_version"] == 1
    assert metadata["obs_dim"] == 13
    assert metadata["policy_interface"] == "hover-v1"
    assert metadata["policy_rate_hz"] == 20.0
    assert metadata["source_commit"] == "abc123"
    assert metadata["source_dirty"] is False
    assert metadata["artifact_run_id"] == "run-1"


def test_git_metadata_marks_mounted_repository_safe(monkeypatch, tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    calls = []
    outputs = iter(("abc123\n", " M changed.py\n"))

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(train_rl.subprocess, "run", fake_run)

    metadata = train_rl._git_source_metadata(repo_root=tmp_path)  # noqa: SLF001

    safe_directory = f"safe.directory={tmp_path.resolve()}"
    assert calls[0][0][:3] == ["git", "-c", safe_directory]
    assert calls[1][0][:3] == ["git", "-c", safe_directory]
    assert metadata == {"source_commit": "abc123", "source_dirty": True}


def test_checkpoint_export_resolves_existing_policy_run(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    out_root = tmp_path / "training" / "press-approach"
    checkpoint = out_root / "run-1" / "logs" / "model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    run_dir = train_rl._checkpoint_run_dir(  # noqa: SLF001 - helper contract.
        checkpoint_path=checkpoint,
        out_root=out_root,
    )

    assert run_dir == out_root / "run-1"


def test_checkpoint_export_rejects_checkpoint_outside_policy_root(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    checkpoint = tmp_path / "other" / "run-1" / "logs" / "model_299.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    with pytest.raises(ValueError, match="outside policy artifact root"):
        train_rl._checkpoint_run_dir(  # noqa: SLF001 - helper contract.
            checkpoint_path=checkpoint,
            out_root=tmp_path / "training" / "press-approach",
        )


def test_checkpoint_export_reuses_contract_captured_at_training_start(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()
    metadata = {
        "policy_interface": "press-v1",
        "curriculum_stage": "approach",
        "obs_schema_version": 1,
        "obs_dim": 34,
        "policy_rate_hz": 17.0,
        "action_kind": "velocity",
        "action_dim": 4,
        "action_scales": {
            "max_body_velocity_mps": 0.5,
            "max_yaw_rate_radps": 1.5,
            "max_body_rate_radps": 6.0,
            "hover_thrust_fraction": 0.4,
        },
        "source_commit": "captured",
        "source_dirty": False,
        "controller": _controller_metadata(),
    }
    train_rl._write_artifact_metadata(  # noqa: SLF001
        run_dir=tmp_path,
        metadata=metadata,
    )

    stored, policy_metadata = train_rl._stored_artifact_metadata(  # noqa: SLF001
        run_dir=tmp_path
    )

    assert stored == metadata
    assert policy_metadata.contract.policy_rate_hz == 17.0
    assert policy_metadata.contract.max_body_velocity_mps == 0.5
    assert policy_metadata.controller_settings == _controller_settings()


def test_checkpoint_export_rejects_run_without_stored_contract(tmp_path: Path) -> None:
    train_rl = _load_train_rl_module()

    with pytest.raises(ValueError, match="original training contract cannot be recovered"):
        train_rl._stored_artifact_metadata(run_dir=tmp_path)  # noqa: SLF001


def test_training_persists_contract_before_ppo_without_exporting_or_publishing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    action = SimpleNamespace(
        kind="velocity",
        action_dim=4,
        max_body_velocity_mps=0.75,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_fraction=0.4,
    )
    training_cfg = SimpleNamespace(
        task_id="press-task",
        policy_name="press-approach",
        training_mode="scratch",
        seed=7,
        max_iterations=2,
        controller_artifact=".build/alignment/test/best",
    )
    config = SimpleNamespace(
        rl=SimpleNamespace(
            action=action,
            policy_rate_hz=17.0,
            training=training_cfg,
            press=SimpleNamespace(
                curriculum_stage="approach",
                policy=SimpleNamespace(
                    actor=SimpleNamespace(output_initialization="scratch-neutral-output")
                ),
            ),
        )
    )
    env = SimpleNamespace(close=lambda: None)
    env_cfg = SimpleNamespace(
        observation_schema_version=1,
        observation_space=34,
        policy_interface="press-v1",
        policy_config=SimpleNamespace(
            actor=SimpleNamespace(output_initialization="scratch-neutral-output")
        ),
    )
    runner = object()
    before_ppo = {}
    export_calls = []

    monkeypatch.setattr(train_rl, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(train_rl, "_build_env", lambda **_kwargs: (env, env_cfg))
    monkeypatch.setattr(
        train_rl,
        "_load_training_controller",
        lambda **_kwargs: (_controller_artifact(), "digest"),
    )
    monkeypatch.setattr(
        train_rl,
        "_git_source_metadata",
        lambda **_kwargs: {"source_commit": "captured", "source_dirty": False},
    )
    import robotics_experiments.rl.training.rsl_rl as training

    def build_runner(*_args, log_dir, **_kwargs):
        run_dir = Path(log_dir).parent
        before_ppo.update(json.loads((run_dir / "policy_meta.json").read_text()))
        return runner

    monkeypatch.setattr(training, "build_runner", build_runner)
    initialization_calls = []
    monkeypatch.setattr(
        training,
        "initialize_actor_output",
        lambda **kwargs: initialization_calls.append(kwargs),
    )
    monkeypatch.setattr(
        training,
        "train",
        lambda *_args: SimpleNamespace(iterations=2),
    )
    monkeypatch.setattr(
        training,
        "export_policy",
        lambda _runner, _run_dir, *, metadata: export_calls.append(metadata),
    )
    args = SimpleNamespace(
        seed=None,
        max_iterations=None,
        num_envs=None,
        export_checkpoint=None,
        initialize_checkpoint=None,
        resume_checkpoint=None,
        controller_artifact=None,
    )

    train_rl._run(args=args, config=config)  # noqa: SLF001

    assert before_ppo["policy_rate_hz"] == 17.0
    assert before_ppo["action_scales"]["max_body_velocity_mps"] == 0.75
    assert before_ppo["artifact_run_id"]
    assert before_ppo["controller"]["controller_digest"] == "digest"
    assert before_ppo["training_mode"] == "scratch"
    assert before_ppo["initialization"] == {"method": "scratch-neutral-output"}
    assert initialization_calls == [{"runner": runner, "method": "scratch-neutral-output"}]
    assert export_calls == []
    assert not (tmp_path / ".build" / "training" / "press-approach" / "latest").exists()


def test_checkpoint_load_skips_optimizer_state(tmp_path: Path) -> None:
    from robotics_experiments.rl.training import rsl_rl

    calls = []
    runner = SimpleNamespace(load=lambda *args, **kwargs: calls.append((args, kwargs)))
    checkpoint = tmp_path / "model_299.pt"

    rsl_rl.load_checkpoint(
        runner=runner,
        checkpoint_path=checkpoint,
        device="cuda:0",
    )

    assert calls == [((str(checkpoint),), {"load_optimizer": False, "map_location": "cuda:0"})]


def test_resume_restores_optimizer_state_and_applies_phase_learning_rate(tmp_path: Path) -> None:
    from robotics_experiments.rl.training import rsl_rl

    calls = []
    optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-4}, {"lr": 1.0e-4}])
    algorithm = SimpleNamespace(optimizer=optimizer, learning_rate=1.0e-4)
    runner = SimpleNamespace(
        alg=algorithm,
        load=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    checkpoint = tmp_path / "model_299.pt"

    rsl_rl.resume_training_from_checkpoint(
        runner=runner,
        checkpoint_path=checkpoint,
        device="cuda:0",
        learning_rate=3.0e-5,
    )

    assert calls == [((str(checkpoint),), {"load_optimizer": True, "map_location": "cuda:0"})]
    assert algorithm.learning_rate == 3.0e-5
    assert [group["lr"] for group in optimizer.param_groups] == [3.0e-5, 3.0e-5]


def test_actor_only_initialization_preserves_fresh_training_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    actor = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ELU(),
        torch.nn.Linear(4, 2),
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ELU(),
        torch.nn.Linear(4, 1),
    )
    std = torch.nn.Parameter(torch.full((2,), 0.2))
    policy = SimpleNamespace(actor=actor, critic=critic, std=std)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters(), std])
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer),
        current_learning_iteration=0,
    )
    critic_before = {
        name: value.detach().clone() for name, value in policy.critic.state_dict().items()
    }
    std_before = policy.std.detach().clone()
    optimizer_before = optimizer.state_dict()
    actor_state = {
        f"actor.{name}": torch.full_like(value, 0.75)
        for name, value in policy.actor.state_dict().items()
    }
    checkpoint = tmp_path / "model_50.pt"
    torch.save(
        {
            "model_state_dict": actor_state
            | {
                f"critic.{name}": torch.full_like(value, 9.0)
                for name, value in policy.critic.state_dict().items()
            }
            | {"std": torch.full_like(policy.std, 9.0)},
            "optimizer_state_dict": {"source": "must not load"},
            "iter": 50,
        },
        checkpoint,
    )

    rsl_rl.initialize_actor_from_checkpoint(
        runner=runner,
        checkpoint_path=checkpoint,
        device="cpu",
    )

    for value in policy.actor.state_dict().values():
        assert torch.all(value == 0.75)
    for name, value in policy.critic.state_dict().items():
        assert torch.equal(value, critic_before[name])
    assert torch.equal(policy.std, std_before)
    assert optimizer.state_dict() == optimizer_before
    assert runner.current_learning_iteration == 0


def test_scratch_neutral_output_initialization_is_near_zero_and_unbiased() -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    actor = torch.nn.Sequential(
        torch.nn.Linear(34, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 4),
    )
    runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(actor=actor)))

    rsl_rl.initialize_actor_output(
        runner=runner,
        method="scratch-neutral-output",
    )

    output = actor[-1]
    assert torch.count_nonzero(output.bias) == 0
    gram = output.weight @ output.weight.T
    torch.testing.assert_close(
        gram,
        torch.eye(4) * 0.01**2,
        atol=1.0e-10,
        rtol=1.0e-5,
    )
    action_mean = actor(torch.zeros((128, 34)))
    assert float(action_mean.abs().max()) < 0.01


@pytest.mark.parametrize(
    ("training_mode", "initialization_checkpoint", "resume_checkpoint", "match"),
    [
        ("scratch", Path("model_1.pt"), None, "scratch training forbids"),
        ("scratch", None, Path("model_1.pt"), "scratch training forbids"),
        ("actor-only", None, None, "actor-only training requires"),
        ("resume", None, None, "resume training requires"),
        ("unknown", None, None, "unsupported"),
    ],
)
def test_training_mode_validates_checkpoint_contract(
    training_mode: str,
    initialization_checkpoint: Path | None,
    resume_checkpoint: Path | None,
    match: str,
) -> None:
    train_rl = _load_train_rl_module()

    with pytest.raises(ValueError, match=match):
        train_rl._validate_training_mode(  # noqa: SLF001
            training_mode=training_mode,
            initialization_checkpoint=initialization_checkpoint,
            resume_checkpoint=resume_checkpoint,
        )


def test_valid_training_modes_accept_their_checkpoint_contract() -> None:
    train_rl = _load_train_rl_module()

    train_rl._validate_training_mode(  # noqa: SLF001
        training_mode="scratch",
        initialization_checkpoint=None,
        resume_checkpoint=None,
    )
    train_rl._validate_training_mode(  # noqa: SLF001
        training_mode="actor-only",
        initialization_checkpoint=Path("model_1.pt"),
        resume_checkpoint=None,
    )
    train_rl._validate_training_mode(  # noqa: SLF001
        training_mode="resume",
        initialization_checkpoint=None,
        resume_checkpoint=Path("model_1.pt"),
    )


def test_resume_requires_matching_interface_controller_and_selection_fingerprint() -> None:
    train_rl = _load_train_rl_module()
    from robotics_experiments.config import load_config
    from robotics_experiments.rl import PolicyContract, PolicyInterface
    from robotics_experiments.rl.training._fingerprint import _press_comparison_fingerprint

    config = load_config(REPO_ROOT / "configs" / "train_press_approach_quad_safety.yaml")
    action = config.rl.action
    contract = PolicyContract(
        policy_interface=PolicyInterface.PRESS_V1,
        obs_schema_version=1,
        obs_dim=34,
        policy_rate_hz=config.rl.policy_rate_hz,
        action_kind=action.kind,
        action_dim=action.action_dim,
        max_body_velocity_mps=action.max_body_velocity_mps,
        max_yaw_rate_radps=action.max_yaw_rate_radps,
        max_body_rate_radps=action.max_body_rate_radps,
        hover_thrust_fraction=action.hover_thrust_fraction,
        curriculum_stage="approach",
    )
    fingerprint = _press_comparison_fingerprint(config=config, contract=contract)
    env_cfg = SimpleNamespace(
        policy_interface="press-v1",
        observation_schema_version=1,
        observation_space=34,
    )

    source = SimpleNamespace(
        contract=contract,
        controller_digest="controller",
        values={"checkpoint_evaluation": {"comparison_fingerprint": fingerprint}},
    )
    train_rl._validate_resume_contract(  # noqa: SLF001
        source_metadata=source,
        config=config,
        env_cfg=env_cfg,
        controller_digest="controller",
    )

    with pytest.raises(ValueError, match="policy interface"):
        train_rl._validate_resume_contract(  # noqa: SLF001
            source_metadata=SimpleNamespace(
                contract=replace(contract, obs_dim=35),
                controller_digest="controller",
                values=source.values,
            ),
            config=config,
            env_cfg=env_cfg,
            controller_digest="controller",
        )
    with pytest.raises(ValueError, match="controller digest"):
        train_rl._validate_resume_contract(  # noqa: SLF001
            source_metadata=SimpleNamespace(
                contract=contract,
                controller_digest="different",
                values=source.values,
            ),
            config=config,
            env_cfg=env_cfg,
            controller_digest="controller",
        )
    with pytest.raises(ValueError, match="comparison fingerprint"):
        train_rl._validate_resume_contract(  # noqa: SLF001
            source_metadata=SimpleNamespace(
                contract=contract,
                controller_digest="controller",
                values={"checkpoint_evaluation": {"comparison_fingerprint": "different"}},
            ),
            config=config,
            env_cfg=env_cfg,
            controller_digest="controller",
        )


def test_actor_only_initialization_rejects_incompatible_architecture(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    actor = torch.nn.Sequential(torch.nn.Linear(3, 2))
    runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(actor=actor)))
    checkpoint = tmp_path / "model_50.pt"
    torch.save(
        {
            "model_state_dict": {
                "actor.0.weight": torch.zeros((2, 4)),
                "actor.0.bias": torch.zeros(2),
            }
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="dimensions or actor architecture"):
        rsl_rl.initialize_actor_from_checkpoint(
            runner=runner,
            checkpoint_path=checkpoint,
            device="cpu",
        )


def test_cross_namespace_initialization_records_complete_source_provenance(
    tmp_path: Path,
) -> None:
    train_rl = _load_train_rl_module()
    training_root = tmp_path / ".build" / "training"
    run_dir = training_root / "press-source" / "source-run"
    checkpoint = run_dir / "logs" / "model_50.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    source_record = {
        "policy_interface": "press-v1",
        "curriculum_stage": "approach",
        "obs_schema_version": 1,
        "obs_dim": 34,
        "policy_rate_hz": 20.0,
        "action_kind": "velocity",
        "action_dim": 4,
        "action_scales": {
            "max_body_velocity_mps": 2.0,
            "max_yaw_rate_radps": 1.5,
            "max_body_rate_radps": 6.0,
            "hover_thrust_fraction": 0.4,
        },
        "source_commit": "source-commit",
        "source_dirty": False,
        "controller": _controller_metadata(),
    }
    train_rl._write_artifact_metadata(  # noqa: SLF001
        run_dir=run_dir,
        metadata=source_record,
    )

    provenance, loaded_metadata = train_rl._checkpoint_provenance(  # noqa: SLF001
        checkpoint_path=checkpoint,
        training_root=training_root,
        repo_root=tmp_path,
        method="actor-only",
        option_name="--initialize-checkpoint",
    )

    assert provenance == {
        "method": "actor-only",
        "source_policy": "press-source",
        "source_stage": "approach",
        "source_run": "source-run",
        "checkpoint_iteration": 50,
        "checkpoint_path": (".build/training/press-source/source-run/logs/model_50.pt"),
        "source_commit": "source-commit",
    }
    assert loaded_metadata.contract.obs_dim == 34
    json.dumps({"initialization": provenance}, allow_nan=False)


@pytest.mark.parametrize(
    ("source_obs_dim", "source_action_dim", "match"),
    [
        (13, 4, "observation dimension"),
        (34, 3, "action dimension"),
    ],
)
def test_actor_initialization_rejects_incompatible_contract_dimensions(
    source_obs_dim: int,
    source_action_dim: int,
    match: str,
) -> None:
    train_rl = _load_train_rl_module()
    source_contract = SimpleNamespace(obs_dim=source_obs_dim, action_dim=source_action_dim)

    with pytest.raises(ValueError, match=match):
        train_rl._validate_initialization_contract(  # noqa: SLF001
            source_contract=source_contract,
            env_cfg=SimpleNamespace(observation_space=34),
            action_dim=4,
        )


def test_training_adapter_uses_yaml_owned_schedule() -> None:
    from robotics_experiments.rl.training import rsl_rl

    training_cfg = SimpleNamespace(
        num_steps_per_env=32,
        save_interval=25,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.995,
        lam=0.95,
        entropy_coef=0.001,
        clip_param=0.2,
        value_loss_coef=1.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    policy_cfg = SimpleNamespace(
        activation="elu",
        actor=SimpleNamespace(
            hidden_dims=[256, 256, 128],
            initial_noise_std=0.2,
            output_initialization="scratch-neutral-output",
        ),
        critic=SimpleNamespace(hidden_dims=[256, 256, 128]),
    )

    runner_cfg = rsl_rl._train_cfg(  # noqa: SLF001 - adapter mapping contract.
        training_config=training_cfg,
        policy_config=policy_cfg,
    )

    assert runner_cfg["algorithm"]["learning_rate"] == 1.0e-4
    assert runner_cfg["algorithm"]["schedule"] == "fixed"
    assert runner_cfg["policy"]["init_noise_std"] == 0.2
    assert runner_cfg["policy"]["actor_hidden_dims"] == [256, 256, 128]
    assert runner_cfg["policy"]["critic_hidden_dims"] == [256, 256, 128]
    assert runner_cfg["policy"]["activation"] == "elu"
    assert runner_cfg["policy"]["class_name"] == "ActorCritic"
    assert runner_cfg["obs_groups"] == {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    assert runner_cfg["save_interval"] == 25


def test_export_contains_only_the_34_to_4_actor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    actor = torch.nn.Sequential(
        torch.nn.Linear(34, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 4),
    )
    critic = torch.nn.Sequential(
        torch.nn.Linear(34, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 1),
    )
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=SimpleNamespace(actor=actor, critic=critic))
    )

    rsl_rl.export_policy(runner, tmp_path, metadata={"obs_dim": 34})
    exported = torch.jit.load(str(tmp_path / "policy.pt"))

    assert exported(torch.zeros((2, 34))).shape == (2, 4)
    assert critic(torch.zeros((2, 34))).shape == (2, 1)


def test_deterministic_actor_returns_actor_mean_in_evaluation_mode() -> None:
    from robotics_experiments.rl.training import rsl_rl

    class FakeActor:
        def __init__(self) -> None:
            self.evaluating = False

        def eval(self) -> None:
            self.evaluating = True

    actor = FakeActor()
    runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(actor=actor)))

    observed = rsl_rl.deterministic_actor(runner=runner)

    assert observed is actor
    assert actor.evaluating

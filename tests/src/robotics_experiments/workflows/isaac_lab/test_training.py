"""Host tests for training stages, pipeline sequencing, and learner restoration."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from rsl_rl.networks import EmpiricalNormalization

from robotics_experiments.config import RlPlateauCriteriaConfig, RlTrainingPhaseConfig
from robotics_experiments.flight_parameters import load_flight_parameters
from robotics_experiments.rl.evaluation import EvaluationMetric, PolicyEvaluation
from robotics_experiments.rl.press import create_press_acceptance_criteria
from robotics_experiments.rl.training import rsl_rl
from robotics_experiments.rl.training.checkpoint import (
    LearnerProgress,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from robotics_experiments.workflows.isaac_lab import curriculum
from robotics_experiments.workflows.isaac_lab import training as train_rl
from robotics_experiments.workflows.isaac_lab.training import _file_sha256
from tests.support.paths import REPO_ROOT


def _phase(
    *,
    maximum: int = 20,
    interval: int = 10,
) -> RlTrainingPhaseConfig:
    return RlTrainingPhaseConfig(
        max_iterations=maximum,
        evaluation_interval=interval,
        learning_rate=1.0e-4,
        schedule="fixed",
        entropy_coef=0.0,
        update_observation_normalization=True,
    )


def _inputs(tmp_path: Path):
    flight_parameters = load_flight_parameters(REPO_ROOT / "tests/fixtures/flight_parameters")
    return SimpleNamespace(
        controller=SimpleNamespace(
            selected_settings=SimpleNamespace(hover_thrust_normalized=0.5),
            run_id="controller-run",
            comparison_fingerprint="controller-fingerprint",
        ),
        controller_path=tmp_path / "resolved/controller.json",
        controller_digest="controller-digest",
        flight_parameters=flight_parameters,
        flight_parameters_path=tmp_path / "resolved/flight-parameters",
        flight_parameters_digest="flight-parameters-digest",
        hover_reset_samples=SimpleNamespace(summary=lambda **_kwargs: {"sample_count": 1.0}),
    )


def _acceptance_criteria(config):
    return create_press_acceptance_criteria(
        config=config.rl.press.acceptance,
        flight_parameters=load_flight_parameters(REPO_ROOT / "tests/fixtures/flight_parameters"),
    )


@pytest.mark.parametrize("resume", ["never", "must"])
def test_trackio_writer_preserves_metric_names_steps_and_run_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume: str,
) -> None:
    logged: list[tuple[dict[str, float], int]] = []
    finished = []
    initialized = {}
    run = SimpleNamespace(log=lambda metrics, step: logged.append((metrics, step)))

    def init(**kwargs):
        initialized.update(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules,
        "trackio",
        SimpleNamespace(init=init, finish=lambda: finished.append(True)),
    )
    monitoring_dir = tmp_path / "monitoring"
    writer = train_rl._TrackioWriter(  # noqa: SLF001
        monitoring_dir=monitoring_dir,
        project="press-hover",
        run_name="attempt_01",
        campaign_run_id="campaign",
        training_seed=42,
        resume=resume,
        update_offset=100,
    )
    writer.add_scalar("validation/Mean longest continuous hover (seconds)", 4.25, 25)
    writer.flush()
    writer.close()

    assert writer.log_dir == str(monitoring_dir)
    assert initialized == {
        "project": "press-hover",
        "name": "attempt_01",
        "config": {"campaign_run_id": "campaign", "training_seed": 42},
        "resume": resume,
        "embed": False,
        "auto_log_gpu": False,
        "auto_log_cpu": False,
    }
    assert logged == [
        (
            {
                "validation/Mean longest continuous hover (seconds)": 4.25,
                "training/Completed curriculum updates": 125,
            },
            125,
        )
    ]
    assert finished == [True]


def test_procedural_smoke_keeps_every_phase_boundary_at_one_update() -> None:
    phases = train_rl._procedural_smoke_phases(  # noqa: SLF001
        phases=(_phase(maximum=20, interval=10), _phase(maximum=30, interval=5))
    )
    assert tuple(phase.max_iterations for phase in phases) == (1, 1)
    assert tuple(phase.evaluation_interval for phase in phases) == (1, 1)


def test_training_metrics_are_small_and_namespaced() -> None:
    source = (
        REPO_ROOT / "src/robotics_experiments/workflows/isaac_lab/tasks/switch_press_env.py"
    ).read_text()
    assert '"training/Fraction of samples inside hover limits"' in source
    assert '"training/Mean worst pose error relative to hover limit (1 = limit)"' in source
    assert '"training/reward/Total reward per step"' in source
    assert '"train/cumulative_outcome/' not in source
    assert '"train/target/' not in source
    assert "_percent" not in source
    assert "100.0 * self._" not in source


def test_evaluation_log_reports_plateau_only_after_the_decision(capsys, tmp_path: Path) -> None:
    evaluation = PolicyEvaluation(
        iteration=20,
        checkpoint_path=tmp_path / "model_20.pt",
        policy_digest="digest",
        metrics=(
            EvaluationMetric("episode_longest_hover_duration_mean_s", 15.0),
            EvaluationMetric("hover_envelope_utilization_mean", 0.25),
            EvaluationMetric("success_rate", 0.95),
            EvaluationMetric("collision_rate", 0.01),
            EvaluationMetric("other_flight_failure_rate", 0.0),
            EvaluationMetric("timeout_rate", 0.04),
        ),
        reward_means=(),
        source="fresh-validation",
    )
    artifact_dir = tmp_path / "candidate"
    train_rl._report_evaluation(  # noqa: SLF001
        evaluation=evaluation,
        phase_index=0,
        phase_iteration=20,
        plateau_state=train_rl.PlateauState(reference_value=0.3, stale_evaluations=0),
        run_dir=tmp_path,
        artifact_dir=artifact_dir,
        validation=SimpleNamespace(checks=(), passed=False),
    )
    metrics_log = capsys.readouterr().err
    assert "longest_hover_mean_s=15.000" in metrics_log
    assert "progress=" not in metrics_log
    assert "stale=" not in metrics_log

    train_rl._report_plateau(  # noqa: SLF001
        criteria=RlPlateauCriteriaConfig(
            metric_name="hover_envelope_utilization_mean",
            direction="minimize",
            minimum_delta=0.003,
            patience_evaluations=5,
        ),
        evaluation=evaluation,
        made_progress=True,
        plateau_state=train_rl.PlateauState(reference_value=0.25, stale_evaluations=0),
        plateau_reached=False,
    )
    plateau_log = capsys.readouterr().err
    assert (
        "metric=hover_envelope_utilization_mean direction=minimize current=0.250000" in plateau_log
    )
    assert "progress=true reference=0.250000 stale=0 reached=false" in plateau_log


def test_experiment_provenance_records_resolved_configuration_and_inputs(
    *, monkeypatch, tmp_path: Path
) -> None:
    from robotics_experiments.config import load_config

    config_path = REPO_ROOT / "configs/train/stage/press_hover_quad.yaml"
    config = load_config(config_path)
    inputs = _inputs(tmp_path)

    def unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("experiment provenance must not launch subprocesses")

    monkeypatch.setattr(train_rl.subprocess, "run", unexpected_subprocess)
    provenance = train_rl._experiment_provenance(  # noqa: SLF001 - persisted experiment record
        config=config,
        config_path=config_path,
        inputs=inputs,
        procedural_smoke=False,
    )
    recorded = json.loads(json.dumps(provenance, allow_nan=False))
    assert recorded["config_path"] == str(config_path.resolve())
    assert recorded["resolved_config"]["rl"]["training"]["gamma"] == config.rl.training.gamma
    assert recorded["resolved_config"]["rl"]["press"]["goal"]["stable_steps"] == 600
    assert list(recorded["resolved_config"]["vehicles"]) == list(config.vehicles)
    assert recorded["runtime"]["python"] == sys.version
    assert recorded["inputs"] == {
        "controller": {
            "path": str(inputs.controller_path),
            "run_id": "controller-run",
            "sha256": "controller-digest",
            "comparison_fingerprint": "controller-fingerprint",
        },
        "flight_parameters": {
            "path": str(inputs.flight_parameters_path),
            "run_id": inputs.flight_parameters.run_id,
            "sha256": "flight-parameters-digest",
            "fingerprint": inputs.flight_parameters.artifact_fingerprint,
        },
    }


@pytest.mark.parametrize("normalize", [False, True])
def test_checkpoint_restore_loads_optimizer_and_iteration(
    *, tmp_path: Path, normalize: bool
) -> None:
    policy: Any = torch.nn.Module()
    policy.actor = torch.nn.Linear(2, 1)
    policy.critic = torch.nn.Linear(2, 1)
    policy.std = torch.nn.Parameter(torch.tensor([0.2]))
    policy.actor_obs_normalizer = EmpiricalNormalization(2) if normalize else torch.nn.Identity()
    policy.critic_obs_normalizer = EmpiricalNormalization(2) if normalize else torch.nn.Identity()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-4)
    observations = torch.tensor([[0.1, -0.2], [0.3, 0.4]])

    def update_learner() -> None:
        optimizer.zero_grad()
        (
            policy.actor(observations).sum() + policy.critic(observations).sum() + policy.std.sum()
        ).backward()
        optimizer.step()
        if normalize:
            with torch.inference_mode():
                policy.actor_obs_normalizer.update(observations)
                policy.critic_obs_normalizer.update(observations)

    update_learner()
    expected = {key: value.clone() for key, value in policy.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    checkpoint = tmp_path / "model_299.pt"
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer),
        current_learning_iteration=17,
        learner_updates=60,
    )
    save_checkpoint(runner=runner, checkpoint_path=checkpoint, iteration=17, curriculum_updates=100)
    update_learner()
    runner.current_learning_iteration = 99

    restored = restore_checkpoint(runner=runner, checkpoint_path=checkpoint)
    assert restored == LearnerProgress(iteration=17, learner_updates=60, curriculum_updates=100)
    assert runner.learner_updates == 60
    assert runner.current_learning_iteration == 17
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, expected[key])
        assert not value.is_inference()
    restored_optimizer = optimizer.state_dict()
    assert restored_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, state in restored_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, expected_optimizer["state"][parameter_id][name])
    assert all(state["step"].item() == 1 for state in optimizer.state.values())
    update_learner()
    assert all(state["step"].item() == 2 for state in optimizer.state.values())


@pytest.mark.parametrize("normalize", [False, True])
def test_phase_configuration_preserves_optimizer_and_controls_normalization(
    *, normalize: bool
) -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-4)
    policy = torch.nn.Module()
    policy.actor_obs_normalizer = EmpiricalNormalization(2) if normalize else torch.nn.Identity()
    policy.critic_obs_normalizer = EmpiricalNormalization(2) if normalize else torch.nn.Identity()
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            policy=policy,
            optimizer=optimizer,
            learning_rate=1.0e-4,
            entropy_coef=0.001,
            schedule="adaptive",
        )
    )
    phase = _phase()
    original_optimizer = runner.alg.optimizer
    rsl_rl.configure_phase(runner=runner, phase=phase)
    assert runner.alg.optimizer is original_optimizer
    assert runner.alg.optimizer.param_groups[0]["lr"] == phase.learning_rate
    assert runner.alg.schedule == "fixed"
    assert runner.alg.entropy_coef == 0.0
    observations = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    if normalize:
        with torch.inference_mode():
            policy.actor_obs_normalizer.update(observations)
            policy.critic_obs_normalizer.update(observations)
    expected = {key: value.clone() for key, value in policy.state_dict().items()}
    rsl_rl.configure_phase(
        runner=runner, phase=replace(phase, update_observation_normalization=False)
    )
    policy.train()  # RSL-RL does this at every new learning chunk.
    if normalize:
        with torch.inference_mode():
            policy.actor_obs_normalizer.update(observations + 10)
            policy.critic_obs_normalizer.update(observations + 10)
    assert policy.training
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    optimizer.zero_grad()
    parameter.square().backward()
    optimizer.step()
    assert parameter.item() != 1.0
    assert runner.alg.optimizer is original_optimizer
    rsl_rl.configure_phase(runner=runner, phase=phase)
    if normalize:
        with torch.inference_mode():
            policy.actor_obs_normalizer.update(observations + 10)
            policy.critic_obs_normalizer.update(observations + 10)
        assert policy.actor_obs_normalizer.count.item() == 4
        assert policy.critic_obs_normalizer.count.item() == 4


def test_refinement_restores_then_configures_then_resets(monkeypatch, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    calls = []
    runner = object()

    class FakeEnv:
        def reset(self):
            calls.append(("reset", torch.is_inference_mode_enabled()))

    checkpoint = tmp_path / "model.pt"

    def restore_checkpoint(**kwargs):
        calls.append(("restore", kwargs))
        return LearnerProgress(iteration=17, learner_updates=60, curriculum_updates=100)

    monkeypatch.setattr(
        train_rl,
        "restore_checkpoint",
        restore_checkpoint,
    )
    monkeypatch.setattr(
        train_rl.training,
        "configure_phase",
        lambda **kwargs: calls.append(("configure", kwargs)),
    )
    phase = _phase()
    restored = train_rl._restore_phase_branch(  # noqa: SLF001
        runner=runner,
        env=FakeEnv(),
        parent_checkpoint=checkpoint,
        phase=phase,
        phase_number=2,
    )
    assert restored == 17
    assert [call[0] for call in calls] == [
        "restore",
        "configure",
        "reset",
    ]
    assert calls[0][1]["checkpoint_path"] == checkpoint
    assert calls[1][1]["phase"] is phase
    assert calls[1][1]["phase_number"] == 2
    assert calls[-1] == ("reset", True)


def test_training_chunks_do_not_repeat_boundary_iteration_numbers() -> None:
    from robotics_experiments.rl.training import rsl_rl

    starts: list[tuple[int, dict[str, int | bool]]] = []

    runner = SimpleNamespace(current_learning_iteration=0, log_dir="logs")

    def learn(
        *,
        num_learning_iterations: int,
        init_at_random_ep_len: bool,
    ) -> None:
        starts.append(
            (
                runner.current_learning_iteration,
                {
                    "num_learning_iterations": num_learning_iterations,
                    "init_at_random_ep_len": init_at_random_ep_len,
                },
            )
        )

    runner.learn = learn
    first = rsl_rl.train_chunk(
        runner=runner,
        completed_iterations=0,
        iterations=10,
        randomize_initial_episode_lengths=True,
    )
    second = rsl_rl.train_chunk(
        runner=runner,
        completed_iterations=10,
        iterations=5,
        randomize_initial_episode_lengths=False,
    )
    assert starts == [
        (1, {"num_learning_iterations": 10, "init_at_random_ep_len": True}),
        (11, {"num_learning_iterations": 5, "init_at_random_ep_len": False}),
    ]
    assert first == 10
    assert second == 5


@pytest.mark.parametrize("normalize", [False, True])
def test_training_adapter_disables_internal_periodic_checkpoint_cadence(*, normalize) -> None:
    training_cfg = SimpleNamespace(
        num_steps_per_env=32,
        gamma=0.995,
        lam=0.95,
        clip_param=0.2,
        value_loss_coef=1.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    policy_cfg = SimpleNamespace(
        activation="elu",
        normalize_observations=normalize,
        actor=SimpleNamespace(
            hidden_dims=[256, 256, 128],
            initial_noise_std=0.2,
            output_initialization="scratch-neutral-output",
        ),
        critic=SimpleNamespace(hidden_dims=[256, 256, 128]),
    )
    runner_cfg = rsl_rl._train_cfg(  # noqa: SLF001 - adapter mapping contract
        training_config=training_cfg,
        phase=_phase(),
        policy_config=policy_cfg,
    )
    assert runner_cfg["save_interval"] == 2**31 - 1
    assert runner_cfg["algorithm"]["learning_rate"] == 1.0e-4
    assert runner_cfg["policy"]["actor_obs_normalization"] is normalize
    assert runner_cfg["policy"]["critic_obs_normalization"] is normalize


def test_scratch_neutral_output_initialization_is_near_zero_and_unbiased() -> None:
    torch = pytest.importorskip("torch")
    from robotics_experiments.rl.training import rsl_rl

    actor = torch.nn.Sequential(
        torch.nn.Linear(35, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 4),
    )
    runner = SimpleNamespace(alg=SimpleNamespace(policy=SimpleNamespace(actor=actor)))
    rsl_rl.initialize_actor_output(runner=runner, method="scratch-neutral-output")
    output = actor[-1]
    assert torch.count_nonzero(output.bias) == 0
    action_mean = actor(torch.zeros((128, 35)))
    assert float(action_mean.abs().max()) < 0.01


@pytest.mark.parametrize("normalize", [False, True])
def test_export_freezes_actor_input_normalization(*, tmp_path: Path, normalize) -> None:
    actor = torch.nn.Sequential(torch.nn.Linear(35, 4))
    critic = torch.nn.Sequential(torch.nn.Linear(35, 1))
    normalizer = EmpiricalNormalization(35) if normalize else torch.nn.Identity()
    observations = torch.linspace(-0.02, 0.04, 70).reshape(2, 35)
    if normalize:
        normalizer.update(observations)
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            policy=SimpleNamespace(actor=actor, actor_obs_normalizer=normalizer, critic=critic)
        )
    )
    expected = actor(normalizer(observations)).detach()
    metadata = {"obs_dim": 35, "marker": "contract"}
    rsl_rl.export_policy(runner, tmp_path, metadata=metadata)
    module = torch.jit.load(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(module(observations), expected)
    torch.testing.assert_close(rsl_rl.deterministic_actor(runner=runner)(observations), expected)
    if normalize:
        normalizer.update(observations + 10.0)
        torch.testing.assert_close(actor(normalizer(observations)), expected)
        normalizer.train()
        normalizer.update(observations + 10.0)
        torch.testing.assert_close(module(observations), expected)
        assert not torch.allclose(actor(normalizer(observations)), expected)
    assert {path.name for path in tmp_path.iterdir()} == {"policy.pt", "policy_meta.json"}
    assert json.loads((tmp_path / "policy_meta.json").read_text()) == metadata


@pytest.fixture
def saved_learner(monkeypatch, tmp_path):
    """A complete diagnostic learner with an exact exported actor."""
    config = train_rl.load_config(REPO_ROOT / "configs/train/stage/press_hover_quad.yaml")
    inputs = _inputs(tmp_path)
    contract = train_rl.rl.PolicyContract(
        policy_interface=train_rl.rl.PolicyInterface.PRESS,
        obs_schema_version=3,
        obs_dim=2,
        policy_rate_hz=20.0,
        action_kind="velocity",
        action_dim=4,
        max_body_velocity_mps=0.5,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_normalized=0.5,
        curriculum_stage="hover",
    )
    monkeypatch.setattr(train_rl, "press_policy_contract", lambda **_kwargs: contract)
    # Contract compatibility is covered by strict artifact tests; retain real tensor files here.
    monkeypatch.setattr(
        train_rl.rl,
        "validate_policy_artifact",
        lambda **_kwargs: SimpleNamespace(
            contract=contract,
            controller_digest=inputs.controller_digest,
            flight_parameters_digest=inputs.flight_parameters_digest,
        ),
    )
    policy = torch.nn.Module()
    policy.actor = torch.nn.Linear(2, 1)
    policy.critic = torch.nn.Linear(2, 1)
    policy.std = torch.nn.Parameter(torch.tensor([0.1]))
    policy.actor_obs_normalizer = EmpiricalNormalization(2)
    policy.critic_obs_normalizer = EmpiricalNormalization(2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    observations = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    policy.actor_obs_normalizer.update(observations)
    policy.critic_obs_normalizer.update(observations)
    (
        policy.actor(observations).sum() + policy.critic(observations).sum() + policy.std.sum()
    ).backward()
    optimizer.step()
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer), learner_updates=60
    )
    run_dir = tmp_path / "producer"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint.pt"
    save_checkpoint(runner=runner, checkpoint_path=checkpoint, iteration=17, curriculum_updates=100)
    artifact_dir = run_dir / "candidate"
    rsl_rl.export_policy(runner, artifact_dir, metadata={"obs_dim": 2})
    report_path = run_dir / "training_report.json"
    report = {
        "artifact_run_id": run_dir.name,
        "status": "validation_failed",
        "acceptance": None,
        "schema_version": 28,
        "report_kind": "training",
        "progress": {"completed_updates": 100},
        "selected": {
            "checkpoint": "checkpoint.pt",
            "checkpoint_digest": _file_sha256(checkpoint),
            "policy": "candidate/policy.pt",
            "policy_digest": _file_sha256(artifact_dir / "policy.pt"),
            "metadata_digest": _file_sha256(artifact_dir / "policy_meta.json"),
        },
        "provenance": {"resolved_config": train_rl._config_values(config)},  # noqa: SLF001
    }
    report_path.write_text(json.dumps(report))
    return train_rl, config, inputs, report_path, runner


@pytest.mark.parametrize("change", ["bytes", "actor", "normalization", "optimizer", "architecture"])
def test_initial_learner_rejects_corruption_or_incompatibility(saved_learner, change):
    train_rl, config, inputs, report_path, _runner = saved_learner
    learner = train_rl.pin_learner(report_path=report_path)  # noqa: SLF001
    checkpoint = Path(learner.checkpoint)
    if change == "bytes":
        checkpoint.write_bytes(b"changed")
    elif change == "architecture":
        config = replace(
            config,
            rl=replace(
                config.rl,
                press=replace(
                    config.rl.press, policy=replace(config.rl.press.policy, activation="relu")
                ),
            ),
        )
    else:
        payload = torch.load(checkpoint, weights_only=True)
        if change == "optimizer":
            del payload["optimizer_state_dict"]
        else:
            key = "actor.weight" if change == "actor" else "actor_obs_normalizer._mean"
            payload["model_state_dict"][key] += 1
        torch.save(payload, checkpoint)
        learner = replace(learner, checkpoint_digest=_file_sha256(checkpoint))  # noqa: SLF001
    with pytest.raises(
        ValueError, match="changed|incompatible|complete|does not match|differs|digest"
    ):
        train_rl.validate_learner(learner=learner, config=config, inputs=inputs)  # noqa: SLF001


@pytest.mark.parametrize("status", ["validation_failed", "test_failed", "interrupted", "training"])
def test_learner_compatibility_does_not_require_acceptance(saved_learner, status):
    module, config, inputs, path, _runner = saved_learner
    report = json.loads(path.read_text())
    report["status"] = status
    path.write_text(json.dumps(report))
    learner = module.pin_learner(report_path=path)
    module.validate_learner(learner=learner, config=config, inputs=inputs)


@pytest.mark.parametrize(
    "outcome", ["complete", "rejected", "error", "interrupt", "publication_error"]
)
@pytest.mark.parametrize("initialized", [False, True])
def test_curriculum_forwards_exact_learner_and_stops(
    saved_learner, monkeypatch, tmp_path, outcome, initialized
):
    from robotics_experiments.config import CurriculumConfig, CurriculumStageConfig

    module, config, inputs, producer_path, _runner = saved_learner
    learner = module.pin_learner(report_path=producer_path)
    root_config = CurriculumConfig(
        policy_name="one-namespace",
        initial_learner=str(producer_path) if initialized else None,
        stages=[
            CurriculumStageConfig(
                name=name, config_path="configs/train/stage/press_hover_quad.yaml"
            )
            for name in ("banana", "landing", "third")
        ],
    )
    prepared = []
    monkeypatch.setattr(curriculum, "load_config", lambda _path: config)
    monkeypatch.setattr(
        curriculum, "prepare_policy_inputs", lambda **kwargs: prepared.append(kwargs) or inputs
    )
    calls = []
    publications = []

    def train(**kwargs):
        assert len(prepared) == 3
        assert kwargs["config"].rl.press.task == "hover"
        assert kwargs["inputs"] is inputs
        assert kwargs["initial_learner"] == (learner if initialized or calls else None)
        assert kwargs["update_offset"] == 100 * len(calls)
        calls.append(kwargs)
        if len(calls) == 2 and outcome in {"error", "interrupt"}:
            raise KeyboardInterrupt if outcome == "interrupt" else RuntimeError("worker failed")
        return module.TrainingResult(
            selected_checkpoint=learner,
            accepted=not (len(calls) == 2 and outcome == "rejected"),
            completed_updates=100,
            report_path=producer_path,
        )

    def publish(**kwargs):
        publications.append(kwargs)
        if len(calls) == 2 and outcome == "publication_error":
            raise RuntimeError("publication failed")

    monkeypatch.setattr(curriculum, "run_training", train)
    monkeypatch.setattr(curriculum, "publish_training", publish)
    # Config paths must exist, while generated runs stay isolated in this test root.
    (tmp_path / "configs/train/stage").mkdir(parents=True)
    (tmp_path / "configs/train/stage/press_hover_quad.yaml").write_text("{}")
    kwargs = dict(
        config=root_config,
        config_path=producer_path,
        repo_root=tmp_path,
        run_id="curriculum",
        procedural_smoke=False,
    )
    if outcome in {"error", "interrupt", "publication_error"}:
        with pytest.raises(KeyboardInterrupt if outcome == "interrupt" else RuntimeError):
            curriculum.run_curriculum(**kwargs)
    else:
        result = curriculum.run_curriculum(**kwargs)
        assert result.complete is (outcome == "complete")
    report = json.loads(
        (tmp_path / ".build/training/one-namespace/curriculum/curriculum_report.json").read_text()
    )
    assert (
        report["status"]
        == {
            "complete": "complete",
            "rejected": "stopped",
            "error": "failed",
            "interrupt": "interrupted",
            "publication_error": "failed",
        }[outcome]
    )
    assert len(calls) == (3 if outcome == "complete" else 2)
    assert [call["run_dir"].name for call in calls] == [
        f"curriculum.stage_{i+1:02d}" for i in range(len(calls))
    ]
    assert report["stages"][0]["published"]
    if outcome == "publication_error":
        assert report["stages"][1]["accepted"] and not report["stages"][1]["published"]
    for i, publication in enumerate(publications):
        assert [stage["name"] for stage in publication["passed_stages"]] == [
            "banana",
            "landing",
            "third",
        ][: i + 1]
        assert all(
            stage["report_digest"] == learner.report_digest
            for stage in publication["passed_stages"]
        )


@pytest.mark.parametrize(
    "outcome",
    [
        "accepted",
        "validation_failed",
        "test_failed",
        "interrupted",
        "error",
        "missing",
        "refinement",
    ],
)
def test_training_campaign_selection_and_failures(monkeypatch, tmp_path, outcome):
    from robotics_experiments.config import load_config
    from robotics_experiments.workflows.isaac_lab.training import _write_json

    config_path = REPO_ROOT / "configs/train/stage/press_approach_quad.yaml"
    original = load_config(config_path)
    config = replace(
        original,
        rl=replace(original.rl, training=replace(original.rl.training, attempt_seeds=(42, 43, 44))),
    )
    run_dir = tmp_path / "policy/stage"
    commands = []

    class Worker:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.index = int(command[command.index("--attempt-index") + 1])
            self.waited = False
            assert "--attempt-seed" not in command
            assert kwargs["env"]["RL_CAMPAIGN_RUN_DIR"] == str(run_dir)
            if outcome == "missing":
                return
            seed = config.rl.training.attempt_seeds[self.index]
            directory = run_dir / f"attempt_{self.index + 1:02d}_seed_{seed}"
            directory.mkdir()
            checkpoint = directory / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            (directory / "policy.pt").write_bytes(f"actor-{seed}".encode())
            (directory / "policy_meta.json").write_text("{}")
            passed = self.index == 1 and outcome != "validation_failed"
            record = {
                "iteration": 60,
                "checkpoint": str(checkpoint.relative_to(run_dir)),
                "policy": str((directory / "policy.pt").relative_to(run_dir)),
                "policy_digest": _file_sha256(directory / "policy.pt"),
                "metrics": {"hover_envelope_utilization_mean": float(3 - self.index)},
                "reward_means": {},
                "source": "fresh-validation",
                "validation_passed": passed,
                "checks": [],
                "phase_index": 0 if outcome == "refinement" else 2,
            }
            report = {
                "schema_version": 28,
                "report_kind": "attempt",
                "artifact_run_id": run_dir.name,
                "attempt_index": self.index,
                "training_seed": seed,
                "worker_pid": 999000 + self.index,
                "evaluations": [record],
                "phase_results": [],
                "selected_evaluation_index": 0,
                "progress": {"completed_updates": 100, "update_offset": 50 + self.index * 100},
                "status": "validation_passed" if passed else "validation_failed",
            }
            _write_json(path=directory / "attempt_report.json", payload=report)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def wait(self, **_kwargs):
            if not self.waited and self.index == 1:
                self.waited = True
                if outcome == "interrupted":
                    raise KeyboardInterrupt
                if outcome == "error":
                    return 1
            return 0

        def terminate(self):
            pass

    monkeypatch.setattr(train_rl.subprocess, "Popen", Worker)
    monkeypatch.setattr(train_rl, "probe_telemetry", lambda: {})
    monkeypatch.setattr(train_rl, "press_policy_contract", lambda **_kwargs: None)
    monkeypatch.setattr(train_rl, "_press_comparison_fingerprint", lambda **_kwargs: "fingerprint")
    tested = []

    def test(**kwargs):
        tested.append(kwargs)
        evaluation = replace(kwargs["selection"].evaluation, source="fresh-test")
        return evaluation, {"status": "complete", "summary": {"gpu_utilization_rate_mean": 0.5}}

    monkeypatch.setattr(train_rl, "_run_final_test", test)
    monkeypatch.setattr(
        train_rl,
        "evaluate_press_acceptance",
        lambda **_kwargs: train_rl.AcceptanceResult(
            passed=outcome != "test_failed", checks=(), reasons=()
        ),
    )
    monkeypatch.setattr(
        train_rl, "_TrackioWriter", lambda **_kwargs: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(train_rl, "write_evaluation_scalars", lambda **_kwargs: None)
    monkeypatch.setattr(train_rl, "write_system_scalars", lambda **_kwargs: None)
    kwargs = dict(
        config=config,
        config_path=config_path,
        inputs=_inputs(tmp_path),
        initial_learner=None,
        run_dir=run_dir,
        monitoring_dir=tmp_path / "monitoring",
        update_offset=50,
        procedural_smoke=False,
    )
    result = None
    if outcome in {"interrupted", "error", "missing"}:
        with pytest.raises(KeyboardInterrupt if outcome == "interrupted" else RuntimeError):
            train_rl.run_training(**kwargs)
    else:
        result = train_rl.run_training(**kwargs)
    report = json.loads((run_dir / "training_report.json").read_text())
    assert "accepted" not in report and "reward_configuration" not in report
    assert all(isinstance(reference, str) for reference in report["attempts"])
    assert report["status"] == {
        "refinement": "validation_failed",
        "error": "failed",
        "missing": "failed",
    }.get(outcome, outcome)
    assert bool(report["selected"]) is (outcome != "missing")
    if result:
        assert result.accepted is (outcome == "accepted")
        assert result.selected_checkpoint is not None
        assert result.completed_updates == 100 * len(commands)
    assert bool(tested) is (outcome in {"accepted", "test_failed"})
    if tested:
        assert tested[0]["selection"].attempt_index == 1
        assert report["test_evaluation"]["policy_digest"] == report["selected"]["policy_digest"]


def test_checkpoint_rng_is_safe_evidence_and_does_not_replace_new_attempt_seed(saved_learner):
    import random

    import numpy as np

    _module, _config, _inputs_value, path, runner = saved_learner
    checkpoint = path.parent / "checkpoint.pt"
    state = load_checkpoint(checkpoint_path=checkpoint)
    assert set(state) == {
        "schema_version",
        "model_state_dict",
        "optimizer_state_dict",
        "progress",
        "rng_state",
    }
    random.seed(87)
    np.random.seed(87)
    torch.manual_seed(87)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(87)
    np.random.seed(87)
    torch.manual_seed(87)
    restore_checkpoint(runner=runner, checkpoint_path=checkpoint)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    rsl_rl.configure_phase(
        runner=runner, phase=replace(_phase(), learning_rate=2e-5, entropy_coef=0.02)
    )
    assert runner.alg.learning_rate == 2e-5
    assert runner.alg.entropy_coef == 0.02
    assert runner.alg.optimizer.param_groups[0]["lr"] == 2e-5
    assert runner.learner_updates == 60


def test_publication_keeps_report_and_actor_bytes_and_retains_comparable_best(
    saved_learner, tmp_path
):
    import shutil

    module, _config, _inputs_value, report_path, _runner = saved_learner
    report = json.loads(report_path.read_text())
    report.update(
        status="accepted",
        acceptance={"passed": True, "checks": []},
        comparison_fingerprint="same-comparison",
        attempts=["attempt.json"],
    )
    report["selected"].update(
        attempt_index=0,
        evaluation_index=0,
        metric={"name": "hover_envelope_utilization_mean", "direction": "minimize", "value": 1.0},
    )
    report["test_evaluation"] = {
        "policy_digest": report["selected"]["policy_digest"],
        "source": "fresh-test",
        "metrics": {"success_rate": 1.0},
        "reward_means": {},
    }
    (report_path.parent / "attempt.json").write_text(
        json.dumps(
            {
                "evaluations": [
                    {"iteration": 17, "metrics": {"success_rate": 1.0}, "reward_means": {}}
                ]
            }
        )
    )
    report_path.write_text(json.dumps(report))
    learner = module.pin_learner(report_path=report_path)
    passed = [
        {
            "name": "stabilize",
            "report_path": str(report_path),
            "report_digest": learner.report_digest,
            "policy_digest": learner.policy_digest,
        }
    ]
    module.publish_training(
        result=module.TrainingResult(
            selected_checkpoint=learner,
            accepted=True,
            completed_updates=100,
            report_path=report_path,
        ),
        curriculum_run_id="curriculum",
        passed_stages=passed,
    )
    assert _file_sha256(report_path) == learner.report_digest
    assert _file_sha256(report_path.parent / "policy.pt") == learner.policy_digest
    assert (tmp_path / "latest").resolve() == report_path.parent
    assert (tmp_path / "best").resolve() == report_path.parent
    second = tmp_path / "second"
    shutil.copytree(report_path.parent, second)
    report["artifact_run_id"] = second.name
    report["selected"]["metric"]["value"] = 2.0
    second_report = second / "training_report.json"
    second_report.write_text(json.dumps(report))
    second_learner = module.pin_learner(report_path=second_report)
    passed.append(
        {
            "name": "reduce-corrections",
            "report_path": str(second_report),
            "report_digest": second_learner.report_digest,
            "policy_digest": second_learner.policy_digest,
        }
    )
    module.publish_training(
        result=module.TrainingResult(
            selected_checkpoint=second_learner,
            accepted=True,
            completed_updates=100,
            report_path=second_report,
        ),
        curriculum_run_id="curriculum",
        passed_stages=passed,
    )
    assert (tmp_path / "latest").resolve() == second
    assert (tmp_path / "best").resolve() == report_path.parent
    assert _file_sha256(second_report) == second_learner.report_digest
    assert json.loads((second / "policy_meta.json").read_text())["curriculum"] == {
        "run_id": "curriculum",
        "passed_stages": passed,
    }


@pytest.mark.parametrize(
    "field", ["model_state_dict", "optimizer_state_dict", "progress", "rng_state"]
)
def test_incomplete_checkpoint_fails_without_partial_fallback(saved_learner, field):
    _module, _config, _inputs_value, report_path, _runner = saved_learner
    path = report_path.parent / "checkpoint.pt"
    payload = load_checkpoint(checkpoint_path=path)
    del payload[field]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incomplete"):
        load_checkpoint(checkpoint_path=path)


def test_checkpoint_rejects_missing_or_wrong_optimizer_moments(saved_learner):
    _module, _config, _inputs_value, report_path, runner = saved_learner
    path = report_path.parent / "checkpoint.pt"
    payload = load_checkpoint(checkpoint_path=path)
    state = next(iter(payload["optimizer_state_dict"]["state"].values()))
    state["exp_avg"] = torch.zeros(19)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer moment shape"):
        restore_checkpoint(runner=runner, checkpoint_path=path)
    del state["step"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="moments or step"):
        load_checkpoint(checkpoint_path=path)


def test_cleanup_preserves_primary_exception():
    from robotics_experiments.workflows.isaac_lab.training import _close_resource

    def fail():
        raise RuntimeError("cleanup")

    with pytest.raises(ValueError, match="primary"):
        try:
            raise ValueError("primary")
        finally:
            _close_resource(SimpleNamespace(close=fail))
    with pytest.raises(RuntimeError, match="cleanup"):
        _close_resource(SimpleNamespace(close=fail))

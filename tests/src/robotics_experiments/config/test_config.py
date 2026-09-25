import json
import shutil
from dataclasses import FrozenInstanceError, replace
from typing import get_type_hints

import pytest
from omegaconf import OmegaConf

from robotics_experiments.config import load_config
from tests.support.paths import REPO_ROOT

_ROOT = REPO_ROOT / "configs"
_EXPERIMENTS = sorted(path for path in _ROOT.glob("*.yaml") if path.name != "base.yaml") + sorted(
    (_ROOT / "train/stage").glob("*.yaml")
)


def _write_experiment(tmp_path, name, vehicles):
    root = tmp_path / "configs"
    shutil.copytree(_ROOT / "experiment", root / "experiment", dirs_exist_ok=True)
    shutil.copytree(_ROOT / "airframes", root / "airframes", dirs_exist_ok=True)
    shutil.copy2(_ROOT / "base.yaml", root / "base.yaml")
    defaults = ["base"]
    defaults.extend(
        {f"experiment/vehicle@vehicles.{label}": "holybro_x500_v2"} for label in vehicles
    )
    defaults.append("_self_")
    path = root / f"{name}.yaml"
    OmegaConf.save(
        {
            "defaults": defaults,
            "scene": "@isaac/assets/TestScenes/MinimalIntegration.usda",
            "vehicles": vehicles,
        },
        path,
    )
    return path


def _vehicle(vehicle_id, namespace, x, *, prim_name=None, **extra):
    spawn = {"position_xy": [x, 0.0]}
    if prim_name is not None:
        spawn["prim_name"] = prim_name
    return {
        "id": vehicle_id,
        "ros": {"namespace": namespace},
        "spawn": spawn,
        **extra,
    }


def test_load_config_is_isolated_and_immutable(monkeypatch):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from robotics_experiments.config import load_config

    with initialize_config_dir(config_dir=str(_ROOT), version_base=None):
        foreign_hydra = GlobalHydra.instance().hydra
        config = load_config(_ROOT / "takeoff_quad.yaml")
        assert GlobalHydra.instance().hydra is foreign_hydra

    vehicle = config.vehicles["x500_01"]
    with pytest.raises(FrozenInstanceError):
        config.scene = "different.usda"
    with pytest.raises(FrozenInstanceError):
        vehicle.bridge.lockstep.lockstep_timeout_s = 99.0
    with pytest.raises(TypeError):
        vehicle.routine.steps[0]["kind"] = "land"
    with pytest.raises(TypeError):
        config.vehicles["other"] = vehicle


def test_load_config_caches_by_explicit_path():
    from robotics_experiments.config import load_config

    path = _ROOT / "takeoff_quad.yaml"
    assert load_config(path) is load_config(path)


def test_load_config_rejects_missing_path(tmp_path):
    from robotics_experiments.config import load_config

    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_config(missing)


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda path: path.stem)
def test_every_shipped_experiment_composes(path):
    from robotics_experiments.config import ExperimentConfig, load_config

    config = load_config(path)
    assert isinstance(config, ExperimentConfig)
    assert config.scene
    assert config.vehicles


def test_default_experiment_has_only_global_root_config():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "takeoff_quad.yaml")
    assert config.switches.spawn_seed == 0
    assert config.rl.training.attempt_seeds == (42,)
    assert set(vars(config)) - {"_config_frozen"} == {
        "sim",
        "switches",
        "rl",
        "scene",
        "vehicles",
        "driver",
        "flight_benchmark",
        "alignment",
    }
    assert config.flight_benchmark.airborne_duration_s == 5.0
    assert config.alignment.capture.sensor_seed == 42
    assert config.alignment.capture.pusher_deployed is True
    assert config.alignment.capture.minimum_neutral_s == 3.0
    assert config.alignment.capture.max_linear_speed_mps == 0.02
    assert config.alignment.capture.max_angular_speed_radps == 0.03
    assert config.alignment.execution.worker_count == 1
    assert config.alignment.execution.candidate_capacity == 128
    assert config.alignment.execution.candidate_capacity_benchmark == (128, 256, 384, 512)
    assert config.alignment.execution.gpu_target_percent == 95.0
    vehicle = config.vehicles["x500_01"]
    assert vehicle.spawn.position_xy == (0.0, 0.0)
    assert vehicle.spawn.usd_path.endswith("/holybro_x500_v2.usda")
    assert vehicle.bridge.rotor_constant
    assert vehicle.px4.failsafe.offboard_loss_action == "land"
    assert vehicle.px4.failsafe.offboard_loss_action_param == 4
    assert vehicle.px4.failsafe.offboard_loss_timeout_s == 1.0
    assert vehicle.px4.logging_profile == 1
    assert vehicle.routine.failure_recovery_timeout_s == 30.0
    assert [step["kind"] for step in vehicle.routine.steps] == ["arm", "takeoff", "hover"]
    assert config.rl.training.vehicle_id == 0
    assert config.rl.policy_rate_hz == 20.0
    assert config.driver.control_rate_hz == 50.0
    assert config.driver.offboard_setpoint_rate_hz == 50.0


def test_alignment_experiment_uses_deterministic_sensor_excitation():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "alignment_px4_x500_velocity.yaml")
    bridge = config.vehicles["x500_01"].bridge

    assert config.sim.enhanced_determinism
    assert not bridge.sensor_noise_enabled
    assert bridge.sensor_excitation_mode == "deterministic"
    assert bridge.barometer_noise_std_pa == 0.1
    assert bridge.magnetometer_noise_scale == 0.1
    assert config.alignment.capture.sensor_seed == 42
    assert config.vehicles["x500_01"].px4.logging_profile == 17


def test_bridge_rejects_unknown_sensor_excitation_mode(tmp_path):
    from robotics_experiments.config import load_config

    path = _write_experiment(
        tmp_path,
        "invalid_sensor_excitation",
        {"x500": _vehicle(0, "/x500", 0.0)},
    )
    px4 = path.parent / "experiment" / "px4" / "default.yaml"
    px4.write_text(
        px4.read_text().replace(
            "sensor_excitation_mode: stochastic",
            "sensor_excitation_mode: random",
        )
    )

    with pytest.raises(
        ValueError,
        match="sensor_excitation_mode must be 'stochastic' or 'deterministic'",
    ):
        load_config(path)


def test_alignment_capture_rejects_seed_outside_numpy_range(tmp_path):
    from robotics_experiments.config import load_config

    path = _write_experiment(tmp_path, "invalid_alignment_seed", {})
    base = path.parent / "base.yaml"
    base.write_text(base.read_text().replace("sensor_seed: 42", "sensor_seed: -1"))

    with pytest.raises(ValueError, match="32-bit unsigned integer"):
        load_config(path)


def test_bridge_rejects_negative_barometer_noise(tmp_path):
    from robotics_experiments.config import load_config

    path = _write_experiment(
        tmp_path,
        "invalid_barometer_noise",
        {"x500": _vehicle(0, "/x500", 0.0)},
    )
    px4 = path.parent / "experiment" / "px4" / "default.yaml"
    px4.write_text(
        px4.read_text().replace("barometer_noise_std_pa: 1.0", "barometer_noise_std_pa: -1.0")
    )

    with pytest.raises(ValueError, match="barometer_noise_std_pa must be finite and non-negative"):
        load_config(path)


def test_bridge_rejects_negative_magnetometer_noise_scale(tmp_path):
    from robotics_experiments.config import load_config

    path = _write_experiment(
        tmp_path,
        "invalid_magnetometer_noise",
        {"x500": _vehicle(0, "/x500", 0.0)},
    )
    px4 = path.parent / "experiment" / "px4" / "default.yaml"
    px4.write_text(
        px4.read_text().replace(
            "magnetometer_noise_scale: 1.0",
            "magnetometer_noise_scale: -1.0",
        )
    )

    with pytest.raises(
        ValueError,
        match="magnetometer_noise_scale must be finite and non-negative",
    ):
        load_config(path)


@pytest.mark.parametrize("stage_name", ["hover", "approach"])
def test_press_episode_duration_comes_from_shared_field(*, tmp_path, stage_name: str):
    root = tmp_path / "configs"
    shutil.copytree(_ROOT, root)
    path = root / f"train/stage/press_{stage_name}_quad.yaml"
    original = load_config(_ROOT / "train/stage" / path.name)
    duration = original.rl.episode_length_s
    experiment_path = path
    experiment_path.write_text(
        experiment_path.read_text().replace(
            f"episode_length_s: {duration}", "episode_length_s: 50.0"
        )
    )

    config = load_config(path)
    assert config.rl.episode_length_s == 50.0
    assert config.rl.training.attempt_seeds == (42,)
    assert config.rl.press.evaluation.validation_seeds == (101, 202, 303)
    assert config.rl.press.evaluation.test_seeds == (404, 505, 606)
    assert config.rl.press.evaluation.episodes_per_seed == 100


def test_press_approach_config_preserves_actor_contract_and_stage_settings():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "train/stage/press_approach_quad.yaml")
    training = config.rl.training
    policy = config.rl.press.policy

    assert config.rl.action.action_dim == 4
    assert config.rl.action.max_body_velocity_mps == 0.5
    assert not config.rl.lab_vehicle.enable_gyroscopic_forces
    assert config.rl.inputs.controller_artifact == ".build/alignment/x500-px4-v1/best"
    assert config.rl.inputs.flight_parameters_artifact.endswith("/latest")
    assert config.rl.press.reset.physical_contact_enabled
    assert config.rl.num_envs == 32768
    assert config.rl.press.task == "approach"
    assert config.rl.episode_length_s == 45.0
    assert tuple(source.source for source in config.rl.press.reset.mix) == (
        "measured_navigation_handoff",
        "pinned_flight_parameters",
    )
    assert tuple(source.weight for source in config.rl.press.reset.mix) == (0.8, 0.2)
    assert config.rl.press.goal.stable_steps / config.rl.policy_rate_hz == 30.0
    assert tuple(phase.max_iterations for phase in training.phases) == (50, 30, 100)
    assert tuple(phase.learning_rate for phase in training.phases) == (
        1.0e-4,
        3.0e-5,
        1.0e-5,
    )
    assert tuple(phase.entropy_coef for phase in training.phases) == (0.0, 0.0, 0.0)
    assert tuple(phase.evaluation_interval for phase in training.phases) == (10, 5, 5)
    assert training.plateau.metric_name == "hover_envelope_utilization_mean"
    assert training.plateau.direction == "minimize"
    assert training.plateau.minimum_delta == 0.003
    assert training.plateau.patience_evaluations == 10
    hover_training = load_config(_ROOT / "train/stage/press_hover_quad.yaml").rl.training
    assert tuple(phase.max_iterations for phase in hover_training.phases) == (50, 100)
    assert tuple(phase.learning_rate for phase in hover_training.phases) == (3.0e-4, 1.0e-5)
    assert tuple(phase.update_observation_normalization for phase in hover_training.phases) == (
        True,
        False,
    )
    assert tuple(phase.evaluation_interval for phase in hover_training.phases) == (5, 5)
    assert hover_training.plateau.metric_name == "hover_envelope_utilization_mean"
    assert hover_training.plateau.direction == "minimize"
    assert hover_training.plateau.minimum_delta == 0.003
    assert hover_training.plateau.patience_evaluations == 5
    assert training.telemetry_sample_interval_s == 1.0
    assert training.gpu_utilization_target_rate == 0.95
    assert training.attempt_seeds == (42,)
    assert policy.actor.initial_noise_std == 0.03
    assert policy.actor.output_initialization == "scratch-neutral-output"
    assert policy.actor.hidden_dims == (256, 256, 128)
    assert policy.critic.hidden_dims == (256, 256, 128)


def test_press_hover_training_configuration():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "train/stage/press_hover_quad.yaml")
    approach = load_config(_ROOT / "train/stage/press_approach_quad.yaml")
    assert approach.rl.press.reward.stabilization_weight == 0.01
    assert approach.rl.press.reward.outside_envelope_weight == 0.03
    assert approach.rl.press.reward.action_effort_weight == 1.0
    assert config.rl.press.acceptance == approach.rl.press.acceptance
    assert config.rl.press.evaluation == approach.rl.press.evaluation
    policy = config.rl.press.policy
    reward = config.rl.press.reward
    assert config.rl.training.num_learning_epochs == 4
    assert policy.normalize_observations is True
    assert config.rl.num_envs == 4096
    assert config.rl.training.num_steps_per_env == 256
    assert config.rl.training.num_mini_batches == 8
    assert config.rl.training.attempt_seeds == (42,)
    assert config.rl.episode_length_s == 40.0
    assert policy.activation == "elu"
    assert reward.position_progress_weight == 0.1
    assert reward.hover_stabilization == "boundary_weighted"
    assert reward.stabilization_weight == 0.03
    assert reward.outside_envelope_weight == 0.09
    assert reward.action_effort_weight == 1.0
    assert reward.predictive_barrier_weight == 0.1
    assert reward.hold_progress_weight == 0.01
    assert reward.terminal_success_weight == 1.5
    assert reward.terminal_failure_weight == 30.0
    assert config.rl.press.acceptance.minimum_range_m == 0.40
    assert config.rl.press.goal.stable_steps == 600
    assert config.rl.press.goal.alignment_error_max_m == 0.01
    assert config.rl.press.goal.relative_attitude_error_max_rad == pytest.approx(
        0.017453292519943295
    )
    assert config.rl.press.goal.linear_speed_max_mps == 0.03
    assert config.rl.press.goal.angular_speed_max_radps == 0.05
    reference = config.rl.press.goal.velocity_reference
    assert reference.range_center_m == 0.48
    assert reference.wall_normal_prediction_horizon_s == 0.4
    assert reference.wall_normal_alignment_gate_m == 0.03
    assert reference.wall_normal_gain_per_s == 1.25
    assert reference.wall_normal_min_mps == -0.04
    assert reference.wall_normal_max_mps == 0.04
    assert reference.tangential_gain_per_s == 2.0
    assert reference.tangential_speed_max_mps == 0.10
    assert config.rl.press.handoff.switch_range_min_m == 0.44
    assert config.rl.press.handoff.switch_range_max_m == 0.67
    assert config.rl.press.handoff.lateral_align_max_m == 0.08
    assert config.rl.press.handoff.wall_normal_speed_max_mps == 0.02
    assert config.rl.press.handoff.lateral_speed_max_mps == 0.02
    assert config.rl.press.handoff.angular_speed_max_radps == 0.05
    assert config.rl.press.handoff.relative_attitude_error_max_rad == pytest.approx(
        0.020702318561895833
    )
    assert config.rl.press.handoff.body_z_alignment_min == 0.965926
    navigation_noise = config.rl.press.navigation_noise
    assert navigation_noise.source == (
        "benchmarks/press_navigation_handoff/waypoint_settle_20260812.json"
    )
    assert navigation_noise.sample_count == 10
    assert navigation_noise.position_margin_m == 0.005
    assert navigation_noise.rotation_margin_rad == pytest.approx(0.004363323129985824)
    assert navigation_noise.switch_range_min_m == pytest.approx(0.46459411621093605)
    assert navigation_noise.switch_range_max_m == pytest.approx(0.4989912414550766)
    assert navigation_noise.lateral_align_max_m == pytest.approx(0.03374635443902746)
    assert navigation_noise.relative_attitude_error_max_rad == pytest.approx(0.01633899543191001)
    assert not hasattr(config.rl.press.reset, "initial_linear_velocity_max_mps")
    assert not hasattr(config.rl.press.reset, "initial_angular_velocity_max_radps")
    assert config.rl.press.acceptance.success_rate_min == 0.95
    assert config.rl.press.acceptance.collision_rate_max == 0.0
    assert config.rl.press.evaluation.validation_seeds == (101, 202, 303)
    assert config.rl.press.evaluation.test_seeds == (404, 505, 606)
    assert config.rl.press.evaluation.episodes_per_seed == 100
    assert config.rl.hover.policy.actor.hidden_dims == (128, 128)


def test_press_hover_outside_only_changes_only_reward_choice():
    from robotics_experiments.config import load_config

    baseline = load_config(_ROOT / "train/stage/press_hover_quad.yaml")
    comparison = load_config(_ROOT / "train/stage/press_hover_outside_only_quad.yaml")
    assert comparison.rl.press.reward.hover_stabilization == "outside_only"
    assert replace(comparison.rl.press.reward, hover_stabilization="boundary_weighted") == (
        baseline.rl.press.reward
    )
    assert comparison.rl.press.task == baseline.rl.press.task
    assert comparison.rl.training == baseline.rl.training


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "hover_stabilization: boundary_weighted",
            "hover_stabilization: unknown",
            "hover_stabilization",
        ),
        (
            "outside_envelope_weight: 0.03",
            "outside_envelope_weight: 0.005",
            "outside_envelope_weight",
        ),
    ],
)
def test_press_rejects_invalid_hover_reward(tmp_path, old, new, message):
    from robotics_experiments.config import load_config

    path = _write_experiment(
        tmp_path,
        "invalid_hover_reward",
        {"x500": _vehicle(0, "/x500", 0.0)},
    )
    reward_config = path.parent / "experiment" / "rl" / "press" / "approach.yaml"
    reward_config.write_text(reward_config.read_text().replace(old, new))

    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("relative_path", "old", "new", "message"),
    [
        (
            "train/stage/press_approach_quad.yaml",
            "attempt_seeds: [42]",
            "attempt_seeds: [42, 42]",
            "attempt_seeds must be nonempty and unique",
        ),
        (
            "experiment/rl/press/approach.yaml",
            "validation_seeds: [101, 202, 303]",
            "validation_seeds: [42, 202, 303]",
            "mutually disjoint",
        ),
        (
            "experiment/rl/press/approach.yaml",
            "test_seeds: [404, 505, 606]",
            "test_seeds: [404, 505]",
            "exactly three unique seeds",
        ),
    ],
)
def test_press_campaign_rejects_invalid_seed_workloads(
    tmp_path,
    relative_path: str,
    old: str,
    new: str,
    message: str,
):
    from robotics_experiments.config import load_config

    root = tmp_path / "configs"
    shutil.copytree(_ROOT, root)
    path = root / relative_path
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match=message):
        load_config(root / "train/stage/press_approach_quad.yaml")


def test_training_and_manual_press_configs_share_one_actor_entry_contract():
    from robotics_experiments.config import load_config
    from robotics_experiments.rl.press import HandoffGate

    training = load_config(_ROOT / "train/stage/press_approach_quad.yaml")
    deployment = load_config(_ROOT / "policy_press_approach_quad.yaml")

    assert HandoffGate.from_config(training.rl.press.handoff) == HandoffGate.from_config(
        deployment.rl.press.handoff
    )
    assert deployment.vehicles["x500_01"].routine.takeoff_timeout_s == 20.0
    steps = deployment.vehicles["x500_01"].routine.steps
    assert tuple(step["kind"] for step in steps) == (
        "arm",
        "takeoff",
        "hover",
        "goto",
        "hover",
        "goto",
        "hover",
        "policy",
        "goto",
        "land",
    )
    assert steps[3]["y"] == 17.24
    assert steps[3]["ground_speed_m_s"] == 0.5
    assert steps[3]["position_tol_m"] == 0.05
    assert steps[5]["y"] == 17.39
    assert steps[5]["ground_speed_m_s"] == 0.08
    assert steps[5]["position_tol_m"] == 0.06
    assert steps[5]["timeout_s"] == 20.0
    assert steps[6]["hold_s"] == 2.0
    switch_y = deployment.switches.spawn_bounds[0].min[1]
    staged_range_m = switch_y - steps[5]["y"]
    assert staged_range_m == pytest.approx(0.60)
    navigation_noise = deployment.rl.press.navigation_noise
    assert navigation_noise.sample_count >= 10
    assert navigation_noise.position_margin_m > 0.0
    assert (
        deployment.rl.press.handoff.switch_range_min_m
        <= navigation_noise.switch_range_min_m - navigation_noise.position_margin_m
    )
    assert (
        deployment.rl.press.handoff.switch_range_max_m
        >= navigation_noise.switch_range_max_m + navigation_noise.position_margin_m
    )
    assert (
        deployment.rl.press.handoff.lateral_align_max_m
        >= navigation_noise.lateral_align_max_m + navigation_noise.position_margin_m
    )
    assert steps[8]["ground_speed_m_s"] == 0.5
    assert "ground_speed_m_s" not in steps[2]
    assert all(float(step.get("timeout_s", 20.0)) <= 45.0 for step in steps)


def test_press_navigation_noise_config_matches_accepted_measurement_summary():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "policy_press_approach_quad.yaml")
    noise = config.rl.press.navigation_noise
    summary = json.loads((REPO_ROOT / noise.source).read_text(encoding="utf-8"))
    observed = summary["observed"]
    required = summary["required_handoff_and_training_envelope"]

    assert summary["sample_count"] == noise.sample_count
    assert summary["position_margin_m"] == noise.position_margin_m
    assert observed["switch_range_min_m"] == noise.switch_range_min_m
    assert observed["switch_range_max_m"] == noise.switch_range_max_m
    assert observed["lateral_align_max_m"] == noise.lateral_align_max_m
    assert config.rl.press.handoff.switch_range_min_m <= required["switch_range_min_m"]
    assert config.rl.press.handoff.switch_range_max_m >= required["switch_range_max_m"]
    assert config.rl.press.handoff.lateral_align_max_m >= required["lateral_align_max_m"]
    assert summary["route"]["settle_hold_s"] > 0.0


def test_policy_transfer_config_uses_distinct_entry_targets():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "policy_transfer_press_approach_quad.yaml")
    steps = config.vehicles["x500_01"].routine.steps

    assert config.scene == "@isaac/assets/LightSwitchWarehouse/Warehouse.usd"
    assert config.rl.press.reset.physical_contact_enabled
    assert tuple(steps[index]["label"] for index in (7, 12, 17)) == (
        "policy-transfer-0.460",
        "policy-transfer-0.475",
        "policy-transfer-0.490",
    )
    assert tuple(steps[index]["y"] for index in (5, 10, 15)) == (
        17.563,
        17.507,
        17.476,
    )
    assert all(steps[index]["position_tol_m"] == 0.05 for index in (5, 10, 15))
    assert all(steps[index]["timeout_s"] == 45.0 for index in (7, 12, 17))
    assert all(
        steps[index]["artifact"] == ".build/training/press-approach/latest" for index in (7, 12, 17)
    )

    hover = load_config(_ROOT / "policy_transfer_press_hover_quad.yaml")
    hover_steps = hover.vehicles["x500_01"].routine.steps
    assert hover.rl.press.task == "hover"
    assert all(
        hover_steps[index]["artifact"] == ".build/training/press-hover/latest"
        for index in (7, 12, 17)
    )


@pytest.mark.parametrize(
    "override",
    [
        {"rl": {"training": {"actor_hidden_dims": [64, 64]}}},
        {"rl": {"press": {"reward": {"alive_weight": 1.0}}}},
        {"rl": {"press": {"hover_task": {"reward_alive": 1.0}}}},
        {"rl": {"press": {"reset": {"initial_linear_velocity_max_mps": 0.0}}}},
    ],
)
def test_old_rl_config_paths_fail_strict_composition(tmp_path, override):
    from hydra.errors import ConfigCompositionException

    from robotics_experiments.config import load_config

    path = _write_experiment(tmp_path, "old_rl_path", {})
    OmegaConf.save(OmegaConf.merge(OmegaConf.load(path), override), path)

    with pytest.raises(ConfigCompositionException):
        load_config(path)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            {"rl": {"press": {"policy": {"actor": {"hidden_dims": [256, 0]}}}}},
            "actor.hidden_dims values must be positive",
        ),
        (
            {"rl": {"hover": {"policy": {"critic": {"hidden_dims": [-1]}}}}},
            "critic.hidden_dims values must be positive",
        ),
        (
            {"rl": {"press": {"policy": {"activation": "  "}}}},
            "activation must be non-empty",
        ),
        (
            {"rl": {"hover": {"policy": {"actor": {"initial_noise_std": -0.1}}}}},
            "initial_noise_std must be finite and non-negative",
        ),
        (
            {"rl": {"hover": {"policy": {"actor": {"initial_noise_std": float("inf")}}}}},
            "initial_noise_std must be finite and non-negative",
        ),
        (
            {"rl": {"inputs": {"controller_artifact": "  "}}},
            "controller_artifact must be non-empty",
        ),
        (
            {
                "rl": {
                    "training": {
                        "phases": [
                            {
                                "max_iterations": 1,
                                "evaluation_interval": 1,
                                "learning_rate": 1.0e-4,
                                "schedule": "unknown",
                                "entropy_coef": 0.0,
                                "update_observation_normalization": True,
                            }
                        ]
                    }
                }
            },
            "schedule",
        ),
        (
            {"rl": {"press": {"policy": {"actor": {"output_initialization": "task-controller"}}}}},
            "output_initialization",
        ),
    ],
)
def test_task_policy_validation(tmp_path, override, match):
    from robotics_experiments.config import load_config

    path = _write_experiment(tmp_path, "invalid_policy", {})
    OmegaConf.save(OmegaConf.merge(OmegaConf.load(path), override), path)

    with pytest.raises(ValueError, match=match):
        load_config(path)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            {"rl": {"press": {"task": "future"}}},
            "rl.press.task must be hover or approach",
        ),
        (
            {"rl": {"press": {"goal": {"range_min_m": 0.50}}}},
            "range bounds",
        ),
        (
            {"rl": {"press": {"goal": {"velocity_reference": {"range_center_m": 0.43}}}}},
            "range_center_m",
        ),
        (
            {
                "rl": {
                    "press": {
                        "goal": {
                            "velocity_reference": {
                                "wall_normal_min_mps": 0.1,
                            }
                        }
                    }
                }
            },
            "wall-normal limits",
        ),
        (
            {"rl": {"press": {"navigation_noise": {"sample_count": 9}}}},
            "sample_count must be at least 10",
        ),
        (
            {"rl": {"press": {"navigation_noise": {"source": "/tmp/sample.json"}}}},
            "source must be a repo-relative path",
        ),
        (
            {
                "rl": {
                    "press": {
                        "navigation_noise": {
                            "switch_range_max_m": 0.80,
                        }
                    }
                }
            },
            "cover the measured navigation maximum",
        ),
        (
            {
                "rl": {
                    "press": {
                        "navigation_noise": {
                            "lateral_align_max_m": 0.50,
                        }
                    }
                }
            },
            "cover measured navigation alignment",
        ),
        (
            {
                "rl": {
                    "press": {
                        "navigation_noise": {
                            "relative_attitude_error_max_rad": 0.50,
                        }
                    }
                }
            },
            "cover measured navigation attitude error",
        ),
        (
            {"rl": {"press": {"reward": {"position_progress_weight": -0.1}}}},
            "finite and non-negative",
        ),
        (
            {
                "rl": {
                    "press": {
                        "goal": {
                            "velocity_reference": {
                                "wall_normal_prediction_horizon_s": 0.0,
                            }
                        }
                    }
                }
            },
            "wall_normal_prediction_horizon_s must be finite and positive",
        ),
        (
            {
                "rl": {
                    "press": {
                        "acceptance": {
                            "collision_rate_max": 1.1,
                        }
                    }
                }
            },
            "between 0 and 1",
        ),
    ],
)
def test_press_approach_validation(tmp_path, override, match):
    from robotics_experiments.config import load_config

    path = _write_experiment(tmp_path, "invalid_press_approach", {})
    OmegaConf.save(OmegaConf.merge(OmegaConf.load(path), override), path)
    with pytest.raises(ValueError, match=match):
        load_config(path)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"driver": {"control_rate_hz": 0.0}}, "control_rate_hz"),
        ({"driver": {"offboard_setpoint_rate_hz": float("inf")}}, "finite and positive"),
        ({"rl": {"policy_rate_hz": float("nan")}}, "finite and positive"),
        (
            {"driver": {"control_rate_hz": 10.0}, "rl": {"policy_rate_hz": 20.0}},
            "policy_rate_hz must not exceed",
        ),
        (
            {
                "driver": {
                    "control_rate_hz": 40.0,
                    "offboard_setpoint_rate_hz": 50.0,
                }
            },
            "offboard_setpoint_rate_hz must not exceed",
        ),
        ({"driver": {"control_rate_hz": 300.0}}, "must not exceed sim.physics_hz"),
    ],
)
def test_rate_validation(tmp_path, overrides, match):
    from robotics_experiments.config import load_config

    path = _write_experiment(tmp_path, "invalid_rates", {})
    authored = OmegaConf.load(path)
    authored = OmegaConf.merge(authored, overrides)
    OmegaConf.save(authored, path)
    with pytest.raises(ValueError, match=match):
        load_config(path)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"switches": {"num_switches": 0}}, "num_switches must be positive"),
        ({"switches": {"press_confirm_steps": 0}}, "press_confirm_steps must be positive"),
        ({"switches": {"max_spawn_retries": 0}}, "max_spawn_retries must be positive"),
        ({"switches": {"spawn_bounds": []}}, "spawn_bounds must contain"),
        (
            {"switches": {"spawn_bounds": [{"min": [1.0, 0.0, 0.0], "max": [0.0, 1.0, 1.0]}]}},
            "min corner must not exceed max corner",
        ),
    ],
)
def test_switch_semantic_validation(tmp_path, overrides, match):
    from robotics_experiments.config import load_config

    path = _write_experiment(tmp_path, "invalid_switches", {})
    authored = OmegaConf.merge(OmegaConf.load(path), overrides)
    OmegaConf.save(authored, path)

    with pytest.raises(ValueError, match=match):
        load_config(path)


@pytest.mark.parametrize(
    ("vehicle_override", "match"),
    [
        (
            {"px4": {"failsafe": {"offboard_loss_action": "climb"}}},
            "offboard_loss_action must be one of",
        ),
        (
            {"px4": {"failsafe": {"offboard_loss_timeout_s": 61.0}}},
            "offboard_loss_timeout_s must be",
        ),
        (
            {"px4": {"logging_profile": 4096}},
            "logging_profile must be",
        ),
        (
            {"routine": {"failure_recovery_timeout_s": 0.0}},
            "failure_recovery_timeout_s must be",
        ),
    ],
)
def test_per_vehicle_failsafe_and_recovery_validation(tmp_path, vehicle_override, match):
    from robotics_experiments.config import load_config

    vehicles = {
        "x500": _vehicle(
            0,
            "/x500",
            0.0,
            **vehicle_override,
        )
    }
    with pytest.raises(ValueError, match=match):
        load_config(_write_experiment(tmp_path, "invalid_recovery", vehicles))


def test_vehicle_mapping_is_typed_and_routine_steps_are_not_any():
    from robotics_experiments.config import ExperimentConfig, VehicleConfig

    assert "VehicleConfig" in str(get_type_hints(ExperimentConfig)["vehicles"])
    routine_type = get_type_hints(VehicleConfig)["routine"]
    assert "RoutineStepConfig" in str(get_type_hints(routine_type)["steps"])


def test_policy_routine_uses_label_and_target_enu_contract():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "policy_hover_quad.yaml")
    policy_step = next(
        step for step in config.vehicles["x500_01"].routine.steps if step["kind"] == "policy"
    )

    assert policy_step["label"] == "hover"
    assert "name" not in policy_step
    assert "target" not in policy_step


def test_empty_fleet_is_valid(tmp_path):
    from robotics_experiments.config import load_config

    config = load_config(_write_experiment(tmp_path, "empty", {}))
    assert config.vehicles == {}


def test_vehicle_order_and_derived_identity(tmp_path):
    from robotics_experiments.config import load_config

    vehicles = {
        "lead": _vehicle(5, "/fleet/lead", 2.0, prim_name="Lead"),
        "wing": _vehicle(1, "/fleet/wing", 4.0, prim_name="Wing"),
    }
    config = load_config(_write_experiment(tmp_path, "ordered", vehicles))
    assert list(config.vehicles) == ["lead", "wing"]
    lead, wing = config.vehicles.values()
    assert (lead.mavlink_port, lead.gcs_port) == (4565, 14555)
    assert (wing.mavlink_port, wing.gcs_port) == (4561, 14551)
    assert lead.debug_log_path != wing.debug_log_path


@pytest.mark.parametrize(
    ("vehicles", "match"),
    [
        (
            {
                "a": _vehicle(0, "/a", 0.0, prim_name="A"),
                "b": _vehicle(0, "/b", 2.0, prim_name="B"),
            },
            "duplicate vehicle id",
        ),
        (
            {
                "a": _vehicle(0, "/a", 0.0, prim_name="Same"),
                "b": _vehicle(1, "/b", 2.0, prim_name="Same"),
            },
            "duplicate vehicle spawn.prim_name",
        ),
        (
            {
                "a": _vehicle(0, "/fleet/a_b", 0.0, prim_name="A"),
                "b": _vehicle(1, "/fleet_a/b", 2.0, prim_name="B"),
            },
            "duplicate vehicle ROS frame prefix",
        ),
    ],
)
def test_fleet_validation(tmp_path, vehicles, match):
    from robotics_experiments.config import load_config

    with pytest.raises(ValueError, match=match):
        load_config(_write_experiment(tmp_path, "invalid", vehicles))


@pytest.mark.parametrize(
    ("namespace", "match"),
    [
        ("x500", "must be absolute"),
        ("/", "must not be root"),
        ("/x500/", "must not end with"),
        ("/fleet//x500", "empty segments"),
        (" /x500", "whitespace"),
    ],
)
def test_namespace_validation(tmp_path, namespace, match):
    from robotics_experiments.config import load_config

    vehicles = {"x500": _vehicle(0, namespace, 0.0)}
    with pytest.raises(ValueError, match=match):
        load_config(_write_experiment(tmp_path, "namespace", vehicles))


def test_per_vehicle_routines_are_independent(tmp_path):
    from robotics_experiments.config import load_config

    vehicles = {
        "manual": _vehicle(0, "/manual", 0.0, prim_name="Manual"),
        "flying": _vehicle(
            1,
            "/flying",
            2.0,
            prim_name="Flying",
            routine={"steps": [{"kind": "arm"}], "max_tilt_deg": 20.0},
        ),
    }
    config = load_config(_write_experiment(tmp_path, "routines", vehicles))
    assert config.vehicles["manual"].routine.steps == ()
    assert config.vehicles["flying"].routine.steps == ({"kind": "arm"},)
    assert config.vehicles["flying"].routine.max_tilt_deg == 20.0


def test_mixed_airframes_drive_bridge_and_px4_launch():
    from robotics_experiments.config import load_config

    config = load_config("configs/navigate_quad_hex.yaml")
    quad = config.vehicles["x500_01"]
    hexa = config.vehicles["s550_01"]
    assert hexa.spawn.usd_path.endswith("/hexacopter_s550.usda")
    assert len(quad.bridge.max_rotor_velocity) == 4
    assert len(hexa.bridge.max_rotor_velocity) == 6
    assert [
        (vehicle.id, vehicle.px4_sim_model, vehicle.px4_sys_autostart)
        for vehicle in config.vehicles.values()
    ] == [
        (0, "none_x500_v2", 22000),
        (1, "none_hexa_x", 22001),
    ]


def test_disabled_and_unknown_assets_affect_px4_launch(tmp_path):
    from robotics_experiments.config import load_config

    vehicles = {
        "disabled": _vehicle(
            0,
            "/disabled",
            0.0,
            bridge={"enabled": False},
            spawn={
                "position_xy": [0.0, 0.0],
                "usd_path": "/assets/unknown.usda",
            },
        )
    }
    config = load_config(_write_experiment(tmp_path, "disabled", vehicles))
    vehicle = config.vehicles["disabled"]
    assert (vehicle.px4_sim_model, vehicle.px4_sys_autostart) == ("", 0)

    vehicles["disabled"]["bridge"] = {"enabled": True}
    with pytest.raises(ValueError, match="unknown.usda"):
        load_config(_write_experiment(tmp_path, "unknown", vehicles))

import shutil
from dataclasses import FrozenInstanceError
from typing import get_type_hints

import pytest

from tests.support.paths import REPO_ROOT

_ROOT = REPO_ROOT / "configs"
_EXPERIMENTS = sorted(path for path in _ROOT.glob("*.yaml") if path.name != "base.yaml")


def _write_experiment(tmp_path, name, vehicles):
    from omegaconf import OmegaConf

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


def test_press_approach_config_preserves_actor_contract_and_stage_settings():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "train_press_approach_quad.yaml")
    training = config.rl.training
    reward = config.rl.press.reward
    policy = config.rl.press.policy

    assert config.rl.action.action_dim == 4
    assert config.rl.action.max_body_velocity_mps == 2.0
    assert training.controller_artifact == ".build/alignment/x500-velocity-px4-v7/best"
    assert config.rl.press.reset.physical_contact_enabled
    assert config.rl.num_envs == 32768
    assert training.policy_name == "press-approach"
    assert training.training_mode == "scratch"
    assert config.rl.press.curriculum_stage == "approach"
    assert training.max_iterations == 300
    assert training.learning_rate == 1.0e-4
    assert training.entropy_coef == 0.0001
    assert training.save_interval == 10
    assert training.schedule == "adaptive"
    assert policy.actor.initial_noise_std == 0.03
    assert policy.actor.output_initialization == "scratch-neutral-output"
    assert policy.actor.hidden_dims == (256, 256, 128)
    assert policy.critic.hidden_dims == (256, 256, 128)
    assert policy.activation == "elu"
    assert reward.range_progress_weight == 50.0
    assert reward.range_barrier_margin_m == 0.02
    assert reward.range_barrier_weight == 10.0
    assert reward.velocity_tracking_weight == 100.0
    assert reward.success_weight == 150.0
    assert reward.collision_weight == 300.0
    assert reward.flight_failure_weight == 100.0
    assert config.rl.press.goal.stable_steps == 5
    reference = config.rl.press.goal.velocity_reference
    assert reference.range_center_m == 0.41
    assert reference.wall_normal_alignment_gate_m == 0.03
    assert reference.wall_normal_gain_per_s == 1.25
    assert reference.wall_normal_min_mps == -0.04
    assert reference.wall_normal_max_mps == 0.08
    assert reference.tangential_gain_per_s == 2.0
    assert reference.tangential_speed_max_mps == 0.10
    assert config.rl.press.handoff.switch_range_min_m == 0.45
    assert config.rl.press.handoff.switch_range_max_m == 0.50
    assert config.rl.press.handoff.lateral_align_max_m == 0.08
    assert config.rl.press.handoff.wall_normal_speed_max_mps == 0.02
    assert config.rl.press.handoff.lateral_speed_max_mps == 0.02
    assert config.rl.press.handoff.angular_speed_max_radps == 0.05
    assert config.rl.press.handoff.body_z_alignment_min == 0.965926
    assert not hasattr(config.rl.press.reset, "initial_linear_velocity_max_mps")
    assert not hasattr(config.rl.press.reset, "initial_angular_velocity_max_radps")
    assert config.rl.press.training_reset.hard_case_fraction == 0.0
    assert config.rl.press.training_reset.range_min_m == 0.45
    assert config.rl.press.training_reset.range_max_m == 0.47
    assert config.rl.press.training_reset.tangential_error_min_m == 0.06
    assert config.rl.press.training_reset.tangential_error_max_m == 0.08
    assert config.rl.press.evaluation.success_rate_min == 0.95
    assert config.rl.press.evaluation.collision_rate_max == 0.01
    assert config.rl.press.evaluation.selection_metric == "held-out-press-task-quality-v3"
    assert config.rl.press.evaluation.collision_penalty == 4.0
    assert config.rl.press.evaluation.completion_time_weight == 0.15
    assert config.rl.hover.policy.actor.hidden_dims == (128, 128)


def test_press_approach_finetune_config_uses_rate_pilot_actor_only_recipe():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "train_press_approach_quad_finetune.yaml")

    assert config.rl.training.training_mode == "actor-only"
    assert config.rl.training.max_iterations == 15
    assert config.rl.training.learning_rate == 3.0e-5
    assert config.rl.training.entropy_coef == 0.0
    assert config.rl.training.save_interval == 5
    assert config.rl.training.schedule == "fixed"
    assert config.rl.press.policy.actor.initial_noise_std == 0.01
    assert config.rl.press.reward.flight_failure_weight == 300.0
    assert config.rl.press.reward.velocity_tracking_weight == 300.0
    assert config.rl.press.training_reset.hard_case_fraction == 0.5
    reference = config.rl.press.goal.velocity_reference
    assert reference.range_center_m == 0.415
    assert reference.wall_normal_max_mps == 0.04
    assert config.rl.press.goal.range_min_m == 0.40
    assert config.rl.press.goal.range_max_m == 0.42


def test_training_and_manual_press_configs_share_one_actor_entry_contract():
    from robotics_experiments.config import load_config
    from robotics_experiments.rl.press import HandoffGate

    training = load_config(_ROOT / "train_press_approach_quad.yaml")
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
    assert steps[3]["y"] == 17.40
    assert steps[3]["ground_speed_m_s"] == 0.5
    assert steps[3]["position_tol_m"] == 0.05
    assert steps[5]["y"] == 17.51
    assert steps[5]["ground_speed_m_s"] == 0.08
    assert steps[5]["position_tol_m"] == 0.02
    assert steps[5]["timeout_s"] == 10.0
    assert steps[8]["ground_speed_m_s"] == 0.5
    assert "ground_speed_m_s" not in steps[2]
    assert all(float(step.get("timeout_s", 20.0)) <= 20.0 for step in steps)


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
    from omegaconf import OmegaConf

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
            {"rl": {"training": {"controller_artifact": "  "}}},
            "controller_artifact must be non-empty",
        ),
        (
            {"rl": {"training": {"training_mode": "resume"}}},
            "training_mode",
        ),
        (
            {"rl": {"press": {"policy": {"actor": {"output_initialization": "task-controller"}}}}},
            "output_initialization",
        ),
    ],
)
def test_task_policy_validation(tmp_path, override, match):
    from omegaconf import OmegaConf

    from robotics_experiments.config import load_config

    path = _write_experiment(tmp_path, "invalid_policy", {})
    OmegaConf.save(OmegaConf.merge(OmegaConf.load(path), override), path)

    with pytest.raises(ValueError, match=match):
        load_config(path)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"rl": {"press": {"curriculum_stage": "future"}}}, "unknown"),
        (
            {"rl": {"press": {"goal": {"range_min_m": 0.43}}}},
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
            {"rl": {"press": {"handoff": {"switch_range_min_m": 0.40}}}},
            "switch_range_min_m",
        ),
        (
            {"rl": {"press": {"training_reset": {"hard_case_fraction": 1.1}}}},
            "hard_case_fraction",
        ),
        (
            {"rl": {"press": {"training_reset": {"range_min_m": 0.48}}}},
            "range bounds",
        ),
        (
            {
                "rl": {
                    "press": {
                        "training_reset": {
                            "tangential_error_min_m": 0.081,
                        }
                    }
                }
            },
            "tangential bounds",
        ),
        (
            {"rl": {"press": {"reward": {"action_delta_weight": -0.1}}}},
            "finite and non-negative",
        ),
        (
            {"rl": {"press": {"reward": {"range_barrier_margin_m": 0.0}}}},
            "range_barrier_margin_m must be finite and positive",
        ),
        (
            {
                "rl": {
                    "press": {
                        "evaluation": {
                            "collision_rate_max": 1.1,
                        }
                    }
                }
            },
            "between 0 and 1",
        ),
        (
            {
                "rl": {
                    "press": {
                        "evaluation": {
                            "selection_metric": "unknown",
                        }
                    }
                }
            },
            "selection_metric",
        ),
        (
            {
                "rl": {
                    "press": {
                        "evaluation": {
                            "collision_penalty": -1.0,
                        }
                    }
                }
            },
            "finite and non-negative",
        ),
    ],
)
def test_press_approach_validation(tmp_path, override, match):
    from omegaconf import OmegaConf

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
    from omegaconf import OmegaConf

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
    from omegaconf import OmegaConf

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

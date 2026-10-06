import functools
import math
import re
from collections.abc import Iterable
from dataclasses import fields
from os import PathLike
from pathlib import Path
from typing import cast

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_config_search_path
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict

from robotics_experiments.config._curriculum import CurriculumConfig
from robotics_experiments.config._environment import ExperimentConfig, VehicleConfig
from robotics_experiments.config._immutability import _freeze_config_graph
from robotics_experiments.config._scene import SwitchModelConfig
from robotics_experiments.config._schemas import _register_schemas

__all__ = ["load_config", "load_curriculum_config", "load_switch_model"]

_PX4_OFFBOARD_LOSS_ACTION_PARAMS = {
    "position": 0,
    "altitude": 1,
    "stabilized": 2,
    "return": 3,
    "land": 4,
    "hold": 5,
}


@functools.cache
def _airframes_by_output_filename(airframe_config_dir: Path) -> dict[str, dict]:
    airframes: dict[str, dict] = {}
    for path in sorted(airframe_config_dir.glob("*.yaml")):
        data = cast(dict, OmegaConf.to_container(OmegaConf.load(path), resolve=True))
        airframes[str(data["output_filename"])] = data
    return airframes


def _airframe_for_drone_asset(usd_path: str, airframe_config_dir: Path) -> dict | None:
    return _airframes_by_output_filename(airframe_config_dir).get(Path(usd_path).name)


# Standard gravity for the grams-force -> Newton unit conversion of the cited
# per-rotor thrust. This is the definition of a gram-force (the unit the motor
# datasheet uses), not the sim's gravity: motor thrust is a property of the
# motor/prop, independent of what gravity the scene runs at.
_STANDARD_GRAVITY_M_S2 = 9.80665


def _compute_rotor_constants(
    max_thrust_per_rotor_g: float,
    max_rotor_velocity: Iterable[float],
) -> list[float]:
    """Derive the Pegasus per-rotor thrust coefficient from the cited thrust spec.

    ``T_i = rotor_constant_i * omega_i^2``, so at the per-rotor maximum angular
    velocity the thrust equals the authored static maximum
    (``max_thrust_per_rotor_g`` grams-force). This keeps ``rotor_constant`` a pure
    derivation of a vendor-citable thrust number rather than a hand-tuned scalar.
    """
    if max_thrust_per_rotor_g <= 0.0:
        raise ValueError(
            "px4 max_thrust_per_rotor_g must be positive, " f"got {max_thrust_per_rotor_g!r}"
        )
    max_velocities = [float(v) for v in max_rotor_velocity]
    if not max_velocities:
        raise ValueError("px4 max_rotor_velocity must contain at least one rotor velocity")
    bad_velocity = next((v for v in max_velocities if v <= 0.0), None)
    if bad_velocity is not None:
        raise ValueError(f"px4 max_rotor_velocity values must be positive, got {bad_velocity!r}")

    max_thrust_n = max_thrust_per_rotor_g / 1000.0 * _STANDARD_GRAVITY_M_S2
    return [max_thrust_n / (v * v) for v in max_velocities]


def _validate_fleet(vehicles: dict[str, VehicleConfig]) -> None:
    """Raise ``ValueError`` on any structurally invalid resolved fleet.

    Operates on the derived runtime fields (``mavlink_port`` / ``debug_log_path``)
    so it never recomputes ``connection_baseport + id``. Invalid configuration
    must fail loudly at resolution time rather than surface later as a port
    collision or an overwritten log.
    """
    ids: set[int] = set()
    endpoints: set[tuple[str, str, int]] = set()
    prim_names: set[str] = set()
    log_paths: set[str] = set()
    ros_namespaces: set[str] = set()
    ros_frame_prefixes: set[str] = set()
    for label, v in vehicles.items():
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (v.pusher.effort_limit_n, v.pusher.velocity_limit_mps)
        ):
            raise ValueError(f"vehicle {label!r} pusher limits must be finite and positive")
        if not 0 <= v.px4.logging_profile <= 4095:
            raise ValueError(
                f"vehicle {label!r} px4.logging_profile must be between 0 and 4095, "
                f"got {v.px4.logging_profile!r}"
            )
        if not math.isfinite(v.px4.failsafe.offboard_loss_timeout_s) or not (
            0.0 <= v.px4.failsafe.offboard_loss_timeout_s <= 60.0
        ):
            raise ValueError(
                f"vehicle {label!r} px4.failsafe.offboard_loss_timeout_s must be "
                "finite and between 0 and 60 seconds, got "
                f"{v.px4.failsafe.offboard_loss_timeout_s!r}"
            )
        if not math.isfinite(v.routine.failure_recovery_timeout_s) or (
            v.routine.failure_recovery_timeout_s <= 0.0
        ):
            raise ValueError(
                f"vehicle {label!r} routine.failure_recovery_timeout_s must be "
                f"finite and positive, got {v.routine.failure_recovery_timeout_s!r}"
            )
        if (
            not math.isfinite(v.bridge.barometer_noise_std_pa)
            or v.bridge.barometer_noise_std_pa < 0.0
        ):
            raise ValueError(
                f"vehicle {label!r} bridge.barometer_noise_std_pa must be "
                f"finite and non-negative, got {v.bridge.barometer_noise_std_pa!r}"
            )
        if v.bridge.sensor_excitation_mode not in {"stochastic", "deterministic"}:
            raise ValueError(
                f"vehicle {label!r} bridge.sensor_excitation_mode must be "
                "'stochastic' or 'deterministic', got "
                f"{v.bridge.sensor_excitation_mode!r}"
            )
        if (
            not math.isfinite(v.bridge.magnetometer_noise_scale)
            or v.bridge.magnetometer_noise_scale < 0.0
        ):
            raise ValueError(
                f"vehicle {label!r} bridge.magnetometer_noise_scale must be "
                f"finite and non-negative, got {v.bridge.magnetometer_noise_scale!r}"
            )
        if len(v.spawn.position_xy) != 2:
            raise ValueError(
                f"vehicle {label!r} position_xy must have length 2, got {v.spawn.position_xy!r}"
            )
        if v.id < 0:
            raise ValueError(f"vehicle id must be non-negative, got {v.id}")
        if v.id in ids:
            raise ValueError(f"duplicate vehicle id={v.id}")
        ids.add(v.id)

        endpoint = (v.bridge.connection_type, v.bridge.connection_ip, v.mavlink_port)
        if endpoint in endpoints:
            raise ValueError(f"duplicate vehicle MAVLink endpoint {endpoint} (vehicle id={v.id})")
        endpoints.add(endpoint)

        if v.spawn.prim_name in prim_names:
            raise ValueError(
                f"duplicate vehicle spawn.prim_name {v.spawn.prim_name!r} (vehicle id={v.id})"
            )
        prim_names.add(v.spawn.prim_name)

        if v.ros.namespace in ros_namespaces:
            raise ValueError(
                f"duplicate vehicle ros.namespace {v.ros.namespace!r} (vehicle id={v.id})"
            )
        ros_namespaces.add(v.ros.namespace)

        ros_frame_prefix = _ros_frame_prefix_from_namespace(v.ros.namespace)
        if ros_frame_prefix in ros_frame_prefixes:
            raise ValueError(
                f"duplicate vehicle ROS frame prefix {ros_frame_prefix!r} " f"(vehicle id={v.id})"
            )
        ros_frame_prefixes.add(ros_frame_prefix)

        # debug_log_path already has the {vehicle} token rendered to id; the
        # date/time tokens stay literal here. Two vehicles whose template lacks
        # {vehicle} render identically and would overwrite each other's logs.
        # Empty path disables logging (no collision).
        if v.debug_log_path:
            if v.debug_log_path in log_paths:
                raise ValueError(
                    f"colliding vehicle debug_log_path {v.bridge.debug_log_path!r} "
                    f"(vehicle id={v.id}); add a {{vehicle}} token to disambiguate"
                )
            log_paths.add(v.debug_log_path)


def _validate_rates(experiment: ExperimentConfig) -> None:
    rates = {
        "sim.physics_hz": experiment.sim.physics_hz,
        "driver.control_rate_hz": experiment.driver.control_rate_hz,
        "driver.offboard_setpoint_rate_hz": experiment.driver.offboard_setpoint_rate_hz,
        "rates.command_rate_hz": experiment.rates.command_rate_hz,
        "rates.sample_rate_hz": experiment.rates.sample_rate_hz,
    }
    for name, value in rates.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if experiment.rates.command_rate_hz > experiment.driver.control_rate_hz:
        raise ValueError("rates.command_rate_hz must not exceed driver.control_rate_hz")
    if experiment.rates.sample_rate_hz > experiment.sim.physics_hz:
        raise ValueError("rates.sample_rate_hz must not exceed sim.physics_hz")
    if experiment.driver.offboard_setpoint_rate_hz > experiment.driver.control_rate_hz:
        raise ValueError("driver.offboard_setpoint_rate_hz must not exceed driver.control_rate_hz")
    if experiment.driver.control_rate_hz > experiment.sim.physics_hz:
        raise ValueError("driver.control_rate_hz must not exceed sim.physics_hz")


def _validate_alignment(experiment: ExperimentConfig) -> None:
    capture = experiment.alignment.capture
    if not 0 <= capture.sensor_seed <= 2**32 - 1:
        raise ValueError("alignment.capture.sensor_seed must be a 32-bit unsigned integer")
    positive_capture = {
        "minimum_neutral_s": capture.minimum_neutral_s,
        "stable_hold_s": capture.stable_hold_s,
        "timeout_s": capture.timeout_s,
        "recenter_gain_hz": capture.recenter_gain_hz,
        "recenter_max_speed_mps": capture.recenter_max_speed_mps,
        "max_linear_speed_mps": capture.max_linear_speed_mps,
        "max_angular_speed_radps": capture.max_angular_speed_radps,
        "max_tilt_deg": capture.max_tilt_deg,
    }
    for name, value in positive_capture.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"alignment.capture.{name} must be finite and positive")
    if capture.timeout_s <= capture.minimum_neutral_s + capture.stable_hold_s:
        raise ValueError(
            "alignment.capture.timeout_s must exceed minimum_neutral_s plus stable_hold_s"
        )
    execution = experiment.alignment.execution
    if execution.worker_count <= 0:
        raise ValueError("alignment.execution.worker_count must be positive")
    if execution.candidate_capacity <= 0:
        raise ValueError("alignment.execution.candidate_capacity must be positive")
    if (
        not execution.candidate_capacity_benchmark
        or any(value <= 0 for value in execution.candidate_capacity_benchmark)
        or len(set(execution.candidate_capacity_benchmark))
        != len(execution.candidate_capacity_benchmark)
    ):
        raise ValueError(
            "alignment.execution.candidate_capacity_benchmark must contain "
            "unique positive values"
        )
    if (
        not math.isfinite(execution.gpu_target_percent)
        or not 0.0 < execution.gpu_target_percent <= 100.0
    ):
        raise ValueError("alignment.execution.gpu_target_percent must be in (0, 100]")
    if not math.isfinite(execution.gpu_sample_interval_s) or execution.gpu_sample_interval_s <= 0.0:
        raise ValueError("alignment.execution.gpu_sample_interval_s must be finite and positive")


def _validate_training_phases(*, phases, path: str) -> None:
    if not phases:
        raise ValueError(f"{path} must contain at least one phase")
    for index, phase in enumerate(phases):
        phase_path = f"{path}[{index}]"
        if phase.max_iterations <= 0 or phase.evaluation_interval <= 0:
            raise ValueError(
                f"{phase_path} max_iterations and evaluation_interval must be positive"
            )
        if phase.evaluation_interval > phase.max_iterations:
            raise ValueError(f"{phase_path}.evaluation_interval must not exceed max_iterations")
        if not math.isfinite(phase.learning_rate) or phase.learning_rate <= 0.0:
            raise ValueError(f"{phase_path}.learning_rate must be finite and positive")
        if phase.schedule not in {"fixed", "adaptive"}:
            raise ValueError(f"{phase_path}.schedule must be 'fixed' or 'adaptive'")
        if not math.isfinite(phase.entropy_coef) or phase.entropy_coef < 0.0:
            raise ValueError(f"{phase_path}.entropy_coef must be finite and non-negative")


def _validate_plateau(*, plateau, path: str) -> None:
    if not plateau.metric_name.strip():
        raise ValueError(f"{path}.metric_name must be non-empty")
    if plateau.direction not in {"minimize", "maximize"}:
        raise ValueError(f"{path}.direction must be 'minimize' or 'maximize'")
    if not math.isfinite(plateau.minimum_delta) or plateau.minimum_delta <= 0.0:
        raise ValueError(f"{path}.minimum_delta must be finite and positive")
    if plateau.patience_evaluations <= 0:
        raise ValueError(f"{path}.patience_evaluations must be positive")


def _validate_rl_policies(experiment: ExperimentConfig) -> None:
    training = experiment.rl.training
    if not experiment.rl.inputs.controller_artifact.strip():
        raise ValueError("rl.inputs.controller_artifact must be non-empty")
    if not experiment.rl.inputs.flight_parameters_artifact.strip():
        raise ValueError("rl.inputs.flight_parameters_artifact must be non-empty")
    attempt_seeds = tuple(training.attempt_seeds)
    if not attempt_seeds or len(set(attempt_seeds)) != len(attempt_seeds):
        raise ValueError("rl.training.attempt_seeds must be nonempty and unique")
    _validate_training_phases(phases=training.phases, path="rl.training.phases")
    _validate_plateau(plateau=training.plateau, path="rl.training.plateau")
    if (
        not math.isfinite(training.evaluation_worker_timeout_s)
        or training.evaluation_worker_timeout_s <= 0.0
    ):
        raise ValueError("rl.training.evaluation_worker_timeout_s must be finite and positive")
    if (
        not math.isfinite(training.telemetry_sample_interval_s)
        or training.telemetry_sample_interval_s <= 0.0
    ):
        raise ValueError("rl.training.telemetry_sample_interval_s must be finite and positive")
    if not math.isfinite(training.gpu_utilization_target_rate) or not (
        0.0 < training.gpu_utilization_target_rate <= 1.0
    ):
        raise ValueError("rl.training.gpu_utilization_target_rate must be in (0, 1]")
    policies = {
        "rl.hover.policy": experiment.rl.hover.policy,
        "rl.press.policy": experiment.rl.press.policy,
    }
    for path, policy in policies.items():
        if not policy.activation.strip():
            raise ValueError(f"{path}.activation must be non-empty")
        dimensions = {
            f"{path}.actor.hidden_dims": policy.actor.hidden_dims,
            f"{path}.critic.hidden_dims": policy.critic.hidden_dims,
        }
        for dimensions_path, hidden_dims in dimensions.items():
            bad_dimension = next((dimension for dimension in hidden_dims if dimension <= 0), None)
            if bad_dimension is not None:
                raise ValueError(
                    f"{dimensions_path} values must be positive, got {bad_dimension!r}"
                )
        noise = policy.actor.initial_noise_std
        if not math.isfinite(noise) or noise < 0.0:
            raise ValueError(
                f"{path}.actor.initial_noise_std must be finite and non-negative, got {noise!r}"
            )
        if policy.actor.output_initialization != "scratch-neutral-output":
            raise ValueError(
                f"{path}.actor.output_initialization must be " "'scratch-neutral-output'"
            )


def _validate_press_task(experiment: ExperimentConfig) -> None:
    press = experiment.rl.press
    if press.task not in {"hover", "approach", "contact", "press"}:
        raise ValueError("rl.press.task must be hover, approach, contact or press")
    for field in fields(press.contact):
        value = getattr(press.contact, field.name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"rl.press.contact.{field.name} must be finite and positive")
    for name in ("first_contact_rate_min", "retreat_rate_min"):
        if getattr(press.contact, name) > 1:
            raise ValueError(f"rl.press.contact.{name} must be at most 1")
    if press.termination.contact_data_capacity_per_body <= 0:
        raise ValueError("contact_data_capacity_per_body must be positive")
    if press.task in {"contact", "press"} and not press.reset.physical_contact_enabled:
        raise ValueError("contact tasks require physical contact")
    if any(
        source.source not in {"pinned_flight_parameters", "measured_navigation_handoff"}
        or not math.isfinite(source.weight)
        or source.weight <= 0
        for source in press.reset.mix
    ):
        raise ValueError("rl.press.reset.mix requires known sources and positive finite weights")
    if len({source.source for source in press.reset.mix}) != len(press.reset.mix):
        raise ValueError("rl.press.reset.mix sources must be unique")
    if not math.isfinite(experiment.rl.episode_length_s) or experiment.rl.episode_length_s <= 0:
        raise ValueError("rl.episode_length_s must be finite and positive")
    if not math.isclose(
        sum(source.weight for source in press.reset.mix), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("press reset weights must sum to one")
    reward = press.reward
    if reward.hover_stabilization not in {"uniform", "outside_only", "boundary_weighted"}:
        raise ValueError(
            "unsupported rl.press.reward.hover_stabilization: " f"{reward.hover_stabilization!r}"
        )
    if not math.isfinite(reward.stabilization_weight) or reward.stabilization_weight <= 0.0:
        raise ValueError("rl.press.reward.stabilization_weight must be finite and positive")
    if (
        not math.isfinite(reward.outside_envelope_weight)
        or reward.outside_envelope_weight <= reward.stabilization_weight
    ):
        raise ValueError(
            "rl.press.reward.outside_envelope_weight must be finite and greater than "
            "stabilization_weight"
        )
    handoff = press.handoff
    if not (
        math.isfinite(handoff.switch_range_min_m)
        and math.isfinite(handoff.switch_range_max_m)
        and 0.0 < handoff.switch_range_min_m <= handoff.switch_range_max_m
    ):
        raise ValueError("rl.press.handoff switch range bounds must be positive and ordered")
    positive_handoff_values = {
        "lateral_align_max_m": handoff.lateral_align_max_m,
        "wall_normal_speed_max_mps": handoff.wall_normal_speed_max_mps,
        "lateral_speed_max_mps": handoff.lateral_speed_max_mps,
        "angular_speed_max_radps": handoff.angular_speed_max_radps,
        "relative_attitude_error_max_rad": handoff.relative_attitude_error_max_rad,
    }
    for name, value in positive_handoff_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"rl.press.handoff.{name} must be finite and positive")
    if not math.isfinite(handoff.pose_age_max_s) or handoff.pose_age_max_s < 0.0:
        raise ValueError("rl.press.handoff.pose_age_max_s must be finite and non-negative")
    bounded_handoff_values = {
        "pose_confidence_min": handoff.pose_confidence_min,
        "body_z_alignment_min": handoff.body_z_alignment_min,
        "estimator_health_min": handoff.estimator_health_min,
    }
    for name, value in bounded_handoff_values.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"rl.press.handoff.{name} must be between 0 and 1")

    navigation_noise = press.navigation_noise
    if not navigation_noise.source.strip():
        raise ValueError("rl.press.navigation_noise.source must be non-empty")
    navigation_source = Path(navigation_noise.source)
    if navigation_source.is_absolute() or ".." in navigation_source.parts:
        raise ValueError("rl.press.navigation_noise.source must be a repo-relative path")
    if navigation_noise.sample_count < 10:
        raise ValueError("rl.press.navigation_noise.sample_count must be at least 10")
    if (
        not math.isfinite(navigation_noise.position_margin_m)
        or navigation_noise.position_margin_m <= 0.0
    ):
        raise ValueError("rl.press.navigation_noise.position_margin_m must be finite and positive")
    if (
        not math.isfinite(navigation_noise.rotation_margin_rad)
        or navigation_noise.rotation_margin_rad <= 0.0
    ):
        raise ValueError(
            "rl.press.navigation_noise.rotation_margin_rad must be finite and positive"
        )
    if not (
        math.isfinite(navigation_noise.switch_range_min_m)
        and math.isfinite(navigation_noise.switch_range_max_m)
        and 0.0 < navigation_noise.switch_range_min_m <= navigation_noise.switch_range_max_m
    ):
        raise ValueError(
            "rl.press.navigation_noise switch range bounds must be positive and ordered"
        )
    if (
        not math.isfinite(navigation_noise.lateral_align_max_m)
        or navigation_noise.lateral_align_max_m < 0.0
    ):
        raise ValueError(
            "rl.press.navigation_noise.lateral_align_max_m must be finite and non-negative"
        )
    if (
        not math.isfinite(navigation_noise.relative_attitude_error_max_rad)
        or navigation_noise.relative_attitude_error_max_rad < 0.0
    ):
        raise ValueError(
            "rl.press.navigation_noise.relative_attitude_error_max_rad must be finite "
            "and non-negative"
        )
    margin_m = navigation_noise.position_margin_m
    if handoff.switch_range_min_m > navigation_noise.switch_range_min_m - margin_m:
        raise ValueError(
            "rl.press.handoff.switch_range_min_m must cover the measured navigation "
            "minimum plus navigation_noise.position_margin_m"
        )
    if handoff.switch_range_max_m < navigation_noise.switch_range_max_m + margin_m:
        raise ValueError(
            "rl.press.handoff.switch_range_max_m must cover the measured navigation "
            "maximum plus navigation_noise.position_margin_m"
        )
    if handoff.lateral_align_max_m < navigation_noise.lateral_align_max_m + margin_m:
        raise ValueError(
            "rl.press.handoff.lateral_align_max_m must cover measured navigation "
            "alignment plus navigation_noise.position_margin_m"
        )
    if handoff.relative_attitude_error_max_rad < (
        navigation_noise.relative_attitude_error_max_rad + navigation_noise.rotation_margin_rad
    ):
        raise ValueError(
            "rl.press.handoff.relative_attitude_error_max_rad must cover measured navigation "
            "attitude error plus navigation_noise.rotation_margin_rad"
        )

    goal = press.goal
    if not 0.0 < goal.range_min_m <= goal.range_max_m:
        raise ValueError("rl.press.goal range bounds must be positive and ordered")
    acceptance = press.acceptance
    if not math.isfinite(acceptance.minimum_range_m) or not (
        0.0 < acceptance.minimum_range_m <= goal.range_min_m
    ):
        raise ValueError(
            "rl.press.acceptance.minimum_range_m must be positive and at most goal.range_min_m"
        )
    reference = goal.velocity_reference
    if not goal.range_min_m <= reference.range_center_m <= goal.range_max_m:
        raise ValueError(
            "rl.press.goal.velocity_reference.range_center_m must lie inside " "the goal range"
        )
    if not (
        math.isfinite(reference.wall_normal_min_mps)
        and math.isfinite(reference.wall_normal_max_mps)
        and reference.wall_normal_min_mps < reference.wall_normal_max_mps
        and reference.wall_normal_min_mps <= 0.0 <= reference.wall_normal_max_mps
    ):
        raise ValueError(
            "rl.press.goal.velocity_reference wall-normal limits must be " "finite and ordered"
        )
    if not 0.0 <= goal.body_z_alignment_min <= 1.0:
        raise ValueError("rl.press.goal.body_z_alignment_min must be between 0 and 1")
    positive_goal_values = {
        "alignment_error_max_m": goal.alignment_error_max_m,
        "relative_attitude_error_max_rad": goal.relative_attitude_error_max_rad,
        "linear_speed_max_mps": goal.linear_speed_max_mps,
        "angular_speed_max_radps": goal.angular_speed_max_radps,
        "velocity_reference.wall_normal_prediction_horizon_s": (
            reference.wall_normal_prediction_horizon_s
        ),
        "velocity_reference.wall_normal_alignment_gate_m": (reference.wall_normal_alignment_gate_m),
        "velocity_reference.wall_normal_gain_per_s": (reference.wall_normal_gain_per_s),
        "velocity_reference.tangential_gain_per_s": reference.tangential_gain_per_s,
        "velocity_reference.tangential_speed_max_mps": (reference.tangential_speed_max_mps),
    }
    for name, value in positive_goal_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"rl.press.goal.{name} must be finite and positive")
    if goal.stable_steps <= 0:
        raise ValueError("rl.press.goal.stable_steps must be positive")
    for field in fields(press.reward):
        name = field.name
        if name == "hover_stabilization":
            continue
        value = getattr(press.reward, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"rl.press.reward.{name} must be finite and non-negative, got {value!r}"
            )
    evaluation = press.evaluation
    for name in (
        "feasibility_position_gain_per_s",
        "feasibility_velocity_damping",
        "feasibility_yaw_gain_per_s",
    ):
        value = getattr(evaluation, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"rl.press.evaluation.{name} must be finite and non-negative")
    validation_seeds = tuple(evaluation.validation_seeds)
    test_seeds = tuple(evaluation.test_seeds)
    if not validation_seeds or len(set(validation_seeds)) != len(validation_seeds):
        raise ValueError("rl.press.evaluation.validation_seeds must be nonempty and unique")
    if len(test_seeds) != 3 or len(set(test_seeds)) != len(test_seeds):
        raise ValueError("rl.press.evaluation.test_seeds must contain exactly three unique seeds")
    seed_groups = (
        set(experiment.rl.training.attempt_seeds),
        set(validation_seeds),
        set(test_seeds),
    )
    overlaps = (
        left & right for index, left in enumerate(seed_groups) for right in seed_groups[index + 1 :]
    )
    if any(overlaps):
        raise ValueError("rl training, validation, and test seeds must be mutually disjoint")
    if evaluation.episodes_per_seed < 1:
        raise ValueError("rl.press.evaluation.episodes_per_seed must be positive")
    for name, value in {
        "success_rate_min": acceptance.success_rate_min,
        "collision_rate_max": acceptance.collision_rate_max,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"rl.press.acceptance.{name} must be between 0 and 1")


def _validate_switches(experiment: ExperimentConfig) -> None:
    switches = experiment.switches
    switches.model.validate()
    for field in fields(switches.verification):
        value = getattr(switches.verification, field.name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"switches.verification.{field.name} must be finite and positive")
    if switches.verification.near_stop_margin_m >= switches.model.travel_m:
        raise ValueError("switch verification margin must lie inside travel")
    if (
        switches.model.back_extent_m + switches.model.contact_offset_m
        >= experiment.rl.press.reset.switch_wall_offset_m
    ):
        raise ValueError("switch back plate must clear the surrounding wall")
    positive_integers = {
        "switches.num_switches": switches.num_switches,
        "switches.max_spawn_retries": switches.max_spawn_retries,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    if not switches.spawn_bounds:
        raise ValueError("switches.spawn_bounds must contain at least one bound")
    for index, bound in enumerate(switches.spawn_bounds):
        if len(bound.min) != 3 or len(bound.max) != 3:
            raise ValueError(f"switches.spawn_bounds[{index}] corners must each have length 3")
        if any(minimum > maximum for minimum, maximum in zip(bound.min, bound.max, strict=True)):
            raise ValueError(
                f"switches.spawn_bounds[{index}] min corner must not exceed max corner"
            )


def _resolve_ros_namespace(*, namespace, vehicle_id: int) -> str:
    if not isinstance(namespace, str):
        raise ValueError(
            f"vehicle id={vehicle_id} ros.namespace must be a string, "
            f"got {type(namespace).__name__}"
        )
    if namespace != namespace.strip():
        raise ValueError(
            f"vehicle id={vehicle_id} ros.namespace must not contain leading or "
            f"trailing whitespace, got {namespace!r}"
        )
    if not namespace.startswith("/"):
        raise ValueError(
            f"vehicle id={vehicle_id} ros.namespace must be absolute, got {namespace!r}"
        )
    if namespace == "/":
        raise ValueError(f"vehicle id={vehicle_id} ros.namespace must not be root '/'")
    if namespace.endswith("/"):
        raise ValueError(
            f"vehicle id={vehicle_id} ros.namespace must not end with '/', got {namespace!r}"
        )
    if "//" in namespace:
        raise ValueError(
            f"vehicle id={vehicle_id} ros.namespace must not contain empty segments, "
            f"got {namespace!r}"
        )
    return namespace


def _ros_frame_prefix_from_namespace(namespace: str) -> str:
    return namespace.removeprefix("/").replace("/", "_")


def _enrich_fleet(cfg, airframe_config_dir: Path) -> None:
    """Fill the derived fields on each already structured vehicle."""
    for label, entry in cfg.vehicles.items():
        action = str(entry.px4.failsafe.offboard_loss_action)
        try:
            action_param = _PX4_OFFBOARD_LOSS_ACTION_PARAMS[action]
        except KeyError as exc:
            raise ValueError(
                f"vehicle {label!r} px4.failsafe.offboard_loss_action must be one of "
                f"{sorted(_PX4_OFFBOARD_LOSS_ACTION_PARAMS)}, got {action!r}"
            ) from exc
        entry.px4.failsafe.offboard_loss_action_param = action_param
        airframe = _airframe_for_drone_asset(str(entry.spawn.usd_path), airframe_config_dir)
        px4_airframe_defaults = (airframe or {}).get("px4", {}).get("bridge", {})
        entry.bridge = OmegaConf.merge(entry.bridge, px4_airframe_defaults)
        entry.bridge.rotor_constant = _compute_rotor_constants(
            float(entry.bridge.max_thrust_per_rotor_g),
            list(entry.bridge.max_rotor_velocity),
        )
        vehicle_id = int(entry["id"])
        _resolve_ros_namespace(namespace=entry.ros.namespace, vehicle_id=vehicle_id)
        entry.mavlink_port = entry.bridge.connection_baseport + vehicle_id
        entry.gcs_port = entry.bridge.gcs_baseport + vehicle_id
        entry.debug_log_path = entry.bridge.debug_log_path.replace("{vehicle}", str(vehicle_id))
        if entry.bridge.enabled:
            asset = Path(str(entry.spawn.usd_path)).name
            if airframe is None:
                raise ValueError(
                    f"no PX4 SITL airframe mapping for drone asset {entry.spawn.usd_path!r} "
                    f"(prim {entry.spawn.prim_name!r}); add it to configs/airframes"
                )
            sitl = airframe.get("px4", {}).get("sitl")
            if not sitl:
                raise ValueError(
                    f"airframe config for drone asset {asset!r} has no px4.sitl mapping; "
                    "add px4.sitl.sim_model and px4.sitl.sys_autostart"
                )
            entry.px4_sim_model = str(sitl["sim_model"])
            entry.px4_sys_autostart = int(sitl["sys_autostart"])


def _config_root(config_path: Path) -> Path:
    return next(
        (parent for parent in config_path.parents if (parent / "base.yaml").is_file()),
        config_path.parent,
    )


def _compose_config(config_path: Path):
    """Compose with an isolated Hydra loader, leaving ``GlobalHydra`` untouched."""

    config_root = _config_root(config_path)
    search_path = create_config_search_path(f"file://{config_root}")
    loader = ConfigLoaderImpl(config_search_path=search_path)
    hydra = Hydra(task_name="robotics_experiments", config_loader=loader)
    cfg = hydra.compose_config(
        config_name=str(config_path.relative_to(config_root).with_suffix("")),
        overrides=[],
        run_mode=RunMode.RUN,
        from_shell=False,
    )
    with open_dict(cfg):
        cfg.pop("hydra")
    return cfg


@functools.cache
def load_config(path: str | PathLike[str], /) -> ExperimentConfig:
    """Compose, enrich, validate, and freeze one explicit experiment config.

    ``path`` is the core boundary and does not read an environment variable or
    mutate Hydra's process-global singleton. Relative paths resolve from the
    repository root, matching the runtime's ``HYDRA_CONFIG=configs/...`` form.
    The returned dataclass graph rejects attribute mutation, and all nested
    lists and mappings are converted to immutable tuples and mapping proxies.
    """

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[3] / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment config does not exist: {config_path}")

    _register_schemas()
    cfg = _compose_config(config_path)
    if OmegaConf.get_type(cfg) is not ExperimentConfig:
        raise ValueError("expected an experiment configuration using base_experiment")
    airframe_config_dir = _config_root(config_path) / "airframes"

    _enrich_fleet(cfg, airframe_config_dir)
    experiment = cast(ExperimentConfig, OmegaConf.to_object(cfg))
    _validate_fleet(experiment.vehicles)
    _validate_rates(experiment)
    _validate_alignment(experiment)
    _validate_rl_policies(experiment)
    _validate_press_task(experiment)
    _validate_switches(experiment)
    return _freeze_config_graph(experiment)


@functools.cache
def load_curriculum_config(path: str | PathLike[str], /) -> CurriculumConfig:
    """Compose and freeze the explicit curriculum root, without fleet enrichment."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[3] / config_path
    config_path = config_path.resolve(strict=True)
    _register_schemas()
    cfg = _compose_config(config_path)
    if OmegaConf.get_type(cfg) is not CurriculumConfig:
        raise ValueError("expected a curriculum configuration using base_curriculum")
    curriculum = cast(CurriculumConfig, OmegaConf.to_object(cfg))
    if re.fullmatch(r"[A-Za-z0-9_-]+", curriculum.policy_name) is None:
        raise ValueError("policy_name must contain only letters, digits, '_' or '-'")
    names = [stage.name for stage in curriculum.stages]
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("curriculum stages must be nonempty with unique nonempty names")
    if any(not stage.config_path.strip() for stage in curriculum.stages):
        raise ValueError("curriculum stage config_path must be nonempty")
    if curriculum.initial_learner is not None and not curriculum.initial_learner.strip():
        raise ValueError("initial_learner must be null or a producer report path")
    return _freeze_config_graph(curriculum)


@functools.cache
def load_switch_model(path: str | PathLike[str], /) -> SwitchModelConfig:
    """Compose and validate a named switch model without generating assets."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[3] / config_path
    _register_schemas()
    cfg = _compose_config(config_path.resolve(strict=True))
    model = cast(SwitchModelConfig, OmegaConf.to_object(cfg.experiment.switches.model))
    model.validate()
    return _freeze_config_graph(model)

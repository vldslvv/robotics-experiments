import functools
import math
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

from robotics_experiments.config._environment import ExperimentConfig, VehicleConfig
from robotics_experiments.config._immutability import _freeze_config_graph
from robotics_experiments.config._schemas import _register_schemas

__all__ = ["load_config"]

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
        "rl.policy_rate_hz": experiment.rl.policy_rate_hz,
    }
    for name, value in rates.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if experiment.rl.policy_rate_hz > experiment.driver.control_rate_hz:
        raise ValueError("rl.policy_rate_hz must not exceed driver.control_rate_hz")
    if experiment.driver.offboard_setpoint_rate_hz > experiment.driver.control_rate_hz:
        raise ValueError("driver.offboard_setpoint_rate_hz must not exceed driver.control_rate_hz")
    if experiment.driver.control_rate_hz > experiment.sim.physics_hz:
        raise ValueError("driver.control_rate_hz must not exceed sim.physics_hz")


def _validate_rl_policies(experiment: ExperimentConfig) -> None:
    if not experiment.rl.training.alignment_artifact.strip():
        raise ValueError("rl.training.alignment_artifact must be non-empty")
    if experiment.rl.training.training_mode not in {"scratch", "actor-only"}:
        raise ValueError("rl.training.training_mode must be either 'scratch' or 'actor-only'")
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


def _validate_press_approach(experiment: ExperimentConfig) -> None:
    press = experiment.rl.press
    if press.curriculum_stage != "approach":
        raise ValueError(
            f"unknown rl.press.curriculum_stage={press.curriculum_stage!r}; " "expected 'approach'"
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

    goal = press.goal
    if not 0.0 < goal.range_min_m <= goal.range_max_m:
        raise ValueError("rl.press.goal range bounds must be positive and ordered")
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
        "linear_speed_max_mps": goal.linear_speed_max_mps,
        "angular_speed_max_radps": goal.angular_speed_max_radps,
        "velocity_reference.wall_normal_gain_per_s": (reference.wall_normal_gain_per_s),
        "velocity_reference.tangential_gain_per_s": reference.tangential_gain_per_s,
        "velocity_reference.tangential_speed_max_mps": (reference.tangential_speed_max_mps),
    }
    for name, value in positive_goal_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"rl.press.goal.{name} must be finite and positive")
    if goal.stable_steps <= 0:
        raise ValueError("rl.press.goal.stable_steps must be positive")

    if handoff.switch_range_min_m <= goal.range_max_m:
        raise ValueError(
            "rl.press.handoff.switch_range_min_m must exceed rl.press.goal.range_max_m"
        )

    for field in fields(press.reward):
        name = field.name
        value = getattr(press.reward, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"rl.press.reward.{name} must be finite and non-negative, got {value!r}"
            )
    evaluation = press.evaluation
    if evaluation.episodes_per_seed <= 0 or not evaluation.seeds:
        raise ValueError("rl.press.evaluation requires seeds and positive episodes_per_seed")
    for name, value in {
        "success_rate_min": evaluation.success_rate_min,
        "collision_rate_max": evaluation.collision_rate_max,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"rl.press.evaluation.{name} must be between 0 and 1")
    if evaluation.selection_metric != "held-out-press-task-quality-v2":
        raise ValueError(
            "rl.press.evaluation.selection_metric must be " "'held-out-press-task-quality-v2'"
        )
    for name in (
        "collision_penalty",
        "flight_failure_penalty",
        "timeout_penalty",
        "non_success_penalty",
        "terminal_range_weight",
        "trajectory_alignment_weight",
        "velocity_tracking_weight",
        "upright_error_weight",
        "action_delta_weight",
        "completion_time_weight",
    ):
        value = getattr(evaluation, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"rl.press.evaluation.{name} must be finite and non-negative")


def _validate_switches(experiment: ExperimentConfig) -> None:
    switches = experiment.switches
    positive_integers = {
        "switches.num_switches": switches.num_switches,
        "switches.press_confirm_steps": switches.press_confirm_steps,
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


def _compose_config(config_path: Path):
    """Compose with an isolated Hydra loader, leaving ``GlobalHydra`` untouched."""

    search_path = create_config_search_path(f"file://{config_path.parent}")
    loader = ConfigLoaderImpl(config_search_path=search_path)
    hydra = Hydra(task_name="robotics_experiments", config_loader=loader)
    cfg = hydra.compose_config(
        config_name=config_path.stem,
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
    airframe_config_dir = config_path.parent / "airframes"

    _enrich_fleet(cfg, airframe_config_dir)
    experiment = cast(ExperimentConfig, OmegaConf.to_object(cfg))
    _validate_fleet(experiment.vehicles)
    _validate_rates(experiment)
    _validate_rl_policies(experiment)
    _validate_press_approach(experiment)
    _validate_switches(experiment)
    return _freeze_config_graph(experiment)

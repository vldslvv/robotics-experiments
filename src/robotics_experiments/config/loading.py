import functools
import math
from collections.abc import Iterable
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
    for entry in cfg.vehicles.values():
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
    return _freeze_config_graph(experiment)

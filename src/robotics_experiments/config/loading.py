import functools
import os
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_config_search_path
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict

from robotics_experiments.config._immutability import _freeze_config_graph
from robotics_experiments.config._schemas import _register_schemas
from robotics_experiments.config.types import (
    ExperimentConfig,
    Px4SitlLaunchConfig,
    VehicleConfig,
)

__all__ = [
    "get_config",
    "get_enabled_px4_launch_configs",
    "load_config",
    "reset_config_cache",
]

_DEFAULT_CONF_DIR = Path(__file__).resolve().parents[3] / "configs"
_DEFAULT_CONFIG_NAME = "takeoff_quad"
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"


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


def _resolve_config_path() -> Path:
    raw_path = os.environ.get(_CONFIG_PATH_ENV)
    if raw_path is None:
        return _DEFAULT_CONF_DIR / f"{_DEFAULT_CONFIG_NAME}.yaml"

    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return path.resolve()


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

        if v.ros_frame_prefix in ros_frame_prefixes:
            raise ValueError(
                f"duplicate vehicle ROS frame prefix {v.ros_frame_prefix!r} " f"(vehicle id={v.id})"
            )
        ros_frame_prefixes.add(v.ros_frame_prefix)

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


def _resolve_ros_namespace(namespace, *, vehicle_id: int) -> str:
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
        ros_namespace = _resolve_ros_namespace(entry.ros.namespace, vehicle_id=vehicle_id)
        entry.mavlink_port = entry.bridge.connection_baseport + vehicle_id
        entry.px4_instance_index = vehicle_id
        entry.gcs_port = entry.bridge.gcs_baseport + vehicle_id
        entry.ros_frame_prefix = _ros_frame_prefix_from_namespace(ros_namespace)
        entry.debug_log_path = entry.bridge.debug_log_path.replace("{vehicle}", str(vehicle_id))


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


def load_config(path: str | os.PathLike[str]) -> ExperimentConfig:
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
    return _freeze_config_graph(experiment)


@functools.cache
def get_config() -> ExperimentConfig:
    """Return the process-wide composed :class:`ExperimentConfig`, loading once.

    The first call composes Hydra against the selected top-level experiment
    config and caches the resulting dataclass.
    Subsequent calls return the cached instance - safe to call from any extension's
    ``on_startup`` and from per-step hot paths.

    Returns:
        The composed ``ExperimentConfig`` instance.
    """
    return load_config(_resolve_config_path())


def _resolve_px4_sitl_launch_config(
    vehicle: VehicleConfig, airframe_config_dir: Path
) -> Px4SitlLaunchConfig:
    """Derive the PX4 SITL launch airframe for one resolved vehicle.

    The drone USD asset name selects the airframe record in ``configs/airframes``,
    whose ``px4.sitl`` block supplies the PX4 ``PX4_SIM_MODEL`` /
    ``PX4_SYS_AUTOSTART`` pair so each asset launches its generated PX4 control
    allocation instead of falling back to a stock quad. An unknown PX4-flyable
    asset raises ``ValueError``
    (naming the asset and prim) rather than silently flying the wrong airframe.
    """
    asset = Path(vehicle.spawn.usd_path).name
    airframe = _airframe_for_drone_asset(vehicle.spawn.usd_path, airframe_config_dir)
    if airframe is None:
        raise ValueError(
            f"no PX4 SITL airframe mapping for drone asset {vehicle.spawn.usd_path!r} "
            f"(prim {vehicle.spawn.prim_name!r}); add it to configs/airframes"
        )
    sitl = airframe.get("px4", {}).get("sitl")
    if not sitl:
        raise ValueError(
            f"airframe config for drone asset {asset!r} has no px4.sitl mapping; "
            "add px4.sitl.sim_model and px4.sitl.sys_autostart"
        )
    return Px4SitlLaunchConfig(
        instance_index=vehicle.px4_instance_index,
        sim_model=str(sitl["sim_model"]),
        sys_autostart=int(sitl["sys_autostart"]),
    )


def get_enabled_px4_launch_configs() -> list[Px4SitlLaunchConfig]:
    """Return PX4 SITL launch specs for the PX4-enabled vehicles, in fleet order.

    The host-side launch scripts iterate these to start one PX4 container per
    vehicle, threading each spec's ``instance_index`` / ``sim_model`` /
    ``sys_autostart`` in as ``PX4_INSTANCE`` / ``PX4_SIM_MODEL`` /
    ``PX4_SYS_AUTOSTART`` so each container launches the airframe matching its
    drone asset.
    """
    airframe_config_dir = _resolve_config_path().parent / "airframes"
    return [
        _resolve_px4_sitl_launch_config(v, airframe_config_dir)
        for v in get_config().vehicles.values()
        if v.bridge.enabled
    ]


def reset_config_cache() -> None:
    """Drop the cached config so the next :func:`get_config` recomposes.

    Intended for tests and interactive reload workflows; production extensions
    should not need to call this.
    """
    get_config.cache_clear()
    _airframes_by_output_filename.cache_clear()

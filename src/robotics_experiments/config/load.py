import functools
import os
from collections.abc import Iterable
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from robotics_experiments.config._schemas import _register_schemas
from robotics_experiments.config.types import (
    DroneConfig,
    EnvConfig,
    Px4BridgeConfig,
    Px4SitlLaunchConfig,
    ResolvedVehicle,
    RoutineConfig,
)

__all__ = [
    "get_config",
    "get_enabled_px4_instance_indices",
    "get_enabled_px4_launch_configs",
    "resolve_drone_bridge_config",
    "resolve_px4_sitl_launch_config",
]

_DEFAULT_CONF_DIR = Path(__file__).resolve().parents[3] / "configs"
_DEFAULT_CONFIG_NAME = "takeoff_quad"
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_AIRFRAME_CONFIG_DIR = _DEFAULT_CONF_DIR / "airframes"


@functools.cache
def _airframes_by_output_filename() -> dict[str, dict]:
    airframes: dict[str, dict] = {}
    for path in sorted(_AIRFRAME_CONFIG_DIR.glob("*.yaml")):
        data = cast(dict, OmegaConf.to_container(OmegaConf.load(path), resolve=True))
        airframes[str(data["output_filename"])] = data
    return airframes


def _airframe_for_drone_asset(usd_path: str) -> dict | None:
    return _airframes_by_output_filename().get(Path(usd_path).name)


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


def _validate_fleet(vehicles: list[ResolvedVehicle]) -> None:
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
    for v in vehicles:
        if len(v.spawn_xy) != 2:
            raise ValueError(f"vehicle id={v.id} spawn_xy must have length 2, got {v.spawn_xy!r}")
        if v.id < 0:
            raise ValueError(f"vehicle id must be non-negative, got {v.id}")
        if v.id in ids:
            raise ValueError(f"duplicate vehicle id={v.id}")
        ids.add(v.id)

        endpoint = (v.px4.connection_type, v.px4.connection_ip, v.mavlink_port)
        if endpoint in endpoints:
            raise ValueError(f"duplicate vehicle MAVLink endpoint {endpoint} (vehicle id={v.id})")
        endpoints.add(endpoint)

        if v.prim_name in prim_names:
            raise ValueError(
                f"duplicate vehicle drone.prim_name {v.prim_name!r} (vehicle id={v.id})"
            )
        prim_names.add(v.prim_name)

        if v.ros_namespace in ros_namespaces:
            raise ValueError(
                f"duplicate vehicle ros.namespace {v.ros_namespace!r} (vehicle id={v.id})"
            )
        ros_namespaces.add(v.ros_namespace)

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
                    f"colliding vehicle debug_log_path {v.px4.debug_log_path!r} "
                    f"(vehicle id={v.id}); add a {{vehicle}} token to disambiguate"
                )
            log_paths.add(v.debug_log_path)


def _resolve_ros_namespace(entry, *, vehicle_id: int) -> str:
    ros = entry.get("ros")
    if ros is None or "namespace" not in ros:
        raise ValueError(f"vehicle id={vehicle_id} must define ros.namespace")

    namespace = ros["namespace"]
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


def _resolve_fleet(cfg) -> list[ResolvedVehicle]:
    """Enrich the composed config's ``vehicles`` into a resolved fleet.

    The single enrichment step from authored Hydra data to a self-describing
    runtime fleet. Each ``vehicles`` entry is a *partial* override; its
    ``drone`` / ``px4`` partials are merged over the structured global
    ``cfg.drone`` / ``cfg.px4`` (``OmegaConf.merge`` both fills defaults and
    rejects unknown keys) into a pure resolved config, from which the derived
    runtime fields (``mavlink_port``, ``px4_instance_index``, the
    ``px4.gcs_baseport + id`` GCS port, the rendered ``debug_log_path``,
    ``prim_name``) are computed once. An empty list is a valid empty fleet
    (0..N is handled uniformly).
    """
    raw = list(cfg.get("vehicles") or [])
    resolved: list[ResolvedVehicle] = []
    for entry in raw:
        drone = OmegaConf.merge(cfg.drone, entry.get("drone", {}))
        airframe = _airframe_for_drone_asset(str(drone.usd_path))
        px4_airframe_defaults = (airframe or {}).get("px4", {}).get("bridge", {})
        px4 = OmegaConf.merge(cfg.px4, px4_airframe_defaults, entry.get("px4", {}))
        # rotor_constant is not authored: derive it from the cited per-rotor max
        # thrust + max_rotor_velocity + standard gravity, so a datasheet thrust change
        # is a single-line edit (see _compute_rotor_constants).
        px4.rotor_constant = _compute_rotor_constants(
            float(px4.max_thrust_per_rotor_g),
            list(px4.max_rotor_velocity),
        )
        routine = OmegaConf.merge(cfg.routine, entry.get("routine", {}))
        drone_cfg = cast(DroneConfig, OmegaConf.to_object(drone))
        px4_cfg = cast(Px4BridgeConfig, OmegaConf.to_object(px4))
        routine_cfg = cast(RoutineConfig, OmegaConf.to_object(routine))
        vehicle_id = int(entry["id"])
        ros_namespace = _resolve_ros_namespace(entry, vehicle_id=vehicle_id)
        resolved.append(
            ResolvedVehicle(
                id=vehicle_id,
                spawn_xy=tuple(entry["spawn_xy"]),
                drone=drone_cfg,
                px4=px4_cfg,
                routine=routine_cfg,
                mavlink_port=px4_cfg.connection_baseport + vehicle_id,
                px4_instance_index=vehicle_id,
                gcs_port=px4_cfg.gcs_baseport + vehicle_id,
                ros_namespace=ros_namespace,
                ros_frame_prefix=_ros_frame_prefix_from_namespace(ros_namespace),
                debug_log_path=px4_cfg.debug_log_path.replace("{vehicle}", str(vehicle_id)),
                prim_name=drone_cfg.prim_name,
            )
        )
    _validate_fleet(resolved)
    return resolved


@functools.cache
def get_config() -> EnvConfig:
    """Return the process-wide composed :class:`EnvConfig`, loading once.

    The first call composes Hydra against the selected top-level experiment
    config and caches the resulting dataclass.
    Subsequent calls return the cached instance - safe to call from any extension's
    ``on_startup`` and from per-step hot paths.

    Returns:
        The composed ``EnvConfig`` instance.
    """
    _register_schemas()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_path = _resolve_config_path()
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        cfg = compose(config_name=config_path.stem)

    # to_object turns the loosely typed cfg.vehicles into raw dicts; replace
    # them with the enriched, validated resolved fleet before caching/returning.
    env = cast(EnvConfig, OmegaConf.to_object(cfg))
    # Derive rotor_constant for the global default px4 too (the per-vehicle
    # resolution does the same for each fleet entry), so the authored config
    # never duplicates the computed thrust coefficient.
    env.px4.rotor_constant = _compute_rotor_constants(
        env.px4.max_thrust_per_rotor_g,
        env.px4.max_rotor_velocity,
    )
    env.vehicles = _resolve_fleet(cfg)
    return env


def resolve_px4_sitl_launch_config(vehicle: ResolvedVehicle) -> Px4SitlLaunchConfig:
    """Derive the PX4 SITL launch airframe for one resolved vehicle.

    The drone USD asset name selects the airframe record in ``configs/airframes``,
    whose ``px4.sitl`` block supplies the PX4 ``PX4_SIM_MODEL`` /
    ``PX4_SYS_AUTOSTART`` pair so each asset launches its generated PX4 control
    allocation instead of falling back to a stock quad. An unknown PX4-flyable
    asset raises ``ValueError``
    (naming the asset and prim) rather than silently flying the wrong airframe.
    """
    asset = Path(vehicle.drone.usd_path).name
    airframe = _airframe_for_drone_asset(vehicle.drone.usd_path)
    if airframe is None:
        raise ValueError(
            f"no PX4 SITL airframe mapping for drone asset {vehicle.drone.usd_path!r} "
            f"(prim {vehicle.prim_name!r}); add it to configs/airframes"
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


def resolve_drone_bridge_config(usd_path: str) -> Px4BridgeConfig:
    """Resolve the per-asset PX4 bridge config for a drone USD asset.

    The same airframe enrichment :func:`_resolve_fleet` applies per vehicle
    (global ``px4`` merged with the asset's ``configs/airframes`` bridge defaults,
    then ``rotor_constant`` computed from the cited thrust), exposed for consumers
    that spawn a drone outside the fleet path -- notably the Isaac Lab press env,
    which needs the airframe's thrust constants (``rotor_constant`` /
    ``max_rotor_velocity`` / ``rolling_moment_coefficient`` / ``linear_drag``) for
    its in-Lab actuation instead of the global iris fallback. Falls back to the
    global ``px4`` defaults for an asset with no airframe mapping.
    """
    global_px4 = get_config().px4
    airframe = _airframe_for_drone_asset(usd_path)
    bridge_defaults = (airframe or {}).get("px4", {}).get("bridge", {})
    px4 = OmegaConf.merge(global_px4, bridge_defaults)
    px4.rotor_constant = _compute_rotor_constants(
        float(px4.max_thrust_per_rotor_g),
        list(px4.max_rotor_velocity),
    )
    return cast(Px4BridgeConfig, OmegaConf.to_object(px4))


def get_enabled_px4_launch_configs() -> list[Px4SitlLaunchConfig]:
    """Return PX4 SITL launch specs for the PX4-enabled vehicles, in fleet order.

    The host-side launch scripts iterate these to start one PX4 container per
    vehicle, threading each spec's ``instance_index`` / ``sim_model`` /
    ``sys_autostart`` in as ``PX4_INSTANCE`` / ``PX4_SIM_MODEL`` /
    ``PX4_SYS_AUTOSTART`` so each container launches the airframe matching its
    drone asset.
    """
    return [resolve_px4_sitl_launch_config(v) for v in get_config().vehicles if v.px4.enabled]


def get_enabled_px4_instance_indices() -> list[int]:
    """Return the PX4 ``-i`` instance indices of the PX4-enabled vehicles.

    The host-side launch scripts iterate these (the resolved
    ``px4_instance_index``, in fleet order) instead of a blind ``0..count``
    loop, so non-contiguous vehicle ids keep their derived MAVLink/PX4 port
    offsets.
    """
    return [c.instance_index for c in get_enabled_px4_launch_configs()]


def _reset_config() -> None:
    """Drop the cached config so the next :func:`get_config` recomposes.

    Intended for tests and for interactive reload workflows; production
    extensions should not need to call this.
    """
    get_config.cache_clear()
    _airframes_by_output_filename.cache_clear()

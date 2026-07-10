"""Pure airframe configuration parsing and mass-property calculations."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

__all__ = [
    "AirframeConfig",
    "ArmGeometry",
    "ArmPlacement",
    "BoomPart",
    "BoxPart",
    "BuilderConfig",
    "CylinderPart",
    "PusherConfig",
    "PusherDriveConfig",
    "compute_body_mass_properties",
    "load_builder_config",
]

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "usd_builder"
Vec3 = tuple[float, float, float]


@dataclass
class BoxPart:
    name: str
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    mass: float
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class CylinderPart:
    name: str
    radius: float
    height: float
    position: tuple[float, float, float]
    mass: float
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BoomPart:
    # The arm tube. Massless geometry (mass folded into the frame), so no `mass`
    # field. `length` is the tip-to-tip radial reach of the boom (origin to motor);
    # the builder lays it down along the arm's local +X and derives the capsule's
    # cylinder height from it.
    name: str
    radius: float
    length: float


@dataclass
class ArmGeometry:
    # Geometry of one arm, authored once and shared by every arm of the airframe.
    # Every position is in the arm-local frame, where +X points outboard along the
    # boom; the per-arm azimuth lives on the arm Xform (see ArmPlacement), so the
    # ESC needs no rotation to lie parallel to the boom.
    boom: BoomPart
    motor: CylinderPart  # the brushless can, at the boom tip
    esc: BoxPart  # ESC riding on the boom
    propeller: CylinderPart  # disc above the motor
    prop_num_blades: int  # cosmetic blade count for the propeller visual


@dataclass
class ArmPlacement:
    # One physical arm: where the shared ArmGeometry is mounted and which motor it
    # carries. `azimuth_deg` is the sole source of the arm's heading.
    name: str
    azimuth_deg: float
    spin_dir: str  # "CCW" | "CW"
    px4_motor_index: int


@dataclass
class AirframeConfig:
    name: str
    output_filename: str
    prim_name: str

    # Articulation-level rigid-body mass/CoM/inertia are NOT authored: they are
    # computed from the component masses + geometry by compute_body_mass_properties
    # so every physics number traces to a cited per-component value. (Isaac Lab's
    # _drone_cfg.py adds no mass override; the env's EventCfg randomization scales
    # the spawned mass per-env at reset.)

    # Frame
    center_plate: BoxPart

    # Propulsion: one shared arm geometry plus a placement per physical arm.
    arm_geometry: ArmGeometry
    arms: list[ArmPlacement]

    # Avionics
    pixhawk: BoxPart
    jetson: BoxPart

    # Sensors
    realsense: BoxPart
    optical_flow: BoxPart
    lidar: CylinderPart

    # Power
    battery: BoxPart
    wiring: BoxPart

    # Pusher (None == no pusher mechanism authored for this airframe). When
    # present, `enabled` is the per-drone on/off switch: a disabled pusher keeps
    # its geometry/drive config but is not authored into the USD.
    pusher: PusherConfig | None = None


@dataclass
class PusherDriveConfig:
    travel: float
    stiffness: float
    damping: float


@dataclass
class PusherConfig:
    enabled: bool
    pad: CylinderPart
    drive: PusherDriveConfig


@dataclass
class BuilderConfig:
    output_dir: str
    airframe: AirframeConfig


# --- Config loading ---------------------------------------------------------


def _as_tuple3(data: Any, label: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in data)
    if len(values) != 3:
        raise ValueError(f"{label} must have exactly 3 values, got {values}")
    return values


def _box(data: dict[str, Any], label: str) -> BoxPart:
    return BoxPart(
        name=str(data["name"]),
        size=_as_tuple3(data["size"], f"{label}.size"),
        position=_as_tuple3(data["position"], f"{label}.position"),
        mass=float(data["mass"]),
        rotation_deg=_as_tuple3(data.get("rotation_deg", (0.0, 0.0, 0.0)), f"{label}.rotation_deg"),
    )


def _cylinder(data: dict[str, Any], label: str) -> CylinderPart:
    return CylinderPart(
        name=str(data["name"]),
        radius=float(data["radius"]),
        height=float(data["height"]),
        position=_as_tuple3(data["position"], f"{label}.position"),
        mass=float(data["mass"]),
        rotation_deg=_as_tuple3(data.get("rotation_deg", (0.0, 0.0, 0.0)), f"{label}.rotation_deg"),
    )


def _boom(data: dict[str, Any], label: str) -> BoomPart:
    return BoomPart(
        name=str(data["name"]),
        radius=float(data["radius"]),
        length=float(data["length"]),
    )


def _arm_geometry(data: dict[str, Any], label: str) -> ArmGeometry:
    propeller = data["propeller"]
    return ArmGeometry(
        boom=_boom(data["boom"], f"{label}.boom"),
        motor=_cylinder(data["motor"], f"{label}.motor"),
        esc=_box(data["esc"], f"{label}.esc"),
        propeller=_cylinder(propeller, f"{label}.propeller"),
        prop_num_blades=int(propeller["num_blades"]),
    )


def _arm_placement(data: dict[str, Any], label: str) -> ArmPlacement:
    return ArmPlacement(
        name=str(data["name"]),
        azimuth_deg=float(data["azimuth_deg"]),
        spin_dir=str(data["spin_dir"]),
        px4_motor_index=int(data["px4_motor_index"]),
    )


def _pusher_config(data: dict[str, Any] | None) -> PusherConfig | None:
    if data is None:
        return None
    drive = data["drive"]
    return PusherConfig(
        enabled=bool(data["enabled"]),
        pad=_cylinder(data["pad"], "airframes.pusher.pad"),
        drive=PusherDriveConfig(
            travel=float(drive["travel"]),
            stiffness=float(drive["stiffness"]),
            damping=float(drive["damping"]),
        ),
    )


def _airframe_config(data: dict[str, Any]) -> AirframeConfig:
    return AirframeConfig(
        name=str(data["name"]),
        output_filename=str(data["output_filename"]),
        prim_name=str(data["prim_name"]),
        center_plate=_box(data["center_plate"], "airframes.center_plate"),
        arm_geometry=_arm_geometry(data["arm_geometry"], "airframes.arm_geometry"),
        arms=[_arm_placement(arm, f"airframes.arms[{i}]") for i, arm in enumerate(data["arms"])],
        pixhawk=_box(data["pixhawk"], "airframes.pixhawk"),
        jetson=_box(data["jetson"], "airframes.jetson"),
        realsense=_box(data["realsense"], "airframes.realsense"),
        optical_flow=_box(data["optical_flow"], "airframes.optical_flow"),
        lidar=_cylinder(data["lidar"], "airframes.lidar"),
        battery=_box(data["battery"], "airframes.battery"),
        wiring=_box(data["wiring"], "airframes.wiring"),
        pusher=_pusher_config(data.get("pusher")),
    )


def _builder_config(data: dict[str, Any]) -> BuilderConfig:
    return BuilderConfig(
        output_dir=str(data["output_dir"]),
        airframe=_airframe_config(data["airframes"]),
    )


def load_builder_config(
    *,
    config_dir: str | os.PathLike[str] = CONFIG_DIR,
    overrides: list[str] | None = None,
) -> BuilderConfig:
    """Compose the Hydra USD-builder config and convert it to dataclasses."""

    config_dir_path = Path(config_dir).resolve()
    hydra_config_dir = config_dir_path.parent
    config_name = f"{config_dir_path.name}/config"

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(hydra_config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return _builder_config(OmegaConf.to_container(cfg, resolve=True))


def _airframe_config_dir(config_dir: str | os.PathLike[str] = CONFIG_DIR) -> Path:
    config_dir_path = Path(config_dir).resolve()
    return config_dir_path.parent / "airframes"


def _available_airframe_config_names(config_dir: str | os.PathLike[str] = CONFIG_DIR) -> list[str]:
    airframe_dir = _airframe_config_dir(config_dir)
    return sorted(path.stem for path in airframe_dir.glob("*.yaml") if path.is_file())


# Body mass properties are derived from the same component placements consumed
# by USD authoring. The separate pusher pad and massless booms are excluded.
def _euler_xyz_matrix(rotation_deg: tuple[float, float, float]) -> np.ndarray:
    """Rotation matrix for USD ``XformCommonAPI`` SetRotate (XYZ Euler degrees)."""
    rx, ry, rz = (math.radians(angle) for angle in rotation_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _part_principal_inertia(part: BoxPart | CylinderPart) -> np.ndarray:
    """Diagonal inertia tensor of a part about its own center, in its local frame."""
    if isinstance(part, BoxPart):
        sx, sy, sz = part.size
        return np.diag(
            [
                part.mass * (sy * sy + sz * sz) / 12.0,
                part.mass * (sx * sx + sz * sz) / 12.0,
                part.mass * (sx * sx + sy * sy) / 12.0,
            ]
        )
    r, h = part.radius, part.height  # cylinder axis is local Z
    return np.diag(
        [
            part.mass * (3.0 * r * r + h * h) / 12.0,
            part.mass * (3.0 * r * r + h * h) / 12.0,
            part.mass * r * r / 2.0,
        ]
    )


def _part_body_properties(
    part: BoxPart | CylinderPart, frame_rot: np.ndarray, body_pos: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, body-frame inertia about the part center, body-frame center position).

    ``frame_rot`` is the enclosing frame rotation (the arm azimuth, or identity);
    it is composed with the part's own ``rotation_deg`` to orient the local
    inertia tensor into the body frame.
    """
    rot = frame_rot @ _euler_xyz_matrix(part.rotation_deg)
    inertia_body = rot @ _part_principal_inertia(part) @ rot.T
    return part.mass, inertia_body, np.asarray(body_pos, dtype=float)


def compute_body_mass_properties(
    cfg: AirframeConfig,
) -> tuple[float, Vec3, Vec3]:
    """Compute (total_mass, center_of_mass, diagonal_inertia) of the drone body.

    Sums every body component using the SAME resolved body-frame placements the
    authoring path uses (arm parts rotated by ``azimuth_deg``; the center-plate-top
    z-shift for pixhawk/jetson/lidar), then parallel-axes each part's inertia to
    the mass-weighted CoM. Off-diagonal products of inertia are ~0 by the airframe
    symmetry and are discarded (PhysX reads only the diagonal in the principal
    frame).
    """
    identity = np.eye(3)
    plate_top = cfg.center_plate.size[2] / 2.0
    parts: list[tuple[float, np.ndarray, np.ndarray]] = [
        _part_body_properties(cfg.center_plate, identity, np.asarray(cfg.center_plate.position)),
    ]

    for placement in cfg.arms:
        arm_rot = _euler_xyz_matrix((0.0, 0.0, placement.azimuth_deg))
        for part in (cfg.arm_geometry.motor, cfg.arm_geometry.esc, cfg.arm_geometry.propeller):
            body_pos = arm_rot @ np.asarray(part.position, dtype=float)
            parts.append(_part_body_properties(part, arm_rot, body_pos))

    parts.append(
        _part_body_properties(
            cfg.pixhawk,
            identity,
            np.asarray(_shift_z(cfg.pixhawk.position, plate_top)),
        )
    )
    parts.append(
        _part_body_properties(
            cfg.jetson,
            identity,
            np.asarray(_shift_z(cfg.jetson.position, plate_top)),
        )
    )
    parts.append(_part_body_properties(cfg.realsense, identity, np.asarray(cfg.realsense.position)))
    parts.append(
        _part_body_properties(cfg.optical_flow, identity, np.asarray(cfg.optical_flow.position))
    )
    parts.append(
        _part_body_properties(
            cfg.lidar,
            identity,
            np.asarray(_shift_z(cfg.lidar.position, plate_top)),
        )
    )
    parts.append(_part_body_properties(cfg.battery, identity, np.asarray(cfg.battery.position)))
    parts.append(_part_body_properties(cfg.wiring, identity, np.asarray(cfg.wiring.position)))

    total_mass = sum(mass for mass, _, _ in parts)
    com = sum((mass * pos for mass, _, pos in parts), start=np.zeros(3)) / total_mass

    inertia = np.zeros((3, 3))
    for mass, inertia_body, pos in parts:
        offset = pos - com
        inertia += inertia_body + mass * (
            float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
        )

    com_tuple = (float(com[0]), float(com[1]), float(com[2]))
    diag = (float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2]))
    return float(total_mass), com_tuple, diag


def _shift_z(pos: Vec3, dz: float) -> Vec3:
    return (pos[0], pos[1], pos[2] + dz)


_P = TypeVar("_P", BoxPart, CylinderPart)


def _replace_pos(part: _P, new_pos: Vec3) -> _P:
    return replace(part, position=new_pos)

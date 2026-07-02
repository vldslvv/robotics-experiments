# build_copter_usd_standalone.py - Hydra-config-driven drone USD generator
#
# Run with standalone OpenUSD Python bindings:
#   python isaac/runtime/standalone/build_copter_usd_standalone.py [config_name] [overrides...]
#
# Available configs: holybro_x500_v2 (default), hexacopter_s550
#
# --- Boundary with Isaac Lab (workflows/isaac_lab/tasks/_drone_cfg.py) ---
# This file authors the USD once. Anything topological / geometric lives here:
#   - Prim tree, meshes, collision shapes
#   - Joint topology (body pairs, axis, limits, relative frames)
#   - Custom attributes (e.g. PX4 rotor mapping metadata on props)
#   - Articulation root marker (ArticulationRootAPI on the wrapper prim)
# The Isaac Lab ArticulationCfg (_drone_cfg.py) wraps the USD at spawn time:
#   - Actuator stiffness / damping / effort limits
#   - Initial pose and joint positions
#   - Rigid body solver flags (CCD, max velocities, gravity toggle)
# Rule: stable names in USD -> Isaac Lab matches joints/bodies by regex.

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ASSETS_DIR = "isaac/assets/LightSwitchWarehouse"
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "usd_builder"


# --- Component dataclasses --------------------------------------------------
# Each real-world part owns its name, shape, position, and mass. Box vs.
# cylinder vs. capsule are kept as separate classes - not generalized - because
# their dimensions don't share a common shape.
#
# Position frames (what each component's `position` is relative to):
#   center_plate    -> body root
#   arms[i]         -> body root (midpoint of the boom)
#   motors[i]       -> body root (motor origin, top of the arm)
#   motors[i].esc   -> that motor's origin
#   motors[i].propeller -> that motor's origin
#   pixhawk         -> top surface of the center plate (z=0 == flush)
#   jetson          -> top surface of the center plate
#   realsense       -> body root
#   optical_flow    -> body root
#   lidar           -> top surface of the center plate
#   battery         -> body root
#   wiring          -> body root
#   pusher.pad      -> body root


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


def _build_all_airframe_usds(
    *,
    config_dir: str | os.PathLike[str] = CONFIG_DIR,
    overrides: list[str] | None = None,
) -> list[str]:
    paths: list[str] = []
    for name in _available_airframe_config_names(config_dir):
        config = load_builder_config(
            config_dir=config_dir,
            overrides=[f"airframes={name}", *(overrides or [])],
        )
        paths.append(build_drone_usd(config.airframe, output_dir=config.output_dir))
    return paths


# --- Body mass properties (computed, not authored) --------------------------
# total_mass / center_of_mass / diagonal_inertia are derived from the component
# masses + geometry so every physics number traces to a cited per-component leaf
# (change a motor mass -> regenerate -> mass/CoM/inertia all follow). The pusher
# pad is excluded: it is a separate rigid body with its own mass, not part of the
# drone body. The boom is massless (folded into the frame plate).


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


# --- USD authoring helpers --------------------------------------------------

Vec3 = tuple[float, float, float]


def _set_mass(prim: Usd.Prim, mass_kg: float) -> None:
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass_kg)


def _apply_pose(
    xformable: UsdGeom.Xformable,
    position: Vec3,
    rotation_deg: Vec3 = (0.0, 0.0, 0.0),
) -> None:
    UsdGeom.XformCommonAPI(xformable).SetTranslate(Gf.Vec3d(*position))
    if rotation_deg != (0.0, 0.0, 0.0):
        UsdGeom.XformCommonAPI(xformable).SetRotate(Gf.Vec3f(*rotation_deg))


def _author_box(stage: Usd.Stage, prim_path: str, part: BoxPart) -> UsdGeom.Xform:
    xform = UsdGeom.Xform.Define(stage, prim_path)
    _apply_pose(xform, part.position, part.rotation_deg)
    _set_mass(xform.GetPrim(), part.mass)
    geom = UsdGeom.Cube.Define(stage, f"{prim_path}/geom")
    geom.CreateSizeAttr(1.0)
    geom.AddScaleOp().Set(Gf.Vec3f(*part.size))
    return xform


def _author_cylinder(stage: Usd.Stage, prim_path: str, part: CylinderPart) -> UsdGeom.Cylinder:
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(part.radius)
    cyl.CreateHeightAttr(part.height)
    _apply_pose(cyl, part.position, part.rotation_deg)
    _set_mass(cyl.GetPrim(), part.mass)
    return cyl


def _author_propeller_visual(
    stage: Usd.Stage, prop_path: str, prop_part: CylinderPart, num_blades: int
) -> None:
    """Author the cosmetic propeller geometry under the rotor marker Xform.

    A small central hub cylinder plus ``num_blades`` thin blade boxes fanned
    evenly about Z. All geometry is collision-free and visual only; it derives
    its dimensions from the propeller radius/height (no extra geometry knobs) and
    spins as a unit because it is parented under the marker's RotateZ op.
    """
    if num_blades < 1:
        raise ValueError(
            f"propeller num_blades must be >= 1, got {num_blades} for {prop_part.name}"
        )

    hub = UsdGeom.Cylinder.Define(stage, f"{prop_path}/hub")
    hub.CreateRadiusAttr(prop_part.radius * 0.12)
    hub.CreateHeightAttr(prop_part.height)

    blade_length = prop_part.radius
    blade_width = prop_part.radius * 0.14
    for blade_idx in range(num_blades):
        angle_deg = blade_idx * (360.0 / num_blades)
        blade = UsdGeom.Xform.Define(stage, f"{prop_path}/blade_{blade_idx + 1}")
        # Order [rotateZ, translate]: the blade box is offset along +X so its
        # inner end meets the hub, then the whole blade is rotated about Z to its
        # slot in the fan.
        blade.AddRotateZOp().Set(angle_deg)
        blade.AddTranslateOp().Set(Gf.Vec3d(blade_length * 0.5, 0.0, 0.0))
        geom = UsdGeom.Cube.Define(stage, f"{prop_path}/blade_{blade_idx + 1}/geom")
        geom.CreateSizeAttr(1.0)
        geom.AddScaleOp().Set(Gf.Vec3f(blade_length, blade_width, prop_part.height))


def _author_arm(
    stage: Usd.Stage,
    arms_root: str,
    idx: int,
    geom: ArmGeometry,
    placement: ArmPlacement,
) -> None:
    """Author one arm assembly under its own azimuth Xform.

    Everything below ``arm_{i}`` is expressed in the arm-local frame (+X outboard
    along the boom). The per-arm heading lives only on the ``arm_{i}`` Xform, so
    the boom, motor, ESC, and propeller all inherit it -- the ESC is parallel to
    the boom by construction and the motor sits at the boom tip.
    """
    arm_path = f"{arms_root}/arm_{idx+1}"
    arm = UsdGeom.Xform.Define(stage, arm_path)
    UsdGeom.XformCommonAPI(arm).SetRotate(Gf.Vec3f(0.0, 0.0, placement.azimuth_deg))

    # Boom: laid along local +X, centered at length/2. A capsule's long axis is
    # local Z, so rotate 90 about Y; the cylinder height excludes the two caps so
    # the rounded tips span exactly origin -> motor.
    boom = geom.boom
    cap = UsdGeom.Capsule.Define(stage, f"{arm_path}/boom")
    cap.CreateRadiusAttr(boom.radius)
    cap.CreateHeightAttr(boom.length - 2 * boom.radius)
    _apply_pose(cap, (boom.length / 2.0, 0.0, 0.0), (0.0, 90.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())

    # Motor can at the boom tip; ESC riding flat on the boom (its local X is the
    # boom direction, so no rotation is needed -- the arm azimuth orients it).
    _author_cylinder(stage, f"{arm_path}/motor", geom.motor)
    _author_box(stage, f"{arm_path}/esc", geom.esc)

    # Propeller marker. The Xform keeps the prop path, PX4 marker attributes (from
    # the placement), translation, mass, the cosmetic RotateZ spin op (driven each
    # physics tick), and the hub/blade visual. rotor_discovery and the bridge read
    # the marker's translation, which the RotateZ op leaves unchanged.
    prop_part = geom.propeller
    prop = UsdGeom.Xform.Define(stage, f"{arm_path}/prop")
    _apply_pose(prop, prop_part.position, prop_part.rotation_deg)
    prop.AddRotateZOp().Set(0.0)
    prop.GetPrim().CreateAttribute("drone:rotorMarker", Sdf.ValueTypeNames.Bool).Set(True)
    prop.GetPrim().CreateAttribute("drone:px4MotorIndex", Sdf.ValueTypeNames.Int).Set(
        placement.px4_motor_index
    )
    prop.GetPrim().CreateAttribute("drone:spinDir", Sdf.ValueTypeNames.String).Set(
        placement.spin_dir
    )
    _set_mass(prop.GetPrim(), prop_part.mass)
    _author_propeller_visual(stage, f"{arm_path}/prop", prop_part, geom.prop_num_blades)


def build_drone_usd(
    cfg: AirframeConfig,
    *,
    output_dir: str = ASSETS_DIR,
) -> str:
    stage_path = os.path.join(output_dir, cfg.output_filename)
    wrapper_path = f"/{cfg.prim_name}"
    root_path = f"{wrapper_path}/body"

    os.makedirs(output_dir, exist_ok=True)
    stage = Usd.Stage.CreateNew(stage_path)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Wrapper holds body + pusher pad as sibling rigid bodies, linked by the
    # prismatic joint. ArticulationRootAPI here makes Isaac Lab treat the pair
    # as one articulation.
    wrapper = UsdGeom.Xform.Define(stage, wrapper_path)
    UsdPhysics.ArticulationRootAPI.Apply(wrapper.GetPrim())

    # Drone body rigid body - articulation-level mass/inertia/CoM, computed from
    # the cited component masses + geometry (not authored).
    total_mass, center_of_mass, diagonal_inertia = compute_body_mass_properties(cfg)
    root = UsdGeom.Xform.Define(stage, root_path)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(total_mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal_inertia))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))

    # Frame: center plate (collidable). The arm booms live under the per-arm
    # assemblies below, not here.
    frame_path = f"{root_path}/frame"
    UsdGeom.Xform.Define(stage, frame_path)
    plate = _author_box(stage, f"{frame_path}/center_plate", cfg.center_plate)
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())

    # Arms: one shared geometry placed per arm under its own azimuth Xform. Each
    # assembly carries its boom, motor, ESC, and propeller marker.
    arms_path = f"{root_path}/arms"
    UsdGeom.Xform.Define(stage, arms_path)
    for i, placement in enumerate(cfg.arms):
        _author_arm(stage, arms_path, i, cfg.arm_geometry, placement)

    # Avionics.
    avionics_path = f"{root_path}/avionics"
    UsdGeom.Xform.Define(stage, avionics_path)
    pixhawk_pos = _shift_z(cfg.pixhawk.position, cfg.center_plate.size[2] / 2)
    _author_box(stage, f"{avionics_path}/pixhawk", _replace_pos(cfg.pixhawk, pixhawk_pos))
    jetson_pos = _shift_z(cfg.jetson.position, cfg.center_plate.size[2] / 2)
    _author_box(stage, f"{avionics_path}/jetson", _replace_pos(cfg.jetson, jetson_pos))

    # Sensors.
    sensors_path = f"{root_path}/sensors"
    UsdGeom.Xform.Define(stage, sensors_path)
    UsdGeom.Xform.Define(stage, f"{sensors_path}/imu_frame")  # IMU lives in Pixhawk
    _author_box(stage, f"{sensors_path}/realsense_d435i", cfg.realsense)
    _author_box(stage, f"{sensors_path}/pmw3901_optical_flow", cfg.optical_flow)
    lidar_pos = _shift_z(cfg.lidar.position, cfg.center_plate.size[2] / 2)
    _author_cylinder(stage, f"{sensors_path}/rplidar_a2", _replace_pos(cfg.lidar, lidar_pos))

    # Power.
    power_path = f"{root_path}/power"
    UsdGeom.Xform.Define(stage, power_path)
    _author_box(stage, f"{power_path}/battery", cfg.battery)
    _author_box(stage, f"{power_path}/wiring", cfg.wiring)

    # Pusher (sibling rigid body + prismatic joint). Authored only when the
    # airframe defines a pusher and it is enabled, so a single config parameter
    # (airframes.pusher.enabled) builds the drone with or without it.
    if cfg.pusher is not None and cfg.pusher.enabled:
        pad = cfg.pusher.pad
        pusher_drive = cfg.pusher.drive
        pusher_grp = f"{wrapper_path}/pusher"
        UsdGeom.Xform.Define(stage, pusher_grp)
        pad_path = f"{pusher_grp}/pad"

        pad_xform = UsdGeom.Xform.Define(stage, pad_path)
        UsdPhysics.RigidBodyAPI.Apply(pad_xform.GetPrim())
        _set_mass(pad_xform.GetPrim(), pad.mass)
        _apply_pose(pad_xform, pad.position, pad.rotation_deg)

        pad_geom = UsdGeom.Cylinder.Define(stage, f"{pad_path}/geom")
        pad_geom.CreateRadiusAttr(pad.radius)
        pad_geom.CreateHeightAttr(pad.height)
        UsdPhysics.CollisionAPI.Apply(pad_geom.GetPrim())

        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{wrapper_path}/pusher_joint")
        joint.CreateBody0Rel().SetTargets([root_path])
        joint.CreateBody1Rel().SetTargets([pad_path])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(pusher_drive.travel)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pad.position))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(pusher_drive.stiffness)
        drive.CreateDampingAttr(pusher_drive.damping)

    stage.SetDefaultPrim(wrapper.GetPrim())
    stage.Export(stage_path)
    # OpenUSD's .usda text exporter ends the file with a trailing blank line
    # ("}\n\n"); normalize to a single trailing newline so the generated asset is
    # already end-of-file-fixer clean and regenerating it is a no-op diff.
    stage_file = Path(stage_path)
    stage_file.write_text(stage_file.read_text().rstrip("\n") + "\n")
    return stage_path


def _shift_z(pos: Vec3, dz: float) -> Vec3:
    return (pos[0], pos[1], pos[2] + dz)


_P = TypeVar("_P", BoxPart, CylinderPart)


def _replace_pos(part: _P, new_pos: Vec3) -> _P:
    return replace(part, position=new_pos)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a drone USD asset using standalone OpenUSD bindings."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_configs",
        help="Generate every airframe config option under the Hydra airframes config group.",
    )
    parser.add_argument(
        "config_name",
        nargs="?",
        default=None,
        help="Airframe config group option to generate, e.g. holybro_x500_v2.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the generated .usda file will be written.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(CONFIG_DIR),
        help="Hydra config directory for USD-builder configs.",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def _hydra_overrides_from_args(args: argparse.Namespace) -> list[str]:
    overrides = list(args.overrides)
    if args.config_name:
        overrides.insert(0, f"airframes={args.config_name}")
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    return overrides


if __name__ == "__main__":
    args = _parse_args()
    overrides = _hydra_overrides_from_args(args)

    if args.all_configs:
        if args.config_name:
            raise SystemExit("--all cannot be combined with a positional airframe config name")
        for path in _build_all_airframe_usds(config_dir=args.config_dir, overrides=overrides):
            print(f"Saved: {path}")
    else:
        config = load_builder_config(
            config_dir=args.config_dir,
            overrides=overrides,
        )
        path = build_drone_usd(
            config.airframe,
            output_dir=config.output_dir,
        )
        print(f"Saved: {path}")

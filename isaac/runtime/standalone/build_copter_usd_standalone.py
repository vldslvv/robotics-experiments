# build_copter_usd_standalone.py - Hydra-config-driven drone USD generator
#
# Run with standalone OpenUSD Python bindings:
#   python isaac/runtime/standalone/build_copter_usd_standalone.py [config_name] [overrides...]
#
# Available configs: quadcopter_500mm (default), hexacopter_s550
#
# --- Boundary with Isaac Lab (drone_cfg.py) ---
# This file authors the USD once. Anything topological / geometric lives here:
#   - Prim tree, meshes, collision shapes
#   - Joint topology (body pairs, axis, limits, relative frames)
#   - Custom attributes (e.g. PX4 rotor mapping metadata on props)
#   - Articulation root marker (ArticulationRootAPI on the wrapper prim)
# The Isaac Lab ArticulationCfg (drone_cfg.py) overrides at spawn time:
#   - Mass / inertia / CoM (so EventTerm can randomize per-env for sim-to-real)
#   - Actuator stiffness / damping / effort limits
#   - Initial pose and joint positions
#   - Rigid body solver flags (CCD, max velocities, gravity toggle)
# Rule: stable names in USD -> Isaac Lab matches joints/bodies by regex.

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

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
#   pusher_pad      -> body root


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
class CapsulePart:
    # Massless geometry (mass folded into the frame), so no `mass` field.
    name: str
    radius: float
    height: float
    position: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]


@dataclass
class Motor:
    name: str
    position: tuple[float, float, float]  # motor origin, relative to body root
    motor: CylinderPart  # the brushless can
    esc: BoxPart  # ESC riding next to this motor
    propeller: CylinderPart  # disc above the motor
    spin_dir: str  # "CCW" | "CW"
    px4_motor_index: int
    prop_num_blades: int  # cosmetic blade count for the propeller visual


@dataclass
class DroneConfig:
    name: str
    output_filename: str
    prim_name: str

    # Articulation-level rigid-body properties (overridden per-env by drone_cfg.py).
    total_mass: float
    diagonal_inertia: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]

    # Frame
    center_plate: BoxPart
    arms: list[CapsulePart]

    # Propulsion (each Motor bundles its own ESC + propeller).
    motors: list[Motor]

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

    # Pusher (None == no pusher mechanism on this airframe).
    pusher_pad: CylinderPart | None = None


@dataclass
class PusherDriveConfig:
    travel: float
    stiffness: float
    damping: float


@dataclass
class BuilderConfig:
    output_dir: str
    pusher_drive: PusherDriveConfig
    drone: DroneConfig


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


def _capsule(data: dict[str, Any], label: str) -> CapsulePart:
    return CapsulePart(
        name=str(data["name"]),
        radius=float(data["radius"]),
        height=float(data["height"]),
        position=_as_tuple3(data["position"], f"{label}.position"),
        rotation_deg=_as_tuple3(data["rotation_deg"], f"{label}.rotation_deg"),
    )


def _motor(data: dict[str, Any], label: str) -> Motor:
    return Motor(
        name=str(data["name"]),
        position=_as_tuple3(data["position"], f"{label}.position"),
        motor=_cylinder(data["motor"], f"{label}.motor"),
        esc=_box(data["esc"], f"{label}.esc"),
        propeller=_cylinder(data["propeller"], f"{label}.propeller"),
        spin_dir=str(data["spin_dir"]),
        px4_motor_index=int(data["px4_motor_index"]),
        prop_num_blades=int(data["propeller"]["num_blades"]),
    )


def _drone_config(data: dict[str, Any]) -> DroneConfig:
    pusher_pad = data.get("pusher_pad")
    return DroneConfig(
        name=str(data["name"]),
        output_filename=str(data["output_filename"]),
        prim_name=str(data["prim_name"]),
        total_mass=float(data["total_mass"]),
        diagonal_inertia=_as_tuple3(data["diagonal_inertia"], "drone.diagonal_inertia"),
        center_of_mass=_as_tuple3(data["center_of_mass"], "drone.center_of_mass"),
        center_plate=_box(data["center_plate"], "drone.center_plate"),
        arms=[_capsule(arm, f"drone.arms[{i}]") for i, arm in enumerate(data["arms"])],
        motors=[_motor(motor, f"drone.motors[{i}]") for i, motor in enumerate(data["motors"])],
        pixhawk=_box(data["pixhawk"], "drone.pixhawk"),
        jetson=_box(data["jetson"], "drone.jetson"),
        realsense=_box(data["realsense"], "drone.realsense"),
        optical_flow=_box(data["optical_flow"], "drone.optical_flow"),
        lidar=_cylinder(data["lidar"], "drone.lidar"),
        battery=_box(data["battery"], "drone.battery"),
        wiring=_box(data["wiring"], "drone.wiring"),
        pusher_pad=None if pusher_pad is None else _cylinder(pusher_pad, "drone.pusher_pad"),
    )


def _builder_config(data: dict[str, Any]) -> BuilderConfig:
    pusher_drive = data["pusher_drive"]
    return BuilderConfig(
        output_dir=str(data["output_dir"]),
        pusher_drive=PusherDriveConfig(
            travel=float(pusher_drive["travel"]),
            stiffness=float(pusher_drive["stiffness"]),
            damping=float(pusher_drive["damping"]),
        ),
        drone=_drone_config(data["drone"]),
    )


def load_builder_config(
    *,
    config_dir: str | os.PathLike[str] = CONFIG_DIR,
    overrides: list[str] | None = None,
) -> BuilderConfig:
    """Compose the Hydra USD-builder config and convert it to dataclasses."""

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return _builder_config(OmegaConf.to_container(cfg, resolve=True))


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


def _author_capsule(stage: Usd.Stage, prim_path: str, part: CapsulePart) -> UsdGeom.Capsule:
    cap = UsdGeom.Capsule.Define(stage, prim_path)
    cap.CreateRadiusAttr(part.radius)
    cap.CreateHeightAttr(part.height)
    _apply_pose(cap, part.position, part.rotation_deg)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _author_propeller_visual(stage: Usd.Stage, prop_path: str, motor: Motor) -> None:
    """Author the cosmetic propeller geometry under the rotor marker Xform.

    A small central hub cylinder plus ``prop_num_blades`` thin blade boxes fanned
    evenly about Z. All geometry is collision-free and visual only; it derives
    its dimensions from the propeller radius/height (no extra geometry knobs) and
    spins as a unit because it is parented under the marker's RotateZ op.
    """
    prop_part = motor.propeller
    num_blades = int(motor.prop_num_blades)
    if num_blades < 1:
        raise ValueError(f"propeller num_blades must be >= 1, got {num_blades} for {motor.name}")

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


def _author_motor(stage: Usd.Stage, parent_path: str, idx: int, motor: Motor) -> None:
    grp_path = f"{parent_path}/motor_{idx+1}"
    UsdGeom.Xform.Define(stage, grp_path)

    # The motor cylinder sits at the motor origin (body-relative).
    motor_part = motor.motor
    motor_world_pos = (
        motor.position[0] + motor_part.position[0],
        motor.position[1] + motor_part.position[1],
        motor.position[2] + motor_part.position[2],
    )
    motor_geom = UsdGeom.Cylinder.Define(stage, f"{grp_path}/motor")
    motor_geom.CreateRadiusAttr(motor_part.radius)
    motor_geom.CreateHeightAttr(motor_part.height)
    _apply_pose(motor_geom, motor_world_pos, motor_part.rotation_deg)
    _set_mass(motor_geom.GetPrim(), motor_part.mass)

    # ESC: position is local to the motor, so add to motor.position.
    esc_part = motor.esc
    esc_world_pos = (
        motor.position[0] + esc_part.position[0],
        motor.position[1] + esc_part.position[1],
        motor.position[2] + esc_part.position[2],
    )
    esc_path = f"{grp_path}/esc"
    esc_xform = UsdGeom.Xform.Define(stage, esc_path)
    _apply_pose(esc_xform, esc_world_pos, esc_part.rotation_deg)
    _set_mass(esc_xform.GetPrim(), esc_part.mass)
    esc_geom = UsdGeom.Cube.Define(stage, f"{esc_path}/geom")
    esc_geom.CreateSizeAttr(1.0)
    esc_geom.AddScaleOp().Set(Gf.Vec3f(*esc_part.size))

    # Propeller: position is local to the motor. The marker is an Xform (keeping
    # the prop prim's path, PX4 marker attributes, translate, and mass) so it can
    # carry both the rotor-spin RotateZ op and the child propeller visual (a hub
    # plus fanned-out blades). rotor_discovery and the bridge read the marker's
    # translation, which the RotateZ op leaves unchanged.
    prop_part = motor.propeller
    prop_world_pos = (
        motor.position[0] + prop_part.position[0],
        motor.position[1] + prop_part.position[1],
        motor.position[2] + prop_part.position[2],
    )
    prop = UsdGeom.Xform.Define(stage, f"{grp_path}/prop")
    _apply_pose(prop, prop_world_pos, prop_part.rotation_deg)
    # Cosmetic spin op, authored after the translate so spinning never disturbs
    # the marker position. The runtime drives this op every physics tick.
    prop.AddRotateZOp().Set(0.0)
    prop.GetPrim().CreateAttribute("drone:rotorMarker", Sdf.ValueTypeNames.Bool).Set(True)
    prop.GetPrim().CreateAttribute("drone:px4MotorIndex", Sdf.ValueTypeNames.Int).Set(
        motor.px4_motor_index
    )
    prop.GetPrim().CreateAttribute("drone:spinDir", Sdf.ValueTypeNames.String).Set(motor.spin_dir)
    _set_mass(prop.GetPrim(), prop_part.mass)
    _author_propeller_visual(stage, f"{grp_path}/prop", motor)


def build_drone_usd(
    cfg: DroneConfig,
    *,
    output_dir: str = ASSETS_DIR,
    pusher_drive: PusherDriveConfig,
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

    # Drone body rigid body - articulation-level mass/inertia/CoM.
    root = UsdGeom.Xform.Define(stage, root_path)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(cfg.total_mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*cfg.diagonal_inertia))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*cfg.center_of_mass))

    # Frame: center plate (collidable) + arms (collidable, massless).
    frame_path = f"{root_path}/frame"
    UsdGeom.Xform.Define(stage, frame_path)
    plate = _author_box(stage, f"{frame_path}/center_plate", cfg.center_plate)
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())

    arms_path = f"{frame_path}/arms"
    UsdGeom.Xform.Define(stage, arms_path)
    for i, arm in enumerate(cfg.arms):
        _author_capsule(stage, f"{arms_path}/arm_{i+1}", arm)

    # Propulsion.
    prop_path = f"{root_path}/propulsion"
    UsdGeom.Xform.Define(stage, prop_path)
    for i, motor in enumerate(cfg.motors):
        _author_motor(stage, prop_path, i, motor)

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

    # Pusher (sibling rigid body + prismatic joint).
    if cfg.pusher_pad is not None:
        pad = cfg.pusher_pad
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
        "config_name",
        nargs="?",
        default=None,
        help="Drone config group option to generate, e.g. quadcopter_500mm.",
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
        overrides.insert(0, f"drone={args.config_name}")
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    return overrides


if __name__ == "__main__":
    args = _parse_args()

    config = load_builder_config(
        config_dir=args.config_dir,
        overrides=_hydra_overrides_from_args(args),
    )
    path = build_drone_usd(
        config.drone,
        output_dir=config.output_dir,
        pusher_drive=config.pusher_drive,
    )
    print(f"Saved: {path}")

# build_drone_usd.py — config-driven drone USD generator for Isaac Sim
#
# Run inside isaac-sim docker:
#   /isaac-sim/python.sh proj/robotics-experiments/isaac_scripts/build_copter_usd_no_pxr.py [config_name]
#
# Available configs: quadcopter_500mm (default), hexacopter_s550
#
# --- Boundary with Isaac Lab (drone_cfg.py) ---
# This file authors the USD once. Anything topological / geometric lives here:
#   - Prim tree, meshes, collision shapes
#   - Joint topology (body pairs, axis, limits, relative frames)
#   - Custom attributes (e.g. drone:spinDir on props)
#   - Articulation root marker (ArticulationRootAPI on the wrapper prim)
# The Isaac Lab ArticulationCfg (drone_cfg.py) overrides at spawn time:
#   - Mass / inertia / CoM (so EventTerm can randomize per-env for sim-to-real)
#   - Actuator stiffness / damping / effort limits
#   - Initial pose and joint positions
#   - Rigid body solver flags (CCD, max velocities, gravity toggle)
# Rule: stable names in USD → Isaac Lab matches joints/bodies by regex.

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from typing import TypeVar

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

ASSETS_DIR = "/isaac-sim/proj/assets/LightSwitchWarehouse"

# Pusher actuator tuning (drive parameters, not part geometry).
PUSHER_TRAVEL = 0.015
PUSHER_STIFFNESS = 800.0
PUSHER_DAMPING = 40.0


# --- Component dataclasses --------------------------------------------------
# Each real-world part owns its name, shape, position, and mass. Box vs.
# cylinder vs. capsule are kept as separate classes — not generalized — because
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
    position: tuple[float, float, float]   # motor origin, relative to body root
    motor: CylinderPart                    # the brushless can
    esc: BoxPart                           # ESC riding next to this motor
    propeller: CylinderPart                # disc above the motor
    spin_dir: str                          # "CCW" | "CW"


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


# --- Config instances -------------------------------------------------------

def _quadcopter_500mm() -> DroneConfig:
    body_half = 0.06
    plate_thickness = 0.006

    motor_radius = 0.015
    motor_height = 0.03
    rotor_radius = 0.1143  # 9" prop
    motor_xy = 0.25
    motor_z = 0.02

    motor_specs = [
        ("M1 FR CCW",  ( motor_xy,  motor_xy, motor_z), "CCW"),
        ("M2 FL CW",   (-motor_xy,  motor_xy, motor_z), "CW"),
        ("M3 RL CCW",  (-motor_xy, -motor_xy, motor_z), "CCW"),
        ("M4 RR CW",   ( motor_xy, -motor_xy, motor_z), "CW"),
    ]

    motors = [
        Motor(
            name=name,
            position=pos,
            motor=CylinderPart(
                name="X2212-class brushless motor",
                radius=motor_radius,
                height=motor_height,
                position=(0.0, 0.0, 0.0),
                mass=0.057,
            ),
            esc=BoxPart(
                name="BLHeli_S 20A ESC",
                # Sits inboard along the arm, 40% of the way back toward the
                # plate, just above the plate's mid-plane.
                size=(0.025, 0.015, 0.004),
                position=(-0.4 * pos[0], -0.4 * pos[1], 0.005 - pos[2]),
                mass=0.007,
            ),
            propeller=CylinderPart(
                name="9045 propeller",
                radius=rotor_radius,
                height=0.002,
                position=(0.0, 0.0, 0.02),  # 2 cm above the motor can
                mass=0.008,
            ),
            spin_dir=spin,
        )
        for (name, pos, spin) in motor_specs
    ]

    arms = [
        CapsulePart(
            name=f"Arm {i+1}",
            radius=0.008,
            height=math.hypot(m.position[0], m.position[1]) * 1.8,
            position=(m.position[0] / 2, m.position[1] / 2, 0.0),
            rotation_deg=(0.0, 90.0, math.degrees(math.atan2(m.position[1], m.position[0]))),
        )
        for i, m in enumerate(motors)
    ]

    return DroneConfig(
        name="quadcopter_500mm",
        output_filename="quadcopter_500mm.usda",
        prim_name="quadcopter",
        total_mass=1.5,
        diagonal_inertia=(0.0347, 0.0458, 0.0977),
        center_of_mass=(0.0, 0.0, 0.01),
        center_plate=BoxPart(
            name="S500-class carbon center plate",
            size=(body_half * 2, body_half * 2, plate_thickness),
            position=(0.0, 0.0, 0.0),
            mass=0.350,  # whole airframe mass rolled in here
        ),
        arms=arms,
        motors=motors,
        pixhawk=BoxPart(
            name="Pixhawk 6C",
            size=(0.050, 0.038, 0.016),
            position=(0.0, 0.0, 0.016 / 2),  # flush on top of center plate
            mass=0.035,
        ),
        jetson=BoxPart(
            name="Jetson Orin Nano Super",
            size=(0.100, 0.080, 0.030),
            position=(-0.03, 0.0, 0.030 / 2),
            mass=0.165,
        ),
        realsense=BoxPart(
            name="Intel RealSense D435i",
            size=(0.090, 0.025, 0.025),
            position=(body_half + 0.01, 0.0, 0.01),
            mass=0.072,
            rotation_deg=(0.0, -15.0, 0.0),
        ),
        optical_flow=BoxPart(
            name="PMW3901 optical flow",
            size=(0.020, 0.020, 0.005),
            position=(0.0, 0.0, -body_half),
            mass=0.005,
            rotation_deg=(180.0, 0.0, 0.0),
        ),
        lidar=CylinderPart(
            name="Slamtec RPLiDAR A2",
            radius=0.038,
            height=0.041,
            position=(0.0, 0.0, 0.041 / 2),  # sits on top of plate
            mass=0.190,
        ),
        battery=BoxPart(
            name="4S 5000 mAh LiPo",
            size=(0.140, 0.045, 0.025),
            position=(0.0, 0.0, -body_half * 0.5),
            mass=0.300,
        ),
        wiring=BoxPart(
            name="PDB + power module",
            size=(0.050, 0.030, 0.010),
            position=(0.0, 0.0, 0.0),
            mass=0.100,
        ),
        pusher_pad=CylinderPart(
            name="Spring-loaded pusher pad",
            radius=0.015,
            height=0.008,
            position=(body_half + 0.05, 0.0, 0.01),
            mass=0.040,
        ),
    )


def _hexacopter_s550() -> DroneConfig:
    body_half = 0.08
    plate_thickness = 0.008

    motor_radius = 0.018
    motor_height = 0.035
    rotor_radius = 0.127  # 10" prop
    arm_len = 0.275       # 550 mm diagonal -> 275 mm arm
    motor_z = 0.02

    spin_dirs = ["CCW", "CW", "CCW", "CW", "CCW", "CW"]
    motor_specs = []
    for i, a_deg in enumerate(range(0, 360, 60)):
        a = math.radians(a_deg)
        x = arm_len * math.cos(a)
        y = arm_len * math.sin(a)
        motor_specs.append(
            (f"M{i+1} {a_deg}deg {spin_dirs[i]}", (x, y, motor_z), spin_dirs[i])
        )

    motors = [
        Motor(
            name=name,
            position=pos,
            motor=CylinderPart(
                name="X2212-class brushless motor",
                radius=motor_radius,
                height=motor_height,
                position=(0.0, 0.0, 0.0),
                mass=0.057,  # 342 g / 6
            ),
            esc=BoxPart(
                name="BLHeli_S 20A ESC",
                # Sits inboard along the arm, 40% of the way back toward the
                # plate, just above the plate's mid-plane.
                size=(0.025, 0.015, 0.004),
                position=(-0.4 * pos[0], -0.4 * pos[1], 0.005 - pos[2]),
                mass=0.007,  # 42 g / 6
            ),
            propeller=CylinderPart(
                name="1045 propeller",
                radius=rotor_radius,
                height=0.002,
                position=(0.0, 0.0, 0.02),  # 2 cm above the motor can
                mass=0.008,  # 48 g / 6
            ),
            spin_dir=spin,
        )
        for (name, pos, spin) in motor_specs
    ]

    arms = [
        CapsulePart(
            name=f"Arm {i+1}",
            radius=0.01,
            height=math.hypot(m.position[0], m.position[1]) * 1.8,
            position=(m.position[0] / 2, m.position[1] / 2, 0.0),
            rotation_deg=(0.0, 90.0, math.degrees(math.atan2(m.position[1], m.position[0]))),
        )
        for i, m in enumerate(motors)
    ]

    return DroneConfig(
        name="hexacopter_s550",
        output_filename="hexacopter_s550.usda",
        prim_name="hexacopter",
        total_mass=1.929,
        diagonal_inertia=(0.075, 0.075, 0.14),
        center_of_mass=(0.0, 0.0, 0.01),
        center_plate=BoxPart(
            name="S550 airframe center plate",
            size=(body_half * 2, body_half * 2, plate_thickness),
            position=(0.0, 0.0, 0.0),
            mass=0.480,
        ),
        arms=arms,
        motors=motors,
        pixhawk=BoxPart(
            name="Pixhawk 6C",
            size=(0.050, 0.038, 0.016),
            position=(0.0, 0.0, 0.016 / 2),
            mass=0.035,
        ),
        jetson=BoxPart(
            name="Jetson Orin Nano Super",
            size=(0.100, 0.080, 0.030),
            position=(-0.03, 0.0, 0.030 / 2),
            mass=0.165,
        ),
        realsense=BoxPart(
            name="Intel RealSense D435i",
            size=(0.090, 0.025, 0.025),
            position=(body_half + 0.01, 0.0, 0.01),
            mass=0.072,
            rotation_deg=(0.0, -15.0, 0.0),
        ),
        optical_flow=BoxPart(
            name="PMW3901 optical flow",
            size=(0.020, 0.020, 0.005),
            position=(0.0, 0.0, -body_half),
            mass=0.005,
            rotation_deg=(180.0, 0.0, 0.0),
        ),
        lidar=CylinderPart(
            name="Slamtec RPLiDAR A2",
            radius=0.038,
            height=0.041,
            position=(0.0, 0.0, 0.041 / 2),
            mass=0.190,
        ),
        battery=BoxPart(
            name="4S 5000 mAh LiPo",
            size=(0.140, 0.045, 0.025),
            position=(0.0, 0.0, -body_half * 0.5),
            mass=0.390,
        ),
        wiring=BoxPart(
            name="PDB + power module",
            size=(0.050, 0.030, 0.010),
            position=(0.0, 0.0, 0.0),
            mass=0.120,
        ),
        pusher_pad=CylinderPart(
            name="Spring-loaded pusher pad",
            radius=0.015,
            height=0.008,
            position=(body_half + 0.05, 0.0, 0.01),
            mass=0.040,
        ),
    )


QUADCOPTER_500MM = _quadcopter_500mm()
HEXACOPTER_S550 = _hexacopter_s550()

CONFIGS: dict[str, DroneConfig] = {
    "quadcopter_500mm": QUADCOPTER_500MM,
    "hexacopter_s550": HEXACOPTER_S550,
}


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


def _author_cylinder(
    stage: Usd.Stage, prim_path: str, part: CylinderPart
) -> UsdGeom.Cylinder:
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(part.radius)
    cyl.CreateHeightAttr(part.height)
    _apply_pose(cyl, part.position, part.rotation_deg)
    _set_mass(cyl.GetPrim(), part.mass)
    return cyl


def _author_capsule(
    stage: Usd.Stage, prim_path: str, part: CapsulePart
) -> UsdGeom.Capsule:
    cap = UsdGeom.Capsule.Define(stage, prim_path)
    cap.CreateRadiusAttr(part.radius)
    cap.CreateHeightAttr(part.height)
    _apply_pose(cap, part.position, part.rotation_deg)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    return cap


def _author_motor(
    stage: Usd.Stage, parent_path: str, idx: int, motor: Motor
) -> None:
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

    # Propeller: position is local to the motor.
    prop_part = motor.propeller
    prop_world_pos = (
        motor.position[0] + prop_part.position[0],
        motor.position[1] + prop_part.position[1],
        motor.position[2] + prop_part.position[2],
    )
    prop = UsdGeom.Cylinder.Define(stage, f"{grp_path}/prop")
    prop.CreateRadiusAttr(prop_part.radius)
    prop.CreateHeightAttr(prop_part.height)
    _apply_pose(prop, prop_world_pos, prop_part.rotation_deg)
    prop.GetPrim().CreateAttribute(
        "drone:spinDir", Sdf.ValueTypeNames.String
    ).Set(motor.spin_dir)
    _set_mass(prop.GetPrim(), prop_part.mass)


def build_drone_usd(cfg: DroneConfig) -> str:
    stage_path = f"{ASSETS_DIR}/{cfg.output_filename}"
    wrapper_path = f"/{cfg.prim_name}"
    root_path = f"{wrapper_path}/body"

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Wrapper holds body + pusher pad as sibling rigid bodies, linked by the
    # prismatic joint. ArticulationRootAPI here makes Isaac Lab treat the pair
    # as one articulation.
    wrapper = UsdGeom.Xform.Define(stage, wrapper_path)
    UsdPhysics.ArticulationRootAPI.Apply(wrapper.GetPrim())

    # Drone body rigid body — articulation-level mass/inertia/CoM.
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
    _author_box(stage, f"{avionics_path}/pixhawk",
                _replace_pos(cfg.pixhawk, pixhawk_pos))
    jetson_pos = _shift_z(cfg.jetson.position, cfg.center_plate.size[2] / 2)
    _author_box(stage, f"{avionics_path}/jetson",
                _replace_pos(cfg.jetson, jetson_pos))

    # Sensors.
    sensors_path = f"{root_path}/sensors"
    UsdGeom.Xform.Define(stage, sensors_path)
    UsdGeom.Xform.Define(stage, f"{sensors_path}/imu_frame")  # IMU lives in Pixhawk
    _author_box(stage, f"{sensors_path}/realsense_d435i", cfg.realsense)
    _author_box(stage, f"{sensors_path}/pmw3901_optical_flow", cfg.optical_flow)
    lidar_pos = _shift_z(cfg.lidar.position, cfg.center_plate.size[2] / 2)
    _author_cylinder(stage, f"{sensors_path}/rplidar_a2",
                     _replace_pos(cfg.lidar, lidar_pos))

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
        joint.CreateUpperLimitAttr(PUSHER_TRAVEL)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pad.position))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(PUSHER_STIFFNESS)
        drive.CreateDampingAttr(PUSHER_DAMPING)

    stage.SetDefaultPrim(wrapper.GetPrim())
    stage.Export(stage_path)
    return stage_path


def _shift_z(pos: Vec3, dz: float) -> Vec3:
    return (pos[0], pos[1], pos[2] + dz)


_P = TypeVar("_P", BoxPart, CylinderPart)


def _replace_pos(part: _P, new_pos: Vec3) -> _P:
    return replace(part, position=new_pos)


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "quadcopter_500mm"
    if config_name not in CONFIGS:
        print(f"Unknown config '{config_name}'. Available: {', '.join(CONFIGS)}")
        sys.exit(1)

    path = build_drone_usd(CONFIGS[config_name])
    print(f"Saved: {path}")

    simulation_app.close()

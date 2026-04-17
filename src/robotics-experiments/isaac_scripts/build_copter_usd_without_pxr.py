# build_drone_usd.py — config-driven drone USD generator for Isaac Sim
#
# Run inside isaac-sim docker:
#   /isaac-sim/python.sh proj/robotics-experiments/isaac_scripts/build_copter_usd_without_pxr.py [config_name]
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
from dataclasses import dataclass, field

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

ASSETS_DIR = "/isaac-sim/proj/assets/LightSwitchWarehouse"

# --- Physical dimensions of emulated hardware (meters) ---
# Each tuple is (x, y, z) extent of a bounding box for the real component.
ESC_SIZE = (0.025, 0.015, 0.004)             # BLHeli_S 20A ESC
PIXHAWK_SIZE = (0.050, 0.038, 0.016)         # Pixhawk 6C flight controller
JETSON_SIZE = (0.100, 0.080, 0.030)          # Jetson Orin Nano Super devkit
REALSENSE_SIZE = (0.090, 0.025, 0.025)       # Intel RealSense D435i
OPTICAL_FLOW_SIZE = (0.020, 0.020, 0.005)    # PMW3901 board
RPLIDAR_RADIUS = 0.038                       # Slamtec RPLiDAR A2
RPLIDAR_HEIGHT = 0.041
BATTERY_SIZE = (0.140, 0.045, 0.025)         # 4S 5000 mAh LiPo pack
WIRING_SIZE = (0.050, 0.030, 0.010)          # PDB + power module stand-in
PROP_THICKNESS = 0.002                       # propeller disc thickness
PROP_Z_OFFSET = 0.02                         # propeller sits this far above motor origin

# --- Mounting / placement offsets (meters, relative to center plate) ---
JETSON_X_OFFSET = -0.03                      # jetson slid aft of pixhawk on the plate
REALSENSE_X_OFFSET = 0.01                    # forward of frame edge
REALSENSE_Z_OFFSET = 0.01                    # slight rise above plate top
REALSENSE_PITCH_DEG = -15                    # nose-down tilt for obstacle view
PUSHER_X_OFFSET = 0.05                       # pad sits this far forward of frame edge
PUSHER_Z_OFFSET = 0.01                       # pad lifted off plate
PUSHER_PAD_RADIUS = 0.015
PUSHER_PAD_HEIGHT = 0.008
PUSHER_TRAVEL = 0.015                        # prismatic joint upper limit
PUSHER_STIFFNESS = 800.0
PUSHER_DAMPING = 40.0


@dataclass
class DroneConfig:
    name: str
    output_filename: str
    body_half: float
    center_plate_thickness: float
    total_mass: float
    diagonal_inertia: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    motor_positions: list[tuple[float, float, float]]
    spin_dirs: list[str]
    arm_radius: float
    arm_height_factor: float
    rotor_radius: float
    motor_radius: float
    motor_height: float
    # BOM per-component masses (kg)
    frame_mass: float
    motor_mass: float
    esc_mass: float
    prop_mass: float
    fcu_mass: float
    companion_mass: float
    camera_mass: float
    lidar_mass: float
    optical_flow_mass: float
    battery_mass: float
    wiring_mass: float
    has_pusher: bool
    pusher_mass: float
    prim_name: str


QUADCOPTER_500MM = DroneConfig(
    name="quadcopter_500mm",
    output_filename="quadcopter_500mm.usda",
    body_half=0.06,
    center_plate_thickness=0.006,
    total_mass=1.5,
    diagonal_inertia=(0.0347, 0.0458, 0.0977),
    center_of_mass=(0, 0, 0.01),
    motor_positions=[
        ( 0.25,  0.25, 0.02),   # M1 FR CCW
        (-0.25,  0.25, 0.02),   # M2 FL CW
        (-0.25, -0.25, 0.02),   # M3 RL CCW
        ( 0.25, -0.25, 0.02),   # M4 RR CW
    ],
    spin_dirs=["CCW", "CW", "CCW", "CW"],
    arm_radius=0.008,
    arm_height_factor=1.8,
    rotor_radius=0.1143,        # 9" prop
    motor_radius=0.015,
    motor_height=0.03,
    frame_mass=0.350,
    motor_mass=0.057,
    esc_mass=0.007,
    prop_mass=0.008,
    fcu_mass=0.035,
    companion_mass=0.165,
    camera_mass=0.072,
    lidar_mass=0.190,
    optical_flow_mass=0.005,
    battery_mass=0.300,
    wiring_mass=0.100,
    has_pusher=True,
    pusher_mass=0.040,
    prim_name="quadcopter",
)

_hex_arm = 0.275                # 550mm diagonal → 275mm arm
HEXACOPTER_S550 = DroneConfig(
    name="hexacopter_s550",
    output_filename="hexacopter_s550.usda",
    body_half=0.08,
    center_plate_thickness=0.008,
    total_mass=1.929,           # BOM sum
    diagonal_inertia=(0.075, 0.075, 0.14),
    center_of_mass=(0, 0, 0.01),
    motor_positions=[
        (_hex_arm * math.cos(math.radians(a)),
         _hex_arm * math.sin(math.radians(a)),
         0.02)
        for a in range(0, 360, 60)   # M1–M6 at 0°, 60°, …, 300°
    ],
    spin_dirs=["CCW", "CW", "CCW", "CW", "CCW", "CW"],
    arm_radius=0.01,
    arm_height_factor=1.8,
    rotor_radius=0.127,         # 10" prop
    motor_radius=0.018,
    motor_height=0.035,
    # BOM: S550 airframe 480g; 6×X2212 342g; 6×BLHeli_S 42g; 6×9045 48g;
    # Pixhawk 6C 35g; Jetson Orin Nano Super 165g; D435i 72g; RPLiDAR A2 190g;
    # PMW3901 5g; pusher 40g; 4S 5000mAh 390g; wiring/power module 120g.
    frame_mass=0.480,
    motor_mass=0.057,           # 342g / 6
    esc_mass=0.007,             # 42g / 6
    prop_mass=0.008,            # 48g / 6
    fcu_mass=0.035,
    companion_mass=0.165,
    camera_mass=0.072,
    lidar_mass=0.190,
    optical_flow_mass=0.005,
    battery_mass=0.390,
    wiring_mass=0.120,
    has_pusher=True,
    pusher_mass=0.040,
    prim_name="hexacopter",
)

CONFIGS: dict[str, DroneConfig] = {
    "quadcopter_500mm": QUADCOPTER_500MM,
    "hexacopter_s550": HEXACOPTER_S550,
}


def _set_mass(prim, mass_kg: float) -> None:
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass_kg)


def _add_box_geom(stage, parent_path: str, size_xyz: tuple[float, float, float]) -> None:
    geom = UsdGeom.Cube.Define(stage, f"{parent_path}/geom")
    geom.CreateSizeAttr(1.0)
    geom.AddScaleOp().Set(Gf.Vec3f(*size_xyz))


def build_drone_usd(cfg: DroneConfig) -> str:
    stage_path = f"{ASSETS_DIR}/{cfg.output_filename}"
    wrapper_path = f"/{cfg.prim_name}"
    root_path = f"{wrapper_path}/body"

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # --- Wrapper (default prim, plain Xform) so drone body and pusher pad
    #     live as sibling rigid bodies under a single reference root. ---
    wrapper = UsdGeom.Xform.Define(stage, wrapper_path)
    # Mark the wrapper as the articulation root so Isaac Lab's Articulation
    # wrapper sees `/body` and `/pusher/pad` as one articulation joined by the
    # prismatic pusher joint. Without this, Isaac Lab treats them as two
    # independent RigidObjects and the joint is invisible to actuator cfgs.
    UsdPhysics.ArticulationRootAPI.Apply(wrapper.GetPrim())

    # --- Drone body rigid body ---
    root = UsdGeom.Xform.Define(stage, root_path)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(cfg.total_mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*cfg.diagonal_inertia))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*cfg.center_of_mass))

    # --- Frame (S550-style airframe: center plate + arms, no landing gear).
    # Mass `frame_mass` covers the whole airframe; arms themselves are massless
    # geometry (their mass is rolled into `frame_mass`). ---
    frame_path = f"{root_path}/frame"
    UsdGeom.Xform.Define(stage, frame_path)

    # Center plate: flat carbon/FR4 sheet. Size = (2*body_half) x (2*body_half)
    # x center_plate_thickness m. Mass = cfg.frame_mass kg (whole airframe).
    frame_body = UsdGeom.Cube.Define(stage, f"{frame_path}/center_plate")
    frame_body.CreateSizeAttr(1.0)
    frame_body.AddScaleOp().Set(
        Gf.Vec3f(cfg.body_half * 2, cfg.body_half * 2, cfg.center_plate_thickness)
    )
    UsdPhysics.CollisionAPI.Apply(frame_body.GetPrim())
    _set_mass(frame_body.GetPrim(), cfg.frame_mass)

    # Arms: tubular booms from center plate to each motor mount. Capsule of
    # radius `arm_radius` m, length = hypot(x, y) * arm_height_factor. Massless
    # geometry (mass folded into frame_mass).
    arms_path = f"{frame_path}/arms"
    UsdGeom.Xform.Define(stage, arms_path)
    for i, (x, y, _z) in enumerate(cfg.motor_positions):
        arm = UsdGeom.Capsule.Define(stage, f"{arms_path}/arm_{i+1}")
        arm.CreateRadiusAttr(cfg.arm_radius)
        arm_length = math.hypot(x, y)
        arm.CreateHeightAttr(arm_length * cfg.arm_height_factor)
        angle = math.degrees(math.atan2(y, x))
        UsdGeom.XformCommonAPI(arm).SetRotate(Gf.Vec3f(0, 90, angle))
        UsdGeom.XformCommonAPI(arm).SetTranslate(Gf.Vec3d(x / 2, y / 2, 0))
        UsdPhysics.CollisionAPI.Apply(arm.GetPrim())

    # --- Propulsion group: per-motor arm + motor + ESC + prop ---
    prop_path = f"{root_path}/propulsion"
    UsdGeom.Xform.Define(stage, prop_path)

    for i, (x, y, z) in enumerate(cfg.motor_positions):
        motor_grp_path = f"{prop_path}/motor_{i+1}"
        UsdGeom.Xform.Define(stage, motor_grp_path)

        # Brushless motor (e.g. X2212 class): cylinder of radius `motor_radius`
        # m and height `motor_height` m. Mass = cfg.motor_mass kg per unit.
        motor = UsdGeom.Cylinder.Define(stage, f"{motor_grp_path}/motor")
        motor.CreateRadiusAttr(cfg.motor_radius)
        motor.CreateHeightAttr(cfg.motor_height)
        UsdGeom.XformCommonAPI(motor).SetTranslate(Gf.Vec3d(x, y, z))
        _set_mass(motor.GetPrim(), cfg.motor_mass)

        # ESC (BLHeli_S 20A class): 25 x 15 x 4 mm board.
        # Mass = cfg.esc_mass kg per unit.
        esc_path = f"{motor_grp_path}/esc"
        esc = UsdGeom.Xform.Define(stage, esc_path)
        UsdGeom.XformCommonAPI(esc).SetTranslate(Gf.Vec3d(x * 0.6, y * 0.6, 0.005))
        _set_mass(esc.GetPrim(), cfg.esc_mass)
        _add_box_geom(stage, esc_path, ESC_SIZE)

        # Propeller (e.g. 9045 / 1045): thin disk, radius `rotor_radius` m,
        # 2 mm thick. Mass = cfg.prop_mass kg per unit.
        prop = UsdGeom.Cylinder.Define(stage, f"{motor_grp_path}/prop")
        prop.CreateRadiusAttr(cfg.rotor_radius)
        prop.CreateHeightAttr(PROP_THICKNESS)
        UsdGeom.XformCommonAPI(prop).SetTranslate(Gf.Vec3d(x, y, z + PROP_Z_OFFSET))
        prop.GetPrim().CreateAttribute(
            "drone:spinDir", Sdf.ValueTypeNames.String
        ).Set(cfg.spin_dirs[i])
        _set_mass(prop.GetPrim(), cfg.prop_mass)

    # --- Avionics: flight controller + companion computer ---
    avionics_path = f"{root_path}/avionics"
    UsdGeom.Xform.Define(stage, avionics_path)

    # Pixhawk 6C flight controller: 50 x 38 x 16 mm enclosure.
    # Mass = cfg.fcu_mass kg.
    pixhawk_path = f"{avionics_path}/pixhawk"
    pixhawk = UsdGeom.Xform.Define(stage, pixhawk_path)
    pixhawk_mount_z = cfg.center_plate_thickness / 2 + PIXHAWK_SIZE[2] / 2
    UsdGeom.XformCommonAPI(pixhawk).SetTranslate(Gf.Vec3d(0, 0, pixhawk_mount_z))
    _set_mass(pixhawk.GetPrim(), cfg.fcu_mass)
    _add_box_geom(stage, pixhawk_path, PIXHAWK_SIZE)

    # Jetson Orin Nano Super (companion computer, devkit carrier):
    # 100 x 80 x 30 mm. Mass = cfg.companion_mass kg.
    jetson_path = f"{avionics_path}/jetson"
    jetson = UsdGeom.Xform.Define(stage, jetson_path)
    jetson_mount_z = cfg.center_plate_thickness / 2 + JETSON_SIZE[2] / 2
    UsdGeom.XformCommonAPI(jetson).SetTranslate(
        Gf.Vec3d(JETSON_X_OFFSET, 0, jetson_mount_z)
    )
    _set_mass(jetson.GetPrim(), cfg.companion_mass)
    _add_box_geom(stage, jetson_path, JETSON_SIZE)

    # --- Sensors group ---
    sensors_path = f"{root_path}/sensors"
    UsdGeom.Xform.Define(stage, sensors_path)

    imu = UsdGeom.Xform.Define(stage, f"{sensors_path}/imu_frame")
    # IMU mass is inside Pixhawk; no separate mass entry.
    _ = imu

    # Intel RealSense D435i (depth + IMU camera): 90 x 25 x 25 mm bar.
    # Mass = cfg.camera_mass kg. Mounted forward, tilted -15 deg in pitch.
    cam_path = f"{sensors_path}/realsense_d435i"
    cam = UsdGeom.Xform.Define(stage, cam_path)
    UsdGeom.XformCommonAPI(cam).SetTranslate(
        Gf.Vec3d(cfg.body_half + REALSENSE_X_OFFSET, 0, REALSENSE_Z_OFFSET)
    )
    UsdGeom.XformCommonAPI(cam).SetRotate(Gf.Vec3f(0, REALSENSE_PITCH_DEG, 0))
    _set_mass(cam.GetPrim(), cfg.camera_mass)
    _add_box_geom(stage, cam_path, REALSENSE_SIZE)

    # PMW3901 optical-flow sensor: 20 x 20 x 5 mm board, downward-facing
    # (rotated 180 deg about X). Mass = cfg.optical_flow_mass kg.
    flow_path = f"{sensors_path}/pmw3901_optical_flow"
    flow = UsdGeom.Xform.Define(stage, flow_path)
    UsdGeom.XformCommonAPI(flow).SetTranslate(Gf.Vec3d(0, 0, -cfg.body_half))
    UsdGeom.XformCommonAPI(flow).SetRotate(Gf.Vec3f(180, 0, 0))
    _set_mass(flow.GetPrim(), cfg.optical_flow_mass)
    _add_box_geom(stage, flow_path, OPTICAL_FLOW_SIZE)

    # Slamtec RPLiDAR A2 (2D 360 deg lidar): cylinder, radius 38 mm,
    # height 41 mm. Mass = cfg.lidar_mass kg. Mounted on top of the frame.
    lidar_path = f"{sensors_path}/rplidar_a2"
    lidar = UsdGeom.Xform.Define(stage, lidar_path)
    lidar_mount_z = cfg.center_plate_thickness / 2 + RPLIDAR_HEIGHT / 2
    UsdGeom.XformCommonAPI(lidar).SetTranslate(Gf.Vec3d(0, 0, lidar_mount_z))
    _set_mass(lidar.GetPrim(), cfg.lidar_mass)
    lidar_geom = UsdGeom.Cylinder.Define(stage, f"{lidar_path}/geom")
    lidar_geom.CreateRadiusAttr(RPLIDAR_RADIUS)
    lidar_geom.CreateHeightAttr(RPLIDAR_HEIGHT)

    # --- Pusher mechanism (spring-loaded) ---
    if cfg.has_pusher:
        # Pad is a sibling of the drone body (both under the wrapper) so the
        # prismatic joint connects two distinct rigid bodies.
        pusher_grp = f"{wrapper_path}/pusher"
        UsdGeom.Xform.Define(stage, pusher_grp)

        # Pusher pad (spring-loaded contact tip): disk, radius 15 mm,
        # height 8 mm. Mass = cfg.pusher_mass kg. Prismatic joint along body X.
        pusher_pad_path = f"{pusher_grp}/pad"
        pad_xform = UsdGeom.Xform.Define(stage, pusher_pad_path)
        UsdPhysics.RigidBodyAPI.Apply(pad_xform.GetPrim())
        _set_mass(pad_xform.GetPrim(), cfg.pusher_mass)
        pusher_pad_x = cfg.body_half + PUSHER_X_OFFSET
        UsdGeom.XformCommonAPI(pad_xform).SetTranslate(
            Gf.Vec3d(pusher_pad_x, 0, PUSHER_Z_OFFSET)
        )

        pad_geom = UsdGeom.Cylinder.Define(stage, f"{pusher_pad_path}/geom")
        pad_geom.CreateRadiusAttr(PUSHER_PAD_RADIUS)
        pad_geom.CreateHeightAttr(PUSHER_PAD_HEIGHT)
        UsdPhysics.CollisionAPI.Apply(pad_geom.GetPrim())

        # Joint prim sits at a flat, stable path under the articulation root so
        # Isaac Lab's `joint_names_expr=["pusher_joint"]` resolves unambiguously.
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{wrapper_path}/pusher_joint")
        joint.CreateBody0Rel().SetTargets([root_path])
        joint.CreateBody1Rel().SetTargets([pusher_pad_path])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(PUSHER_TRAVEL)
        # Local attachment frames: on body0 the joint sits where the pad rests;
        # on body1 (the pad itself) the joint sits at the pad's origin.
        joint.CreateLocalPos0Attr(Gf.Vec3f(pusher_pad_x, 0.0, PUSHER_Z_OFFSET))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(PUSHER_STIFFNESS)
        drive.CreateDampingAttr(PUSHER_DAMPING)

    # --- Power: battery + wiring/power module/mounting ---
    power_path = f"{root_path}/power"
    UsdGeom.Xform.Define(stage, power_path)

    # Battery (4S 5000 mAh LiPo class): 140 x 45 x 25 mm pack, slung below the
    # center plate. Mass = cfg.battery_mass kg.
    battery_path = f"{power_path}/battery"
    battery = UsdGeom.Xform.Define(stage, battery_path)
    UsdGeom.XformCommonAPI(battery).SetTranslate(Gf.Vec3d(0, 0, -cfg.body_half * 0.5))
    _set_mass(battery.GetPrim(), cfg.battery_mass)
    _add_box_geom(stage, battery_path, BATTERY_SIZE)

    # Wiring / PDB / power module / mounting hardware: 50 x 30 x 10 mm
    # stand-in. Mass = cfg.wiring_mass kg.
    wiring_path = f"{power_path}/wiring"
    wiring = UsdGeom.Xform.Define(stage, wiring_path)
    _set_mass(wiring.GetPrim(), cfg.wiring_mass)
    _add_box_geom(stage, wiring_path, WIRING_SIZE)

    stage.SetDefaultPrim(wrapper.GetPrim())
    stage.Export(stage_path)
    return stage_path


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "quadcopter_500mm"
    if config_name not in CONFIGS:
        print(f"Unknown config '{config_name}'. Available: {', '.join(CONFIGS)}")
        sys.exit(1)

    path = build_drone_usd(CONFIGS[config_name])
    print(f"Saved: {path}")

    simulation_app.close()

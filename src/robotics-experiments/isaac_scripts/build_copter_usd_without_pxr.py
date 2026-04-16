# build_drone_usd.py — config-driven drone USD generator for Isaac Sim
#
# Run inside isaac-sim docker:
#   /isaac-sim/python.sh proj/robotics-experiments/isaac_scripts/build_copter_usd_without_pxr.py [config_name]
#
# Available configs: quadcopter_500mm (default), hexacopter_s550

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

ASSETS_DIR = "/isaac-sim/proj/assets/LightSwitchWarehouse"


@dataclass
class DroneConfig:
    name: str
    output_filename: str
    body_half: float
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
    has_pusher: bool


QUADCOPTER_500MM = DroneConfig(
    name="quadcopter_500mm",
    output_filename="quadcopter_500mm.usda",
    body_half=0.06,
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
    has_pusher=True,
)

_hex_arm = 0.275                # 550mm diagonal → 275mm arm
HEXACOPTER_S550 = DroneConfig(
    name="hexacopter_s550",
    output_filename="hexacopter_s550.usda",
    body_half=0.08,
    total_mass=2.5,
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
    has_pusher=False,
)

CONFIGS: dict[str, DroneConfig] = {
    "quadcopter_500mm": QUADCOPTER_500MM,
    "hexacopter_s550": HEXACOPTER_S550,
}


def build_drone_usd(cfg: DroneConfig) -> str:
    stage_path = f"{ASSETS_DIR}/{cfg.output_filename}"

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # --- Root rigid body ---
    root = UsdGeom.Xform.Define(stage, "/quadcopter")
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(cfg.total_mass)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*cfg.diagonal_inertia))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*cfg.center_of_mass))

    # --- Body box ---
    body = UsdGeom.Cube.Define(stage, "/quadcopter/body")
    body.CreateSizeAttr(1.0)
    body.AddScaleOp().Set(Gf.Vec3f(cfg.body_half * 2, cfg.body_half * 2, cfg.body_half))
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())

    # --- Arms + motor housings + rotor discs ---
    for i, (x, y, z) in enumerate(cfg.motor_positions):
        arm = UsdGeom.Capsule.Define(stage, f"/quadcopter/arm_{i+1}")
        arm.CreateRadiusAttr(cfg.arm_radius)
        arm_length = math.hypot(x, y)
        arm.CreateHeightAttr(arm_length * cfg.arm_height_factor)
        angle = math.degrees(math.atan2(y, x))
        UsdGeom.XformCommonAPI(arm).SetRotate(Gf.Vec3f(0, 90, angle))
        UsdGeom.XformCommonAPI(arm).SetTranslate(Gf.Vec3d(x / 2, y / 2, 0))
        UsdPhysics.CollisionAPI.Apply(arm.GetPrim())

        motor = UsdGeom.Cylinder.Define(stage, f"/quadcopter/motor_{i+1}")
        motor.CreateRadiusAttr(cfg.motor_radius)
        motor.CreateHeightAttr(cfg.motor_height)
        UsdGeom.XformCommonAPI(motor).SetTranslate(Gf.Vec3d(x, y, z))

        rotor = UsdGeom.Cylinder.Define(stage, f"/quadcopter/rotor_{i+1}")
        rotor.CreateRadiusAttr(cfg.rotor_radius)
        rotor.CreateHeightAttr(0.002)
        UsdGeom.XformCommonAPI(rotor).SetTranslate(Gf.Vec3d(x, y, z + 0.02))
        rotor.GetPrim().CreateAttribute(
            "drone:spinDir", Sdf.ValueTypeNames.String
        ).Set(cfg.spin_dirs[i])

    # --- Sensor mount frames ---
    UsdGeom.Xform.Define(stage, "/quadcopter/imu_frame")

    cam = UsdGeom.Xform.Define(stage, "/quadcopter/camera_frame")
    UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(cfg.body_half + 0.01, 0, 0.01))
    UsdGeom.XformCommonAPI(cam).SetRotate(Gf.Vec3f(0, -15, 0))

    flow = UsdGeom.Xform.Define(stage, "/quadcopter/optical_flow_frame")
    UsdGeom.XformCommonAPI(flow).SetTranslate(Gf.Vec3d(0, 0, -cfg.body_half))
    UsdGeom.XformCommonAPI(flow).SetRotate(Gf.Vec3f(180, 0, 0))

    lidar = UsdGeom.Xform.Define(stage, "/quadcopter/lidar_frame")
    UsdGeom.XformCommonAPI(lidar).SetTranslate(Gf.Vec3d(0, 0, cfg.body_half + 0.04))

    # --- Pusher mechanism (optional) ---
    if cfg.has_pusher:
        pusher_path = "/quadcopter/pusher_pad"
        pad = UsdGeom.Cylinder.Define(stage, pusher_path)
        pad.CreateRadiusAttr(0.015)
        pad.CreateHeightAttr(0.008)
        UsdGeom.XformCommonAPI(pad).SetTranslate(Gf.Vec3d(cfg.body_half + 0.05, 0, 0.01))
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(pad.GetPrim())

        joint = UsdPhysics.PrismaticJoint.Define(stage, "/quadcopter/pusher_joint")
        joint.CreateBody0Rel().SetTargets(["/quadcopter/body"])
        joint.CreateBody1Rel().SetTargets([pusher_path])
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(0.015)

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(800.0)
        drive.CreateDampingAttr(40.0)

    stage.SetDefaultPrim(root.GetPrim())
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

# build_quadcopter_usd.py — revised to run inside SimulationApp
# so it uses omni.usd instead of raw pxr

# Run using the following command inside isaac-sim docker:
# /isaac-sim/python.sh proj/robotics-experiments/isaac_scripts/build_quadcopter_usd_without_pxr.py

# This script can be remade to use only pxr,
# and that will make it runnable at any environment
# where pxr is installed

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "hide_ui": True})

# ALL imports after SimulationApp
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf  # pxr is loaded by Kit at this point
import math

# This path concides with the mounts for Isaac Sim container
STAGE_PATH = "/isaac-sim/proj/assets/LightSwitchWarehouse/quadcopter_500mm.usda"
ARM_LENGTH  = 0.25      # 500mm diagonal → 250mm arm
ROTOR_R     = 0.1143   # 9" prop radius
BODY_HALF   = 0.06     # 120mm body cube half-extent
TOTAL_MASS  = 1.5      # kg

# Create a new stage via omni.usd (not Usd.Stage.CreateNew directly)
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

# --- Root rigid body ---
root = UsdGeom.Xform.Define(stage, "/quadcopter")
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
mass_api.CreateMassAttr(TOTAL_MASS)
mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0347, 0.0458, 0.0977))
mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0.01))

# --- Body box ---
body = UsdGeom.Cube.Define(stage, "/quadcopter/body")
body.CreateSizeAttr(1.0)
body.AddScaleOp().Set(Gf.Vec3f(BODY_HALF * 2, BODY_HALF * 2, BODY_HALF))
UsdPhysics.CollisionAPI.Apply(body.GetPrim())

# --- 4 arms + motor housings + rotor discs ---
motor_positions = [
    ( ARM_LENGTH,  ARM_LENGTH, 0.02),  # M1 FR CCW
    (-ARM_LENGTH,  ARM_LENGTH, 0.02),  # M2 FL CW
    (-ARM_LENGTH, -ARM_LENGTH, 0.02),  # M3 RL CCW
    ( ARM_LENGTH, -ARM_LENGTH, 0.02),  # M4 RR CW
]
spin_dirs = ["CCW", "CW", "CCW", "CW"]

for i, (x, y, z) in enumerate(motor_positions):
    arm = UsdGeom.Capsule.Define(stage, f"/quadcopter/arm_{i+1}")
    arm.CreateRadiusAttr(0.008)
    arm.CreateHeightAttr(ARM_LENGTH * 1.8)
    angle = math.degrees(math.atan2(y, x))
    UsdGeom.XformCommonAPI(arm).SetRotate(Gf.Vec3f(0, 90, angle))
    UsdGeom.XformCommonAPI(arm).SetTranslate(Gf.Vec3d(x / 2, y / 2, 0))
    UsdPhysics.CollisionAPI.Apply(arm.GetPrim())

    motor = UsdGeom.Cylinder.Define(stage, f"/quadcopter/motor_{i+1}")
    motor.CreateRadiusAttr(0.015)
    motor.CreateHeightAttr(0.03)
    UsdGeom.XformCommonAPI(motor).SetTranslate(Gf.Vec3d(x, y, z))

    rotor = UsdGeom.Cylinder.Define(stage, f"/quadcopter/rotor_{i+1}")
    rotor.CreateRadiusAttr(ROTOR_R)
    rotor.CreateHeightAttr(0.002)
    UsdGeom.XformCommonAPI(rotor).SetTranslate(Gf.Vec3d(x, y, z + 0.02))
    rotor.GetPrim().CreateAttribute(
        "drone:spinDir", Sdf.ValueTypeNames.String
    ).Set(spin_dirs[i])

# --- Sensor mount frames ---
# IMU (center of mass)
UsdGeom.Xform.Define(stage, "/quadcopter/imu_frame")

# Forward RGB-D camera (-15° tilt)
cam = UsdGeom.Xform.Define(stage, "/quadcopter/camera_frame")
UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(BODY_HALF + 0.01, 0, 0.01))
UsdGeom.XformCommonAPI(cam).SetRotate(Gf.Vec3f(0, -15, 0))

# Optical flow (downward)
flow = UsdGeom.Xform.Define(stage, "/quadcopter/optical_flow_frame")
UsdGeom.XformCommonAPI(flow).SetTranslate(Gf.Vec3d(0, 0, -BODY_HALF))
UsdGeom.XformCommonAPI(flow).SetRotate(Gf.Vec3f(180, 0, 0))

# 2D LiDAR (top, horizontal)
lidar = UsdGeom.Xform.Define(stage, "/quadcopter/lidar_frame")
UsdGeom.XformCommonAPI(lidar).SetTranslate(Gf.Vec3d(0, 0, BODY_HALF + 0.04))

# Add pusher -- prim, joints, drive
pusher_path = "/quadcopter/pusher_pad"
pad = UsdGeom.Cylinder.Define(stage, pusher_path)
pad.CreateRadiusAttr(0.015)
pad.CreateHeightAttr(0.008)
UsdGeom.XformCommonAPI(pad).SetTranslate(Gf.Vec3d(BODY_HALF + 0.05, 0, 0.01))
UsdPhysics.CollisionAPI.Apply(pad.GetPrim())

joint = UsdPhysics.PrismaticJoint.Define(stage, "/quadcopter/pusher_joint")
joint.CreateBody0Rel().SetTargets(["/quadcopter"])
joint.CreateBody1Rel().SetTargets([pusher_path])
joint.CreateAxisAttr("X")
joint.CreateLowerLimitAttr(0.0)
joint.CreateUpperLimitAttr(0.015)   # 15mm travel

drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
drive.CreateTypeAttr("force")
drive.CreateStiffnessAttr(800.0)    # N/m
drive.CreateDampingAttr(40.0)       # ~0.7× critical damping

stage.SetDefaultPrim(root.GetPrim())
stage.Export(STAGE_PATH)
print(f"Saved: {STAGE_PATH}")

simulation_app.close()

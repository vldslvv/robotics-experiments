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
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from robotics_experiments.drone._airframe import (
    CONFIG_DIR,
    AirframeConfig,
    ArmGeometry,
    ArmPlacement,
    BoxPart,
    CylinderPart,
    _available_airframe_config_names,
    _replace_pos,
    _shift_z,
    compute_body_mass_properties,
    load_builder_config,
)

ASSETS_DIR = "isaac/assets/LightSwitchWarehouse"


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


# --- USD authoring helpers --------------------------------------------------

Vec3 = tuple[float, float, float]
_PUSHER_MOUNT_COLOR = (0.18, 0.18, 0.20)
_PUSHER_BOOM_COLOR = (0.06, 0.06, 0.07)
_PUSHER_PLUNGER_COLOR = (0.65, 0.68, 0.72)
_PUSHER_PAD_COLOR = (1.0, 0.32, 0.04)


def _set_mass(prim: Usd.Prim, mass_kg: float) -> None:
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass_kg)


def _set_display_color(gprim: UsdGeom.Gprim, color: Vec3) -> None:
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


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
    """Author cosmetic propeller geometry under the animated rotor marker.

    A small central hub cylinder plus ``num_blades`` thin blade boxes fanned
    evenly about Z provide the visual. All dimensions derive from the configured
    propeller, and the subtree spins as a unit because it is parented under the
    marker's RotateZ op. Collision must not be parented here: changing a
    collision shape's ancestor transform invalidates Isaac's physics tensor
    views, so the stationary swept-disc collider is authored beside this marker
    by :func:`_author_arm`.
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

    # The prop marker above is animated for cosmetic spin. Keep the complete
    # swept-disc collision envelope on a stationary sibling under the arm so a
    # visual RotateZ update never rebuilds a PhysX shape or invalidates tensor
    # views. The cylinder is rotationally symmetric, so it represents the same
    # occupied volume without inheriting the cosmetic animation.
    prop_collision = UsdGeom.Cylinder.Define(stage, f"{arm_path}/prop_collision")
    prop_collision.CreateRadiusAttr(prop_part.radius)
    prop_collision.CreateHeightAttr(prop_part.height)
    _apply_pose(prop_collision, prop_part.position, prop_part.rotation_deg)
    UsdPhysics.CollisionAPI.Apply(prop_collision.GetPrim())
    prop_collision.MakeInvisible()


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

    # Pusher tool. The mount and carbon boom are fixed children of the drone
    # body. The plunger and forward-facing pad form the moving rigid body linked
    # by pusher_joint. Keeping the pad collision at /pusher/pad/geom preserves
    # the training/runtime contact-sensor contract.
    if cfg.pusher is not None and cfg.pusher.enabled:
        mount = cfg.pusher.mount
        boom = cfg.pusher.boom
        plunger = cfg.pusher.plunger
        pad = cfg.pusher.pad
        pusher_drive = cfg.pusher.drive

        fixed_path = f"{root_path}/pusher"
        UsdGeom.Xform.Define(stage, fixed_path)
        mount_xform = _author_box(stage, f"{fixed_path}/mount", mount)
        UsdPhysics.CollisionAPI.Apply(mount_xform.GetPrim())
        mount_geom = UsdGeom.Cube(stage.GetPrimAtPath(f"{fixed_path}/mount/geom"))
        _set_display_color(mount_geom, _PUSHER_MOUNT_COLOR)

        boom_geom = _author_cylinder(stage, f"{fixed_path}/fixed_boom", boom)
        UsdPhysics.CollisionAPI.Apply(boom_geom.GetPrim())
        _set_display_color(boom_geom, _PUSHER_BOOM_COLOR)

        pusher_grp = f"{wrapper_path}/pusher"
        UsdGeom.Xform.Define(stage, pusher_grp)
        pad_path = f"{pusher_grp}/pad"

        pad_xform = UsdGeom.Xform.Define(stage, pad_path)
        UsdPhysics.RigidBodyAPI.Apply(pad_xform.GetPrim())
        _set_mass(pad_xform.GetPrim(), pad.mass)
        _apply_pose(pad_xform, pad.position)

        plunger_geom = UsdGeom.Cylinder.Define(stage, f"{pad_path}/plunger")
        plunger_geom.CreateAxisAttr("X")
        plunger_geom.CreateRadiusAttr(plunger.radius)
        plunger_geom.CreateHeightAttr(plunger.length)
        plunger_center_x = -(pad.height + plunger.length) / 2.0
        _apply_pose(plunger_geom, (plunger_center_x, 0.0, 0.0))
        _set_display_color(plunger_geom, _PUSHER_PLUNGER_COLOR)

        pad_geom = UsdGeom.Cylinder.Define(stage, f"{pad_path}/geom")
        pad_geom.CreateAxisAttr("X")
        pad_geom.CreateRadiusAttr(pad.radius)
        pad_geom.CreateHeightAttr(pad.height)
        UsdPhysics.CollisionAPI.Apply(pad_geom.GetPrim())
        _set_display_color(pad_geom, _PUSHER_PAD_COLOR)

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

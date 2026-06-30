import argparse
import importlib
from pathlib import Path

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILDER_PATH = _REPO_ROOT / "isaac/runtime/standalone/build_copter_usd_standalone.py"
_CONFIG_DIR = _REPO_ROOT / "configs/usd_builder"


def _load_builder_module():
    return importlib.import_module("isaac.runtime.standalone.build_copter_usd_standalone")


def test_usd_builder_default_hydra_config_loads_holybro_x500_v2():
    builder = _load_builder_module()

    config = builder.load_builder_config(config_dir=_CONFIG_DIR)

    assert config.output_dir == "isaac/assets/LightSwitchWarehouse"
    assert config.airframe.pusher is not None
    assert config.airframe.pusher.enabled is True
    assert config.airframe.pusher.drive.travel == 0.015
    assert config.airframe.name == "holybro_x500_v2"
    assert config.airframe.output_filename == "holybro_x500_v2.usda"


def test_usd_builder_computes_body_mass_properties_from_components():
    builder = _load_builder_module()
    cfg = builder.load_builder_config(config_dir=_CONFIG_DIR).airframe

    total_mass, com, diagonal_inertia = builder.compute_body_mass_properties(cfg)

    # total_mass is the sum of every body component mass, EXCLUDING the pusher
    # pad (a separate rigid body) and the massless booms. Computing it from the
    # cited component masses is the single source -- editing one mass propagates.
    arm = cfg.arm_geometry
    expected_mass = (
        cfg.center_plate.mass
        + len(cfg.arms) * (arm.motor.mass + arm.esc.mass + arm.propeller.mass)
        + cfg.pixhawk.mass
        + cfg.jetson.mass
        + cfg.realsense.mass
        + cfg.optical_flow.mass
        + cfg.lidar.mass
        + cfg.battery.mass
        + cfg.wiring.mass
    )
    assert total_mass == pytest.approx(expected_mass)
    assert total_mass == pytest.approx(2.2992)

    # Symmetric quad: yaw inertia is the largest principal moment, and the two
    # horizontal moments are close (the forward sensors break exact equality).
    ixx, iyy, izz = diagonal_inertia
    assert izz > ixx
    assert izz > iyy
    assert ixx == pytest.approx(iyy, rel=0.25)

    # CoM sits near the geometric center, exactly on the lateral plane of symmetry.
    assert abs(com[0]) < 0.02
    assert com[1] == pytest.approx(0.0, abs=1e-6)
    assert abs(com[2]) < 0.02


def test_usd_builder_hydra_override_loads_hexacopter():
    builder = _load_builder_module()

    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=["airframes=hexacopter_s550"],
    )

    assert config.airframe.name == "hexacopter_s550"
    assert config.airframe.prim_name == "hexacopter"
    assert len(config.airframe.arms) == 6


def test_usd_builder_quad_px4_motor_metadata_matches_generic_quad_x_order():
    builder = _load_builder_module()
    config = builder.load_builder_config(config_dir=_CONFIG_DIR)

    assert [arm.px4_motor_index for arm in config.airframe.arms] == [2, 1, 3, 0]
    assert [arm.spin_dir for arm in config.airframe.arms] == ["CCW", "CW", "CCW", "CW"]


def test_usd_builder_positional_cli_shorthand_maps_to_hydra_override():
    builder = _load_builder_module()
    args = argparse.Namespace(
        config_name="hexacopter_s550",
        overrides=["airframes.pusher.drive.stiffness=900.0"],
        output_dir="/tmp/usd-out",
    )

    assert builder._hydra_overrides_from_args(args) == [
        "airframes=hexacopter_s550",
        "airframes.pusher.drive.stiffness=900.0",
        "output_dir=/tmp/usd-out",
    ]


def test_usd_builder_hydra_override_cli_does_not_force_default_airframe():
    builder = _load_builder_module()
    args = argparse.Namespace(
        config_name=None,
        overrides=["airframes=hexacopter_s550"],
        output_dir=None,
    )

    assert builder._hydra_overrides_from_args(args) == ["airframes=hexacopter_s550"]


def test_usd_builder_all_discovers_airframe_configs():
    builder = _load_builder_module()

    assert builder._available_airframe_config_names(_CONFIG_DIR) == [
        "hexacopter_s550",
        "holybro_x500_v2",
    ]


def test_usd_builder_all_generates_every_airframe_config(tmp_path):
    builder = _load_builder_module()

    paths = builder._build_all_airframe_usds(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}"],
    )

    assert {Path(path).name for path in paths} == {
        "hexacopter_s550.usda",
        "holybro_x500_v2.usda",
    }
    assert Usd.Stage.Open(str(tmp_path / "hexacopter_s550.usda")).GetDefaultPrim().GetName() == (
        "hexacopter"
    )
    assert Usd.Stage.Open(str(tmp_path / "holybro_x500_v2.usda")).GetDefaultPrim().GetName() == (
        "holybro_x500_v2"
    )


def test_usd_builder_generates_quad_with_rotor_metadata(tmp_path):
    builder = _load_builder_module()
    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}"],
    )

    stage_path = builder.build_drone_usd(
        config.airframe,
        output_dir=config.output_dir,
    )
    stage = Usd.Stage.Open(stage_path)

    assert stage.GetDefaultPrim().GetName() == "holybro_x500_v2"
    # arm_i carries the placement authored at config index i-1; the PX4 motor
    # index is a marker attribute, independent of the arm number.
    expected = {
        "/holybro_x500_v2/body/arms/arm_4/prop": (0, "CW"),
        "/holybro_x500_v2/body/arms/arm_2/prop": (1, "CW"),
        "/holybro_x500_v2/body/arms/arm_1/prop": (2, "CCW"),
        "/holybro_x500_v2/body/arms/arm_3/prop": (3, "CCW"),
    }
    for path, (px4_index, spin_dir) in expected.items():
        prim = stage.GetPrimAtPath(path)
        assert prim.GetAttribute("drone:rotorMarker").Get() is True
        assert prim.GetAttribute("drone:px4MotorIndex").Get() == px4_index
        assert prim.GetAttribute("drone:spinDir").Get() == spin_dir

        # The marker is an Xform carrying a cosmetic RotateZ spin op (initial 0,
        # ordered after the translate) plus the propeller visual (hub + blades).
        assert prim.IsA(UsdGeom.Xform)
        xformable = UsdGeom.Xformable(prim)
        op_order = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
        assert "xformOp:translate" in op_order
        rotate_ops = [
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ
        ]
        assert len(rotate_ops) == 1
        assert op_order.index("xformOp:translate") < op_order.index(rotate_ops[0].GetOpName())
        assert rotate_ops[0].Get() == 0.0

        assert stage.GetPrimAtPath(f"{path}/hub").IsA(UsdGeom.Cylinder)
        blades = [child for child in prim.GetChildren() if child.GetName().startswith("blade_")]
        assert len(blades) == 2  # num_blades from the quad config


def test_usd_builder_default_quad_has_pusher(tmp_path):
    builder = _load_builder_module()
    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}"],
    )

    stage_path = builder.build_drone_usd(config.airframe, output_dir=config.output_dir)
    stage = Usd.Stage.Open(stage_path)

    # Pad rigid body + its collision geom exist under the pusher group.
    pad = stage.GetPrimAtPath("/holybro_x500_v2/pusher/pad")
    assert pad.IsValid()
    assert pad.HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/holybro_x500_v2/pusher/pad/geom").IsA(UsdGeom.Cylinder)

    # Prismatic spring-damper joint with travel from airframes.pusher.drive.travel.
    joint_prim = stage.GetPrimAtPath("/holybro_x500_v2/pusher_joint")
    assert joint_prim.IsValid()
    joint = UsdPhysics.PrismaticJoint(joint_prim)
    assert joint.GetAxisAttr().Get() == "X"
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(config.airframe.pusher.drive.travel)


def test_usd_builder_pusher_disabled_omits_pusher(tmp_path):
    builder = _load_builder_module()
    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}", "airframes.pusher.enabled=false"],
    )
    assert config.airframe.pusher.enabled is False

    stage_path = builder.build_drone_usd(config.airframe, output_dir=config.output_dir)
    stage = Usd.Stage.Open(stage_path)

    # No pusher subtree is authored when disabled.
    assert not stage.GetPrimAtPath("/holybro_x500_v2/pusher_joint").IsValid()
    assert not stage.GetPrimAtPath("/holybro_x500_v2/pusher").IsValid()

    # The rest of the airframe is intact (rotor markers still present).
    for path in (
        "/holybro_x500_v2/body/arms/arm_4/prop",
        "/holybro_x500_v2/body/arms/arm_1/prop",
    ):
        assert stage.GetPrimAtPath(path).GetAttribute("drone:rotorMarker").Get() is True


def test_usd_builder_esc_parallel_to_arm(tmp_path):
    builder = _load_builder_module()
    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}"],
    )

    stage_path = builder.build_drone_usd(config.airframe, output_dir=config.output_dir)
    stage = Usd.Stage.Open(stage_path)

    from pxr import Gf  # noqa: PLC0415

    def _world_dir(path, local):
        m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        origin = m.Transform(Gf.Vec3d(0, 0, 0))
        tip = m.Transform(Gf.Vec3d(*local))
        return Gf.Vec3d(tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]).GetNormalized()

    # Each ESC's long axis (local X) must be parallel to its boom (capsule long
    # axis = local Z), with the azimuth inherited from the shared arm Xform.
    for i in range(1, len(config.airframe.arms) + 1):
        boom_dir = _world_dir(f"/holybro_x500_v2/body/arms/arm_{i}/boom", (0, 0, 1))
        esc_dir = _world_dir(f"/holybro_x500_v2/body/arms/arm_{i}/esc", (1, 0, 0))
        # Parallel as a line (sign-agnostic): |dot| == 1.
        assert abs(boom_dir[0] * esc_dir[0] + boom_dir[1] * esc_dir[1]) == pytest.approx(
            1.0, abs=1e-9
        )

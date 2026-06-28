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


def test_usd_builder_default_hydra_config_loads_quadcopter():
    builder = _load_builder_module()

    config = builder.load_builder_config(config_dir=_CONFIG_DIR)

    assert config.output_dir == "isaac/assets/LightSwitchWarehouse"
    assert config.drone.pusher is not None
    assert config.drone.pusher.enabled is True
    assert config.drone.pusher.drive.travel == 0.015
    assert config.drone.name == "quadcopter_500mm"
    assert config.drone.output_filename == "quadcopter_500mm.usda"


def test_usd_builder_hydra_override_loads_hexacopter():
    builder = _load_builder_module()

    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=["drone=hexacopter_s550"],
    )

    assert config.drone.name == "hexacopter_s550"
    assert config.drone.prim_name == "hexacopter"
    assert len(config.drone.arms) == 6


def test_usd_builder_quad_px4_motor_metadata_matches_generic_quad_x_order():
    builder = _load_builder_module()
    config = builder.load_builder_config(config_dir=_CONFIG_DIR)

    assert [arm.px4_motor_index for arm in config.drone.arms] == [2, 1, 3, 0]
    assert [arm.spin_dir for arm in config.drone.arms] == ["CCW", "CW", "CCW", "CW"]


def test_usd_builder_positional_cli_shorthand_maps_to_hydra_override():
    builder = _load_builder_module()
    args = argparse.Namespace(
        config_name="hexacopter_s550",
        overrides=["drone.pusher.drive.stiffness=900.0"],
        output_dir="/tmp/usd-out",
    )

    assert builder._hydra_overrides_from_args(args) == [
        "drone=hexacopter_s550",
        "drone.pusher.drive.stiffness=900.0",
        "output_dir=/tmp/usd-out",
    ]


def test_usd_builder_hydra_override_cli_does_not_force_default_drone():
    builder = _load_builder_module()
    args = argparse.Namespace(
        config_name=None,
        overrides=["drone=hexacopter_s550"],
        output_dir=None,
    )

    assert builder._hydra_overrides_from_args(args) == ["drone=hexacopter_s550"]


def test_usd_builder_generates_quad_with_rotor_metadata(tmp_path):
    builder = _load_builder_module()
    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}"],
    )

    stage_path = builder.build_drone_usd(
        config.drone,
        output_dir=config.output_dir,
    )
    stage = Usd.Stage.Open(stage_path)

    assert stage.GetDefaultPrim().GetName() == "quadcopter"
    # arm_i carries the placement authored at config index i-1; the PX4 motor
    # index is a marker attribute, independent of the arm number.
    expected = {
        "/quadcopter/body/arms/arm_4/prop": (0, "CW"),
        "/quadcopter/body/arms/arm_2/prop": (1, "CW"),
        "/quadcopter/body/arms/arm_1/prop": (2, "CCW"),
        "/quadcopter/body/arms/arm_3/prop": (3, "CCW"),
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

    stage_path = builder.build_drone_usd(config.drone, output_dir=config.output_dir)
    stage = Usd.Stage.Open(stage_path)

    # Pad rigid body + its collision geom exist under the pusher group.
    pad = stage.GetPrimAtPath("/quadcopter/pusher/pad")
    assert pad.IsValid()
    assert pad.HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/quadcopter/pusher/pad/geom").IsA(UsdGeom.Cylinder)

    # Prismatic spring-damper joint with travel from drone.pusher.drive.travel.
    joint_prim = stage.GetPrimAtPath("/quadcopter/pusher_joint")
    assert joint_prim.IsValid()
    joint = UsdPhysics.PrismaticJoint(joint_prim)
    assert joint.GetAxisAttr().Get() == "X"
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(config.drone.pusher.drive.travel)


def test_usd_builder_pusher_disabled_omits_pusher(tmp_path):
    builder = _load_builder_module()
    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}", "drone.pusher.enabled=false"],
    )
    assert config.drone.pusher.enabled is False

    stage_path = builder.build_drone_usd(config.drone, output_dir=config.output_dir)
    stage = Usd.Stage.Open(stage_path)

    # No pusher subtree is authored when disabled.
    assert not stage.GetPrimAtPath("/quadcopter/pusher_joint").IsValid()
    assert not stage.GetPrimAtPath("/quadcopter/pusher").IsValid()

    # The rest of the airframe is intact (rotor markers still present).
    for path in (
        "/quadcopter/body/arms/arm_4/prop",
        "/quadcopter/body/arms/arm_1/prop",
    ):
        assert stage.GetPrimAtPath(path).GetAttribute("drone:rotorMarker").Get() is True


def test_usd_builder_esc_parallel_to_arm(tmp_path):
    builder = _load_builder_module()
    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}"],
    )

    stage_path = builder.build_drone_usd(config.drone, output_dir=config.output_dir)
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
    for i in range(1, len(config.drone.arms) + 1):
        boom_dir = _world_dir(f"/quadcopter/body/arms/arm_{i}/boom", (0, 0, 1))
        esc_dir = _world_dir(f"/quadcopter/body/arms/arm_{i}/esc", (1, 0, 0))
        # Parallel as a line (sign-agnostic): |dot| == 1.
        assert abs(boom_dir[0] * esc_dir[0] + boom_dir[1] * esc_dir[1]) == pytest.approx(
            1.0, abs=1e-9
        )

import math

import numpy as np
import pytest

from robotics_experiments.px4_bridge.rotor_discovery import discover_px4_rotor_mapping

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def _make_stage(markers):
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Drone")
    body = UsdGeom.Xform.Define(stage, "/World/Drone/body")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    for marker in markers:
        path = marker.get("path", f"/World/Drone/body/propulsion/motor_{marker['index']}/prop")
        parent = "/".join(path.split("/")[:-1])
        UsdGeom.Xform.Define(stage, parent)
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdGeom.XformCommonAPI(UsdGeom.Xformable(prim)).SetTranslate(
            Gf.Vec3d(*marker.get("position", (0.25, -0.25, 0.04)))
        )
        if marker.get("rotor_marker", True):
            prim.CreateAttribute("drone:rotorMarker", Sdf.ValueTypeNames.Bool).Set(True)
        if "index" in marker:
            prim.CreateAttribute("drone:px4MotorIndex", Sdf.ValueTypeNames.Int).Set(marker["index"])
        if "spin" in marker:
            prim.CreateAttribute("drone:spinDir", Sdf.ValueTypeNames.String).Set(marker["spin"])
    return stage


def _valid_quad_markers():
    return [
        {
            "path": "/World/Drone/body/propulsion/motor_4/prop",
            "index": 0,
            "spin": "CW",
            "position": (0.25, -0.25, 0.04),
        },
        {
            "path": "/World/Drone/body/propulsion/motor_2/prop",
            "index": 1,
            "spin": "CW",
            "position": (-0.25, 0.25, 0.04),
        },
        {
            "path": "/World/Drone/body/propulsion/motor_1/prop",
            "index": 2,
            "spin": "CCW",
            "position": (0.25, 0.25, 0.04),
        },
        {
            "path": "/World/Drone/body/propulsion/motor_3/prop",
            "index": 3,
            "spin": "CCW",
            "position": (-0.25, -0.25, 0.04),
        },
    ]


def _valid_hexa_markers():
    # hexacopter_s550 asset: arm at azimuth 60*index deg (FLU), R = 0.275 m,
    # spin alternating CCW/CW from index 0.
    r = 0.275
    specs = [
        (0, 0.0, "CCW"),
        (1, 60.0, "CW"),
        (2, 120.0, "CCW"),
        (3, 180.0, "CW"),
        (4, -120.0, "CCW"),
        (5, -60.0, "CW"),
    ]
    return [
        {
            "path": f"/World/Drone/body/propulsion/motor_{index}/prop",
            "index": index,
            "spin": spin,
            "position": (
                r * math.cos(math.radians(az)),
                r * math.sin(math.radians(az)),
                0.04,
            ),
        }
        for index, az, spin in specs
    ]


def test_valid_hexa_metadata_returns_px4_order_and_rot_dir():
    stage = _make_stage(reversed(_valid_hexa_markers()))

    mapping = discover_px4_rotor_mapping(stage, "/World/Drone")

    assert mapping.num_rotors == 6
    assert mapping.rotor_link_paths == [
        f"/World/Drone/body/propulsion/motor_{index}/prop" for index in range(6)
    ]
    assert mapping.rot_dir == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]


def test_six_rotor_wrong_heading_order_fails():
    markers = _valid_hexa_markers()
    # Swap index 1 and index 2 positions so the per-index headings no longer
    # match the canonical 60*index clocking.
    markers[1], markers[2] = (
        {**markers[1], "position": markers[2]["position"]},
        {**markers[2], "position": markers[1]["position"]},
    )
    stage = _make_stage(markers)

    with pytest.raises(RuntimeError, match="Generic Hexarotor X marker order mismatch"):
        discover_px4_rotor_mapping(stage, "/World/Drone")


def test_valid_quad_metadata_returns_px4_order_and_rot_dir():
    stage = _make_stage(reversed(_valid_quad_markers()))

    mapping = discover_px4_rotor_mapping(stage, "/World/Drone")

    assert mapping.rotor_link_paths == [
        "/World/Drone/body/propulsion/motor_4/prop",
        "/World/Drone/body/propulsion/motor_2/prop",
        "/World/Drone/body/propulsion/motor_1/prop",
        "/World/Drone/body/propulsion/motor_3/prop",
    ]
    assert mapping.rot_dir == [-1.0, -1.0, 1.0, 1.0]
    np.testing.assert_allclose(
        mapping.positions_body[:, :2],
        [[0.25, -0.25], [-0.25, 0.25], [0.25, 0.25], [-0.25, -0.25]],
    )


def test_missing_marker_metadata_fails():
    stage = _make_stage([{**_valid_quad_markers()[0], "rotor_marker": False}])

    with pytest.raises(RuntimeError, match="No PX4 rotor markers"):
        discover_px4_rotor_mapping(stage, "/World/Drone")


def test_duplicate_index_fails():
    markers = _valid_quad_markers()
    markers[1] = {**markers[1], "index": 0}
    stage = _make_stage(markers)

    with pytest.raises(RuntimeError, match="Duplicate drone:px4MotorIndex=0"):
        discover_px4_rotor_mapping(stage, "/World/Drone")


def test_non_contiguous_index_fails():
    markers = _valid_quad_markers()[:3]
    markers[2] = {**markers[2], "index": 3}
    stage = _make_stage(markers)

    with pytest.raises(RuntimeError, match="must be contiguous"):
        discover_px4_rotor_mapping(stage, "/World/Drone")


def test_invalid_spin_direction_fails():
    markers = [{**_valid_quad_markers()[0], "spin": "SIDEWAYS"}]
    stage = _make_stage(markers)

    with pytest.raises(RuntimeError, match="invalid drone:spinDir"):
        discover_px4_rotor_mapping(stage, "/World/Drone")


def test_zero_xy_arm_length_fails():
    markers = [{**_valid_quad_markers()[0], "position": (0.0, 0.0, 0.04)}]
    stage = _make_stage(markers)

    with pytest.raises(RuntimeError, match="zero XY arm length"):
        discover_px4_rotor_mapping(stage, "/World/Drone")


def test_four_rotor_wrong_quadrant_order_fails():
    markers = _valid_quad_markers()
    markers[0] = {**markers[0], "position": (-0.25, -0.25, 0.04)}
    stage = _make_stage(markers)

    with pytest.raises(RuntimeError, match="Generic Quad X marker order mismatch"):
        discover_px4_rotor_mapping(stage, "/World/Drone")

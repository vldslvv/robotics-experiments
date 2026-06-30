"""Discover PX4 rotor topology from authored USD asset metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["DiscoveredRotorMapping", "discover_px4_rotor_mapping"]


@dataclass(frozen=True)
class DiscoveredRotorMapping:
    """Rotor marker paths and Pegasus spin directions in PX4 motor-index order."""

    rotor_link_paths: list[str]
    rot_dir: list[float]
    positions_body: np.ndarray

    @property
    def num_rotors(self) -> int:
        return len(self.rotor_link_paths)


def discover_px4_rotor_mapping(stage, articulation_path: str) -> DiscoveredRotorMapping:
    """Scan a spawned USD articulation for PX4 rotor marker metadata.

    Rotor marker prims are identified by ``drone:rotorMarker = true`` under
    the drone rigid body. Each marker must author ``drone:px4MotorIndex`` and
    ``drone:spinDir``. Returned paths and spin directions are ordered by PX4
    motor index.
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    root_path = str(articulation_path).rstrip("/")
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Articulation prim does not exist: {root_path}")

    body_path = _resolve_body_path(stage, root_path, UsdPhysics)
    body_prim = stage.GetPrimAtPath(body_path)
    body_to_world = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_to_body = body_to_world.GetInverse()

    markers = []
    for prim in _iter_descendants(body_prim):
        marker_attr = prim.GetAttribute("drone:rotorMarker")
        if not marker_attr or marker_attr.Get() is not True:
            continue

        path = prim.GetPath().pathString
        index_attr = prim.GetAttribute("drone:px4MotorIndex")
        index = index_attr.Get() if index_attr and index_attr.HasValue() else None
        if not isinstance(index, int):
            raise RuntimeError(f"Rotor marker {path} is missing integer drone:px4MotorIndex")

        spin_attr = prim.GetAttribute("drone:spinDir")
        spin_dir = spin_attr.Get() if spin_attr and spin_attr.HasValue() else None
        if spin_dir not in {"CW", "CCW"}:
            raise RuntimeError(
                f"Rotor marker {path} has invalid drone:spinDir={spin_dir!r}; "
                "expected 'CW' or 'CCW'"
            )

        rotor_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        position_body = world_to_body.Transform(rotor_to_world.ExtractTranslation())
        position = np.asarray(
            [position_body[0], position_body[1], position_body[2]],
            dtype=np.float64,
        )
        if math.hypot(float(position[0]), float(position[1])) <= 1e-9:
            raise RuntimeError(f"Rotor marker {path} has zero XY arm length")

        markers.append((index, path, spin_dir, position))

    if not markers:
        raise RuntimeError(f"No PX4 rotor markers found under {body_path}")

    _validate_indices([(index, path) for index, path, _spin, _pos in markers])
    markers_by_index = sorted(markers, key=lambda item: item[0])
    paths = [path for _index, path, _spin, _pos in markers_by_index]
    spins = [spin for _index, _path, spin, _pos in markers_by_index]
    positions = np.asarray([pos for _index, _path, _spin, pos in markers_by_index])

    if len(markers_by_index) == 4:
        _validate_generic_quad_x_order(paths, positions)
    elif len(markers_by_index) == 6:
        _validate_generic_hexa_x_order(paths, positions)

    rot_dir = [-1.0 if spin == "CW" else 1.0 for spin in spins]
    return DiscoveredRotorMapping(
        rotor_link_paths=paths,
        rot_dir=rot_dir,
        positions_body=positions,
    )


def _resolve_body_path(stage, articulation_path: str, usd_physics) -> str:
    articulation_prim = stage.GetPrimAtPath(articulation_path)
    if articulation_prim.HasAPI(usd_physics.RigidBodyAPI):
        return articulation_path

    body_path = f"{articulation_path}/body"
    body_prim = stage.GetPrimAtPath(body_path)
    if body_prim.IsValid() and body_prim.HasAPI(usd_physics.RigidBodyAPI):
        return body_path

    raise RuntimeError(
        "Drone articulation must be a rigid body or contain a rigid body child at " f"{body_path}"
    )


def _iter_descendants(prim):
    for child in prim.GetAllChildren():
        yield child
        yield from _iter_descendants(child)


def _validate_indices(indices: Sequence[tuple[int, str]]) -> None:
    seen: dict[int, str] = {}
    for index, path in indices:
        if index < 0:
            raise RuntimeError(f"Rotor marker {path} has negative drone:px4MotorIndex={index}")
        if index in seen:
            raise RuntimeError(f"Duplicate drone:px4MotorIndex={index}: {seen[index]} and {path}")
        seen[index] = path

    expected = set(range(len(indices)))
    actual = set(seen)
    if actual != expected:
        raise RuntimeError(
            "Rotor marker drone:px4MotorIndex values must be contiguous "
            f"0..{len(indices) - 1}; got {sorted(actual)}"
        )


def _validate_generic_quad_x_order(paths: Sequence[str], positions: np.ndarray) -> None:
    # Isaac body frame is FLU. PX4 Generic Quad X order is FR, RL, FL, RR,
    # which maps to (+X,-Y), (-X,+Y), (+X,+Y), (-X,-Y) in FLU.
    expected_signs = ((1, -1), (-1, 1), (1, 1), (-1, -1))
    for index, (path, position, signs) in enumerate(
        zip(paths, positions, expected_signs, strict=False)
    ):
        x, y = float(position[0]), float(position[1])
        expected_x, expected_y = signs
        if _sign(x) != expected_x or _sign(y) != expected_y:
            raise RuntimeError(
                "4-rotor PX4 Generic Quad X marker order mismatch: "
                f"index {index} path {path} has body XY=({x:.6g}, {y:.6g})"
            )


def _validate_generic_hexa_x_order(paths: Sequence[str], positions: np.ndarray) -> None:
    # Isaac body frame is FLU. The hexacopter_s550 asset clocks its arms at
    # azimuth 60*index deg (index 0 on the forward +X axis), matching the
    # CA_ROTOR geometry generated from configs/airframes/hexacopter_s550.yaml.
    # Guard the per-index arm heading so a future asset/airframe mismatch fails
    # loudly here.
    tolerance_deg = 20.0
    for index, (path, position) in enumerate(zip(paths, positions, strict=False)):
        x, y = float(position[0]), float(position[1])
        angle_deg = math.degrees(math.atan2(y, x))
        expected_deg = 60.0 * index
        error_deg = abs((angle_deg - expected_deg + 180.0) % 360.0 - 180.0)
        if error_deg > tolerance_deg:
            raise RuntimeError(
                "6-rotor PX4 Generic Hexarotor X marker order mismatch: "
                f"index {index} path {path} has body XY=({x:.6g}, {y:.6g}) "
                f"(heading {angle_deg:.1f}deg, expected {expected_deg:.1f}deg)"
            )


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0

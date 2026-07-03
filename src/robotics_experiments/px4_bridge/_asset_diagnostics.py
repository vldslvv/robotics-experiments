"""USD asset diagnostics for PX4 bridge initialization."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .frames import _normalize_quat_wxyz, _quat_conjugate_wxyz, _quat_multiply_wxyz

RootBodyMismatchPolicy = Literal["warn", "fail", "ignore"]

_LOGGER = logging.getLogger(__name__)


# NOTE: Known noisy diagnostic. With the spawned quadcopter asset and
# `drone.spawn_stationary = true`, this validation emits a "PX4 root/body pose
# mismatch" WARNING on every bridge start (seen in test-integration-px4-takeoff,
# -instability, and -drone-articulation). The reported delta is a fixed,
# asset-intrinsic, pure-Z TRANSLATION of ~0.1045 m (translation_delta_m ~= 0.1045,
# orientation_delta_rad == 0): spawn lifts the articulation root so the asset's
# lowest geometry clears the ground, while the rigid `body` sub-prim keeps its
# authored local position.
#
# This particular offset is benign: the bridge reads state and applies thrust
# entirely in the /body frame (see IsaacArticulationIO.read_state and
# _resolve_rotor_positions_body), so the root translation never reaches PX4. An
# ORIENTATION mismatch is the genuinely dangerous case (it would corrupt PX4
# attitude/state). The current single `policy` + threshold conflates the two, so
# we cannot silence the benign translation noise without also losing the
# orientation safety check.
#
# Possible improvements (deferred - keep this analysis so it is not re-derived):
#   1. Split severity: handle translation and orientation independently
#      (e.g. orientation_policy default "warn", translation_policy default
#      "ignore"), keeping the orientation check while silencing the static
#      translation offset.
#   2. Make policy + thresholds configurable through the Hydra `px4` group
#      (add fields to Px4BridgeConfig + configs/experiment/px4/default.yaml and
#      thread RootBodyPoseDiagnosticParams from bridge_host -> Px4Bridge ->
#      IsaacArticulationIO, which already accepts it) so a scene can escalate or
#      ignore as needed.
#   3. Fix the asset so the articulation root coincides with the body (offset 0).
#      Out of scope here: that regenerates/commits a USD asset.
@dataclass(frozen=True)
class RootBodyPoseDiagnosticParams:
    """Configuration for root/body pose validation."""

    policy: RootBodyMismatchPolicy = "warn"
    translation_threshold_m: float = 1e-4
    orientation_threshold_rad: float = 1e-4


@dataclass(frozen=True)
class RootBodyPoseDiagnostics:
    """Pose comparison between an articulation root and its simulated body."""

    articulation_path: str
    body_path: str
    root_world_pose: tuple[np.ndarray, np.ndarray]
    body_world_pose: tuple[np.ndarray, np.ndarray]
    body_relative_to_root_pose: tuple[np.ndarray, np.ndarray]
    translation_delta_m: float
    orientation_delta_rad: float
    translation_threshold_m: float
    orientation_threshold_rad: float

    @property
    def has_mismatch(self) -> bool:
        return (
            self.translation_delta_m > self.translation_threshold_m
            or self.orientation_delta_rad > self.orientation_threshold_rad
        )


def validate_root_body_pose(
    stage,
    articulation_path: str,
    body_path: str,
    params: RootBodyPoseDiagnosticParams,
    *,
    logger: logging.Logger = _LOGGER,
) -> RootBodyPoseDiagnostics | None:
    """Validate root/body pose according to ``params``.

    Logs the computed diagnostics and enforces the mismatch policy: ``"warn"``
    logs a warning, ``"fail"`` raises ``RuntimeError``. Returns the computed
    diagnostics, or ``None`` when the policy is ``"ignore"`` (no diagnostic
    work is done).
    """
    if params.policy not in ("warn", "fail", "ignore"):
        raise ValueError(f"Unsupported root/body mismatch policy: {params.policy!r}")
    if params.policy == "ignore":
        return None

    diagnostics = compute_root_body_pose_diagnostics(
        stage,
        articulation_path,
        body_path,
        translation_threshold_m=params.translation_threshold_m,
        orientation_threshold_rad=params.orientation_threshold_rad,
    )
    message = _format_root_body_pose_diagnostics(diagnostics)
    logger.info("PX4 root/body pose diagnostic: %s", message)
    if diagnostics.has_mismatch:
        if params.policy == "warn":
            logger.warning("PX4 root/body pose mismatch: %s", message)
        else:
            logger.error("PX4 root/body pose mismatch: %s", message)
            raise RuntimeError(f"PX4 root/body pose mismatch: {message}")
    return diagnostics


def compute_root_body_pose_diagnostics(
    stage,
    articulation_path: str,
    body_path: str,
    *,
    translation_threshold_m: float = 1e-4,
    orientation_threshold_rad: float = 1e-4,
) -> RootBodyPoseDiagnostics:
    """Compute root/body world-pose and relative-pose diagnostics from a USD stage."""
    from pxr import Usd, UsdGeom

    root_path = str(articulation_path)
    resolved_body_path = str(body_path)
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Articulation prim does not exist: {root_path}")
    body_prim = stage.GetPrimAtPath(resolved_body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"Body prim does not exist: {resolved_body_path}")

    root_to_world = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    body_to_world = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    root_translation = root_to_world.ExtractTranslation()
    body_translation = body_to_world.ExtractTranslation()
    root_position = _vec3_to_array(root_translation)
    body_position = _vec3_to_array(body_translation)
    root_orientation = _matrix_quat_wxyz(root_to_world)
    body_orientation = _matrix_quat_wxyz(body_to_world)

    body_position_root = _vec3_to_array(root_to_world.GetInverse().Transform(body_translation))
    body_orientation_root = _normalize_quat_wxyz(
        _quat_multiply_wxyz(_quat_conjugate_wxyz(root_orientation), body_orientation)
    )
    if body_orientation_root[0] < 0.0:
        body_orientation_root = -body_orientation_root

    return RootBodyPoseDiagnostics(
        articulation_path=root_path,
        body_path=resolved_body_path,
        root_world_pose=(root_position, root_orientation),
        body_world_pose=(body_position, body_orientation),
        body_relative_to_root_pose=(body_position_root, body_orientation_root),
        translation_delta_m=float(np.linalg.norm(body_position - root_position)),
        orientation_delta_rad=_quat_angle_between_rad(root_orientation, body_orientation),
        translation_threshold_m=float(translation_threshold_m),
        orientation_threshold_rad=float(orientation_threshold_rad),
    )


def _format_root_body_pose_diagnostics(diagnostics: RootBodyPoseDiagnostics) -> str:
    relative_position, relative_orientation = diagnostics.body_relative_to_root_pose
    root_position, root_orientation = diagnostics.root_world_pose
    body_position, body_orientation = diagnostics.body_world_pose
    return (
        f"articulation={diagnostics.articulation_path} body={diagnostics.body_path} "
        f"translation_delta_m={diagnostics.translation_delta_m:.9g} "
        f"translation_threshold_m={diagnostics.translation_threshold_m:.9g} "
        f"orientation_delta_rad={diagnostics.orientation_delta_rad:.9g} "
        f"orientation_threshold_rad={diagnostics.orientation_threshold_rad:.9g} "
        f"root_world_pose=(position={root_position.tolist()}, "
        f"orientation_wxyz={root_orientation.tolist()}) "
        f"body_world_pose=(position={body_position.tolist()}, "
        f"orientation_wxyz={body_orientation.tolist()}) "
        f"body_relative_to_root_pose=(position={relative_position.tolist()}, "
        f"orientation_wxyz={relative_orientation.tolist()})"
    )


def _vec3_to_array(value) -> np.ndarray:
    return np.asarray([value[0], value[1], value[2]], dtype=np.float64)


def _matrix_quat_wxyz(matrix) -> np.ndarray:
    quat = matrix.ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    result = np.array(
        [quat.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
        dtype=np.float64,
    )
    return _normalize_quat_wxyz(result)


def _quat_angle_between_rad(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    a = _normalize_quat_wxyz(a_wxyz)
    b = _normalize_quat_wxyz(b_wxyz)
    dot = abs(float(np.dot(a, b)))
    return float(2.0 * math.acos(np.clip(dot, -1.0, 1.0)))

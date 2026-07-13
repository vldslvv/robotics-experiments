"""Isaac Sim articulation read/write - the only Isaac-aware module.

Encapsulates state extraction (pose, velocities) and force application so the
rest of the bridge has no Isaac Sim dependency. Tests can substitute a fake
implementing the same ``read_state`` / ``apply_dynamics`` seam.

Initialization is deferred until the first tick after Play because Isaac's
articulation handles cannot be opened before the physics scene exists.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np

import robotics_experiments.px4_bridge.frames as frames
from robotics_experiments.px4_bridge.asset_diagnostics import (
    RootBodyPoseDiagnosticParams,
    validate_root_body_pose,
)
from robotics_experiments.px4_bridge.rotor_discovery import _resolve_body_path
from robotics_experiments.px4_bridge.thrust_model import RotorDynamics
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = ["IsaacArticulationIO"]

_LOGGER = logging.getLogger(__name__)


class IsaacArticulationIO:
    """Read state from and apply forces to a drone articulation in Isaac Sim.

    Args:
        articulation_path: USD prim path to the drone root articulation.
        rotor_link_paths: USD prim paths of rotor thrust markers **in PX4 motor-index
            order** - front-right CW, rear-left CW, front-left CCW,
            rear-right CCW for ``Generic Quadrotor X``. PX4 rotor geometry is
            expressed in FRD, while Isaac body marker positions are FLU, so
            PX4 right maps to Isaac negative Y. Wrong order makes the drone
            flip on takeoff; validate against airframe geometry.
    """

    def __init__(
        self,
        *,
        articulation_path: str,
        rotor_link_paths: Sequence[str],
        root_body_pose_diagnostics: RootBodyPoseDiagnosticParams | None = None,
        rotor_spin_visual_scale: float = 0.03,
        rotor_spin_signs: Sequence[float] | None = None,
    ):
        self._articulation_path = str(articulation_path)
        self._rotor_link_paths: List[str] = [str(p) for p in rotor_link_paths]
        self.num_rotors = len(self._rotor_link_paths)
        self._initialized = False
        self._root_body_pose_diagnostics = (
            root_body_pose_diagnostics
            if root_body_pose_diagnostics is not None
            else RootBodyPoseDiagnosticParams()
        )
        self._rotor_spin_visual_scale = float(rotor_spin_visual_scale)
        self._configured_rotor_spin_signs = self._coerce_rotor_spin_signs(
            rotor_spin_signs,
            expected_count=len(self._rotor_link_paths),
        )

        self._articulation: Any | None = None
        self._body: Any | None = None
        self._body_path: str | None = None
        self._rotor_positions_body: Optional[np.ndarray] = None
        self._force_apply_count: int = 0
        # Cosmetic rotor-spin state, populated in initialize(): one RotateZ
        # xformOp handle per rotor marker, the per-rotor accumulated visual angle
        # in degrees, and the per-rotor spin-direction sign (+1 CCW / -1 CW). This
        # is visual-only and never feeds back into physics or the lockstep.
        self._rotor_spin_ops: List[Any] = []
        self._rotor_angle_deg: Optional[np.ndarray] = None
        self._rotor_spin_signs: Optional[np.ndarray] = None

    @staticmethod
    def _coerce_rotor_spin_signs(
        rotor_spin_signs: Sequence[float] | None,
        *,
        expected_count: int,
    ) -> np.ndarray | None:
        if rotor_spin_signs is None:
            return None
        signs = np.asarray(rotor_spin_signs, dtype=np.float64)
        if signs.shape != (expected_count,):
            raise ValueError(
                f"rotor_spin_signs expected shape ({expected_count},), got {signs.shape}"
            )
        if not np.all(np.isin(signs, [-1.0, 1.0])):
            raise ValueError("rotor_spin_signs must contain only -1.0 (CW) or 1.0 (CCW)")
        return signs

    # ---- Lifecycle ---------------------------------------------------------

    def is_initialized(self) -> bool:
        """Return whether all articulation handles were acquired successfully."""
        return self._initialized

    def initialize(self) -> None:
        """Acquire the articulation body and rotor thrust marker poses.

        Safe to call multiple times - first successful call wins.
        """
        if self._articulation is not None:
            return

        # Imports are deferred so this module can be imported in non-Isaac
        # contexts (e.g. unit tests for the surrounding logic).
        import omni.usd
        from isaacsim.core.prims import RigidPrim, SingleArticulation
        from pxr import UsdPhysics

        self._articulation = SingleArticulation(prim_path=self._articulation_path)
        if hasattr(self._articulation, "initialize"):
            try:
                self._articulation.initialize()
            except Exception as exc:
                # Some Isaac versions auto-initialize; tolerate failure here
                # and let the first read fail loudly if the handle is stale.
                _LOGGER.warning(
                    "Isaac articulation initialization failed for %s: %s",
                    self._articulation_path,
                    exc,
                )

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Cannot initialize Isaac articulation I/O without an open USD stage")

        body_path = _resolve_body_path(stage, self._articulation_path, UsdPhysics)
        self._body_path = body_path
        self._body = RigidPrim(prim_paths_expr=body_path)
        if hasattr(self._body, "initialize"):
            try:
                self._body.initialize()
            except Exception as exc:
                _LOGGER.warning("Isaac body initialization failed for %s: %s", body_path, exc)

        validate_root_body_pose(
            stage,
            self._articulation_path,
            body_path,
            self._root_body_pose_diagnostics,
            logger=_LOGGER,
        )

        self._rotor_positions_body = self._resolve_rotor_positions_body(stage, body_path)
        self._resolve_rotor_spin_ops(stage)
        self._initialized = True

    def _resolve_rotor_spin_ops(self, stage) -> None:
        """Acquire the cosmetic RotateZ xformOp authored on each rotor marker.

        The drone asset builder authors a RotateZ op on every rotor marker, so
        this requires it and raises when it is missing. Runtime wiring injects
        rotor signs in PX4 motor-index order; direct/manual construction falls
        back to ``drone:spinDir`` with the same convention as
        :mod:`rotor_discovery` (CW -> -1, CCW -> +1). This only touches visual
        transforms and never the physics body.
        """
        from pxr import UsdGeom

        ops: List[Any] = []
        signs: List[float] = []
        for path in self._rotor_link_paths:
            rotor_prim = stage.GetPrimAtPath(path)
            xformable = UsdGeom.Xformable(rotor_prim)
            spin_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                    spin_op = op
                    break
            if spin_op is None:
                raise RuntimeError(f"Rotor marker prim is missing its RotateZ spin op: {path}")
            ops.append(spin_op)

            if self._configured_rotor_spin_signs is None:
                spin_attr = rotor_prim.GetAttribute("drone:spinDir")
                spin_dir = spin_attr.Get() if spin_attr and spin_attr.HasValue() else None
                if spin_dir == "CW":
                    signs.append(-1.0)
                elif spin_dir == "CCW":
                    signs.append(1.0)
                else:
                    raise RuntimeError(
                        f"Rotor marker {path} has invalid drone:spinDir={spin_dir!r}; "
                        "expected 'CW' or 'CCW'"
                    )

        self._rotor_spin_ops = ops
        self._rotor_spin_signs = (
            self._configured_rotor_spin_signs.copy()
            if self._configured_rotor_spin_signs is not None
            else np.asarray(signs, dtype=np.float64)
        )
        self._rotor_angle_deg = np.zeros(len(ops), dtype=np.float64)

    def _resolve_rotor_positions_body(self, stage, body_path: str) -> np.ndarray:
        from pxr import Usd, UsdGeom

        body_prim = stage.GetPrimAtPath(body_path)
        body_xform = UsdGeom.Xformable(body_prim)
        body_to_world = body_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_to_body = body_to_world.GetInverse()

        positions = []
        rotor_debug = []
        for index, path in enumerate(self._rotor_link_paths):
            rotor_prim = stage.GetPrimAtPath(path)
            if not rotor_prim.IsValid():
                raise RuntimeError(f"Rotor thrust marker prim does not exist: {path}")
            rotor_xform = UsdGeom.Xformable(rotor_prim)
            rotor_to_world = rotor_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            position_body = world_to_body.Transform(rotor_to_world.ExtractTranslation())
            position = [position_body[0], position_body[1], position_body[2]]
            positions.append(position)
            rotor_debug.append(
                {
                    "index": index,
                    "path": path,
                    "position_body": [float(v) for v in position],
                }
            )

        _LOGGER.info(
            "PX4 rotor mapping resolved: articulation=%s body=%s rotors=%s",
            self._articulation_path,
            body_path,
            rotor_debug,
        )

        return np.asarray(positions, dtype=np.float64)

    def shutdown(self) -> None:
        """Drop handles. Safe to call repeatedly."""
        self._articulation = None
        self._body = None
        self._body_path = None
        self._rotor_positions_body = None
        self._force_apply_count = 0
        self._rotor_spin_ops = []
        self._rotor_angle_deg = None
        self._rotor_spin_signs = None
        self._initialized = False

    # ---- Per-tick I/O ------------------------------------------------------

    def read_state(self, sim_time_us: int) -> VehicleState:
        """Sample body state into a :class:`VehicleState` (ENU/FLU).

        The PX4 HIL IMU path recomputes acceleration inside the Pegasus IMU
        from velocity deltas, so this state sample only carries pose and
        velocity data.
        """
        if self._articulation is None:
            self.initialize()
        if self._body is None:
            raise RuntimeError("Isaac body handle is not initialized")

        # SingleArticulation is useful for lifecycle/root access, but PX4 state
        # must describe the rigid body that receives rotor thrust, drag, and
        # body torque. If the USD root and /body differ, sampling the root would
        # send PX4 state for a different frame than the one physics actuates.
        position_enu, orientation_wxyz = self._sample_body_world_pose()

        linear_velocity_enu = self._sample_body_linear_velocity()
        # Isaac reports angular velocity in the world frame. PX4 HIL gyro data
        # is body-frame FRD, so keep VehicleState's contract as body-FLU here
        # and let the MAVLink port do the final FLU->FRD sign conversion.
        angular_velocity_enu = self._sample_body_angular_velocity()
        angular_velocity_flu = frames.world_enu_to_body_flu(
            orientation_wxyz,
            angular_velocity_enu,
        )

        return VehicleState(
            position_enu=position_enu,
            orientation_flu_enu_wxyz=orientation_wxyz,
            linear_velocity_enu=linear_velocity_enu,
            angular_velocity_flu=angular_velocity_flu,
            sim_time_us=int(sim_time_us),
        )

    def _require_body_method(self, singular_name: str, plural_name: str):
        if self._body is None:
            raise RuntimeError("Isaac body handle is not initialized")
        singular = getattr(self._body, singular_name, None)
        if callable(singular):
            return singular, False
        plural = getattr(self._body, plural_name, None)
        if callable(plural):
            return plural, True
        body_path = self._body_path or "<unresolved body>"
        raise RuntimeError(
            f"Isaac body {body_path} is missing required API " f"{singular_name}()/{plural_name}()"
        )

    @staticmethod
    def _first_body_row(value) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 2 and array.shape[0] >= 1:
            return array[0]
        return array

    def _sample_body_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        method, is_batched = self._require_body_method("get_world_pose", "get_world_poses")
        pos, quat_wxyz = method()
        if is_batched:
            pos = self._first_body_row(pos)
            quat_wxyz = self._first_body_row(quat_wxyz)
        return np.asarray(pos, dtype=np.float64), np.asarray(quat_wxyz, dtype=np.float64)

    def _sample_body_linear_velocity(self) -> np.ndarray:
        method, is_batched = self._require_body_method(
            "get_linear_velocity",
            "get_linear_velocities",
        )
        velocity = method()
        if is_batched:
            velocity = self._first_body_row(velocity)
        return np.asarray(velocity, dtype=np.float64)

    def _sample_body_angular_velocity(self) -> np.ndarray:
        method, is_batched = self._require_body_method(
            "get_angular_velocity",
            "get_angular_velocities",
        )
        velocity = method()
        if is_batched:
            velocity = self._first_body_row(velocity)
        return np.asarray(velocity, dtype=np.float64)

    @staticmethod
    def _rotor_forces_to_world(
        *,
        body_position_world: np.ndarray,
        body_orientation_wxyz: np.ndarray,
        rotor_positions_body: np.ndarray,
        forces_n: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert Pegasus body-frame rotor effects into Isaac world-space inputs."""
        position_world = np.asarray(body_position_world, dtype=np.float64)
        orientation_wxyz = np.asarray(body_orientation_wxyz, dtype=np.float64)
        positions_body = np.asarray(rotor_positions_body, dtype=np.float64)
        forces = np.asarray(forces_n, dtype=np.float64)
        if positions_body.ndim != 2 or positions_body.shape[1] != 3:
            raise ValueError(
                f"rotor_positions_body expects shape (N, 3), got {positions_body.shape}"
            )
        if forces.shape != (positions_body.shape[0],):
            raise ValueError(
                f"forces_n expected shape ({positions_body.shape[0]},), got {forces.shape}"
            )

        rotor_positions_world = np.asarray(
            [
                position_world + frames.rotate_vector_wxyz(orientation_wxyz, position_body)
                for position_body in positions_body
            ],
            dtype=np.float64,
        )
        rotor_forces_world = np.asarray(
            [
                frames.rotate_vector_wxyz(
                    orientation_wxyz,
                    np.array([0.0, 0.0, float(force_n)], dtype=np.float64),
                )
                for force_n in forces
            ],
            dtype=np.float64,
        )
        return rotor_forces_world, rotor_positions_world

    def apply_forces(self, forces_n: np.ndarray) -> None:
        """Apply per-rotor thrust at each marker's current world-space position.

        Pegasus applies rotor thrust in the rotor/body local +Z direction. This
        bridge uses marker prims rather than Pegasus rotor rigid bodies, so it
        converts body-local marker offsets and force vectors into world-space
        arguments before calling Isaac's force-at-position API.

        Args:
            forces_n: Shape ``(num_rotors,)`` array of forces in Newtons.
        """
        if self._body is None or self._rotor_positions_body is None:
            return
        forces = np.asarray(forces_n, dtype=np.float64)
        if forces.shape != (self.num_rotors,):
            raise ValueError(
                f"apply_forces expected shape ({self.num_rotors},), got {forces.shape}"
            )

        body_position_world, body_orientation_wxyz = self._sample_body_world_pose()
        rotor_forces_world, rotor_positions_world = self._rotor_forces_to_world(
            body_position_world=body_position_world,
            body_orientation_wxyz=body_orientation_wxyz,
            rotor_positions_body=self._rotor_positions_body,
            forces_n=forces,
        )
        self._force_apply_count += 1
        if self._force_apply_count <= 3 or self._force_apply_count % 250 == 0:
            net_force_world = np.sum(rotor_forces_world, axis=0)
            net_torque_world = np.sum(
                np.cross(rotor_positions_world - body_position_world, rotor_forces_world),
                axis=0,
            )
            _LOGGER.debug(
                "PX4 rotor force application: n=%d body_position_world=%s "
                "body_quat_wxyz=%s forces_n=%s first_force_world=%s "
                "first_position_world=%s net_force_world=%s net_torque_world=%s "
                "is_global=True",
                self._force_apply_count,
                body_position_world.tolist(),
                body_orientation_wxyz.tolist(),
                forces.tolist(),
                rotor_forces_world[0].tolist() if len(rotor_forces_world) else [],
                rotor_positions_world[0].tolist() if len(rotor_positions_world) else [],
                net_force_world.tolist(),
                net_torque_world.tolist(),
            )

        for force_world, position_world in zip(
            rotor_forces_world, rotor_positions_world, strict=False
        ):
            try:
                self._body.apply_forces_and_torques_at_pos(
                    forces=np.asarray([force_world], dtype=np.float64),
                    positions=np.asarray([position_world], dtype=np.float64),
                    is_global=True,
                )
            except Exception as exc:
                _LOGGER.warning("Failed to apply rotor force to %s: %s", self._body_path, exc)

    def apply_dynamics(self, dynamics: RotorDynamics) -> None:
        """Apply a full rotor-model result to the drone - the single sink to Isaac.

        Per-rotor thrust is applied at each marker; the body reaction torque and
        linear drag are applied at the body origin so yaw authority and damping
        match the Pegasus thrust model.
        """
        self.apply_forces(dynamics.forces_n)
        self._apply_body_effect(force_n=None, torque_n_m=dynamics.body_torque_n_m)
        self._apply_body_effect(force_n=dynamics.body_drag_n, torque_n_m=None)

    def update_rotor_spin(self, rotor_omega_rad_s: np.ndarray, dt: float) -> None:
        """Advance the cosmetic rotor spin from per-rotor angular velocity.

        Visual-only: integrates each rotor's authored RotateZ op by
        ``sign * degrees(omega) * visual_scale * dt`` and writes it back. The
        ``visual_scale`` factor slows the resolved per-airframe omega to a
        non-aliasing on-screen rate. This never touches forces, the body, or the
        MAVLink lockstep, so it cannot perturb flight dynamics.
        """
        if self._rotor_angle_deg is None or self._rotor_spin_signs is None:
            return
        omega = np.asarray(rotor_omega_rad_s, dtype=np.float64)
        if omega.shape != (len(self._rotor_spin_ops),):
            raise ValueError(
                f"update_rotor_spin expected shape ({len(self._rotor_spin_ops)},), "
                f"got {omega.shape}"
            )
        delta_deg = self._rotor_spin_signs * np.degrees(omega) * self._rotor_spin_visual_scale * dt
        rotor_angle_deg = np.mod(self._rotor_angle_deg + delta_deg, 360.0)
        self._rotor_angle_deg = rotor_angle_deg
        for op, angle in zip(self._rotor_spin_ops, rotor_angle_deg, strict=False):
            op.Set(float(angle))

    def _apply_body_effect(
        self,
        *,
        force_n: np.ndarray | None,
        torque_n_m: np.ndarray | None,
    ) -> None:
        if self._body is None:
            return
        force = (
            np.zeros((1, 3), dtype=np.float64)
            if force_n is None
            else np.asarray([force_n], dtype=np.float64)
        )
        torque = (
            np.zeros((1, 3), dtype=np.float64)
            if torque_n_m is None
            else np.asarray([torque_n_m], dtype=np.float64)
        )
        if force.shape != (1, 3):
            raise ValueError(f"force_n expects shape (3,), got {force.shape[1:]}")
        if torque.shape != (1, 3):
            raise ValueError(f"torque_n_m expects shape (3,), got {torque.shape[1:]}")
        if not np.any(force) and not np.any(torque):
            return
        try:
            self._body.apply_forces_and_torques_at_pos(
                forces=force,
                torques=torque,
                positions=np.zeros((1, 3), dtype=np.float64),
                is_global=False,
            )
        except TypeError:
            # Isaac versions differ in accepted kwargs. Fall back to the older
            # force-only call when possible; torque support is validated in the
            # Isaac smoke test for the deployed version.
            if np.any(force):
                self._body.apply_forces_and_torques_at_pos(
                    forces=force,
                    positions=np.zeros((1, 3), dtype=np.float64),
                    is_global=False,
                )
            elif np.any(torque):
                _LOGGER.warning(
                    "Isaac body torque API is unavailable for %s; skipping torque %s",
                    self._body_path,
                    torque,
                )
        except Exception as exc:
            _LOGGER.warning("Failed to apply body effect to %s: %s", self._body_path, exc)

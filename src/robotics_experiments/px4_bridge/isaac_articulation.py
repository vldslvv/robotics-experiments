"""Isaac Sim articulation read/write - the only Isaac-aware module.

Encapsulates state extraction (pose, velocities) and per-rotor force
application so the rest of the bridge has no Isaac Sim dependency. Tests can
substitute a fake implementing the same two ``read_state`` / ``apply_forces``
methods.

Initialization is deferred until the first tick after Play because Isaac's
articulation handles cannot be opened before the physics scene exists.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np

from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = ["IsaacArticulationIO"]

_LOGGER = logging.getLogger(__name__)


class IsaacArticulationIO:
    """Read state from and apply forces to a drone articulation in Isaac Sim.

    Args:
        articulation_path: USD prim path to the drone root articulation.
        rotor_link_paths: USD prim paths of rotor thrust markers **in PX4 motor-index
            order** - front-right CW, rear-left CW, front-left CCW,
            rear-right CCW for ``Generic Quadrotor X``. Wrong order makes the
            drone flip on takeoff; validate against airframe geometry.
    """

    def __init__(self, *, articulation_path: str, rotor_link_paths: Sequence[str]):
        self._articulation_path = str(articulation_path)
        self._rotor_link_paths: List[str] = [str(p) for p in rotor_link_paths]

        self._articulation: Any | None = None
        self._body: Any | None = None
        self._body_path: str | None = None
        self._rotor_positions_body: Optional[np.ndarray] = None
        self._last_velocity_enu: Optional[np.ndarray] = None
        self._last_sim_time_us: Optional[int] = None

    @property
    def num_rotors(self) -> int:
        return len(self._rotor_link_paths)

    @property
    def is_initialized(self) -> bool:
        return self._articulation is not None

    # ---- Lifecycle ---------------------------------------------------------

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

        body_path = self._resolve_body_path(stage)
        self._body_path = body_path
        self._body = RigidPrim(prim_paths_expr=body_path)
        if hasattr(self._body, "initialize"):
            try:
                self._body.initialize()
            except Exception as exc:
                _LOGGER.warning("Isaac body initialization failed for %s: %s", body_path, exc)

        self._rotor_positions_body = self._resolve_rotor_positions_body(stage, body_path)

    def _resolve_body_path(self, stage) -> str:
        from pxr import UsdPhysics

        articulation_prim = stage.GetPrimAtPath(self._articulation_path)
        if not articulation_prim.IsValid():
            raise RuntimeError(f"Articulation prim does not exist: {self._articulation_path}")
        if articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return self._articulation_path

        body_path = f"{self._articulation_path.rstrip('/')}/body"
        body_prim = stage.GetPrimAtPath(body_path)
        if body_prim.IsValid() and body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return body_path

        raise RuntimeError(
            "Drone articulation must be a rigid body or contain a rigid body child at "
            f"{body_path}"
        )

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
            spin_attr = rotor_prim.GetAttribute("drone:spinDir")
            spin_dir = spin_attr.Get() if spin_attr and spin_attr.HasValue() else None
            rotor_debug.append(
                {
                    "index": index,
                    "path": path,
                    "position_body": [float(v) for v in position],
                    "spin_dir": spin_dir,
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
        self._last_velocity_enu = None
        self._last_sim_time_us = None

    # ---- Per-tick I/O ------------------------------------------------------

    def read_state(self, sim_time_us: int) -> VehicleState:
        """Sample articulation state into a :class:`VehicleState` (ENU/FLU).

        Linear acceleration is finite-differenced from the previous tick's
        linear velocity. The first tick produces zero acceleration.
        """
        if self._articulation is None:
            self.initialize()
        if self._articulation is None:
            raise RuntimeError("Isaac articulation handle is not initialized")

        pos, quat_wxyz = self._articulation.get_world_pose()
        position_enu = np.asarray(pos, dtype=np.float64)
        orientation_wxyz = np.asarray(quat_wxyz, dtype=np.float64)

        linear_velocity_enu = np.asarray(self._articulation.get_linear_velocity(), dtype=np.float64)
        angular_velocity_flu = np.asarray(
            self._articulation.get_angular_velocity(), dtype=np.float64
        )

        # Finite-difference accel from last tick (world ENU frame).
        if self._last_velocity_enu is not None and self._last_sim_time_us is not None:
            dt_s = max(1e-6, (sim_time_us - self._last_sim_time_us) * 1e-6)
            linear_acceleration_enu = (linear_velocity_enu - self._last_velocity_enu) / dt_s
        else:
            linear_acceleration_enu = np.zeros(3, dtype=np.float64)

        self._last_velocity_enu = linear_velocity_enu.copy()
        self._last_sim_time_us = sim_time_us

        return VehicleState(
            position_enu=position_enu,
            orientation_flu_enu_wxyz=orientation_wxyz,
            linear_velocity_enu=linear_velocity_enu,
            angular_velocity_flu=angular_velocity_flu,
            linear_acceleration_enu=linear_acceleration_enu,
            sim_time_us=int(sim_time_us),
        )

    def apply_forces(self, forces_n: np.ndarray) -> None:
        """Apply per-rotor forces to the drone body at each marker's local +Z axis.

        Args:
            forces_n: Shape ``(num_rotors,)`` array of forces in Newtons.
        """
        if self._body is None or self._rotor_positions_body is None:
            return
        if forces_n.shape != (self.num_rotors,):
            raise ValueError(
                f"apply_forces expected shape ({self.num_rotors},), got {forces_n.shape}"
            )

        for position_body, magnitude in zip(
            self._rotor_positions_body, forces_n, strict=False
        ):
            try:
                self._body.apply_forces_and_torques_at_pos(
                    forces=np.array([[0.0, 0.0, float(magnitude)]], dtype=np.float64),
                    positions=np.asarray([position_body], dtype=np.float64),
                    is_global=False,
                )
            except Exception as exc:
                _LOGGER.warning("Failed to apply rotor force to %s: %s", self._body_path, exc)

    def apply_body_force(self, force_n: np.ndarray) -> None:
        """Apply a body-frame force at the drone body origin."""
        self._apply_body_effect(force_n=force_n, torque_n_m=None)

    def apply_body_torque(self, torque_n_m: np.ndarray) -> None:
        """Apply a body-frame torque at the drone body origin."""
        self._apply_body_effect(force_n=None, torque_n_m=torque_n_m)

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

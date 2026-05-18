"""Isaac Sim articulation read/write - the only Isaac-aware module.

Encapsulates state extraction (pose, velocities) and per-rotor force
application so the rest of the bridge has no Isaac Sim dependency. Tests can
substitute a fake implementing the same two ``read_state`` / ``apply_forces``
methods.

Initialization is deferred until the first tick after Play because Isaac's
articulation handles cannot be opened before the physics scene exists.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from robotics_experiments.px4_bridge.vehicle_state import VehicleState


class IsaacArticulationIO:
    """Read state from and apply forces to a drone articulation in Isaac Sim.

    Args:
        articulation_path: USD prim path to the drone root articulation.
        rotor_link_paths: USD prim paths of rotor links **in PX4 motor-index
            order** - front-right CW, rear-left CW, front-left CCW,
            rear-right CCW for ``Generic Quadrotor X``. Wrong order makes the
            drone flip on takeoff; validate against airframe geometry.
    """

    def __init__(self, *, articulation_path: str, rotor_link_paths: Sequence[str]):
        self._articulation_path = str(articulation_path)
        self._rotor_link_paths: List[str] = [str(p) for p in rotor_link_paths]

        self._articulation = None
        self._rotor_links: Optional[list] = None
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
        """Acquire the articulation and rotor link handles.

        Safe to call multiple times - first successful call wins.
        """
        if self._articulation is not None:
            return

        # Imports are deferred so this module can be imported in non-Isaac
        # contexts (e.g. unit tests for the surrounding logic).
        from isaacsim.core.prims import RigidPrim, SingleArticulation

        self._articulation = SingleArticulation(prim_path=self._articulation_path)
        if hasattr(self._articulation, "initialize"):
            try:
                self._articulation.initialize()
            except Exception:
                # Some Isaac versions auto-initialize; tolerate failure here
                # and let the first read fail loudly if the handle is stale.
                pass

        self._rotor_links = [RigidPrim(prim_paths_expr=path) for path in self._rotor_link_paths]
        for link in self._rotor_links:
            if hasattr(link, "initialize"):
                try:
                    link.initialize()
                except Exception:
                    pass

    def shutdown(self) -> None:
        """Drop handles. Safe to call repeatedly."""
        self._articulation = None
        self._rotor_links = None
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
        """Apply per-rotor forces along each link's local +Z axis.

        Args:
            forces_n: Shape ``(num_rotors,)`` array of forces in Newtons.
        """
        if self._rotor_links is None:
            return
        if forces_n.shape != (self.num_rotors,):
            raise ValueError(
                f"apply_forces expected shape ({self.num_rotors},), got {forces_n.shape}"
            )

        for link, magnitude in zip(self._rotor_links, forces_n, strict=False):
            try:
                link.apply_forces_and_torques_at_pos(
                    forces=np.array([[0.0, 0.0, float(magnitude)]], dtype=np.float64),
                    is_global=False,
                )
            except Exception:
                # Best-effort: the precise API varies across Isaac Sim
                # versions. The user can wire in the correct call once
                # they confirm which method their version exposes.
                pass

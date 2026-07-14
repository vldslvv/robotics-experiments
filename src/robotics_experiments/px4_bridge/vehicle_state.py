"""Pure-Python vehicle state passed across the bridge layers.

This module has no Isaac Sim, MAVLink, or Pegasus dependency on purpose: it is
the seam every other layer talks across, and is therefore safe to import from
unit tests, staging scripts, and synthetic state generators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["VehicleState"]


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class VehicleState:
    """Frozen snapshot of the drone's state at a single physics tick.

    All vector fields are length-3 ``np.ndarray`` of dtype float64. The frame
    convention is fixed by the project and matches what Isaac Sim natively
    produces; converting to PX4's NED/FRD happens inside the MAVLink port.

    Attributes:
        position_enu: World position ``[x_east, y_north, z_up]`` in meters.
        orientation_flu_enu_wxyz: Quaternion ``[w, x, y, z]`` rotating the FLU
            body frame into the ENU world frame.
        linear_velocity_enu: World-frame linear velocity in m/s.
        angular_velocity_flu: Body-frame (FLU) angular velocity in rad/s.
        sim_time_us: Simulation time in microseconds. Used as the source of
            truth for ``HIL_SENSOR.time_usec`` - must be monotonic and derived
            from sim time, not wall clock, for lockstep to work.
    """

    position_enu: np.ndarray
    orientation_flu_enu_wxyz: np.ndarray
    linear_velocity_enu: np.ndarray
    angular_velocity_flu: np.ndarray
    sim_time_us: int

    @staticmethod
    def hover_at(position_enu=(0.0, 0.0, 1.0), sim_time_us: int = 0) -> "VehicleState":
        """Return a level, stationary state at the given position.

        Useful for staging tests that exercise the network/protocol layer
        without an actual physics simulation.
        """
        return VehicleState(
            position_enu=_readonly(np.array(position_enu, dtype=np.float64, copy=True)),
            orientation_flu_enu_wxyz=_readonly(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
            linear_velocity_enu=_readonly(np.zeros(3, dtype=np.float64)),
            angular_velocity_flu=_readonly(np.zeros(3, dtype=np.float64)),
            sim_time_us=int(sim_time_us),
        )

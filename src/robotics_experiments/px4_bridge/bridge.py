"""Top-level ``Px4Bridge`` orchestrator implementing Contract 1.

The bridge owns nothing scene-related: it does not load USD, register physics
callbacks, or hold the timeline. Its single responsibility is to convert the
articulation state read from Isaac into PX4-bound HIL messages every tick and
apply the resulting actuator commands as forces.

The class composes three injected collaborators so each layer can be staged
and tested independently:

- ``articulation_io``: reads ``VehicleState``, applies per-rotor forces.
- ``port``: pumps state out, pulls actuator commands back (MAVLink in prod,
  fake in tests).
- ``thrust_model``: maps rotor references to Pegasus-compatible dynamics.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import frames
from .flight_stack import AbstractFlightStackBridge
from .simulator import ArticulationIO
from .thrust_model import RotorThrustModel

__all__ = ["Px4Bridge"]

_LOGGER = logging.getLogger(__name__)


class Px4Bridge:
    """Wires articulation I/O, the flight-stack port, and the thrust model."""

    def __init__(
        self,
        *,
        articulation_path: str,
        rotor_link_paths,
        thrust_model: RotorThrustModel,
        port: AbstractFlightStackBridge,
        articulation_io: Optional[ArticulationIO] = None,
        max_tilt_angle_deg: Optional[float] = None,
    ):
        """Build the bridge. No sockets or articulation handles are touched here.

        Args:
            articulation_path: USD path to the drone root articulation.
            rotor_link_paths: USD paths to rotor thrust markers in PX4 motor-index order.
            thrust_model: Maps rotor commands to per-rotor forces and body effects.
            port: Adapter to the flight stack.
            articulation_io: Optional override for Isaac state I/O - used by
                tests to inject a fake. Defaults to a real
                :class:`IsaacArticulationIO`.
            max_tilt_angle_deg: Maximum allowed tilt angle in degrees. When the
                drone's body-Z axis deviates from world-Z by more than this
                angle, the bridge zeros all rotor forces. ``None`` disables
                the check.
        """
        if thrust_model.num_rotors != port.num_rotors:
            raise ValueError(
                f"thrust_model num_rotors={thrust_model.num_rotors} "
                f"!= port num_rotors={port.num_rotors}"
            )
        if len(rotor_link_paths) != port.num_rotors:
            raise ValueError(
                f"rotor_link_paths has {len(rotor_link_paths)} entries; expected {port.num_rotors}"
            )

        self._port = port
        self._thrust_model = thrust_model
        self._tick_count = 0
        self._dynamics_debug_interval = 50
        self._max_tilt_angle_deg = max_tilt_angle_deg
        self._tilt_cutoff_count = 0
        if articulation_io is None:
            from robotics_experiments.px4_bridge.isaac_articulation import (
                IsaacArticulationIO,
            )

            articulation_io = IsaacArticulationIO(
                articulation_path=articulation_path,
                rotor_link_paths=rotor_link_paths,
            )
        self._io = articulation_io

    # ---- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Open the flight-stack connection. Articulation handles are deferred."""
        self._tick_count = 0
        self._port.start()

    def stop(self) -> None:
        """Close the flight-stack connection and drop articulation handles."""
        self._port.stop()
        self._io.shutdown()

    # ---- Per-tick ----------------------------------------------------------

    def on_physics_step(self, dt: float, sim_time_us: int) -> None:
        """Per-physics-tick callback. Read state, exchange MAVLink, apply forces."""
        if not self._io.is_initialized:
            self._io.initialize()

        state = self._io.read_state(int(sim_time_us))
        if not np.all(np.isfinite(state.position_enu)) or not np.all(
            np.isfinite(state.orientation_flu_enu_wxyz)
        ):
            # NaN guard: skip this tick, keep the last good force unchanged.
            return

        if (
            self._max_tilt_angle_deg is not None
            and self._max_tilt_angle_deg > 0.0
        ):
            tilt = frames.tilt_angle_deg(state.orientation_flu_enu_wxyz)
            if tilt > self._max_tilt_angle_deg:
                self._tilt_cutoff_count += 1
                if self._tilt_cutoff_count <= 10 or self._tilt_cutoff_count % 250 == 0:
                    _LOGGER.warning(
                        "PX4 bridge tilt cutoff: count=%d tilt=%.1f deg "
                        "limit=%.1f deg; zeroing rotor forces",
                        self._tilt_cutoff_count,
                        tilt,
                        self._max_tilt_angle_deg,
                    )
                self._io.apply_forces(np.zeros(self._thrust_model.num_rotors, dtype=np.float64))
                return

        commands = self._port.step(state)
        body_velocity_flu = frames.world_enu_to_body_flu(
            state.orientation_flu_enu_wxyz,
            state.linear_velocity_enu,
        )
        dynamics = self._thrust_model.commands_to_dynamics(
            commands,
            linear_body_velocity_flu=body_velocity_flu,
        )
        self._tick_count += 1
        if self._tick_count <= 10 or self._tick_count % self._dynamics_debug_interval == 0:
            _LOGGER.debug(
                "PX4 bridge dynamics: tick=%d sim_time_us=%d connected=%s armed=%s "
                "commands=%s forces_n=%s body_torque_n_m=%s body_drag_n=%s "
                "position_enu=%s quat_wxyz=%s velocity_enu=%s body_velocity_flu=%s",
                self._tick_count,
                int(sim_time_us),
                self.is_connected,
                self.is_armed,
                np.asarray(commands, dtype=np.float64).tolist(),
                np.asarray(dynamics.forces_n, dtype=np.float64).tolist(),
                np.asarray(dynamics.body_torque_n_m, dtype=np.float64).tolist(),
                np.asarray(dynamics.body_drag_n, dtype=np.float64).tolist(),
                np.asarray(state.position_enu, dtype=np.float64).tolist(),
                np.asarray(state.orientation_flu_enu_wxyz, dtype=np.float64).tolist(),
                np.asarray(state.linear_velocity_enu, dtype=np.float64).tolist(),
                np.asarray(body_velocity_flu, dtype=np.float64).tolist(),
            )
        self._io.apply_forces(dynamics.forces_n)
        if hasattr(self._io, "apply_body_torque"):
            self._io.apply_body_torque(dynamics.body_torque_n_m)
        if hasattr(self._io, "apply_body_force"):
            self._io.apply_body_force(dynamics.body_drag_n)

    # ---- Status ------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._port.is_connected

    @property
    def is_armed(self) -> bool:
        return self._port.is_armed

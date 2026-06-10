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

from typing import Optional

import numpy as np

from . import frames
from .flight_stack import AbstractFlightStackBridge
from .simulator import ArticulationIO
from .thrust_model import RotorThrustModel

__all__ = ["Px4Bridge"]


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

        commands = self._port.step(state)
        body_velocity_flu = frames.world_enu_to_body_flu(
            state.orientation_flu_enu_wxyz,
            state.linear_velocity_enu,
        )
        dynamics = self._thrust_model.commands_to_dynamics(
            commands,
            linear_body_velocity_flu=body_velocity_flu,
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

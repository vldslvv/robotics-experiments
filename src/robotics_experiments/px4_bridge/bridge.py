"""Top-level ``Px4Bridge`` orchestrator implementing Contract 1.

The bridge owns nothing scene-related: it does not load USD, register physics
callbacks, or hold the timeline. Its single responsibility is to convert the
articulation state read from Isaac into PX4-bound HIL messages every tick and
apply the resulting actuator commands as forces.

The class composes three injected collaborators so each layer can be staged
and tested independently:

- ``articulation_io``: reads ``VehicleState``, applies a ``RotorDynamics`` via
  ``apply_dynamics`` (rotor forces at markers plus body torque and drag).
- ``port``: ``exchange`` pumps state out and pulls actuator commands back in
  lockstep, as one indivisible tick (MAVLink in prod, fake in tests).
- ``thrust_model``: maps rotor references to Pegasus-compatible dynamics.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from . import frames
from .contract import ActuatorCommands, BridgeStatus
from .flight_stack import AbstractFlightStackBridge
from .simulator import ArticulationIO
from .thrust_model import RotorDynamics, RotorThrustModel

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
        max_tilt_angle_deg: float = 180.0,
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
                angle, the bridge zeros all rotor forces while continuing the
                MAVLink sensor/actuator exchange every tick, so PX4 lockstep is
                preserved. Defaults to ``180.0`` - the maximum possible tilt, so
                the cutoff never fires (disabled) unless a smaller limit is set.
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
        # Per-tick state stashed by send_sensors for the paired recv_and_apply.
        self._pending_state: Any = None
        self._pending_send_result: Optional[ActuatorCommands] = None
        self._pending_force_cutoff: bool = False
        self._pending_sim_time_us: int = 0
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
        """Per-physics-tick callback. Read state, exchange MAVLink, apply forces.

        The single-vehicle composition of the two legs: :meth:`send_sensors`
        publishes this tick's sensors, then :meth:`recv_and_apply` blocks for the
        paired actuator and applies the resulting forces. A multi-vehicle
        coordinator calls the two legs across vehicles instead, so all sends
        precede all blocking polls.
        """
        self.send_sensors(dt, sim_time_us)
        self.recv_and_apply()

    def send_sensors(self, dt: float, sim_time_us: int) -> None:
        """Send leg: read state and publish this tick's sensors to PX4 (no block).

        Stashes the read state, the tilt force-cutoff decision, and the port's
        send result on ``self`` for the paired :meth:`recv_and_apply`.
        """
        if not self._io.is_initialized:
            self._io.initialize()

        state = self._io.read_state(int(sim_time_us))
        if not np.all(np.isfinite(state.position_enu)) or not np.all(
            np.isfinite(state.orientation_flu_enu_wxyz)
        ):
            # A non-finite state is unrecoverable corruption: fail loudly rather
            # than forwarding NaN/Inf into the MAVLink/EKF lockstep. Per the
            # CLAUDE.md hard rule, unexpected runtime state must not fail
            # silently.
            #
            # Deferred recoverable design: if non-finite states ever turn out to
            # be a recurring, recoverable runtime condition, cache the
            # last-finite VehicleState and, on a non-finite tick, send HIL_SENSOR
            # from the cached state (with sim_time_us bumped to the current tick)
            # while applying zero forces, instead of raising. The logged values
            # below are the breadcrumbs needed to implement that.
            _LOGGER.error(
                "PX4 bridge non-finite vehicle state at sim_time_us=%d: "
                "position_enu=%s orientation_flu_enu_wxyz=%s",
                int(sim_time_us),
                np.asarray(state.position_enu, dtype=np.float64).tolist(),
                np.asarray(state.orientation_flu_enu_wxyz, dtype=np.float64).tolist(),
            )
            raise RuntimeError(
                f"PX4 bridge read a non-finite vehicle state at "
                f"sim_time_us={int(sim_time_us)}; refusing to forward NaN/Inf "
                f"into the PX4 lockstep exchange"
            )

        # The tilt cutoff gates only the *applied* rotor forces. The
        # sensor/actuator exchange must keep running every physics tick so that
        # HIL_SENSOR.time_usec stays strictly increasing and PX4 lockstep is
        # preserved even while the safety condition holds.
        force_cutoff = False
        tilt = frames.tilt_angle_deg(state.orientation_flu_enu_wxyz)
        if tilt > self._max_tilt_angle_deg:
            force_cutoff = True
            self._tilt_cutoff_count += 1
            if self._tilt_cutoff_count <= 10 or self._tilt_cutoff_count % 250 == 0:
                _LOGGER.warning(
                    "PX4 bridge tilt cutoff: count=%d tilt=%.1f deg "
                    "limit=%.1f deg; zeroing rotor forces",
                    self._tilt_cutoff_count,
                    tilt,
                    self._max_tilt_angle_deg,
                )

        # Send leg, using the real state, before gating applied forces: publish
        # this tick's sensors out to PX4. A None result means an actuator must
        # be polled in recv_and_apply; a concrete result is the no-block case.
        self._pending_send_result = self._port.send(state)
        self._pending_state = state
        self._pending_force_cutoff = force_cutoff
        self._pending_sim_time_us = int(sim_time_us)

    def recv_and_apply(self) -> None:
        """Poll leg: block for the paired actuator and apply the resulting forces.

        Consumes the state stashed by :meth:`send_sensors`. Tilt-gated ticks
        apply zeros; otherwise the actuator commands run through the thrust model
        and the resulting dynamics are applied to the articulation.
        """
        state = self._pending_state
        sim_time_us = self._pending_sim_time_us
        # Block for the paired actuator (or take the no-block result from send).
        send_result = self._pending_send_result
        commands = self._port.poll() if send_result is None else send_result
        if self._pending_force_cutoff:
            zero = np.zeros(self._thrust_model.num_rotors, dtype=np.float64)
            self._io.apply_dynamics(
                RotorDynamics(
                    forces_n=zero,
                    body_torque_n_m=np.zeros(3, dtype=np.float64),
                    body_drag_n=np.zeros(3, dtype=np.float64),
                )
            )
            return

        body_velocity_flu = frames.body_velocity_flu_from_state(state)
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
                np.asarray(commands.controls, dtype=np.float64).tolist(),
                np.asarray(dynamics.forces_n, dtype=np.float64).tolist(),
                np.asarray(dynamics.body_torque_n_m, dtype=np.float64).tolist(),
                np.asarray(dynamics.body_drag_n, dtype=np.float64).tolist(),
                np.asarray(state.position_enu, dtype=np.float64).tolist(),
                np.asarray(state.orientation_flu_enu_wxyz, dtype=np.float64).tolist(),
                np.asarray(state.linear_velocity_enu, dtype=np.float64).tolist(),
                np.asarray(body_velocity_flu, dtype=np.float64).tolist(),
            )
        self._io.apply_dynamics(dynamics)

    # ---- Status ------------------------------------------------------------

    @property
    def status(self) -> BridgeStatus:
        return self._port.status

    @property
    def is_connected(self) -> bool:
        return self._port.is_connected

    @property
    def is_armed(self) -> bool:
        return self._port.is_armed

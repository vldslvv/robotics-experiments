"""Settings and seam objects that make up the bridge's public contract.

These are pure-Python dataclasses with no Isaac Sim, MAVLink, or Pegasus
dependency, so they are safe to import from unit tests, staging scripts, and
any consuming logic. Together with :class:`VehicleState` (input) and
:class:`RotorDynamics` (output) they form the bridge seam:

- :class:`LockstepSettings` configures the lockstep exchange.
- :class:`BridgeStatus` is the read-only runtime snapshot of that exchange.
- :class:`ActuatorCommands` is what the flight stack returns each tick (PX4 ->
  bridge) and what the thrust model consumes to produce :class:`RotorDynamics`,
  so the per-tick data flow is fully typed:
  ``VehicleState -> ActuatorCommands -> RotorDynamics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["ActuatorCommands", "BridgeStatus", "LockstepSettings"]


@dataclass(frozen=True)
class ActuatorCommands:
    """Per-motor actuator command references returned by the flight stack.

    These are PX4's ``HIL_ACTUATOR_CONTROLS`` outputs (``controls[0:num_rotors]``)
    after the port's offset/scaling - one entry per rotor in PX4 motor-index
    order. They are commands, not realized forces: the thrust model converts
    them into a :class:`RotorDynamics` (forces, body torque, drag). Their
    physical interpretation is owned by the thrust model - the Pegasus quadratic
    model reads them as rotor angular-velocity references in rad/s, the linear
    model as normalized 0..1 force fractions.

    Attributes:
        controls: Length-``num_rotors`` ``np.ndarray`` of command references.
    """

    controls: np.ndarray


@dataclass(frozen=True)
class LockstepSettings:
    """Tunables for the PX4 lockstep exchange.

    These values default to neutral library fallbacks; the Hydra ``px4.lockstep``
    config group is the runtime source of truth and supplies the project values.

    Attributes:
        lockstep_timeout_s: Hard deadline on receiving an actuator message each
            steady-state tick. On timeout the port disconnects and returns zero
            commands so a stalled flight stack cannot freeze the simulator.
        first_heartbeat_timeout_s: Hard deadline on PX4's first heartbeat after
            the port starts.
        bootstrap_prime_ticks: Number of priming ``HIL_SENSOR`` sends allowed
            before the bridge blocks for PX4's first ``HIL_ACTUATOR_CONTROLS``.
            Must exceed PX4's boot sensor need (~15) or the first-actuator wait
            times out.
        first_actuator_timeout_s: Generous blocking deadline for PX4's first
            actuator (PX4 boot takes seconds of wallclock). Steady-state ticks
            use ``lockstep_timeout_s`` instead.
        max_sustained_actuator_lag_s: Safety tripwire on steady-state actuator
            lag (current sim time minus the newest applied actuator timestamp).
            A regression net, not a happy-path control.
        heartbeat_rate_hz: Wall-clock rate for bridge-originated heartbeats.
        physics_rate_hz: Isaac physics cadence, also the PX4 sensor cadence
            (single source of truth = ``sim.physics_hz``). Derived, not a
            standalone YAML knob.
    """

    lockstep_timeout_s: float = 0.1
    first_heartbeat_timeout_s: float = 5.0
    bootstrap_prime_ticks: int = 20
    first_actuator_timeout_s: float = 5.0
    max_sustained_actuator_lag_s: float = 0.5
    heartbeat_rate_hz: float = 1.0
    physics_rate_hz: float = 250.0


@dataclass(frozen=True)
class BridgeStatus:
    """Read-only snapshot of the lockstep exchange at a point in time.

    Surfaced by `read_bridge_status` (and the bridge status reader) so consuming
    logic - state machines, RL rollouts, diagnostics - can observe the bridge
    without reaching into private MAVLink state.

    Attributes:
        is_connected: Whether the MAVLink connection is currently live.
        is_armed: Whether PX4 currently commands nonzero rotor references.
        first_heartbeat_received: Whether PX4's first heartbeat has arrived.
        sensor_send_count: Total ``HIL_SENSOR`` packets sent.
        actuator_receive_count: Total ``HIL_ACTUATOR_CONTROLS`` packets received.
        last_sensor_send_us: Sim time of the last ``HIL_SENSOR`` sent, or ``None``.
        last_actuator_receive_us: Timestamp of the last actuator received, or
            ``None``.
        actuator_lag_us: ``last_sensor_send_us - last_actuator_receive_us`` when
            both exist, else ``None``. ~1 tick in healthy strict lockstep.
        last_disconnect_reason: Reason string of the last disconnect, or ``None``.
    """

    is_connected: bool
    is_armed: bool
    first_heartbeat_received: bool
    sensor_send_count: int
    actuator_receive_count: int
    last_sensor_send_us: Optional[int]
    last_actuator_receive_us: Optional[int]
    actuator_lag_us: Optional[int]
    last_disconnect_reason: Optional[str]

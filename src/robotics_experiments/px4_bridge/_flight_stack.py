from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from robotics_experiments.px4_bridge.contract import ActuatorCommands, BridgeStatus
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__: list[str] = []


class _AbstractFlightStackBridge(ABC):
    """Stage-able lockstep seam between sim orchestration and a flight stack.

    A tick decomposes into a non-blocking send leg and a blocking poll leg:
    :meth:`send` publishes sensors derived from the state, then :meth:`poll`
    blocks for the paired actuator response. Within a single vehicle, ``send``
    strictly precedes its ``poll``. Splitting the tick lets a multi-vehicle
    coordinator send sensors to every PX4 first and only then block for each
    actuator, so the flight stacks compute concurrently and one stalled PX4
    cannot freeze the others.

    :meth:`exchange` is the single-vehicle convenience that composes the two
    legs back into one send-then-poll call; a one-vehicle scene is byte-for-byte
    the old indivisible exchange.
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def send(self, state: VehicleState) -> Optional[ActuatorCommands]:
        """Send leg: publish sensors derived from ``state``, no blocking receive.

        Returns ``None`` once sensors were sent and a paired actuator must be
        polled (the caller then calls :meth:`poll`). Returns a concrete
        :class:`ActuatorCommands` for the no-block cases (disconnected, a
        non-increasing timestamp, or a pre-handshake/cadence skip) so ``poll``
        is skipped this tick.
        """

    @abstractmethod
    def poll(self) -> ActuatorCommands:
        """Poll leg: block (with a hard timeout) for the paired actuator.

        Only valid after a :meth:`send` returned ``None`` this tick. Returns the
        per-motor :class:`ActuatorCommands`; zeros on a timeout.
        """

    def exchange(self, state: VehicleState) -> ActuatorCommands:
        """Send sensors derived from ``state``, then block for the actuator.

        One causal tick composed from :meth:`send` and :meth:`poll`: the sensor
        send strictly precedes the blocking actuator receive. Returns the
        per-motor :class:`ActuatorCommands`; zeros when disconnected or on
        timeout. This is the single-vehicle convenience; multi-vehicle callers
        drive ``send`` / ``poll`` across vehicles directly.
        """
        result = self.send(state)
        return self.poll() if result is None else result

    @property
    @abstractmethod
    def status(self) -> BridgeStatus: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @property
    @abstractmethod
    def is_armed(self) -> bool: ...

    @property
    @abstractmethod
    def num_rotors(self) -> int: ...

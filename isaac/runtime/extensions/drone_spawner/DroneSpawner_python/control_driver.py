"""Live control adapter: tick one per-drone RoutineDriver per drone from Kit.

This is the Isaac-side, forgiving adapter over the pure per-drone
:class:`~robotics_experiments.commanding.RoutineDriver`. The phase machine and all
routine logic live in ``commanding``; this module only wires the Isaac seams (a
``BridgeHost`` for connectivity, an ``IsaacArticulationIO`` for state, the bridge
clock for sim time, a ``MavlinkCommandPort`` over each drone's derived
``gcs_port``) and ticks every drone's driver from one Kit **app-update**
subscription -- a control tick separate from the bridge's physics-step
subscription, never the bridge's send/recv legs.

It is always *forgiving*: a routine ``FAIL``, a PX4 failsafe, or a phase timeout
is logged via ``carb`` and the offending drone's driver idles (manual control
remains). The strict, asserting counterpart is the standalone scenario's adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import carb

from robotics_experiments.commanding import (
    CommandAck,
    DriverParams,
    MavlinkCommandPort,
    RoutineDriver,
    RoutineObservation,
)

from .bridge_host import BridgeHost

__all__ = ["FleetControlDriver"]

_LOG_PREFIX = "[DroneSpawner.control]"


@dataclass
class _Member:
    """One drone's live control state (driver + the seams it owns)."""

    driver: RoutineDriver
    port: MavlinkCommandPort
    observer: Any


class _LiveDriverEnv:
    """Forgiving :class:`DriverEnvironment` over one live drone.

    Reads connectivity from the ``BridgeHost``, sim time from the bridge clock,
    and the kinematic state from a state-only ``IsaacArticulationIO``; a failure
    is logged (the driver then idles) rather than raised, since runtime is
    recoverable.
    """

    def __init__(self, host: BridgeHost, observer: Any, port: MavlinkCommandPort, label: str):
        self._host = host
        self._observer = observer
        self._port = port
        self._label = label

    def is_bridge_connected(self) -> bool:
        return bool(self._host.is_connected)

    def sim_time_us(self) -> int:
        return int(self._host.sim_time_us)

    def read_observation(
        self, *, sim_time_us: int, last_ack: Optional[CommandAck]
    ) -> RoutineObservation:
        return RoutineObservation(
            vehicle_state=self._observer.read_state(sim_time_us=sim_time_us),
            bridge_status=self._host.status,
            last_ack=last_ack,
            failsafe=self._port.failsafe,
            sim_time_us=sim_time_us,
        )

    def log(self, message: str) -> None:
        carb.log_info(f"{_LOG_PREFIX} {message}")

    def on_failure(self, kind: Any, message: str) -> None:
        carb.log_warn(
            f"{_LOG_PREFIX} {message} ({kind}); stopping routine (manual control remains available)"
        )

    def on_step_advanced(self, step_label: str) -> None:
        carb.log_info(f"{_LOG_PREFIX} [{self._label}] completed step {step_label!r}")


class FleetControlDriver:
    """Ticks one per-drone :class:`RoutineDriver` per drone from the app-update stream.

    Built in the extension's ``_setup_drone`` after the bridge group starts;
    :meth:`stop` drops the subscription and tears down each drone's port/observer.
    Only drones whose resolved ``routine.steps`` is non-empty get a driver, so a
    fleet can mix flying and manual drones.
    """

    def __init__(self, members: list[_Member]) -> None:
        self._members = members
        self._update_sub: Any = None

    @classmethod
    def from_vehicles(cls, entries: list[tuple[Any, BridgeHost, str]]) -> "FleetControlDriver":
        """Build a driver for each ``(vehicle, host, articulation_path)`` with a routine.

        Each px4-enabled vehicle is passed in; only those whose
        ``vehicle.routine.steps`` is non-empty get a :class:`RoutineDriver`.
        """
        from robotics_experiments.px4_bridge import IsaacArticulationIO

        members: list[_Member] = []
        for vehicle, host, articulation_path in entries:
            if not vehicle.routine.steps:
                continue
            label = f"drone{vehicle.id}"
            port = MavlinkCommandPort(
                gcs_endpoint=f"udp:127.0.0.1:{vehicle.gcs_port}",
                heartbeat_period_s=vehicle.routine.gcs_heartbeat_period_s,
                label=label,
            )
            # State-only observer (read_state samples body pose/velocity; rotor
            # paths are unused for it, so none are resolved here).
            observer = IsaacArticulationIO(articulation_path=articulation_path, rotor_link_paths=[])
            env = _LiveDriverEnv(host, observer, port, label)
            driver = RoutineDriver(
                DriverParams.from_routine_config(vehicle.routine), port, env, label=label
            )
            members.append(_Member(driver=driver, port=port, observer=observer))
        return cls(members)

    def start(self) -> None:
        """Subscribe to the app-update stream and tick every drone's driver.

        A no-op (with a warning) when no drone has a routine to fly.
        """
        if not self._members:
            carb.log_warn(
                f"{_LOG_PREFIX} routine.steps is non-empty but no px4-enabled vehicle has a "
                "routine; nothing to fly"
            )
            return
        import omni.kit.app

        carb.log_info(f"{_LOG_PREFIX} starting routine for {len(self._members)} vehicle(s)")
        app = omni.kit.app.get_app()
        self._update_sub = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="drone_spawner.control_driver"
        )

    def stop(self) -> None:
        """Drop the subscription, disconnect every GCS link, and shut down observers."""
        self._update_sub = None
        for member in self._members:
            member.port.disconnect()
            if member.observer is not None:
                try:
                    member.observer.shutdown()
                except Exception as exc:  # noqa: BLE001 - best-effort teardown
                    carb.log_warn(f"{_LOG_PREFIX} observer shutdown failed: {exc}")
        self._members = []

    def _on_update(self, event: Any) -> None:
        """One control tick: advance every drone's phase machine. Never raises into Kit."""
        for member in self._members:
            try:
                member.driver.tick()
            except Exception as exc:  # noqa: BLE001 - a control bug must not spam every frame
                carb.log_error(f"{_LOG_PREFIX} control tick failed: {exc}")

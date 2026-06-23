"""The host-agnostic fleet runtime orchestrator.

:class:`FleetOrchestrator` is the single production pipeline for the drone fleet
runtime, expressed as discrete, ordered phases:

    open_scene -> apply_settings -> spawn_fleet -> start_bridges -> build_drivers
    -> (the shell ticks the drivers) -> teardown

It holds no ``SimulationApp``/Kit state, so both thin shells drive it: the live
``experiment_runtime`` Kit extension (scene+settings at startup, spawn/bridge/
drivers on timeline PLAY, ticking from an app-update subscription) and the
standalone scenario runner (everything in sequence, ticking from its blocking
``simulation_app.update()`` loop). The two genuinely differing concerns are the
shell's tick source and the failure reaction; the latter is the injected
``make_env`` callable that builds a :class:`DriverEnvironment` per flying drone,
so strict (integration) and forgiving (live/regular) runs share this pipeline.

Production logic only: any extra integration-test checks (ground-settle wait,
bridge-heartbeat wait, the pre-arm start-tilt check, the bridge-log scan) live in
the standalone shell and read the per-drone :class:`DroneRuntimeHandle`s.

Isaac-coupled domain leaf: imports ``carb``/``omni`` (here and via
``bridge_host``) at module scope, so it stays off the core import path behind the
lazy ``robotics_experiments.runtime`` package ``__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import carb

from robotics_experiments.commanding import (
    CommandAck,
    DriverEnvironment,
    DriverParams,
    MavlinkCommandPort,
    RoutineDriver,
    RoutineObservation,
)

from ._scene import resolve_scene_path
from .bridge_host import BridgeHost, BridgeHostGroup

__all__ = ["DroneRuntimeHandle", "FleetDriverEnv", "FleetOrchestrator"]

_LOG_PREFIX = "[FleetRuntime]"


def _repo_root() -> Path:
    # orchestrator.py -> runtime -> robotics_experiments -> src -> repo root
    return Path(__file__).resolve().parents[3]


@dataclass
class DroneRuntimeHandle:
    """One vehicle's live runtime state -- the seams a shell reads.

    ``host``/``observer`` exist only for ``px4.enabled`` vehicles (a disabled PX4
    must not construct a ``BridgeHost``); ``port``/``driver`` are set by
    :meth:`FleetOrchestrator.build_drivers` only for flying drones (non-empty
    resolved ``routine.steps``).
    """

    vehicle: Any
    label: str
    articulation_path: str
    host: Optional[BridgeHost] = None
    observer: Any = None
    port: Optional[MavlinkCommandPort] = None
    driver: Optional[RoutineDriver] = None


class FleetDriverEnv:
    """Generic :class:`DriverEnvironment` over one live drone handle.

    The production reads (connectivity, sim time, observation assembly) are
    identical for every host; the three varying reactions are injected callables
    so the live (forgiving) and standalone (strict/forgiving) shells share this
    one class instead of each defining its own env. Built only for a flying drone,
    so its handle always carries a ``host``/``observer``/``port``.
    """

    def __init__(
        self,
        handle: DroneRuntimeHandle,
        *,
        log: Callable[[str], None],
        on_failure: Callable[[Any, str], None],
        on_step_advanced: Callable[[str], None],
    ) -> None:
        if handle.host is None or handle.observer is None or handle.port is None:
            raise RuntimeError(
                f"{_LOG_PREFIX} FleetDriverEnv requires a flying drone "
                f"(host/observer/port); got {handle.label!r}"
            )
        self._host = handle.host
        self._observer = handle.observer
        self._port = handle.port
        self._log_fn = log
        self._on_failure_fn = on_failure
        self._on_step_advanced_fn = on_step_advanced

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
        self._log_fn(message)

    def on_failure(self, kind: Any, message: str) -> None:
        self._on_failure_fn(kind, message)

    def on_step_advanced(self, step_label: str) -> None:
        self._on_step_advanced_fn(step_label)


class FleetOrchestrator:
    """Composes the fleet runtime pipeline; driven by a thin per-host shell."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.stage: Any = None
        self._group: BridgeHostGroup = BridgeHostGroup([])
        self._handles: list[DroneRuntimeHandle] = []

    @property
    def drones(self) -> list[DroneRuntimeHandle]:
        """Every spawned vehicle's handle."""
        return self._handles

    @property
    def bridged(self) -> list[DroneRuntimeHandle]:
        """Handles with a PX4 bridge host (``px4.enabled`` vehicles)."""
        return [h for h in self._handles if h.host is not None]

    @property
    def flying(self) -> list[DroneRuntimeHandle]:
        """Handles with a built routine driver (non-empty ``routine.steps``)."""
        return [h for h in self._handles if h.driver is not None]

    @property
    def all_done(self) -> bool:
        """Whether every flying drone's routine is done (vacuously true if none)."""
        flying = self.flying
        return all(h.driver.is_done for h in flying if h.driver is not None)

    # ---- scene + settings --------------------------------------------------

    def open_scene(self, scene_override: Optional[str] = None) -> None:
        """Open the experiment stage (``env.scene`` unless overridden).

        Fires the USD ``open_stage``; the stage is ready on a later frame, so the
        shell pumps once (standalone) or handles the ``OPENED`` event (live) and
        then calls :meth:`adopt_current_stage`.
        """
        import omni.usd

        scene = scene_override if scene_override else self.env.scene
        path = resolve_scene_path(scene, repo_root=_repo_root())
        if not path.exists():
            raise FileNotFoundError(f"Scene does not exist: {path}")
        carb.log_info(f"{_LOG_PREFIX} opening scene {path}")
        if not omni.usd.get_context().open_stage(str(path)):
            raise RuntimeError(f"Failed to open stage: {path}")

    def adopt_current_stage(self) -> None:
        """Bind ``self.stage`` to the now-open USD stage."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"{_LOG_PREFIX} stage adopted before it finished opening")
        self.stage = stage

    def apply_settings(self) -> None:
        """Apply the configured physics/render settings to the adopted stage."""
        from robotics_experiments.simulation import apply_settings

        apply_settings(self.stage, self.env.sim)

    # ---- fleet -------------------------------------------------------------

    def spawn_fleet(self) -> None:
        """Spawn every vehicle and build a bridge host + observer per enabled PX4."""
        from robotics_experiments.drone import spawn_drone
        from robotics_experiments.px4_bridge import IsaacArticulationIO

        if self.stage is None:
            raise RuntimeError(f"{_LOG_PREFIX} spawn_fleet called before the stage was adopted")
        summary = ", ".join(
            f"id={v.id} prim={v.prim_name} px4_enabled={v.px4.enabled} "
            f"endpoint={v.mavlink_port} spawn_xy={tuple(v.spawn_xy)}"
            for v in self.env.vehicles
        )
        carb.log_info(f"{_LOG_PREFIX} spawning {len(self.env.vehicles)} vehicle(s): {summary}")
        hosts: list[BridgeHost] = []
        self._handles = []
        for vehicle in self.env.vehicles:
            path = spawn_drone(self.stage, vehicle.drone, position=tuple(vehicle.spawn_xy), yaw=0.0)
            # The observer reads the same body the bridge actuates (honoring any
            # px4.articulation_path override) so it never samples a stale frame.
            articulation_path = vehicle.px4.articulation_path or path
            label = f"drone{vehicle.id}"
            handle = DroneRuntimeHandle(
                vehicle=vehicle, label=label, articulation_path=articulation_path
            )
            if vehicle.px4.enabled:
                handle.host = BridgeHost(
                    articulation_path=path,
                    vehicle=vehicle,
                    physics_hz=self.env.sim.physics_hz,
                )
                hosts.append(handle.host)
                # State-only observer (read_state samples body pose/velocity; no
                # rotor paths needed for it).
                handle.observer = IsaacArticulationIO(
                    articulation_path=articulation_path, rotor_link_paths=[]
                )
            self._handles.append(handle)
        # One coordinator owns the single physics-step subscription and runs all
        # sends before all polls. An empty hosts list yields a no-op group.
        self._group = BridgeHostGroup(hosts)

    def start_bridges(self) -> None:
        """Start the bridge group (its hosts and the single physics-step sub)."""
        self._group.start()

    def build_drivers(self, make_env: Callable[[DroneRuntimeHandle], DriverEnvironment]) -> None:
        """Build one :class:`RoutineDriver` per flying drone.

        A drone flies when its resolved ``routine.steps`` is non-empty; it then
        gets a GCS ``MavlinkCommandPort`` and a driver wired through ``make_env``
        (the per-host reaction). A routine on a PX4-disabled vehicle has no bridge
        to fly it and is left manual with a warning.
        """
        for handle in self._handles:
            if not handle.vehicle.routine.steps:
                continue
            if handle.host is None:
                carb.log_warn(
                    f"{_LOG_PREFIX} [{handle.label}] has a routine but px4 is disabled; "
                    "no bridge to fly it -- left manual"
                )
                continue
            handle.port = MavlinkCommandPort(
                gcs_endpoint=f"udp:127.0.0.1:{handle.vehicle.gcs_port}",
                heartbeat_period_s=handle.vehicle.routine.gcs_heartbeat_period_s,
                label=handle.label,
            )
            handle.driver = RoutineDriver(
                DriverParams.from_routine_config(handle.vehicle.routine),
                handle.port,
                make_env(handle),
                label=handle.label,
            )

    def teardown(self) -> None:
        """Stop the bridge group, drop ports/observers, and despawn every vehicle."""
        from robotics_experiments.drone import despawn_drone

        self._group.stop()
        for handle in self._handles:
            if handle.port is not None:
                handle.port.disconnect()
            if handle.observer is not None:
                handle.observer.shutdown()
            if self.stage is not None:
                despawn_drone(self.stage, handle.vehicle.drone)
        self._group = BridgeHostGroup([])
        self._handles = []

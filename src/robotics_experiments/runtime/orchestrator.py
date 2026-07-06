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

Per-drone state is staged and immutable: ``spawn_fleet`` produces one frozen
:class:`SpawnedDrone` per vehicle, and ``build_drivers`` composes each flying
drone's private driver state over its spawned state -- a phase never mutates the
previous phase's output.

Production logic only: any extra integration-test checks (ground-settle wait,
bridge-heartbeat wait, the pre-arm start-tilt check, the bridge-log scan) live in
the standalone shell and read the per-drone :class:`SpawnedDrone` / driver states.

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
    FailureKind,
    MavlinkCommandPort,
    RoutineDriver,
    RoutineObservation,
)
from robotics_experiments.config import EnvConfig, ResolvedVehicle
from robotics_experiments.runtime._scene import resolve_scene_path
from robotics_experiments.runtime.bridge_host import BridgeHost, BridgeHostGroup, BridgeHostParams

__all__ = ["FleetDriverEnv", "FleetOrchestrator", "SpawnedDrone"]

_LOG_PREFIX = "[FleetRuntime]"


def _repo_root() -> Path:
    # orchestrator.py -> runtime -> robotics_experiments -> src -> repo root
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SpawnedDrone:
    """One vehicle's spawn-phase state, produced by :meth:`FleetOrchestrator.spawn_fleet`.

    ``host``/``observer`` exist only for ``px4.enabled`` vehicles (a disabled PX4
    must not construct a ``BridgeHost``). ``articulation_path`` is the
    override-resolved body path shared by the host and the observer, so both read
    the same body the bridge actuates.
    """

    vehicle: ResolvedVehicle
    label: str
    articulation_path: str
    host: Optional[BridgeHost]
    observer: Any


@dataclass(frozen=True)
class _FlyingDrone:
    """One flying drone's driver-phase state, composing its spawn-phase state.

    Produced by :meth:`FleetOrchestrator.build_drivers` only for flying drones
    (non-empty resolved ``routine.steps`` on a ``px4.enabled`` vehicle), so
    ``spawned.host``/``spawned.observer`` are always present here.
    """

    spawned: SpawnedDrone
    port: MavlinkCommandPort
    driver: RoutineDriver


class FleetDriverEnv:
    """Generic :class:`DriverEnvironment` over one flying drone's states.

    The production reads (connectivity, sim time, observation assembly) are
    identical for every host; the three varying reactions are injected callables
    so the live (forgiving) and standalone (strict/forgiving) shells share this
    one class instead of each defining its own env. Built only for a flying
    drone, so the spawned state always carries a ``host``/``observer``.
    """

    def __init__(
        self,
        spawned: SpawnedDrone,
        port: MavlinkCommandPort,
        *,
        log: Callable[[str], None],
        on_failure: Callable[[FailureKind, str], None],
        on_step_advanced: Callable[[str], None],
    ) -> None:
        if spawned.host is None or spawned.observer is None:
            raise RuntimeError(
                f"{_LOG_PREFIX} FleetDriverEnv requires a flying drone "
                f"(host/observer); got {spawned.label!r}"
            )
        self._host = spawned.host
        self._observer = spawned.observer
        self._port = port
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

    def on_failure(self, kind: FailureKind, message: str) -> None:
        self._on_failure_fn(kind, message)

    def on_step_advanced(self, step_label: str) -> None:
        self._on_step_advanced_fn(step_label)


class FleetOrchestrator:
    """Composes the fleet runtime pipeline; driven by a thin per-host shell."""

    def __init__(self, env: EnvConfig) -> None:
        self.env = env
        self.stage: Any = None
        self._group: BridgeHostGroup = BridgeHostGroup([])
        self._spawned: list[SpawnedDrone] = []
        self._flying: list[_FlyingDrone] = []

    @property
    def drones(self) -> list[SpawnedDrone]:
        """Every spawned vehicle's state."""
        return self._spawned

    @property
    def bridged(self) -> list[SpawnedDrone]:
        """Spawned states with a PX4 bridge host (``px4.enabled`` vehicles)."""
        return [s for s in self._spawned if s.host is not None]

    @property
    def flying(self) -> list[_FlyingDrone]:
        """Flying-drone states (non-empty ``routine.steps``), with their drivers."""
        return self._flying

    @property
    def all_done(self) -> bool:
        """Whether every flying drone's routine is done (vacuously true if none)."""
        return all(f.driver.is_done for f in self._flying)

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
        """Spawn every vehicle into a frozen :class:`SpawnedDrone` per vehicle.

        A bridge host (from a frozen :class:`BridgeHostParams`) and a state-only
        observer are built only for ``px4.enabled`` vehicles; the one
        :class:`BridgeHostGroup` coordinates all hosts.
        """
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
        self._spawned = []
        for vehicle in self.env.vehicles:
            path = spawn_drone(self.stage, vehicle.drone, position=tuple(vehicle.spawn_xy), yaw=0.0)
            host: Optional[BridgeHost] = None
            observer: Any = None
            # The observer reads the same body the bridge actuates (honoring any
            # px4.articulation_path override) so it never samples a stale frame;
            # the params resolve that path once for both.
            articulation_path = vehicle.px4.articulation_path or path
            if vehicle.px4.enabled:
                host = BridgeHost(
                    BridgeHostParams(
                        px4=vehicle.px4,
                        vehicle_id=vehicle.id,
                        mavlink_port=vehicle.mavlink_port,
                        debug_log_path=vehicle.debug_log_path,
                        articulation_path=articulation_path,
                        physics_hz=self.env.sim.physics_hz,
                    )
                )
                # State-only observer (read_state samples body pose/velocity; no
                # rotor paths needed for it).
                observer = IsaacArticulationIO(
                    articulation_path=articulation_path, rotor_link_paths=[]
                )
            self._spawned.append(
                SpawnedDrone(
                    vehicle=vehicle,
                    label=f"drone{vehicle.id}",
                    articulation_path=articulation_path,
                    host=host,
                    observer=observer,
                )
            )
        # One coordinator owns the single physics-step subscription and runs all
        # sends before all polls. An empty hosts list yields a no-op group.
        self._group = BridgeHostGroup([s.host for s in self._spawned if s.host is not None])

    def start_bridges(self) -> None:
        """Start the bridge group (its hosts and the single physics-step sub)."""
        self._group.start()

    def _policy_fn_factory(self, descriptor: Any) -> Any:
        """Load a ``{kind: policy}`` step's artifact into its pure step function.

        Injected into every driver's params: composes
        ``robotics_experiments.rl.build_policy_step_fn`` (validated artifact load +
        observation packing + OFFBOARD mapping) over the global ``rl.action``
        config. A repo-relative ``artifact`` path (the committed configs use
        ``.build/training/<policy-name>/latest``) resolves against the repo root.
        """
        from robotics_experiments.rl import build_policy_step_fn

        artifact = Path(str(descriptor["artifact"]))
        if not artifact.is_absolute():
            artifact = _repo_root() / artifact
        return build_policy_step_fn(artifact, self.env.rl.action)

    def build_drivers(
        self, make_env: Callable[[SpawnedDrone, MavlinkCommandPort], DriverEnvironment]
    ) -> None:
        """Compose a frozen driver state per flying drone.

        A drone flies when its resolved ``routine.steps`` is non-empty; it then
        gets a GCS ``MavlinkCommandPort`` and a driver wired through
        ``make_env(spawned, port)`` (the per-host reaction). A routine on a
        PX4-disabled vehicle has no bridge to fly it and is left manual with a
        warning.
        """
        self._flying = []
        for spawned in self._spawned:
            if not spawned.vehicle.routine.steps:
                continue
            if spawned.host is None:
                carb.log_warn(
                    f"{_LOG_PREFIX} [{spawned.label}] has a routine but px4 is disabled; "
                    "no bridge to fly it -- left manual"
                )
                continue
            port = MavlinkCommandPort(
                gcs_endpoint=f"udp:127.0.0.1:{spawned.vehicle.gcs_port}",
                heartbeat_period_s=spawned.vehicle.routine.gcs_heartbeat_period_s,
                label=spawned.label,
            )
            driver = RoutineDriver(
                DriverParams.from_routine_config(
                    spawned.vehicle.routine,
                    driver=self.env.driver,
                    home_lat_deg=spawned.vehicle.px4.home_lat_deg,
                    home_lon_deg=spawned.vehicle.px4.home_lon_deg,
                    home_alt_m=spawned.vehicle.px4.home_alt_m,
                    policy_fn_factory=self._policy_fn_factory,
                ),
                port,
                make_env(spawned, port),
                label=spawned.label,
            )
            self._flying.append(_FlyingDrone(spawned=spawned, port=port, driver=driver))

    def teardown(self) -> None:
        """Stop the bridge group, drop ports/observers, and despawn every vehicle."""
        from robotics_experiments.drone import despawn_drone

        self._group.stop()
        for flying in self._flying:
            flying.port.disconnect()
        for spawned in self._spawned:
            if spawned.observer is not None:
                spawned.observer.shutdown()
            if self.stage is not None:
                despawn_drone(self.stage, spawned.vehicle.drone)
        self._group = BridgeHostGroup([])
        self._spawned = []
        self._flying = []

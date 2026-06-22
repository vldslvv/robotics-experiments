import carb
import omni.ext
import omni.timeline
import omni.usd
from omni.timeline import TimelineEventType
from omni.usd import StageEventType

from robotics_experiments.config import get_config as _get_experiment_config
from robotics_experiments.drone import despawn_drone, spawn_drone

from .bridge_host import BridgeHost, BridgeHostGroup
from .control_driver import FleetControlDriver

__all__ = ["Extension"]


class Extension(omni.ext.IExt):
    """Isaac Sim extension that owns the runtime drone lifecycle.

    Subscribes to stage-open and timeline events. The drone is spawned only
    when playback starts and despawned when playback stops, so opening a stage
    or enabling the extension does not mutate the scene with preview state.

    Iterates the experiment config's ``vehicles`` list (the single spawn path;
    one vehicle = a one-element list): each vehicle is spawned at its own pose
    and, when its ``px4.enabled`` is true, gets its own independent
    :class:`BridgeHost`. The enabled hosts are driven by a single
    :class:`BridgeHostGroup`, which owns the one physics-step subscription and
    runs every host's send leg before every host's poll leg, so the PX4
    instances compute concurrently and one stalled PX4 cannot freeze the others.
    Hosts are opt-in per vehicle so the lower bridge layers can be staged
    independently before the full pipeline is wired.
    """

    def on_startup(self, ext_id: str):
        """Register stage and timeline subscriptions.

        Args:
            ext_id: Extension identifier assigned by the Kit runtime.
        """
        usd_context = omni.usd.get_context()
        self.stage = usd_context.get_stage()
        stage_events = usd_context.get_stage_event_stream()
        self._stage_event_sub = stage_events.create_subscription_to_pop(
            self._on_stage_event,
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline_events = timeline.get_timeline_event_stream()
        self._timeline_event_sub = timeline_events.create_subscription_to_pop(
            self._on_timeline_event,
        )
        # The group owns the single physics-step subscription. Empty and
        # unsubscribed until PLAY builds the per-vehicle hosts; teardown stops it
        # and resets it to an empty group, so it is never None.
        self._bridge_group = BridgeHostGroup([])
        self._spawned_drones: list = []
        # The high-level routine driver, built on PLAY only when the config
        # routine is non-empty (see _setup_drone). None means manual operation.
        self._control_driver: FleetControlDriver | None = None

    def on_shutdown(self):
        """Release subscriptions and clear any runtime drone state.

        Called by the Kit runtime when the extension is disabled. Clearing the
        subscription references causes the underlying event handles to be
        freed.
        """
        self._stage_event_sub = None
        self._timeline_event_sub = None
        self._teardown_drone()

    def _on_stage_event(self, event):
        """Clear runtime state when a new stage finishes opening.

        Args:
            event: Stage event emitted by ``omni.usd``; only
                ``StageEventType.OPENED`` is handled.
        """
        if event.type == int(StageEventType.OPENED):
            self.stage = omni.usd.get_context().get_stage()
            self._teardown_drone()

    def _on_timeline_event(self, event):
        """Spawn or despawn the drone based on timeline transitions.

        Spawns a fresh runtime drone on ``PLAY`` and despawns on ``STOP`` or
        ``PAUSE`` so the prim is removed from the persisted stage.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        if event.type == int(TimelineEventType.PLAY):
            self._setup_drone()
        elif event.type in (int(TimelineEventType.STOP), int(TimelineEventType.PAUSE)):
            self._teardown_drone()

    def _setup_drone(self):
        """Spawn every configured vehicle and start each enabled bridge."""
        if self.stage is None:
            return
        self._teardown_drone()

        env = _get_experiment_config()
        summary = ", ".join(
            f"id={v.id} prim={v.prim_name} "
            f"px4_enabled={v.px4.enabled} endpoint={v.mavlink_port} "
            f"spawn_xy={tuple(v.spawn_xy)}"
            for v in env.vehicles
        )
        carb.log_info(f"[DroneSpawner] spawning {len(env.vehicles)} vehicle(s): {summary}")
        hosts: list[BridgeHost] = []
        # (vehicle, host, articulation_path) per px4-enabled vehicle, handed to the
        # high-level control driver so it can open each drone's GCS link and read
        # its kinematic state without recomputing any derived field.
        control_entries: list[tuple] = []
        for vehicle in env.vehicles:
            path = spawn_drone(self.stage, vehicle.drone, position=tuple(vehicle.spawn_xy), yaw=0.0)
            self._spawned_drones.append(vehicle.drone)
            if not vehicle.px4.enabled:
                continue
            host = BridgeHost(
                articulation_path=path,
                vehicle=vehicle,
                physics_hz=env.sim.physics_hz,
            )
            hosts.append(host)
            # The observer reads the same body the bridge actuates (honoring any
            # px4.articulation_path override), so it never samples a stale frame.
            control_entries.append((vehicle, host, vehicle.px4.articulation_path or path))
        # One coordinator owns the single physics-step subscription and runs all
        # sends before all polls, decoupling the per-vehicle lockstep. An empty
        # hosts list (no px4-enabled vehicle) yields a group whose per-step
        # callback simply iterates nothing.
        self._bridge_group = BridgeHostGroup(hosts)
        self._bridge_group.start()

        # Routine config is the single source of truth: a non-empty steps list
        # builds the control driver (off a separate app-update tick, never the
        # bridge's physics legs); an empty list means manual operation.
        if env.routine.steps:
            self._control_driver = FleetControlDriver.from_vehicles(
                control_entries, routine_cfg=env.routine
            )
            self._control_driver.start()

    def _teardown_drone(self):
        """Stop the control driver and bridge group, then despawn every vehicle."""
        if self._control_driver is not None:
            self._control_driver.stop()
            self._control_driver = None
        self._bridge_group.stop()
        self._bridge_group = BridgeHostGroup([])
        for drone_cfg in self._spawned_drones:
            despawn_drone(self.stage, drone_cfg)
        self._spawned_drones = []

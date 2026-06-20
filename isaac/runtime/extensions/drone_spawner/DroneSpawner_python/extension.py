import omni.ext
import omni.timeline
import omni.usd
from omni.timeline import TimelineEventType
from omni.usd import StageEventType

from robotics_experiments.config import get_config as _get_experiment_config
from robotics_experiments.drone import despawn_drone, spawn_drone

from .bridge_host import BridgeHost

__all__ = ["Extension"]


class Extension(omni.ext.IExt):
    """Isaac Sim extension that owns the runtime drone lifecycle.

    Subscribes to stage-open and timeline events. The drone is spawned only
    when playback starts and despawned when playback stops, so opening a stage
    or enabling the extension does not mutate the scene with preview state.

    Iterates the experiment config's ``vehicles`` list (the single spawn path;
    one vehicle = a one-element list): each vehicle is spawned at its own pose
    and, when its ``px4.enabled`` is true, gets its own independent
    :class:`BridgeHost`. The hosts tick serially in registration order on each
    PhysX step. Hosts are opt-in per vehicle so the lower bridge layers can be
    staged independently before the full pipeline is wired.
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
        self._bridge_hosts: list[BridgeHost] = []
        self._spawned_drones: list = []

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
        for vehicle in env.vehicles:
            path = spawn_drone(self.stage, vehicle.drone, position=tuple(vehicle.spawn_xy), yaw=0.0)
            self._spawned_drones.append(vehicle.drone)
            if not vehicle.px4.enabled:
                continue
            host = BridgeHost(
                articulation_path=path,
                config=vehicle.px4,
                physics_hz=env.sim.physics_hz,
            )
            host.start()
            self._bridge_hosts.append(host)

    def _teardown_drone(self):
        """Stop every bridge host and despawn every spawned vehicle."""
        for host in self._bridge_hosts:
            host.stop()
        self._bridge_hosts = []
        for drone_cfg in self._spawned_drones:
            despawn_drone(self.stage, drone_cfg)
        self._spawned_drones = []

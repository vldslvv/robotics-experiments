from typing import Any

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
import rclpy
from omni.timeline import TimelineEventType
from omni.usd import StageEventType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

from robotics_experiments.config import get_config as _get_experiment_config
from robotics_experiments.environment import (
    SwitchState,
    despawn_switches,
    init_articulations,
    spawn_switches,
    step_switches,
    validate_switches,
)

from .global_variables import EXTENSION_TITLE

__all__ = ["LightSwitchExtension", "LightSwitchNode"]

_DURABILITY_POLICIES = {
    "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
    "volatile": DurabilityPolicy.VOLATILE,
}
_HISTORY_POLICIES = {
    "keep_last": HistoryPolicy.KEEP_LAST,
    "keep_all": HistoryPolicy.KEEP_ALL,
}
_RELIABILITY_POLICIES = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}


class LightSwitchNode(Node):
    """ROS 2 node that owns publishers for light-switch toggle events."""

    def __init__(self, publisher_config: Any):
        """Initialize the ROS node with the fixed name ``light_switch_node``."""
        super().__init__("light_switch_node")
        qos = _build_qos_profile(publisher_config.qos)
        self._state_pub = self.create_publisher(
            Float32MultiArray,
            publisher_config.topic,
            qos,
        )
        self.get_logger().info("LightSwitchNode ready.")

    def publish_toggle(self, index: int, state: bool) -> None:
        """Publish a ``(switch_index, state)`` toggle event on the ROS topic."""
        msg = Float32MultiArray()
        msg.data = [float(index), float(state)]
        self._state_pub.publish(msg)


class LightSwitchExtension(omni.ext.IExt):
    """Isaac Sim extension that spawns light switches and bridges their state to ROS 2.

    Spawns switch/lamp prims on timeline play, initializes their articulations
    once physics is running, polls joint positions to detect button presses,
    and publishes toggle events on ``/light_switches/state``.
    """

    def on_startup(self, ext_id: str):
        """Initialize ROS, create the node, and subscribe to runtime streams.

        Args:
            ext_id: Extension identifier assigned by the Kit runtime.
        """
        carb.log_info(f"[{EXTENSION_TITLE}] Extension starting...")

        usd_context = omni.usd.get_context()
        self._stage = usd_context.get_stage()
        self._switch_state: list[SwitchState] = []
        self._articulations_ready = False
        self._simulation_running = False
        self._switches_spawned = False

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._ros_node: LightSwitchNode | None = LightSwitchNode(
            _get_switch_config().state_publisher
        )

        stage_events = usd_context.get_stage_event_stream()
        self._stage_event_sub = stage_events.create_subscription_to_pop(
            self._on_stage_event,
            name="light_switch_stage",
        )

        timeline = omni.timeline.get_timeline_interface()
        timeline_events = timeline.get_timeline_event_stream()
        self._timeline_event_sub = timeline_events.create_subscription_to_pop(
            self._on_timeline_event,
            name="light_switch_timeline",
        )

        app = omni.kit.app.get_app()
        update_events = app.get_update_event_stream()
        self._update_event_sub = update_events.create_subscription_to_pop(
            self._on_update,
            name="light_switch_update",
        )

        carb.log_info(f"[{EXTENSION_TITLE}] Running. Press Play to begin.")

    def on_shutdown(self):
        """Unsubscribe from Kit streams, destroy the ROS node, and shut down rclpy.

        Called by the Kit runtime when the extension is disabled.
        """
        carb.log_info(f"[{EXTENSION_TITLE}] Shutting down...")

        if self._update_event_sub:
            self._update_event_sub.unsubscribe()
            self._update_event_sub = None

        if self._timeline_event_sub:
            self._timeline_event_sub.unsubscribe()
            self._timeline_event_sub = None

        if self._stage_event_sub:
            self._stage_event_sub.unsubscribe()
            self._stage_event_sub = None

        self._teardown_switches()

        if self._ros_node:
            self._ros_node.destroy_node()
            self._ros_node = None

        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

        carb.log_info(f"[{EXTENSION_TITLE}] Shutdown complete.")

    def _on_stage_event(self, event):
        """Clear runtime switch state when a new stage finishes opening.

        Args:
            event: Stage event emitted by ``omni.usd``; only
                ``StageEventType.OPENED`` is handled.
        """
        if event.type == int(StageEventType.OPENED):
            self._stage = omni.usd.get_context().get_stage()
            self._teardown_switches()

    def _on_timeline_event(self, event):
        """Dispatch timeline transitions to simulation start/stop handlers.

        ``PAUSE`` keeps spawned switches and articulation handles intact; it
        only stops per-frame switch stepping until playback resumes.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        if event.type == int(TimelineEventType.PLAY):
            self._setup_switches()
        elif event.type == int(TimelineEventType.PAUSE):
            self._pause_switches()
        elif event.type == int(TimelineEventType.STOP):
            self._teardown_switches()

    def _setup_switches(self):
        """Spawn switches if needed and arm per-frame switch processing.

        Validates the resulting scene before arming the main loop. On
        validation failure the simulation-running flag is left unset so the
        update loop stays idle.
        """
        if self._stage is None:
            carb.log_error(f"[{EXTENSION_TITLE}] No stage loaded.")
            return

        if not self._switches_spawned:
            self._teardown_switches()
            self._switch_state = spawn_switches(self._stage)

            if not validate_switches(self._stage, self._switch_state):
                carb.log_error(
                    f"[{EXTENSION_TITLE}] Scene validation failed; aborting simulation start."
                )
                self._teardown_switches()
                return

            self._switches_spawned = True
            self._articulations_ready = False

        carb.log_info(f"[{EXTENSION_TITLE}] Simulation started.")
        self._simulation_running = True

    def _pause_switches(self):
        """Pause per-frame switch processing while keeping runtime state intact."""
        self._simulation_running = False

    def _teardown_switches(self):
        """Reset runtime flags and despawn all switches and lamps from the stage."""
        carb.log_info(f"[{EXTENSION_TITLE}] Simulation stopped - resetting state.")
        self._simulation_running = False
        self._articulations_ready = False
        self._switches_spawned = False

        despawn_switches(self._stage, self._switch_state)
        self._switch_state = []

    def _on_update(self, event):
        """Pump ROS callbacks and, when the sim is playing, step switch logic.

        Initializes articulations on the first tick after play begins; later
        ticks poll joint positions and forward toggle events to the ROS
        publisher via :meth:`_publish_toggle`.

        Args:
            event: Per-frame update event from ``omni.kit.app``.
        """
        if self._ros_node is None:
            carb.log_error(f"[{EXTENSION_TITLE}] ROS node is not initialized.")
            return

        rclpy.spin_once(self._ros_node, timeout_sec=0)

        if not self._simulation_running:
            return

        self._step_switches()

    def _step_switches(self):
        """Initialize articulations or run per-frame switch interaction logic."""
        if not self._switches_spawned:
            return

        if not self._articulations_ready:
            if init_articulations(self._switch_state):
                self._articulations_ready = True
            return

        if self._stage is None:
            carb.log_error(f"[{EXTENSION_TITLE}] No stage loaded.")
            return
        step_switches(self._stage, self._switch_state, on_toggle=self._publish_toggle)

    def _publish_toggle(self, index: int, state: bool) -> None:
        """Publish a ``(switch_index, state)`` toggle event on the ROS topic.

        Args:
            index: Zero-based switch index matching the spawn order.
            state: Post-toggle on/off state of the switch.
        """
        if self._ros_node is None:
            carb.log_error(f"[{EXTENSION_TITLE}] ROS node is not initialized.")
            return
        self._ros_node.publish_toggle(index, state)


def _get_switch_config():
    """Return the lightswitch portion of the shared experiment config."""
    return _get_experiment_config().switches


def _normalize_qos_name(name: str, *, field_name: str) -> str:
    """Normalize a configured QoS policy name for enum lookup."""
    if not isinstance(name, str):
        raise ValueError(f"QoS {field_name} must be a string, got {type(name).__name__}.")
    return name.strip().lower().replace("-", "_")


def _resolve_qos_policy(name: str, *, field_name: str, policies: dict[str, Any]) -> Any:
    """Resolve a configured QoS policy name to the corresponding rclpy enum value."""
    key = _normalize_qos_name(name, field_name=field_name)
    try:
        return policies[key]
    except KeyError as exc:
        valid = ", ".join(sorted(policies))
        raise ValueError(f"Invalid QoS {field_name} {name!r}; expected one of: {valid}.") from exc


def _build_qos_profile(qos_config: Any) -> QoSProfile:
    """Build a ROS 2 QoS profile from Hydra configuration."""
    if qos_config.depth <= 0:
        raise ValueError(f"QoS depth must be positive, got {qos_config.depth}.")
    return QoSProfile(
        depth=qos_config.depth,
        durability=_resolve_qos_policy(
            qos_config.durability,
            field_name="durability",
            policies=_DURABILITY_POLICIES,
        ),
        history=_resolve_qos_policy(
            qos_config.history,
            field_name="history",
            policies=_HISTORY_POLICIES,
        ),
        reliability=_resolve_qos_policy(
            qos_config.reliability,
            field_name="reliability",
            policies=_RELIABILITY_POLICIES,
        ),
    )

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
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32MultiArray

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


class LightSwitchNode(Node):
    """ROS 2 node that owns publishers for light-switch toggle events."""

    def __init__(self):
        """Initialize the ROS node with the fixed name ``light_switch_node``."""
        super().__init__("light_switch_node")
        self.get_logger().info("LightSwitchNode ready.")


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
        self.stage = usd_context.get_stage()
        self._switch_state: list[SwitchState] = []
        self._state_pub: Any | None = None
        self._is_initialized = False
        self._sim_playing = False

        if not rclpy.ok():
            rclpy.init()
        self._ros_node: LightSwitchNode | None = LightSwitchNode()

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

        if rclpy.ok():
            rclpy.shutdown()

        carb.log_info(f"[{EXTENSION_TITLE}] Shutdown complete.")

    def _on_stage_event(self, event):
        """Clear runtime switch state when a new stage finishes opening.

        Args:
            event: Stage event emitted by ``omni.usd``; only
                ``StageEventType.OPENED`` is handled.
        """
        if event.type == int(StageEventType.OPENED):
            self.stage = omni.usd.get_context().get_stage()
            self._teardown_switches()

    def _on_timeline_event(self, event):
        """Dispatch timeline transitions to simulation start/stop handlers.

        ``PAUSE`` is treated identically to ``STOP`` because articulations
        must be re-initialized on resume.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        if event.type == int(TimelineEventType.PLAY):
            self._setup_switches()
        elif event.type in (
            int(TimelineEventType.STOP),
            int(TimelineEventType.PAUSE),
        ):
            self._teardown_switches()

    def _setup_switches(self):
        """Despawn any stale prims, spawn fresh switches, and create the ROS publisher.

        Validates the resulting scene before arming the main loop. On
        validation failure the simulation-playing flag is left unset so the
        update loop stays idle.
        """
        if self.stage is None:
            carb.log_error(f"[{EXTENSION_TITLE}] No stage loaded.")
            return

        self._teardown_switches()
        self._switch_state = spawn_switches(self.stage)

        if not validate_switches(self.stage, self._switch_state):
            carb.log_error(
                f"[{EXTENSION_TITLE}] Scene validation failed; aborting simulation start."
            )
            self._teardown_switches()
            return

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        if self._ros_node is None:
            carb.log_error(f"[{EXTENSION_TITLE}] ROS node is not initialized.")
            self._teardown_switches()
            return
        self._state_pub = self._ros_node.create_publisher(
            Float32MultiArray, "/light_switches/state", qos
        )

        carb.log_info(f"[{EXTENSION_TITLE}] Simulation started.")
        self._sim_playing = True

    def _teardown_switches(self):
        """Reset runtime flags and despawn all switches and lamps from the stage."""
        carb.log_info(f"[{EXTENSION_TITLE}] Simulation stopped - resetting state.")
        self._sim_playing = False
        self._is_initialized = False
        self._state_pub = None

        despawn_switches(self.stage, self._switch_state)
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

        if not self._sim_playing:
            return

        self._step_switches()

    def _step_switches(self):
        """Initialize articulations or run per-frame switch interaction logic."""
        if not self._is_initialized:
            if init_articulations(self._switch_state):
                self._is_initialized = True
            return

        if self.stage is None:
            carb.log_error(f"[{EXTENSION_TITLE}] No stage loaded.")
            return
        step_switches(self.stage, self._switch_state, on_toggle=self._publish_toggle)

    def _publish_toggle(self, index: int, state: bool) -> None:
        """Publish a ``(switch_index, state)`` toggle event on the ROS topic.

        Args:
            index: Zero-based switch index matching the spawn order.
            state: Post-toggle on/off state of the switch.
        """
        msg = Float32MultiArray()
        msg.data = [float(index), float(state)]
        if self._state_pub is None:
            carb.log_error(f"[{EXTENSION_TITLE}] ROS node is not initialized.")
            return
        self._state_pub.publish(msg)

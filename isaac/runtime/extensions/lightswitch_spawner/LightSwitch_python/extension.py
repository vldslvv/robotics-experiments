import os
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
from std_msgs.msg import Float32MultiArray

from LightSwitch_python.global_variables import EXTENSION_TITLE
from LightSwitch_python.qos import build_qos_profile
from robotics_experiments.config import load_config
from robotics_experiments.environment import (
    SwitchesRuntimeState,
    adopt_switches_stage,
    create_switches_runtime,
    pause_switches_runtime,
    play_switches_runtime,
    step_switches_runtime,
    teardown_switches_runtime,
)

__all__ = ["LightSwitchExtension", "LightSwitchNode"]

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"


class LightSwitchNode(Node):
    """ROS 2 node that owns publishers for light-switch toggle events."""

    def __init__(self, publisher_config: Any):
        """Initialize the ROS node with the fixed name ``light_switch_node``."""
        super().__init__("light_switch_node")
        qos = build_qos_profile(qos_config=publisher_config.qos)
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
        try:
            config_path = os.environ[_CONFIG_PATH_ENV]
        except KeyError as exc:
            raise RuntimeError(f"missing required environment variable {_CONFIG_PATH_ENV}") from exc
        config = load_config(config_path).switches

        usd_context = omni.usd.get_context()
        self._state: SwitchesRuntimeState = create_switches_runtime(
            config=config,
            stage=usd_context.get_stage(),
        )

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._ros_node: LightSwitchNode | None = LightSwitchNode(config.state_publisher)

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

        self._state = teardown_switches_runtime(state=self._state)

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
            self._state = adopt_switches_stage(
                state=self._state,
                stage=omni.usd.get_context().get_stage(),
            )

    def _on_timeline_event(self, event):
        """Dispatch timeline transitions to simulation start/stop handlers.

        ``PAUSE`` keeps spawned switches and articulation handles intact; it
        only stops per-frame switch stepping until playback resumes.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        if event.type == int(TimelineEventType.PLAY):
            self._state = play_switches_runtime(state=self._state)
        elif event.type == int(TimelineEventType.PAUSE):
            self._state = pause_switches_runtime(state=self._state)
        elif event.type == int(TimelineEventType.STOP):
            self._state = teardown_switches_runtime(state=self._state)

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

        self._state = step_switches_runtime(
            state=self._state,
            on_toggle=self._publish_toggle,
        )

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

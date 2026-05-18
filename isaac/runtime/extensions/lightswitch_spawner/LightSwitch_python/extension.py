from dataclasses import dataclass

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
import rclpy
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


class LightSwitchNode(Node):
    """ROS 2 node that owns publishers for light-switch toggle events."""

    def __init__(self):
        """Initialize the ROS node with the fixed name ``light_switch_node``."""
        super().__init__("light_switch_node")
        self.get_logger().info("LightSwitchNode ready.")


@dataclass
class ExtensionState:
    """Mutable runtime state for :class:`LightSwitchExtension`.

    Attributes:
        state_pub: ROS publisher for switch toggle events, created on simulation start.
        is_initialized: Whether articulations are initialized for the current play.
        sim_playing: Whether the Kit timeline is currently playing.
        update_sub: Subscription to the Kit per-frame update stream.
        timeline_sub: Subscription to the Kit timeline event stream.
        ros_node: The owned ROS 2 node, or ``None`` after shutdown.
    """

    state_pub: object | None = None
    is_initialized: bool = False
    sim_playing: bool = False
    update_sub: object | None = None
    timeline_sub: object | None = None
    ros_node: LightSwitchNode | None = None


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
        carb.log_info("[LightSwitch] Extension starting...")

        self._switch_state: list[SwitchState] = []
        self._state = ExtensionState()

        if not rclpy.ok():
            rclpy.init()
        self._state.ros_node = LightSwitchNode()

        timeline = omni.timeline.get_timeline_interface()
        timeline_events = timeline.get_timeline_event_stream()
        self._state.timeline_sub = timeline_events.create_subscription_to_pop(
            self._on_timeline_event,
            name="light_switch_timeline",
        )

        app = omni.kit.app.get_app()
        update_events = app.get_update_event_stream()
        self._state.update_sub = update_events.create_subscription_to_pop(
            self._on_update,
            name="light_switch_update",
        )

        carb.log_info("[LightSwitch] Running. Press Play to begin.")

    def on_shutdown(self):
        """Unsubscribe from Kit streams, destroy the ROS node, and shut down rclpy.

        Called by the Kit runtime when the extension is disabled.
        """
        carb.log_info("[LightSwitch] Shutting down...")

        if self._state.update_sub:
            self._state.update_sub.unsubscribe()
            self._state.update_sub = None

        if self._state.timeline_sub:
            self._state.timeline_sub.unsubscribe()
            self._state.timeline_sub = None

        if self._state.ros_node:
            self._state.ros_node.destroy_node()
            self._state.ros_node = None

        if rclpy.ok():
            rclpy.shutdown()

        carb.log_info("[LightSwitch] Shutdown complete.")

    def _on_timeline_event(self, event):
        """Dispatch timeline transitions to simulation start/stop handlers.

        ``PAUSE`` is treated identically to ``STOP`` because articulations
        must be re-initialized on resume.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._on_simulation_start()
        elif event.type in (
            int(omni.timeline.TimelineEventType.STOP),
            int(omni.timeline.TimelineEventType.PAUSE),
        ):
            self._on_simulation_stop()

    def _on_simulation_start(self):
        """Despawn any stale prims, spawn fresh switches, and create the ROS publisher.

        Validates the resulting scene before arming the main loop. On
        validation failure the simulation-playing flag is left unset so the
        update loop stays idle.
        """
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            carb.log_error("[LightSwitch] No stage loaded.")
            return

        despawn_switches(stage, self._switch_state)
        self._switch_state = spawn_switches(stage)

        if not validate_switches(stage, self._switch_state):
            carb.log_error("[LightSwitch] Scene validation failed; aborting simulation start.")
            return

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state.state_pub = self._state.ros_node.create_publisher(
            Float32MultiArray, "/light_switches/state", qos
        )

        carb.log_info("[LightSwitch] Simulation started.")
        self._state.sim_playing = True

    def _on_simulation_stop(self):
        """Reset runtime flags and despawn all switches and lamps from the stage."""
        carb.log_info("[LightSwitch] Simulation stopped - resetting state.")
        self._state.sim_playing = False
        self._state.is_initialized = False

        stage = omni.usd.get_context().get_stage()
        despawn_switches(stage, self._switch_state)
        self._switch_state = []

    def _on_update(self, event):
        """Pump ROS callbacks and, when the sim is playing, step switch logic.

        Initializes articulations on the first tick after play begins; later
        ticks poll joint positions and forward toggle events to the ROS
        publisher via :meth:`_publish_toggle`.

        Args:
            event: Per-frame update event from ``omni.kit.app``.
        """
        rclpy.spin_once(self._state.ros_node, timeout_sec=0)

        if not self._state.sim_playing:
            return

        if not self._state.is_initialized:
            if init_articulations(self._switch_state):
                self._state.is_initialized = True
            return

        stage = omni.usd.get_context().get_stage()
        step_switches(stage, self._switch_state, on_toggle=self._publish_toggle)

    def _publish_toggle(self, index: int, state: bool) -> None:
        """Publish a ``(switch_index, state)`` toggle event on the ROS topic.

        Args:
            index: Zero-based switch index matching the spawn order.
            state: Post-toggle on/off state of the switch.
        """
        msg = Float32MultiArray()
        msg.data = [float(index), float(state)]
        self._state.state_pub.publish(msg)

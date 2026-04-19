from dataclasses import dataclass

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
import rclpy
from lightswitch_config import (
    SwitchState,
    despawn_switches,
    init_articulations,
    spawn_switches,
    step_switches,
    validate_switches,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32MultiArray


class LightSwitchNode(Node):
    def __init__(self):
        super().__init__("light_switch_node")
        self.get_logger().info("LightSwitchNode ready.")


@dataclass
class ExtensionState:
    state_pub: object | None = None
    is_initialized: bool = False
    sim_playing: bool = False
    update_sub: object | None = None
    timeline_sub: object | None = None
    ros_node: LightSwitchNode | None = None


class LightSwitchExtension(omni.ext.IExt):

    def on_startup(self, ext_id: str):
        carb.log_info("[LightSwitch] Extension starting...")

        self._switch_state: list[SwitchState] = []
        self._state = ExtensionState()

        if not rclpy.ok():
            rclpy.init()
        self._state.ros_node = LightSwitchNode()

        timeline = omni.timeline.get_timeline_interface()
        self._state.timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event, name="light_switch_timeline"
        )

        app = omni.kit.app.get_app()
        self._state.update_sub = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="light_switch_update"
        )

        carb.log_info("[LightSwitch] Running. Press Play to begin.")

    def on_shutdown(self):
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
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._on_simulation_start()
        elif event.type in (
            int(omni.timeline.TimelineEventType.STOP),
            int(omni.timeline.TimelineEventType.PAUSE),
        ):
            self._on_simulation_stop()

    def _on_simulation_start(self):
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
        carb.log_info("[LightSwitch] Simulation stopped - resetting state.")
        self._state.sim_playing = False
        self._state.is_initialized = False

        stage = omni.usd.get_context().get_stage()
        despawn_switches(stage, self._switch_state)
        self._switch_state = []

    def _on_update(self, event):
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
        msg = Float32MultiArray()
        msg.data = [float(index), float(state)]
        self._state.state_pub.publish(msg)

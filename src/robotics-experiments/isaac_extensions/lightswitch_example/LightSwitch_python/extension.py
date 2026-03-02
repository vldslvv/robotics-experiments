import omni.ext
import omni.kit.app
import omni.timeline
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool
from isaacsim.core.prims import SingleArticulation


SWITCH_PRIM_PATH   = "/Root/LightSwitch"
PRESS_THRESHOLD    = 0.003    # Y-axis travel in meters — adjust sign/magnitude to your joint
PUBLISH_RATE_STEPS = 60


class LightSwitchNode(Node):
    def __init__(self):
        super().__init__("light_switch_node")
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Bool, "/light_switch/status", qos)
        self.get_logger().info("LightSwitchNode ready.")

    def publish(self, state: bool):
        msg = Bool()
        msg.data = state
        self.pub.publish(msg)
        self.get_logger().info(f"Switch → {'ON' if state else 'OFF'}")


class LightSwitchExtension(omni.ext.IExt):

    def on_startup(self, ext_id: str):
        print("[LightSwitch] Extension starting...")

        self._switch_state      = False
        self._last_button_state = False
        self._step_counter      = 0
        self._articulation      = None
        self._is_initialized    = False
        self._update_sub        = None
        self._ros_node          = None

        if not rclpy.ok():
            rclpy.init()
        self._ros_node = LightSwitchNode()

        app = omni.kit.app.get_app()
        self._update_sub = app.get_update_event_stream().create_subscription_to_pop(
            self._on_simulation_step,
            name="light_switch_ros2_update"
        )

        print("[LightSwitch] Running. Press ▶ Play to begin.")

    def on_shutdown(self):
        print("[LightSwitch] Shutting down...")

        if self._update_sub:
            self._update_sub.unsubscribe()
            self._update_sub = None

        if self._ros_node:
            self._ros_node.destroy_node()
            self._ros_node = None

        if rclpy.ok():
            rclpy.shutdown()

        print("[LightSwitch] Shutdown complete.")

    def _on_simulation_step(self, event):
        rclpy.spin_once(self._ros_node, timeout_sec=0)

        timeline = omni.timeline.get_timeline_interface()

        # ── Deferred init ──────────────────────────────────────────────────────
        if not self._is_initialized:
            # Only construct SingleArticulation once the simulation is playing.
            # Constructing it before physics is ready causes a broken __del__
            # crash due to _callbacks never being set on the Articulation object.
            if not timeline.is_playing():
                return

            try:
                self._articulation = SingleArticulation(prim_path=SWITCH_PRIM_PATH)
                self._articulation.initialize()
                print(f"[LightSwitch] Initialized. DOFs: {self._articulation.dof_names}")
                self._is_initialized = True
            except Exception as e:
                print(f"[LightSwitch] Init failed: ({e})")
                self._articulation = None  # discard any half-constructed object
            return  # always defer joint read to the next frame

        # ── If sim stopped, reset so we re-initialize cleanly on next Play ────
        # get_joint_positions() returns None when physics view is torn down,
        # so we must exit before the read block and clear state.
        if not timeline.is_playing():
            self._is_initialized    = False
            self._articulation      = None
            self._last_button_state = False  # avoid phantom toggle on next Play
            self._step_counter      = 0
            return

        # ── Read joint position (guaranteed post-init, post-play) ──────────────
        joint_positions = self._articulation.get_joint_positions()
        button_pos      = joint_positions[0]  # adjust index if DOF order differs

        # ── Toggle latch on leading edge of press ──────────────────────────────
        # Uses >= for +Y travel. Swap to <= and negate PRESS_THRESHOLD if -Y.
        button_pressed = button_pos >= PRESS_THRESHOLD
        if button_pressed and not self._last_button_state:
            self._switch_state = not self._switch_state
            self._ros_node.publish(self._switch_state)

        self._last_button_state = button_pressed

        # ── Periodic heartbeat publish ─────────────────────────────────────────
        self._step_counter += 1
        if self._step_counter >= PUBLISH_RATE_STEPS:
            self._ros_node.publish(self._switch_state)
            self._step_counter = 0

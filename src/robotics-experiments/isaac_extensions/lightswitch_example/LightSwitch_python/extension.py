import os
import random

import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
import rclpy
from pxr import Gf, Sdf, UsdGeom
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32

from isaacsim.core.prims import SingleArticulation


# ── Configuration ──────────────────────────────────────────────────────────────

SWITCH_USD_PATH  = "/isaac-sim/proj/assets/LightSwitchWarehouse/LightSwitch.usd"  # prefab asset path
NUM_SWITCHES     = 5
SPAWN_AREA       = 3.0     # switches spawn within +/- this range on X and Y (meters)
PRESS_THRESHOLD  = 0.01   # Y-axis joint travel to count as pressed (meters)

# Each switch publishes on:
#   /light_switch_{i}/status  (std_msgs/Bool)   — latched ON/OFF state
#   /light_switch_{i}/button  (std_msgs/Float32) — raw joint position (for RL observations)
PUBLISH_RATE_STEPS = 60    # heartbeat publish interval in simulation steps


# ── Per-switch ROS2 publisher pair ─────────────────────────────────────────────

class SwitchPublisher:
    """Owns the two ROS2 publishers for a single switch instance."""

    def __init__(self, node: Node, index: int):
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._status_pub = node.create_publisher(
            Bool, f"/light_switch_{index}/status", qos
        )
        self._button_pub = node.create_publisher(
            Float32, f"/light_switch_{index}/button", 10
        )
        self._index = index

    def publish_status(self, state: bool):
        msg = Bool()
        msg.data = state
        self._status_pub.publish(msg)

    def publish_button(self, position: float):
        msg = Float32()
        msg.data = position
        self._button_pub.publish(msg)


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class LightSwitchNode(Node):
    def __init__(self):
        super().__init__("light_switch_node")
        self.get_logger().info("LightSwitchNode ready.")


# ── Extension ──────────────────────────────────────────────────────────────────

class LightSwitchExtension(omni.ext.IExt):

    def on_startup(self, ext_id: str):
        print("[LightSwitch] Extension starting...")

        # Per-switch state — populated in _spawn_switches and _init_articulations
        self._switch_prim_paths  = []
        self._articulations      = []
        self._publishers         = []
        self._switch_states      = []
        self._last_button_states = []

        # Shared state
        self._step_counter    = 0
        self._is_initialized  = False
        self._sim_playing     = False
        self._update_sub      = None
        self._timeline_sub    = None
        self._ros_node        = None

        if not rclpy.ok():
            rclpy.init()
        self._ros_node = LightSwitchNode()

        # ── Timeline subscription ──────────────────────────────────────────────
        timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event,
            name="light_switch_timeline"
        )

        # ── App update subscription ────────────────────────────────────────────
        app = omni.kit.app.get_app()
        self._update_sub = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="light_switch_update"
        )

        print("[LightSwitch] Running. Press ▶ Play to begin.")

    def on_shutdown(self):
        print("[LightSwitch] Shutting down...")

        if self._update_sub:
            self._update_sub.unsubscribe()
            self._update_sub = None

        if self._timeline_sub:
            self._timeline_sub.unsubscribe()
            self._timeline_sub = None

        if self._ros_node:
            self._ros_node.destroy_node()
            self._ros_node = None

        if rclpy.ok():
            rclpy.shutdown()

        print("[LightSwitch] Shutdown complete.")

    # ── Spawning ───────────────────────────────────────────────────────────────

    def _spawn_switches(self):
        """
        Instantiate NUM_SWITCHES copies of the LightSwitch prefab USD at random
        XY positions. Clears any previously spawned instances first so this is
        safe to call on every episode reset.
        """
        if not os.path.isfile(SWITCH_USD_PATH):
            print(f"[LightSwitch] ERROR: SWITCH_USD_PATH not found: '{SWITCH_USD_PATH}'")
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[LightSwitch] ERROR: No stage loaded.")
            return

        # Remove previously spawned prims
        for path in self._switch_prim_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(Sdf.Path(path))

        self._switch_prim_paths = []

        for i in range(NUM_SWITCHES):
            prim_path = f"/Root/LightSwitch_{i}"

            prim = stage.DefinePrim(prim_path, "Xform")
            prim.GetReferences().AddReference(SWITCH_USD_PATH)

            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(
                random.uniform(-SPAWN_AREA, SPAWN_AREA),
                random.uniform(-SPAWN_AREA, SPAWN_AREA),
                0.0
            ))

            self._switch_prim_paths.append(prim_path)

        print(f"[LightSwitch] Spawned {NUM_SWITCHES} switches.")

    def _validate_scene(self) -> bool:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[LightSwitch] ERROR: No stage loaded.")
            return False

        for path in self._switch_prim_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                print(f"[LightSwitch] ERROR: Prim not found at '{path}'.")
                return False

        return True

    # ── Timeline lifecycle ─────────────────────────────────────────────────────

    def _on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._on_simulation_start()
        elif event.type in (
            int(omni.timeline.TimelineEventType.STOP),
            int(omni.timeline.TimelineEventType.PAUSE),
        ):
            self._on_simulation_stop()

    def _on_simulation_start(self):
        self._spawn_switches()

        if not self._validate_scene():
            return

        # Create per-switch ROS2 publishers
        self._publishers = [
            SwitchPublisher(self._ros_node, i) for i in range(NUM_SWITCHES)
        ]

        # Reset per-switch state
        self._switch_states      = [False] * NUM_SWITCHES
        self._last_button_states = [False] * NUM_SWITCHES

        print("[LightSwitch] Simulation started.")
        self._sim_playing = True

    def _on_simulation_stop(self):
        print("[LightSwitch] Simulation stopped — resetting state.")
        self._sim_playing        = False
        self._is_initialized     = False
        self._articulations      = []    # physics view torn down; drop all references
        self._switch_states      = []
        self._last_button_states = []
        self._step_counter       = 0

    # ── Articulation init ──────────────────────────────────────────────────────

    def _init_articulations(self) -> bool:
        """
        Construct and initialize one SingleArticulation per spawned switch.
        Must be called from within the update loop after Play — not during
        _on_simulation_start — because the physics view is not ready yet at
        the moment the PLAY timeline event fires.
        Returns True on success, False on failure.
        """
        try:
            articulations = []
            for path in self._switch_prim_paths:
                art = SingleArticulation(prim_path=path)
                art.initialize()
                articulations.append(art)

            self._articulations = articulations
            print(f"[LightSwitch] Initialized {len(self._articulations)} articulations.")
            print(f"[LightSwitch] DOFs: {self._articulations[0].dof_names}")
            return True

        except Exception as e:
            print(f"[LightSwitch] Articulation init failed: ({e})")
            self._articulations = []  # discard any half-constructed objects
            return False

    # ── Per-frame update ───────────────────────────────────────────────────────

    def _on_update(self, event):
        rclpy.spin_once(self._ros_node, timeout_sec=0)

        if not self._sim_playing:
            return

        # ── Deferred articulation init (first frame after Play) ────────────────
        if not self._is_initialized:
            if self._init_articulations():
                self._is_initialized = True
            return  # always defer joint read to the next frame

        # ── Per-switch update ──────────────────────────────────────────────────
        for i, art in enumerate(self._articulations):
            joint_positions = art.get_joint_positions()
            button_pos      = joint_positions[0]  # adjust index if DOF order differs

            # Publish raw joint position for RL observations
            self._publishers[i].publish_button(float(button_pos))

            # Toggle latch on leading edge of press
            # Uses >= for +Y travel. Swap to <= and negate PRESS_THRESHOLD if -Y.
            button_pressed = button_pos >= PRESS_THRESHOLD
            if button_pressed and not self._last_button_states[i]:
                self._switch_states[i] = not self._switch_states[i]
                self._publishers[i].publish_status(self._switch_states[i])

            self._last_button_states[i] = button_pressed

        # ── Periodic heartbeat publish ─────────────────────────────────────────
        self._step_counter += 1
        if self._step_counter >= PUBLISH_RATE_STEPS:
            for i, pub in enumerate(self._publishers):
                pub.publish_status(self._switch_states[i])
            self._step_counter = 0

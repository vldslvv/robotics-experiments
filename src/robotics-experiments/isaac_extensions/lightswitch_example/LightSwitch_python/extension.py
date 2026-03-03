import os
import random
from dataclasses import dataclass, field

import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
import rclpy
from pxr import Gf, Sdf, UsdGeom
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32MultiArray

from isaacsim.core.prims import SingleArticulation


# ── Configuration ──────────────────────────────────────────────────────────────

SWITCH_USD_PATH  = "/isaac-sim/proj/assets/LightSwitchWarehouse/LightSwitch.usd"  # prefab asset path
NUM_SWITCHES     = 5
SPAWN_BOUNDS     = [                    # spawn regions along warehouse wall
    Gf.Range3d(
        Gf.Vec3d(-3.0, 17.99, 0.5),    # min (X, Y, Z)
        Gf.Vec3d( 8.0, 17.99, 2.7),    # max (X, Y, Z)
    ),
]
PRESS_THRESHOLD  = 0.01   # Y-axis joint travel to count as pressed (meters)
MIN_SPAWN_DIST  = 0.5    # minimum Euclidean distance between any two switches (meters)
MAX_SPAWN_RETRIES = 100

# Publishes an event on state change:
#   /light_switches/state  (std_msgs/Float32MultiArray)
#   Layout: [index, state]  — state is 0.0 (OFF) or 1.0 (ON).


# ── ROS2 node ──────────────────────────────────────────────────────────────────

class LightSwitchNode(Node):
    def __init__(self):
        super().__init__("light_switch_node")
        self.get_logger().info("LightSwitchNode ready.")


# ── Per-switch state ────────────────────────────────────────────────────────────

@dataclass
class SwitchState:
    prim_path:         str
    articulation:      SingleArticulation | None = field(default=None, repr=False)
    state:             bool = False
    last_button_state: bool = False


# ── Extension-level shared state ───────────────────────────────────────────────

@dataclass
class ExtensionState:
    state_pub:      object | None          = None
    is_initialized: bool                   = False
    sim_playing:    bool                   = False
    update_sub:     object | None          = None
    timeline_sub:   object | None          = None
    ros_node:       LightSwitchNode | None = None


# ── Spawning helpers ───────────────────────────────────────────────────────────

def _sample_positions(
    count: int,
    bounds: list[Gf.Range3d],
    min_distance: float = MIN_SPAWN_DIST,
    max_retries: int = MAX_SPAWN_RETRIES,
) -> list[Gf.Vec3d]:
    """Return `count` random positions, each sampled from a randomly chosen bound.

    Uses rejection sampling to ensure no two positions are closer than
    min_distance in 3D space. Falls back to the best candidate after
    100 retries to avoid hanging when the region is densely packed.
    """
    positions = []

    def _dist(a: Gf.Vec3d, b: Gf.Vec3d) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    for _ in range(count):
        fallback = None
        placed = False
        for _ in range(max_retries):
            b = random.choice(bounds)
            mn, mx = b.min, b.max
            c = Gf.Vec3d(
                random.uniform(mn[0], mx[0]),
                random.uniform(mn[1], mx[1]),
                random.uniform(mn[2], mx[2]),
            )
            if fallback is None:
                fallback = c
            if all(_dist(c, p) >= min_distance for p in positions):
                positions.append(c)
                placed = True
                break

        if not placed:
            print(f"[LightSwitch] WARNING: Could not place switch {len(positions)} "
                  f"with min separation {min_distance} m after {max_retries} retries; "
                  "accepting overlapping position.")
            positions.append(fallback)

    return positions


def _spawn_switches(stage, usd_path: str, positions: list[Gf.Vec3d]) -> list[str]:
    """
    Instantiate one LightSwitch prim per entry in `positions`.
    Returns the list of spawned prim paths, or an empty list on failure.
    """
    if not os.path.isfile(usd_path):
        print(f"[LightSwitch] ERROR: USD asset not found: '{usd_path}'")
        return []

    prim_paths = []
    for i, pos in enumerate(positions):
        prim_path = f"/Root/LightSwitch_{i}"

        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(usd_path)

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(pos)

        prim_paths.append(prim_path)

    print(f"[LightSwitch] Spawned {len(prim_paths)} switches.")
    return prim_paths


def _validate_scene(stage, switch_state: list) -> bool:
    for sw in switch_state:
        prim = stage.GetPrimAtPath(sw.prim_path)
        if not prim.IsValid():
            print(f"[LightSwitch] ERROR: Prim not found at '{sw.prim_path}'.")
            return False
    return True


def _init_articulations(switch_state: list) -> bool:
    """
    Construct and initialize one SingleArticulation per spawned switch.
    Must be called from within the update loop after Play — not during
    _on_simulation_start — because the physics view is not ready yet at
    the moment the PLAY timeline event fires.
    Returns True on success, False on failure.
    """
    try:
        for sw in switch_state:
            art = SingleArticulation(prim_path=sw.prim_path)
            art.initialize()
            sw.articulation = art

        print(f"[LightSwitch] Initialized {len(switch_state)} articulations.")
        print(f"[LightSwitch] DOFs: {switch_state[0].articulation.dof_names}")
        return True

    except Exception as e:
        print(f"[LightSwitch] Articulation init failed: ({e})")
        for sw in switch_state:
            sw.articulation = None  # discard any half-constructed objects
        return False


# ── Extension ──────────────────────────────────────────────────────────────────

class LightSwitchExtension(omni.ext.IExt):

    def on_startup(self, ext_id: str):
        print("[LightSwitch] Extension starting...")

        # Per-switch state — populated in _on_simulation_start and _init_articulations
        self._switch_state: list[SwitchState] = []

        # Shared state
        self._state = ExtensionState()

        if not rclpy.ok():
            rclpy.init()
        self._state.ros_node = LightSwitchNode()

        # ── Timeline subscription ──────────────────────────────────────────────
        timeline = omni.timeline.get_timeline_interface()
        self._state.timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event,
            name="light_switch_timeline"
        )

        # ── App update subscription ────────────────────────────────────────────
        app = omni.kit.app.get_app()
        self._state.update_sub = app.get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="light_switch_update"
        )

        print("[LightSwitch] Running. Press ▶ Play to begin.")

    def on_shutdown(self):
        print("[LightSwitch] Shutting down...")

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

        print("[LightSwitch] Shutdown complete.")

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
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[LightSwitch] ERROR: No stage loaded.")
            return

        # Remove any previously spawned prims before spawning new ones
        for sw in self._switch_state:
            prim = stage.GetPrimAtPath(sw.prim_path)
            if prim.IsValid():
                stage.RemovePrim(Sdf.Path(sw.prim_path))

        switch_positions = _sample_positions(NUM_SWITCHES, SPAWN_BOUNDS)
        switch_prim_paths = _spawn_switches(stage, SWITCH_USD_PATH, switch_positions)
        self._switch_state = [SwitchState(prim_path=p) for p in switch_prim_paths]

        if not _validate_scene(stage, self._switch_state):
            print("[LightSwitch] ERROR: Scene validation failed; aborting simulation start.")
            return

        # Create single combined publisher
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._state.state_pub = self._state.ros_node.create_publisher(
            Float32MultiArray, "/light_switches/state", qos
        )

        print("[LightSwitch] Simulation started.")
        self._state.sim_playing = True

    def _on_simulation_stop(self):
        print("[LightSwitch] Simulation stopped — resetting state.")
        self._state.sim_playing    = False
        self._state.is_initialized = False

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            for sw in self._switch_state:
                prim = stage.GetPrimAtPath(sw.prim_path)
                if prim.IsValid():
                    stage.RemovePrim(Sdf.Path(sw.prim_path))
        self._switch_state = []    # physics view torn down; drop all references

    # ── Per-frame update ───────────────────────────────────────────────────────

    def _on_update(self, event):
        rclpy.spin_once(self._state.ros_node, timeout_sec=0)

        if not self._state.sim_playing:
            return

        # ── Deferred articulation init (first frame after Play) ────────────────
        if not self._state.is_initialized:
            if _init_articulations(self._switch_state):
                self._state.is_initialized = True
            return  # always defer joint read to the next frame

        # ── Per-switch update ──────────────────────────────────────────────────
        for i, sw in enumerate(self._switch_state):
            joint_positions = sw.articulation.get_joint_positions()
            button_pos      = joint_positions[0]  # adjust index if DOF order differs

            # Toggle latch on leading edge of press
            # Uses >= for +Y travel. Swap to <= and negate PRESS_THRESHOLD if -Y.
            button_pressed = button_pos >= PRESS_THRESHOLD
            if button_pressed and not sw.last_button_state:
                sw.state = not sw.state
                msg = Float32MultiArray()
                msg.data = [float(i), float(sw.state)]
                self._state.state_pub.publish(msg)

            sw.last_button_state = button_pressed

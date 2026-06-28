import importlib
import sys
import tomllib
import types
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_extension_manifest_exposes_only_lightswitch_python_module():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "isaac/runtime/extensions/lightswitch_spawner/config/extension.toml"
    )

    manifest = tomllib.loads(manifest_path.read_text())
    module_names = {module["name"] for module in manifest["python"]["module"]}

    assert module_names == {"LightSwitch_python"}


class _TimelineEventType(IntEnum):
    PLAY = 1
    STOP = 2
    PAUSE = 3


class _StageEventType(IntEnum):
    OPENED = 1


class _EventStream:
    def __init__(self):
        self.callbacks = []

    def create_subscription_to_pop(self, callback, name=None):
        self.callbacks.append((callback, name))
        return SimpleNamespace(unsubscribe=lambda: None)


class _UsdContext:
    def __init__(self, stage):
        self.stage = stage
        self.stage_events = _EventStream()

    def get_stage(self):
        return self.stage

    def get_stage_event_stream(self):
        return self.stage_events


class _Timeline:
    def __init__(self):
        self.events = _EventStream()

    def get_timeline_event_stream(self):
        return self.events


class _App:
    def __init__(self):
        self.events = _EventStream()

    def get_update_event_stream(self):
        return self.events


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _Node:
    # Test-local fake. ``instances`` exposes the node(s) the extension created so
    # a test can observe the ROS publisher (and its published messages) without
    # reading the extension's private ``_ros_node`` handle. Both lists are reset
    # per test by ``_import_extension``.
    instances: list["_Node"] = []
    destroyed: list["_Node"] = []

    def __init__(self, name):
        self.name = name
        self.publishers = []
        _Node.instances.append(self)

    def get_logger(self):
        return SimpleNamespace(info=lambda message: None)

    def create_publisher(self, msg_type, topic, qos):
        publisher = _Publisher()
        self.publishers.append((msg_type, topic, qos, publisher))
        return publisher

    def destroy_node(self):
        self.destroyed.append(self)


class _QoSProfile:
    def __init__(self, *, depth, durability, history, reliability):
        self.depth = depth
        self.durability = durability
        self.history = history
        self.reliability = reliability


class _Float32MultiArray:
    def __init__(self):
        self.data = []


def _qos_config(
    *,
    depth=1,
    durability="transient_local",
    reliability="reliable",
    history="keep_last",
):
    return SimpleNamespace(
        depth=depth,
        durability=durability,
        reliability=reliability,
        history=history,
    )


def _publisher_config(topic="/light_switches/state", qos=None):
    if qos is None:
        qos = _qos_config()
    return SimpleNamespace(topic=topic, qos=qos)


def _install_fake_modules(monkeypatch, usd_context, timeline, app, publisher_config):
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.TimelineEventType = _TimelineEventType
    omni_timeline.get_timeline_interface = lambda: timeline
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.StageEventType = _StageEventType
    omni_usd.get_context = lambda: usd_context
    omni_kit = types.ModuleType("omni.kit")
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: app
    omni.ext = omni_ext
    omni.timeline = omni_timeline
    omni.usd = omni_usd
    omni.kit = omni_kit
    omni_kit.app = omni_kit_app

    carb = types.ModuleType("carb")
    carb.log_info = lambda message: None
    carb.log_error = lambda message: None

    rclpy = types.ModuleType("rclpy")
    rclpy.ok_value = False
    rclpy.spin_calls = []
    rclpy.init_calls = 0
    rclpy.shutdown_calls = 0

    def ok():
        return rclpy.ok_value

    def init():
        rclpy.init_calls += 1
        rclpy.ok_value = True

    def shutdown():
        rclpy.shutdown_calls += 1
        rclpy.ok_value = False

    rclpy.ok = ok
    rclpy.init = init
    rclpy.shutdown = shutdown
    rclpy.spin_once = lambda node, timeout_sec=0: rclpy.spin_calls.append((node, timeout_sec))

    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _Node
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.DurabilityPolicy = SimpleNamespace(
        TRANSIENT_LOCAL="transient_local",
        VOLATILE="volatile",
    )
    rclpy_qos.HistoryPolicy = SimpleNamespace(
        KEEP_LAST="keep_last",
        KEEP_ALL="keep_all",
    )
    rclpy_qos.QoSProfile = _QoSProfile
    rclpy_qos.ReliabilityPolicy = SimpleNamespace(
        RELIABLE="reliable",
        BEST_EFFORT="best_effort",
    )

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float32MultiArray = _Float32MultiArray

    fake_environment = types.ModuleType("robotics_experiments.environment")
    fake_environment.SwitchState = SimpleNamespace
    fake_environment.despawn_switches = lambda stage, switch_state: None
    fake_environment.init_articulations = lambda switch_state: True
    fake_environment.spawn_switches = lambda stage: []
    fake_environment.step_switches = lambda stage, switch_state, on_toggle=None: None
    fake_environment.validate_switches = lambda stage, switch_state: True

    fake_config = types.ModuleType("robotics_experiments.config")
    fake_config.get_config = lambda: SimpleNamespace(
        switches=SimpleNamespace(state_publisher=publisher_config)
    )

    modules = {
        "carb": carb,
        "omni": omni,
        "omni.ext": omni_ext,
        "omni.kit": omni_kit,
        "omni.kit.app": omni_kit_app,
        "omni.timeline": omni_timeline,
        "omni.usd": omni_usd,
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "robotics_experiments.config": fake_config,
        "robotics_experiments.environment": fake_environment,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _import_extension(monkeypatch, stage=None, publisher_config=None):
    for name in list(sys.modules):
        if name == "LightSwitch_python" or name.startswith("LightSwitch_python."):
            sys.modules.pop(name, None)

    _Node.instances = []
    _Node.destroyed = []
    if stage is None:
        stage = object()
    usd_context = _UsdContext(stage)
    timeline = _Timeline()
    app = _App()
    if publisher_config is None:
        publisher_config = _publisher_config()
    _install_fake_modules(monkeypatch, usd_context, timeline, app, publisher_config)
    extension_root = (
        Path(__file__).resolve().parents[1] / "isaac/runtime/extensions/lightswitch_spawner"
    )
    monkeypatch.syspath_prepend(str(extension_root))
    module = importlib.import_module("LightSwitch_python.extension")
    return module, usd_context, timeline, app


def _fire_timeline(timeline, event_type):
    for callback, _name in timeline.events.callbacks:
        callback(SimpleNamespace(type=int(event_type)))


def _fire_stage_opened(usd_context):
    for callback, _name in usd_context.stage_events.callbacks:
        callback(SimpleNamespace(type=int(_StageEventType.OPENED)))


def _fire_update(app):
    for callback, _name in app.events.callbacks:
        callback(SimpleNamespace())


def _created_node():
    """The ROS node the extension constructed (the latest, if reconstructed)."""
    return _Node.instances[-1]


def test_startup_subscribes_without_spawning(monkeypatch):
    module, usd_context, timeline, app = _import_extension(monkeypatch)
    calls = []
    monkeypatch.setattr(module, "spawn_switches", lambda stage: calls.append(("spawn", stage)))

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    # Startup subscribes to all three Kit streams but spawns nothing until play,
    # and creates the ROS node with its state publisher.
    assert calls == []
    assert len(usd_context.stage_events.callbacks) == 1
    assert len(timeline.events.callbacks) == 1
    assert len(app.events.callbacks) == 1
    assert _created_node().publishers


def test_play_spawns_validates_and_uses_startup_publisher(monkeypatch):
    stage = object()
    publisher_config = _publisher_config(
        topic="/custom_switch_state",
        qos=_qos_config(
            depth=7,
            durability="volatile",
            reliability="best_effort",
            history="keep_all",
        ),
    )
    module, _usd_context, timeline, _app = _import_extension(
        monkeypatch,
        stage=stage,
        publisher_config=publisher_config,
    )
    switches = [SimpleNamespace(prim_path="/Root/LightSwitch_0", lamp_prim_path="/Root/Lamp_0")]
    calls = []

    def despawn(stage, state):
        calls.append(("despawn", stage, list(state)))

    def spawn(stage):
        calls.append(("spawn", stage))
        return switches

    def validate(stage, state):
        calls.append(("validate", stage, state))
        return True

    monkeypatch.setattr(module, "despawn_switches", despawn)
    monkeypatch.setattr(module, "spawn_switches", spawn)
    monkeypatch.setattr(module, "validate_switches", validate)

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    startup_node = _created_node()
    _fire_timeline(timeline, _TimelineEventType.PLAY)

    # Play clears the stage, spawns, and validates -- in that order -- using the
    # play stage and the switches spawn returned (so the state flows to validate).
    assert calls == [
        ("despawn", stage, []),
        ("spawn", stage),
        ("validate", stage, switches),
    ]
    # Play reuses the startup node/publisher; no second node is constructed.
    assert _Node.instances == [startup_node]
    _msg_type, topic, qos, _publisher = startup_node.publishers[0]
    assert topic == "/custom_switch_state"
    assert qos.depth == 7
    assert qos.durability == "volatile"
    assert qos.reliability == "best_effort"
    assert qos.history == "keep_all"


def test_play_uses_default_hydra_publisher_settings(monkeypatch):
    stage = object()
    module, _usd_context, _timeline, _app = _import_extension(monkeypatch, stage=stage)

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    _msg_type, topic, qos, _publisher = _created_node().publishers[0]
    assert topic == "/light_switches/state"
    assert qos.depth == 1
    assert qos.durability == "transient_local"
    assert qos.reliability == "reliable"
    assert qos.history == "keep_last"


def test_validation_failure_tears_down_partial_runtime_state(monkeypatch):
    stage = object()
    module, _usd_context, timeline, app = _import_extension(monkeypatch, stage=stage)
    switches = [SimpleNamespace(prim_path="/Root/LightSwitch_0", lamp_prim_path="/Root/Lamp_0")]
    calls = []

    def despawn(stage, state):
        calls.append(("despawn", list(state)))

    monkeypatch.setattr(module, "despawn_switches", despawn)
    monkeypatch.setattr(module, "spawn_switches", lambda stage: switches)
    monkeypatch.setattr(module, "validate_switches", lambda stage, state: False)
    monkeypatch.setattr(module, "init_articulations", lambda state: calls.append("init") or True)
    monkeypatch.setattr(
        module, "step_switches", lambda stage, state, on_toggle=None: calls.append("step")
    )

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    _fire_timeline(timeline, _TimelineEventType.PLAY)

    # A failed validation despawns the just-spawned switches (the second despawn).
    assert calls == [("despawn", []), ("despawn", switches)]
    # The loop is left idle: subsequent updates neither init nor step.
    _fire_update(app)
    assert calls == [("despawn", []), ("despawn", switches)]
    # The ROS node survives the failed start.
    assert _created_node().publishers


def test_repeated_play_does_not_double_spawn(monkeypatch):
    stage = object()
    module, _usd_context, timeline, _app = _import_extension(monkeypatch, stage=stage)
    switches = [SimpleNamespace(prim_path="/Root/LightSwitch_0", lamp_prim_path="/Root/Lamp_0")]
    calls = []

    monkeypatch.setattr(module, "despawn_switches", lambda stage, state: calls.append("despawn"))
    monkeypatch.setattr(module, "spawn_switches", lambda stage: calls.append("spawn") or switches)
    monkeypatch.setattr(
        module,
        "validate_switches",
        lambda stage, state: calls.append("validate") or True,
    )

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    _fire_timeline(timeline, _TimelineEventType.PLAY)
    _fire_timeline(timeline, _TimelineEventType.PLAY)

    # The second play does not re-spawn: spawn/validate run exactly once.
    assert calls == ["despawn", "spawn", "validate"]


def test_pause_keeps_switch_state_and_articulations(monkeypatch):
    stage = object()
    module, _usd_context, timeline, app = _import_extension(monkeypatch, stage=stage)
    switches = ["switch"]
    calls = []

    monkeypatch.setattr(module, "despawn_switches", lambda stage, state: calls.append("despawn"))
    monkeypatch.setattr(module, "spawn_switches", lambda stage: switches)
    monkeypatch.setattr(module, "validate_switches", lambda stage, state: True)
    monkeypatch.setattr(module, "init_articulations", lambda state: calls.append("init") or True)
    monkeypatch.setattr(
        module, "step_switches", lambda stage, state, on_toggle=None: calls.append("step")
    )

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    _fire_timeline(timeline, _TimelineEventType.PLAY)
    _fire_update(app)  # init articulations
    _fire_update(app)  # step (running)

    # The play teardown despawns once, then articulations init, then a step runs.
    assert calls == ["despawn", "init", "step"]

    _fire_timeline(timeline, _TimelineEventType.PAUSE)
    _fire_update(app)  # paused: no further step

    # Pause keeps state (no despawn) and halts per-frame stepping.
    assert calls == ["despawn", "init", "step"]


def test_play_after_pause_resumes_without_reinitializing(monkeypatch):
    stage = object()
    module, _usd_context, timeline, app = _import_extension(monkeypatch, stage=stage)
    switches = ["switch"]
    calls = []

    monkeypatch.setattr(module, "despawn_switches", lambda stage, state: calls.append("despawn"))
    monkeypatch.setattr(module, "spawn_switches", lambda stage: switches)
    monkeypatch.setattr(module, "validate_switches", lambda stage, state: True)
    monkeypatch.setattr(module, "init_articulations", lambda state: calls.append("init") or True)
    monkeypatch.setattr(
        module, "step_switches", lambda stage, state, on_toggle=None: calls.append("step")
    )

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    _fire_timeline(timeline, _TimelineEventType.PLAY)
    _fire_update(app)  # init articulations
    _fire_update(app)  # step
    _fire_timeline(timeline, _TimelineEventType.PAUSE)

    _fire_timeline(timeline, _TimelineEventType.PLAY)  # resume
    _fire_update(app)  # steps immediately, no re-spawn and no re-init

    # Only one despawn (no re-spawn) and one init (no re-initialize); stepping
    # resumes directly.
    assert calls == ["despawn", "init", "step", "step"]


def test_stop_stage_open_and_shutdown_teardown(monkeypatch):
    stage = object()
    module, usd_context, timeline, app = _import_extension(monkeypatch, stage=stage)
    switches = ["switch"]
    despawns = []

    monkeypatch.setattr(
        module, "despawn_switches", lambda stg, state: despawns.append((stg, list(state)))
    )
    monkeypatch.setattr(module, "spawn_switches", lambda stg: switches)
    monkeypatch.setattr(module, "validate_switches", lambda stg, state: True)

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    # STOP despawns the spawned switches with the play stage.
    _fire_timeline(timeline, _TimelineEventType.PLAY)
    _fire_timeline(timeline, _TimelineEventType.STOP)
    assert despawns[-1] == (stage, ["switch"])

    # A new stage opening re-targets the stage and tears down again: the despawn
    # carries the new stage (observable proof the stage handle was adopted).
    _fire_timeline(timeline, _TimelineEventType.PLAY)
    usd_context.stage = "new-stage"
    _fire_stage_opened(usd_context)
    assert despawns[-1] == ("new-stage", ["switch"])

    # Shutdown tears down on that adopted stage and destroys the ROS node; a
    # second shutdown is idempotent (no raise).
    _fire_timeline(timeline, _TimelineEventType.PLAY)
    node = _created_node()
    extension.on_shutdown()
    assert despawns[-1] == ("new-stage", ["switch"])
    assert node in _Node.destroyed
    extension.on_shutdown()


def test_update_spins_ros_initializes_then_steps(monkeypatch):
    stage = object()
    module, _usd_context, timeline, app = _import_extension(monkeypatch, stage=stage)
    switches = ["switch"]
    calls = []

    monkeypatch.setattr(module, "spawn_switches", lambda stg: switches)
    monkeypatch.setattr(module, "validate_switches", lambda stg, state: True)

    def init_articulations(state):
        calls.append(("init", list(state)))
        return True

    monkeypatch.setattr(module, "init_articulations", init_articulations)
    monkeypatch.setattr(
        module,
        "step_switches",
        lambda stg, state, on_toggle=None: calls.append(("step", stg, list(state), on_toggle)),
    )

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    _fire_timeline(timeline, _TimelineEventType.PLAY)

    _fire_update(app)  # first running tick initializes articulations
    _fire_update(app)  # next tick steps

    assert calls[0] == ("init", ["switch"])
    _step_label, step_stage, step_state, on_toggle = calls[1]
    assert (_step_label, step_stage, step_state) == ("step", stage, ["switch"])
    # The step's on_toggle is the publisher callback: invoking it publishes.
    on_toggle(2, True)
    assert _created_node().publishers[0][3].messages[-1].data == [2.0, 1.0]
    # Each update tick spun the ROS node once.
    node = _created_node()
    assert sys.modules["rclpy"].spin_calls == [(node, 0), (node, 0)]


def test_publish_toggle_sends_float_array(monkeypatch):
    module, _usd_context, timeline, app = _import_extension(monkeypatch)
    switches = ["switch"]
    monkeypatch.setattr(module, "spawn_switches", lambda stage: switches)
    monkeypatch.setattr(module, "validate_switches", lambda stage, state: True)
    monkeypatch.setattr(module, "init_articulations", lambda state: True)
    # A real switch toggle flows through step_switches' on_toggle to the publisher.
    monkeypatch.setattr(
        module, "step_switches", lambda stage, state, on_toggle=None: on_toggle(2, True)
    )

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    _fire_timeline(timeline, _TimelineEventType.PLAY)
    _fire_update(app)  # init articulations
    _fire_update(app)  # step -> on_toggle -> publish

    publisher = _created_node().publishers[0][3]
    assert len(publisher.messages) == 1
    assert publisher.messages[0].data == [2.0, 1.0]


@pytest.mark.parametrize(
    ("qos", "message"),
    [
        (_qos_config(depth=0), "QoS depth must be positive"),
        (_qos_config(durability="forever"), "Invalid QoS durability"),
        (_qos_config(reliability="sometimes"), "Invalid QoS reliability"),
        (_qos_config(history="keep_some"), "Invalid QoS history"),
    ],
)
def test_invalid_qos_config_raises(monkeypatch, qos, message):
    module, _usd_context, _timeline, _app = _import_extension(
        monkeypatch,
        publisher_config=_publisher_config(qos=qos),
    )
    with pytest.raises(ValueError, match=message):
        module.LightSwitchExtension().on_startup("lightswitch_spawner")

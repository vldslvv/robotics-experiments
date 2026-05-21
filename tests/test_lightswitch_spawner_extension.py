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
    destroyed = []

    def __init__(self, name):
        self.name = name
        self.publishers = []

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


def _install_fake_modules(monkeypatch, usd_context, app, publisher_config):
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.TimelineEventType = _TimelineEventType
    omni_timeline.get_timeline_interface = lambda: _Timeline()
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
    rclpy.spin_once = lambda node, timeout_sec=0: rclpy.spin_calls.append(
        (node, timeout_sec)
    )

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

    if stage is None:
        stage = object()
    usd_context = _UsdContext(stage)
    app = _App()
    if publisher_config is None:
        publisher_config = _publisher_config()
    _install_fake_modules(monkeypatch, usd_context, app, publisher_config)
    extension_root = (
        Path(__file__).resolve().parents[1]
        / "isaac/runtime/extensions/lightswitch_spawner"
    )
    monkeypatch.syspath_prepend(str(extension_root))
    module = importlib.import_module("LightSwitch_python.extension")
    return module, usd_context, app


def test_startup_subscribes_without_spawning(monkeypatch):
    module, usd_context, app = _import_extension(monkeypatch)
    calls = []
    monkeypatch.setattr(module, "spawn_switches", lambda stage: calls.append(("spawn", stage)))

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    assert calls == []
    assert extension.stage is usd_context.stage
    assert len(usd_context.stage_events.callbacks) == 1
    assert len(app.events.callbacks) == 1


def test_play_spawns_validates_and_creates_publisher(monkeypatch):
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
    module, _usd_context, _app = _import_extension(
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
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    assert calls == [
        ("despawn", stage, []),
        ("spawn", stage),
        ("validate", stage, switches),
    ]
    assert extension._switch_state == switches
    assert extension._sim_playing
    assert extension._state_pub is not None
    _msg_type, topic, qos, _publisher = extension._ros_node.publishers[0]
    assert topic == "/custom_switch_state"
    assert qos.depth == 7
    assert qos.durability == "volatile"
    assert qos.reliability == "best_effort"
    assert qos.history == "keep_all"


def test_play_uses_default_hydra_publisher_settings(monkeypatch):
    stage = object()
    module, _usd_context, _app = _import_extension(monkeypatch, stage=stage)

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    _msg_type, topic, qos, _publisher = extension._ros_node.publishers[0]
    assert topic == "/light_switches/state"
    assert qos.depth == 1
    assert qos.durability == "transient_local"
    assert qos.reliability == "reliable"
    assert qos.history == "keep_last"


def test_validation_failure_tears_down_partial_runtime_state(monkeypatch):
    stage = object()
    module, _usd_context, _app = _import_extension(monkeypatch, stage=stage)
    switches = [SimpleNamespace(prim_path="/Root/LightSwitch_0", lamp_prim_path="/Root/Lamp_0")]
    calls = []

    def despawn(stage, state):
        calls.append(("despawn", list(state)))

    monkeypatch.setattr(module, "despawn_switches", despawn)
    monkeypatch.setattr(module, "spawn_switches", lambda stage: switches)
    monkeypatch.setattr(module, "validate_switches", lambda stage, state: False)

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

    assert calls == [("despawn", []), ("despawn", switches)]
    assert extension._switch_state == []
    assert not extension._sim_playing
    assert extension._state_pub is None


def test_stop_pause_stage_open_and_shutdown_teardown(monkeypatch):
    stage = object()
    module, usd_context, _app = _import_extension(monkeypatch, stage=stage)
    calls = []

    def despawn(stage, state):
        calls.append((stage, list(state)))

    monkeypatch.setattr(module, "despawn_switches", despawn)

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    extension._switch_state = ["switch"]
    extension._sim_playing = True
    extension._is_initialized = True
    extension._state_pub = object()

    extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PAUSE)))

    assert calls == [(stage, ["switch"])]
    assert extension._switch_state == []
    assert not extension._sim_playing
    assert not extension._is_initialized
    assert extension._state_pub is None

    usd_context.stage = "new-stage"
    extension._switch_state = ["new-switch"]
    extension._on_stage_event(SimpleNamespace(type=int(_StageEventType.OPENED)))

    assert extension.stage == "new-stage"
    assert calls[-1] == ("new-stage", ["new-switch"])

    extension._switch_state = ["shutdown-switch"]
    extension.on_shutdown()

    assert calls[-1] == ("new-stage", ["shutdown-switch"])
    assert extension._ros_node is None


def test_update_spins_ros_initializes_then_steps(monkeypatch):
    stage = object()
    module, _usd_context, _app = _import_extension(monkeypatch, stage=stage)
    calls = []

    def init_articulations(state):
        calls.append(("init", list(state)))
        return True

    monkeypatch.setattr(module, "init_articulations", init_articulations)
    monkeypatch.setattr(
        module,
        "step_switches",
        lambda stage, state, on_toggle=None: calls.append(("step", stage, list(state), on_toggle)),
    )

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    extension._switch_state = ["switch"]
    extension._sim_playing = True

    extension._on_update(SimpleNamespace())
    extension._on_update(SimpleNamespace())

    assert calls[0] == ("init", ["switch"])
    assert calls[1] == ("step", stage, ["switch"], extension._publish_toggle)
    assert sys.modules["rclpy"].spin_calls == [
        (extension._ros_node, 0),
        (extension._ros_node, 0),
    ]


def test_publish_toggle_sends_float_array(monkeypatch):
    module, _usd_context, _app = _import_extension(monkeypatch)
    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    publisher = _Publisher()
    extension._state_pub = publisher

    extension._publish_toggle(2, True)

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
    module, _usd_context, _app = _import_extension(
        monkeypatch,
        publisher_config=_publisher_config(qos=qos),
    )
    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    with pytest.raises(ValueError, match=message):
        extension._on_timeline_event(SimpleNamespace(type=int(_TimelineEventType.PLAY)))

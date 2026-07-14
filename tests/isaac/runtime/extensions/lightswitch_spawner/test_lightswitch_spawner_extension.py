import importlib
import sys
import tomllib
import types
from enum import IntEnum
from types import SimpleNamespace

import pytest

from tests.support.paths import REPO_ROOT


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


class _Node:
    instances: list["_Node"] = []
    destroyed: list["_Node"] = []

    def __init__(self, name):
        self.name = name
        self.publishers = []
        self.instances.append(self)

    def get_logger(self):
        return SimpleNamespace(info=lambda message: None)

    def create_publisher(self, msg_type, topic, qos):
        publisher = SimpleNamespace(messages=[], publish=lambda msg: publisher.messages.append(msg))
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
    *, depth=1, durability="transient_local", reliability="reliable", history="keep_last"
):
    return SimpleNamespace(
        depth=depth,
        durability=durability,
        reliability=reliability,
        history=history,
    )


def _publisher_config(*, topic="/light_switches/state", qos=None):
    return SimpleNamespace(topic=topic, qos=qos or _qos_config())


def _import_extension(monkeypatch, *, stage=None, publisher_config=None):
    for name in list(sys.modules):
        if name == "LightSwitch_python" or name.startswith("LightSwitch_python."):
            sys.modules.pop(name, None)
    _Node.instances = []
    _Node.destroyed = []
    if stage is None:
        stage = object()

    stage_events = _EventStream()
    timeline_events = _EventStream()
    update_events = _EventStream()
    usd_context = SimpleNamespace(
        stage=stage,
        get_stage=lambda: usd_context.stage,
        get_stage_event_stream=lambda: stage_events,
    )

    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.TimelineEventType = _TimelineEventType
    omni_timeline.get_timeline_interface = lambda: SimpleNamespace(
        get_timeline_event_stream=lambda: timeline_events
    )
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.StageEventType = _StageEventType
    omni_usd.get_context = lambda: usd_context
    omni_kit = types.ModuleType("omni.kit")
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: SimpleNamespace(get_update_event_stream=lambda: update_events)
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
    rclpy.init = lambda: setattr(rclpy, "ok_value", True)
    rclpy.ok = lambda: rclpy.ok_value
    rclpy.shutdown = lambda: setattr(rclpy, "ok_value", False)
    rclpy.spin_once = lambda node, timeout_sec=0: rclpy.spin_calls.append((node, timeout_sec))
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _Node
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.DurabilityPolicy = SimpleNamespace(
        TRANSIENT_LOCAL="transient_local", VOLATILE="volatile"
    )
    rclpy_qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep_last", KEEP_ALL="keep_all")
    rclpy_qos.QoSProfile = _QoSProfile
    rclpy_qos.ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best_effort")

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float32MultiArray = _Float32MultiArray

    fake_environment = types.ModuleType("robotics_experiments.environment")
    fake_environment.__path__ = []
    fake_environment.SwitchesRuntimeState = SimpleNamespace
    fake_environment.create_switches_runtime = lambda **kwargs: SimpleNamespace(**kwargs)
    for name in (
        "adopt_switches_stage",
        "pause_switches_runtime",
        "play_switches_runtime",
        "step_switches_runtime",
        "teardown_switches_runtime",
    ):
        setattr(fake_environment, name, lambda **kwargs: kwargs["state"])

    config = publisher_config or _publisher_config()
    fake_config = types.ModuleType("robotics_experiments.config")
    fake_config.load_config = lambda path: SimpleNamespace(
        switches=SimpleNamespace(state_publisher=config)
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
    monkeypatch.setenv("ROBOTICS_EXPERIMENT_CONFIG_PATH", "configs/takeoff_quad.yaml")
    monkeypatch.syspath_prepend(str(REPO_ROOT / "isaac/runtime/extensions/lightswitch_spawner"))
    module = importlib.import_module("LightSwitch_python.extension")
    return module, usd_context, stage_events, timeline_events, update_events


def _fire(stream, event_type=None):
    event = SimpleNamespace(type=int(event_type)) if event_type is not None else SimpleNamespace()
    for callback, _name in stream.callbacks:
        callback(event)


def test_extension_manifest_exposes_only_lightswitch_python_module():
    path = REPO_ROOT / "isaac/runtime/extensions/lightswitch_spawner/config/extension.toml"
    manifest = tomllib.loads(path.read_text())
    assert {module["name"] for module in manifest["python"]["module"]} == {"LightSwitch_python"}


def test_startup_creates_one_state_node_and_three_subscriptions(monkeypatch):
    module, _context, stage_events, timeline_events, update_events = _import_extension(monkeypatch)
    calls = []
    monkeypatch.setattr(
        module,
        "create_switches_runtime",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(name="initial"),
    )

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    assert calls and len(_Node.instances) == 1
    assert all(
        len(stream.callbacks) == 1 for stream in (stage_events, timeline_events, update_events)
    )


def test_timeline_events_rebind_the_single_runtime_state(monkeypatch):
    module, _context, _stage_events, timeline_events, _update_events = _import_extension(
        monkeypatch
    )
    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    calls = []

    def transition(label):
        return lambda **kwargs: calls.append((label, kwargs["state"])) or SimpleNamespace(
            name=label
        )

    monkeypatch.setattr(module, "play_switches_runtime", transition("play"))
    monkeypatch.setattr(module, "pause_switches_runtime", transition("pause"))
    monkeypatch.setattr(module, "teardown_switches_runtime", transition("stop"))

    _fire(timeline_events, _TimelineEventType.PLAY)
    _fire(timeline_events, _TimelineEventType.PAUSE)
    _fire(timeline_events, _TimelineEventType.STOP)

    assert [label for label, _state in calls] == ["play", "pause", "stop"]
    assert calls[1][1].name == "play"
    assert calls[2][1].name == "pause"


def test_stage_and_update_events_delegate_and_rebind(monkeypatch):
    module, context, stage_events, _timeline_events, update_events = _import_extension(monkeypatch)
    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    adopted = SimpleNamespace(name="adopted")
    stepped = SimpleNamespace(name="stepped")
    monkeypatch.setattr(module, "adopt_switches_stage", lambda **kwargs: adopted)
    callback = []
    inputs = []
    monkeypatch.setattr(
        module,
        "step_switches_runtime",
        lambda **kwargs: (
            inputs.append(kwargs["state"]),
            callback.append(kwargs["on_toggle"]),
            stepped,
        )[2],
    )

    context.stage = "new-stage"
    _fire(stage_events, _StageEventType.OPENED)
    _fire(update_events)
    _fire(update_events)

    assert inputs == [adopted, stepped]
    callback[0](2, True)
    assert _Node.instances[0].publishers[0][3].messages[-1].data == [2.0, 1.0]
    assert sys.modules["rclpy"].spin_calls == [(_Node.instances[0], 0)] * 2


def test_shutdown_tears_down_state_and_destroys_node(monkeypatch):
    module, _context, _stage_events, _timeline_events, _update_events = _import_extension(
        monkeypatch
    )
    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    node = _Node.instances[0]
    calls = []
    monkeypatch.setattr(
        module,
        "teardown_switches_runtime",
        lambda **kwargs: calls.append(kwargs["state"]) or kwargs["state"],
    )

    extension.on_shutdown()

    assert calls
    assert node in _Node.destroyed


def test_configured_qos_is_applied(monkeypatch):
    publisher = _publisher_config(
        topic="/custom",
        qos=_qos_config(
            depth=7, durability="volatile", reliability="best_effort", history="keep_all"
        ),
    )
    module, *_ = _import_extension(monkeypatch, publisher_config=publisher)
    module.LightSwitchExtension().on_startup("lightswitch_spawner")

    _msg_type, topic, qos, _publisher = _Node.instances[0].publishers[0]
    assert (topic, qos.depth, qos.durability, qos.reliability, qos.history) == (
        "/custom",
        7,
        "volatile",
        "best_effort",
        "keep_all",
    )


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
    module, *_ = _import_extension(monkeypatch, publisher_config=_publisher_config(qos=qos))
    with pytest.raises(ValueError, match=message):
        module.LightSwitchExtension().on_startup("lightswitch_spawner")

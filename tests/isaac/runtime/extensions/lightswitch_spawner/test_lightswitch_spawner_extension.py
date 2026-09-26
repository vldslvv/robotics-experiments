import importlib
import sys
import tomllib
import types
from types import SimpleNamespace

import pytest

from tests.support.paths import REPO_ROOT


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
        publisher = SimpleNamespace(messages=[])
        publisher.publish = publisher.messages.append
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


def _import_extension(monkeypatch, *, publisher_config=None, events=()):
    for name in list(sys.modules):
        if name == "LightSwitch_python" or name.startswith("LightSwitch_python."):
            sys.modules.pop(name, None)
    _Node.instances = []
    _Node.destroyed = []

    update_events = _EventStream()
    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_kit = types.ModuleType("omni.kit")
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: SimpleNamespace(get_update_event_stream=lambda: update_events)
    omni.ext = omni_ext
    omni.kit = omni_kit
    omni_kit.app = omni_kit_app

    carb = types.ModuleType("carb")
    carb.log_info = lambda message: None

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

    config = publisher_config or _publisher_config()
    fake_config = types.ModuleType("robotics_experiments.config")
    fake_config.load_config = lambda path: SimpleNamespace(
        switches=SimpleNamespace(state_publisher=config)
    )
    fake_runtime = types.ModuleType("ExperimentRuntime_python.extension")
    fake_runtime.get_instance = lambda: SimpleNamespace(
        read_status=lambda: SimpleNamespace(switch_events=events)
    )

    modules = {
        "carb": carb,
        "omni": omni,
        "omni.ext": omni_ext,
        "omni.kit": omni_kit,
        "omni.kit.app": omni_kit_app,
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "std_msgs": std_msgs,
        "std_msgs.msg": std_msgs_msg,
        "robotics_experiments.config": fake_config,
        "ExperimentRuntime_python.extension": fake_runtime,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH", "configs/routines/basic_takeoff_quad.yaml"
    )
    monkeypatch.syspath_prepend(str(REPO_ROOT / "isaac/runtime/extensions/lightswitch_spawner"))
    module = importlib.import_module("LightSwitch_python.extension")
    return module, update_events, rclpy


def _fire(stream):
    for callback, _name in stream.callbacks:
        callback(SimpleNamespace())


def test_extension_manifest_exposes_only_lightswitch_python_module():
    path = REPO_ROOT / "isaac/runtime/extensions/lightswitch_spawner/config/extension.toml"
    manifest = tomllib.loads(path.read_text())
    assert {module["name"] for module in manifest["python"]["module"]} == {"LightSwitch_python"}
    assert "experiment_runtime" in manifest["dependencies"]


def test_startup_creates_ros_publisher_and_one_update_subscription(monkeypatch):
    module, update_events, _rclpy = _import_extension(monkeypatch)

    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    assert len(_Node.instances) == 1
    assert len(_Node.instances[0].publishers) == 1
    assert len(update_events.callbacks) == 1


def test_update_publishes_each_runtime_event_once(monkeypatch):
    events = (
        SimpleNamespace(sequence=1, switch_index=2, state=True),
        SimpleNamespace(sequence=2, switch_index=2, state=False),
    )
    module, update_events, rclpy = _import_extension(monkeypatch, events=events)
    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    _fire(update_events)
    _fire(update_events)

    publisher = _Node.instances[0].publishers[0][3]
    assert [message.data for message in publisher.messages] == [
        [2.0, 1.0],
        [2.0, 0.0],
    ]
    assert rclpy.spin_calls == [(_Node.instances[0], 0)] * 2


def test_update_waits_for_experiment_runtime(monkeypatch):
    module, update_events, _rclpy = _import_extension(monkeypatch)
    sys.modules["ExperimentRuntime_python.extension"].get_instance = lambda: (_ for _ in ()).throw(
        RuntimeError("not started")
    )
    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")

    _fire(update_events)

    assert _Node.instances[0].publishers[0][3].messages == []


def test_shutdown_destroys_node_and_owned_rclpy_context(monkeypatch):
    module, _update_events, rclpy = _import_extension(monkeypatch)
    extension = module.LightSwitchExtension()
    extension.on_startup("lightswitch_spawner")
    node = _Node.instances[0]

    extension.on_shutdown()

    assert node in _Node.destroyed
    assert not rclpy.ok()


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

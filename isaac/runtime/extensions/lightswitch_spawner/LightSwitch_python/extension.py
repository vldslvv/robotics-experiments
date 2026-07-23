"""Thin ROS publisher for switch events owned by ``experiment_runtime``."""

from __future__ import annotations

import os
from typing import Any

import carb
import omni.ext
import omni.kit.app
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from LightSwitch_python.global_variables import EXTENSION_TITLE
from LightSwitch_python.qos import build_qos_profile
from robotics_experiments.config import load_config

__all__ = ["LightSwitchExtension", "LightSwitchNode"]

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"


class LightSwitchNode(Node):
    """ROS 2 node that publishes immutable runtime toggle events."""

    def __init__(self, publisher_config: Any):
        super().__init__("light_switch_node")
        self._state_pub = self.create_publisher(
            Float32MultiArray,
            publisher_config.topic,
            build_qos_profile(qos_config=publisher_config.qos),
        )
        self.get_logger().info("LightSwitchNode ready.")

    def publish_toggle(self, index: int, state: bool) -> None:
        """Publish one ``(switch_index, state)`` transition."""
        message = Float32MultiArray()
        message.data = [float(index), float(state)]
        self._state_pub.publish(message)


class LightSwitchExtension(omni.ext.IExt):
    """Pump ROS and publish new events from the shared fleet runtime snapshot."""

    def on_startup(self, ext_id: str):
        del ext_id
        carb.log_info(f"[{EXTENSION_TITLE}] Extension starting...")
        try:
            config_path = os.environ[_CONFIG_PATH_ENV]
        except KeyError as exc:
            raise RuntimeError(f"missing required environment variable {_CONFIG_PATH_ENV}") from exc
        publisher_config = load_config(config_path).switches.state_publisher
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._ros_node: LightSwitchNode | None = LightSwitchNode(publisher_config)
        self._last_sequence = 0
        self._update_event_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="light_switch_publish")
        )

    def on_shutdown(self):
        self._update_event_sub = None
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def _on_update(self, event):
        del event
        if self._ros_node is None:
            return
        rclpy.spin_once(self._ros_node, timeout_sec=0)
        try:
            from ExperimentRuntime_python.extension import get_instance

            events = get_instance().read_status().switch_events
        except RuntimeError:
            return
        for toggle in events:
            if toggle.sequence <= self._last_sequence:
                continue
            self._ros_node.publish_toggle(toggle.switch_index, toggle.state)
            self._last_sequence = toggle.sequence

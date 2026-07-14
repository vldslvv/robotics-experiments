"""ROS QoS configuration adapter for the light-switch publisher."""

from typing import Any

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

__all__ = ["build_qos_profile"]

_DURABILITY_POLICIES = {
    "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
    "volatile": DurabilityPolicy.VOLATILE,
}
_HISTORY_POLICIES = {
    "keep_last": HistoryPolicy.KEEP_LAST,
    "keep_all": HistoryPolicy.KEEP_ALL,
}
_RELIABILITY_POLICIES = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}


def _normalize_qos_name(*, name: str, field_name: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"QoS {field_name} must be a string, got {type(name).__name__}.")
    return name.strip().lower().replace("-", "_")


def _resolve_qos_policy(*, name: str, field_name: str, policies: dict[str, Any]) -> Any:
    key = _normalize_qos_name(name=name, field_name=field_name)
    try:
        return policies[key]
    except KeyError as exc:
        valid = ", ".join(sorted(policies))
        raise ValueError(f"Invalid QoS {field_name} {name!r}; expected one of: {valid}.") from exc


def build_qos_profile(*, qos_config: Any) -> QoSProfile:
    """Build a ROS 2 QoS profile from Hydra configuration."""
    if qos_config.depth <= 0:
        raise ValueError(f"QoS depth must be positive, got {qos_config.depth}.")
    return QoSProfile(
        depth=qos_config.depth,
        durability=_resolve_qos_policy(
            name=qos_config.durability,
            field_name="durability",
            policies=_DURABILITY_POLICIES,
        ),
        history=_resolve_qos_policy(
            name=qos_config.history,
            field_name="history",
            policies=_HISTORY_POLICIES,
        ),
        reliability=_resolve_qos_policy(
            name=qos_config.reliability,
            field_name="reliability",
            policies=_RELIABILITY_POLICIES,
        ),
    )

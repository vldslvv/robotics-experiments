"""Shared ULog access helpers for the flight-analysis tools."""

from __future__ import annotations

import math

import numpy as np
from pyulog import ULog

__all__ = ["dataset", "index_at", "value_at", "bool_value", "euler_deg"]


def dataset(ulog: ULog, name: str) -> dict:
    for item in ulog.data_list:
        if item.name == name:
            return item.data
    raise KeyError(f"ULog topic not found: {name}")


def index_at(data: dict, seconds: float) -> int:
    timestamp = np.asarray(data["timestamp"], dtype=np.float64)
    return int(np.argmin(np.abs(timestamp - seconds * 1_000_000.0)))


def value_at(data: dict, key: str, index: int, default: float = float("nan")) -> float:
    if key not in data:
        return default
    return float(data[key][index])


def bool_value(data: dict, key: str, index: int) -> int:
    if key not in data:
        return -1
    return int(data[key][index])


def euler_deg(q: list[float]) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))

"""Per-vehicle log-context plumbing for the PX4 bridge.

Stdlib-only (no pymavlink/Pegasus/Isaac), so the bridge layers and the
drone-spawner host can import it freely.

The PX4 bridge layers log through module-global loggers
(``logging.getLogger(__name__)``) that are shared across every vehicle in a
multi-vehicle process. Each vehicle's debug log is a separate file handler on
the shared ``robotics_experiments.px4_bridge`` package logger (see
``mavlink._configure_file_logging``). Without a discriminator every handler
would receive every vehicle's records, so a multi-drone scene merged all
bridges into one file. :func:`vehicle_log_context` marks the dynamic scope of a
single vehicle's bridge work, and :class:`VehicleLogFilter` (installed on each
per-vehicle handler) keeps each file to its own vehicle.

Records emitted outside any context (``vehicle id is None``) are passed by every
handler, so single-port direct usage and unit tests keep their full logs.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Iterator

__all__ = ["VehicleLogFilter", "vehicle_log_context"]

_CURRENT_VEHICLE_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "robotics_experiments_px4_bridge_vehicle_id", default=None
)


@contextmanager
def vehicle_log_context(vehicle_id: int) -> Iterator[None]:
    """Tag every px4_bridge log record emitted in this scope with ``vehicle_id``.

    The per-vehicle debug-log file handlers filter on this value so a
    multi-vehicle scene writes each vehicle's records only to its own file.
    """
    token = _CURRENT_VEHICLE_ID.set(int(vehicle_id))
    try:
        yield
    finally:
        _CURRENT_VEHICLE_ID.reset(token)


class VehicleLogFilter(logging.Filter):
    """Pass records tagged for this handler's vehicle (or untagged records)."""

    def __init__(self, vehicle_id: int):
        super().__init__()
        self._vehicle_id = int(vehicle_id)

    def filter(self, _record: logging.LogRecord) -> bool:
        current = _CURRENT_VEHICLE_ID.get()
        return current is None or current == self._vehicle_id

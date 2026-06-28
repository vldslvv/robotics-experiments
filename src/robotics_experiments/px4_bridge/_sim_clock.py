"""Sim-time clock for the PX4 bridge.

The bridge-facing ``sim_time_us`` is the most fragile invariant in the bridge:
it must be monotonic, expressed in integer microseconds, and derived from sim
time rather than wall clock (PX4 advances its own clock from
``HIL_SENSOR.time_usec``). It is also the value the MAVLink port pins its sensor
cadence to. Historically the counter, the settle offset, and the integer-tick
derivation lived as loose arithmetic in two places. This module owns all three
so the invariant is defined and enforced once.

Pure Python with no Isaac/MAVLink/Pegasus dependency, so it is importable and
unit-testable without a simulator.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["SimClock", "physics_tick_us"]


def physics_tick_us(physics_hz: float) -> int:
    """Return the integer-microsecond physics tick for a physics rate.

    This is the single source of the integer tick. Deriving the sensor rate as
    ``1e6 / tick_us`` (rather than trusting ``1 / physics_hz``) keeps the
    per-tick dt computed from integer-microsecond sim-time deltas exactly equal
    to the sensor period, which the MAVLink port relies on (see ``mavlink.py``).
    """
    if physics_hz <= 0:
        raise ValueError(f"physics_hz must be positive, got {physics_hz}")
    return int(round(1_000_000 / float(physics_hz)))


class SimClock:
    """Accumulates sim time in integer microseconds and guards monotonicity.

    One instance is created per bridge session by the scene-owning host. Each
    physics step calls :meth:`advance`; the host reads :meth:`is_settling` to
    decide whether the optional startup settle delay has elapsed, then takes
    :meth:`bridge_time_us` as the bridge-facing timestamp.
    """

    def __init__(self, *, physics_hz: float, settle_time_s: float = 0.0):
        """Build the clock.

        Args:
            physics_hz: Isaac physics cadence; sets :attr:`tick_us`.
            settle_time_s: Sim-time delay before the bridge-facing clock starts.
                The bridge-facing time is the total accumulated time minus this
                offset, so it is near zero on the first post-settle tick.
        """
        self._tick_us = physics_tick_us(physics_hz)
        self._settle_time_us = int(round(max(0.0, settle_time_s) * 1_000_000))
        self._total_us = 0
        self._tick_count = 0
        self._last_bridge_us: Optional[int] = None

    @property
    def tick_us(self) -> int:
        """Canonical integer-microsecond physics tick."""
        return self._tick_us

    @property
    def tick_count(self) -> int:
        """Number of physics steps advanced so far."""
        return self._tick_count

    @property
    def total_us(self) -> int:
        """Total accumulated sim time since construction, in microseconds."""
        return self._total_us

    @property
    def settle_time_us(self) -> int:
        """Configured startup settle offset, in microseconds."""
        return self._settle_time_us

    @property
    def is_settling(self) -> bool:
        """True while the accumulated time is still inside the settle window."""
        return self._total_us < self._settle_time_us

    def advance(self, dt: float) -> int:
        """Advance by one physics step of ``dt`` seconds; return total sim time."""
        self._total_us += int(round(dt * 1_000_000))
        self._tick_count += 1
        return self._total_us

    def bridge_time_us(self) -> int:
        """Return the bridge-facing timestamp (total minus the settle offset).

        Asserts the value never goes backwards, catching a time-base regression
        at the source instead of relying on a defensive check downstream in the
        MAVLink port.
        """
        bridge_us = self._total_us - self._settle_time_us
        if self._last_bridge_us is not None and bridge_us < self._last_bridge_us:
            raise AssertionError(
                f"SimClock bridge time went backwards: current={bridge_us} "
                f"last={self._last_bridge_us}"
            )
        self._last_bridge_us = bridge_us
        return bridge_us

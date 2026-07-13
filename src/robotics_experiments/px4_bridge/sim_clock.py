"""Frozen simulation-clock state and pure transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

__all__ = [
    "SimClockState",
    "advance_sim_clock",
    "bridge_time_us",
    "create_sim_clock",
    "sim_clock_is_settling",
]


def _physics_tick_us(physics_hz: float) -> int:
    if physics_hz <= 0:
        raise ValueError(f"physics_hz must be positive, got {physics_hz}")
    return int(round(1_000_000 / float(physics_hz)))


@dataclass(frozen=True)
class SimClockState:
    """Immutable integer-microsecond clock progress for one bridge session."""

    tick_us: int
    settle_time_us: int
    total_us: int = 0
    tick_count: int = 0
    last_bridge_us: Optional[int] = None


def create_sim_clock(*, physics_hz: float, settle_time_s: float = 0.0) -> SimClockState:
    """Create a clock at time zero."""
    return SimClockState(
        tick_us=_physics_tick_us(physics_hz),
        settle_time_us=int(round(max(0.0, settle_time_s) * 1_000_000)),
    )


def advance_sim_clock(*, state: SimClockState, dt: float) -> SimClockState:
    """Return the state after one physics step."""
    return replace(
        state,
        total_us=state.total_us + int(round(dt * 1_000_000)),
        tick_count=state.tick_count + 1,
    )


def sim_clock_is_settling(*, state: SimClockState) -> bool:
    """Return whether accumulated time remains inside the settle window."""
    return state.total_us < state.settle_time_us


def bridge_time_us(*, state: SimClockState) -> tuple[SimClockState, int]:
    """Return a state recording and the current bridge-facing timestamp."""
    current = state.total_us - state.settle_time_us
    if state.last_bridge_us is not None and current < state.last_bridge_us:
        raise AssertionError(
            f"SimClockState bridge time went backwards: current={current} "
            f"last={state.last_bridge_us}"
        )
    return replace(state, last_bridge_us=current), current

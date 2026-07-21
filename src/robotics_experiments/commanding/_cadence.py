"""Private exact-average cadence transition shared by cyclic executives."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__: list[str] = []

_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class _CadenceState:
    """Phase retained between source ticks, expressed in nanoseconds."""

    previous_time_ns: int | None = None
    remainder_ns: float = 0.0


def _advance_cadence(
    state: _CadenceState,
    *,
    time_ns: int,
    rate_hz: float,
) -> tuple[_CadenceState, bool]:
    """Advance to ``time_ns`` and emit no more than one update.

    Whole missed periods are discarded while their fractional remainder is
    retained. This prevents catch-up bursts after a delayed source tick without
    introducing long-run drift.
    """
    if state.previous_time_ns is None:
        return _CadenceState(previous_time_ns=time_ns), False
    if time_ns < state.previous_time_ns:
        raise ValueError(
            f"cadence time must be monotonic, got {time_ns} after {state.previous_time_ns}"
        )

    period_ns = _NANOSECONDS_PER_SECOND / rate_hz
    accumulated_ns = state.remainder_ns + (time_ns - state.previous_time_ns)
    due = accumulated_ns >= period_ns or math.isclose(
        accumulated_ns,
        period_ns,
        rel_tol=1e-12,
        abs_tol=1e-6,
    )
    remainder_ns = math.fmod(accumulated_ns, period_ns) if due else accumulated_ns
    if due and math.isclose(
        remainder_ns,
        period_ns,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        remainder_ns = 0.0
    return _CadenceState(time_ns, remainder_ns), due


def _next_interval_steps(
    state: _CadenceState,
    *,
    source_period_ns: int,
    rate_hz: float,
) -> tuple[_CadenceState, int]:
    """Return the source-step count through the next cadence emission."""
    if source_period_ns <= 0:
        raise ValueError(f"source_period_ns must be positive, got {source_period_ns}")
    current_time_ns = state.previous_time_ns
    if current_time_ns is None:
        current_time_ns = 0
        state = _CadenceState(previous_time_ns=current_time_ns)
    period_ns = _NANOSECONDS_PER_SECOND / rate_hz
    remaining_ns = period_ns - state.remainder_ns
    steps = max(1, math.ceil(remaining_ns / source_period_ns))
    advanced, due = _advance_cadence(
        state,
        time_ns=current_time_ns + steps * source_period_ns,
        rate_hz=rate_hz,
    )
    if not due:
        raise RuntimeError("cadence interval calculation did not reach an emission")
    return advanced, steps

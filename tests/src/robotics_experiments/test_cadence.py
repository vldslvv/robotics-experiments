"""Deterministic tests for the shared exact-average cadence transition."""

from robotics_experiments.commanding._cadence import (
    _advance_cadence,
    _CadenceState,
    _next_interval_steps,
)


def _count(*, source_rate_hz: int, target_rate_hz: float, seconds: int) -> int:
    state = _CadenceState(previous_time_ns=0)
    source_period_ns = 1_000_000_000 // source_rate_hz
    emitted = 0
    for tick in range(1, source_rate_hz * seconds + 1):
        state, due = _advance_cadence(
            state,
            time_ns=tick * source_period_ns,
            rate_hz=target_rate_hz,
        )
        emitted += int(due)
    return emitted


def test_integer_and_fractional_rate_ratios_have_exact_counts() -> None:
    assert _count(source_rate_hz=250, target_rate_hz=50.0, seconds=1) == 50
    assert _count(source_rate_hz=50, target_rate_hz=20.0, seconds=1) == 20
    assert _count(source_rate_hz=1000, target_rate_hz=73.0, seconds=10) == 730


def test_fractional_rate_has_no_long_run_drift() -> None:
    assert _count(source_rate_hz=1000, target_rate_hz=59.94, seconds=100) == 5_994


def test_delayed_tick_emits_once_without_catch_up_burst() -> None:
    state = _CadenceState(previous_time_ns=0)
    state, due = _advance_cadence(state, time_ns=1_010_000_000, rate_hz=50.0)
    assert due
    state, due = _advance_cadence(state, time_ns=1_010_000_000, rate_hz=50.0)
    assert not due
    state, due = _advance_cadence(state, time_ns=1_020_000_000, rate_hz=50.0)
    assert due


def test_next_policy_decimation_alternates_thirteen_and_twelve() -> None:
    state = _CadenceState(previous_time_ns=0)
    intervals = []
    for _ in range(6):
        state, steps = _next_interval_steps(
            state,
            source_period_ns=4_000_000,
            rate_hz=20.0,
        )
        intervals.append(steps)
    assert intervals == [13, 12, 13, 12, 13, 12]
    assert state.previous_time_ns == 300_000_000

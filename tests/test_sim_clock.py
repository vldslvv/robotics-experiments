"""Unit tests for the PX4 bridge sim-time clock."""

from __future__ import annotations

import pytest

from robotics_experiments.px4_bridge._sim_clock import SimClock, physics_tick_us


def test_physics_tick_us_rounds_to_integer_microseconds():
    assert physics_tick_us(250.0) == 4000
    # 1e6 / 333 is 3003.003..., which rounds to an integer-microsecond tick.
    assert physics_tick_us(333.0) == 3003


def test_physics_tick_us_rejects_non_positive_rate():
    with pytest.raises(ValueError, match="physics_hz must be positive"):
        physics_tick_us(0.0)
    with pytest.raises(ValueError, match="physics_hz must be positive"):
        physics_tick_us(-10.0)


def test_sim_clock_accumulates_and_reports_tick():
    clock = SimClock(physics_hz=250.0)
    assert clock.tick_us == 4000
    assert clock.total_us == 0
    assert clock.tick_count == 0

    assert clock.advance(0.004) == 4000
    assert clock.advance(0.004) == 8000
    assert clock.tick_count == 2
    assert clock.bridge_time_us() == 8000


def test_sim_clock_without_settle_is_never_settling():
    clock = SimClock(physics_hz=250.0)
    assert clock.settle_time_us == 0
    assert not clock.is_settling
    clock.advance(0.004)
    assert not clock.is_settling


def test_sim_clock_settle_window_and_post_settle_offset():
    # 12 ms settle at 250 Hz (4 ms ticks): settling for the first 3 ticks.
    clock = SimClock(physics_hz=250.0, settle_time_s=0.012)
    assert clock.settle_time_us == 12000

    clock.advance(0.004)  # 4000
    assert clock.is_settling
    clock.advance(0.004)  # 8000
    assert clock.is_settling
    clock.advance(0.004)  # 12000
    assert not clock.is_settling  # reached the window edge

    # The bridge-facing time subtracts the settle offset, so it is near zero
    # on the first post-settle tick.
    assert clock.bridge_time_us() == 0
    clock.advance(0.004)  # 16000
    assert clock.bridge_time_us() == 4000


def test_sim_clock_bridge_time_must_not_go_backwards():
    # advance() is the only way to move time, so monotonicity normally holds;
    # the assertion guards against a regression that mutates state out of band.
    clock = SimClock(physics_hz=250.0)
    clock.advance(0.004)
    assert clock.bridge_time_us() == 4000
    clock._total_us = 1000  # simulate a corrupted time-base
    with pytest.raises(AssertionError, match="went backwards"):
        clock.bridge_time_us()

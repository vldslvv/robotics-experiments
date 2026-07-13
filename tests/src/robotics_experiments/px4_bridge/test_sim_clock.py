"""Unit tests for the PX4 bridge sim-time clock."""

from __future__ import annotations

import pytest

from robotics_experiments.px4_bridge import (
    SimClockState,
    advance_sim_clock,
    bridge_time_us,
    create_sim_clock,
    sim_clock_is_settling,
)


def test_physics_tick_us_rounds_to_integer_microseconds():
    assert create_sim_clock(physics_hz=250.0).tick_us == 4000
    # 1e6 / 333 is 3003.003..., which rounds to an integer-microsecond tick.
    assert create_sim_clock(physics_hz=333.0).tick_us == 3003


def test_physics_tick_us_rejects_non_positive_rate():
    with pytest.raises(ValueError, match="physics_hz must be positive"):
        create_sim_clock(physics_hz=0.0)
    with pytest.raises(ValueError, match="physics_hz must be positive"):
        create_sim_clock(physics_hz=-10.0)


def test_sim_clock_accumulates_and_reports_tick():
    clock = create_sim_clock(physics_hz=250.0)
    assert clock.tick_us == 4000
    assert clock.total_us == 0
    assert clock.tick_count == 0

    initial = clock
    clock = advance_sim_clock(state=clock, dt=0.004)
    assert clock is not initial
    assert initial.total_us == 0
    assert clock.total_us == 4000
    clock = advance_sim_clock(state=clock, dt=0.004)
    assert clock.total_us == 8000
    assert clock.tick_count == 2
    clock, current = bridge_time_us(state=clock)
    assert current == 8000


def test_sim_clock_without_settle_is_never_settling():
    clock = create_sim_clock(physics_hz=250.0)
    assert clock.settle_time_us == 0
    assert not sim_clock_is_settling(state=clock)
    clock = advance_sim_clock(state=clock, dt=0.004)
    assert not sim_clock_is_settling(state=clock)


def test_sim_clock_settle_window_and_post_settle_offset():
    # 12 ms settle at 250 Hz (4 ms ticks): settling for the first 3 ticks.
    clock = create_sim_clock(physics_hz=250.0, settle_time_s=0.012)
    assert clock.settle_time_us == 12000

    clock = advance_sim_clock(state=clock, dt=0.004)  # 4000
    assert sim_clock_is_settling(state=clock)
    clock = advance_sim_clock(state=clock, dt=0.004)  # 8000
    assert sim_clock_is_settling(state=clock)
    clock = advance_sim_clock(state=clock, dt=0.004)  # 12000
    assert not sim_clock_is_settling(state=clock)  # reached the window edge

    # The bridge-facing time subtracts the settle offset, so it is near zero
    # on the first post-settle tick.
    clock, current = bridge_time_us(state=clock)
    assert current == 0
    clock = advance_sim_clock(state=clock, dt=0.004)  # 16000
    clock, current = bridge_time_us(state=clock)
    assert current == 4000


def test_sim_clock_bridge_time_must_not_go_backwards():
    # advance() is the only way to move time, so monotonicity normally holds;
    # the assertion guards against a regression that mutates state out of band.
    clock = advance_sim_clock(state=create_sim_clock(physics_hz=250.0), dt=0.004)
    clock, current = bridge_time_us(state=clock)
    assert current == 4000
    clock = SimClockState(
        tick_us=clock.tick_us,
        settle_time_us=clock.settle_time_us,
        total_us=1000,
        tick_count=clock.tick_count,
        last_bridge_us=clock.last_bridge_us,
    )
    with pytest.raises(AssertionError, match="went backwards"):
        bridge_time_us(state=clock)

"""Pure per-drone RoutineDriver tests: the phase machine over fake seams.

Drive a :class:`RoutineDriver` with a :class:`FakeCommandPort` and a
:class:`FakeDriverEnvironment` through ``WAIT_BRIDGE..DONE`` without Docker,
Isaac, or pymavlink, and exercise the failure reactions (envelope / command /
failsafe / timeout) and the per-step advance callback.
"""

from _fakes import FakeCommandPort, FakeDriverEnvironment, make_routine_observation

from robotics_experiments.commanding import (
    DriverParams,
    FailureKind,
    RoutineDriver,
)
from robotics_experiments.commanding._commands import _MAV_CMD_COMPONENT_ARM_DISARM
from robotics_experiments.commanding._driver import _DriverPhase
from robotics_experiments.commanding._observation import CommandAck
from robotics_experiments.commanding._routine import _EnvelopeGates

_GATES = _EnvelopeGates(
    max_tilt_deg=35.0,
    max_lateral_drift_m=2.0,
    max_vertical_speed_m_s=8.0,
    max_downward_speed_m_s=-4.0,
    max_angular_speed_rad_s=8.0,
    hover_max_abs_vertical_speed_m_s=0.35,
    hover_max_abs_horizontal_speed_m_s=0.30,
)


def _params(steps):
    return DriverParams(
        steps=steps,
        gates=_GATES,
        command_ack_timeout_s=10.0,
        bridge_heartbeat_timeout_s=12.0,
        handshake_timeout_s=12.0,
        prearm_settle_s=0.0,
        takeoff_target_timeout_s=30.0,
        takeoff_timeout_s=30.0,
        hover_settle_s=2.0,
    )


_STEPS = [
    {"kind": "arm"},
    {"kind": "takeoff", "climb_m": 2.5, "min_climb_m": 1.0},
    {"kind": "hover", "hold_s": 1.0},
]


def _make_driver(*, raise_on_failure=False):
    port = FakeCommandPort()
    env = FakeDriverEnvironment(raise_on_failure=raise_on_failure)
    clock = {"now": 0.0}  # frozen wall clock: phase deadlines never fire
    driver = RoutineDriver(_params(_STEPS), port, env, label="d0", monotonic=lambda: clock["now"])
    return driver, port, env, clock


def test_routine_driver_runs_arm_takeoff_hover_to_done():
    driver, port, env, _clock = _make_driver()

    # WAIT_BRIDGE: holds until the bridge connects.
    driver.tick()
    assert driver.phase is _DriverPhase.WAIT_BRIDGE
    env.connected = True
    driver.tick()
    assert driver.phase is _DriverPhase.CONNECT

    # CONNECT opens the port; HANDSHAKE holds until the port handshakes.
    driver.tick()
    assert driver.phase is _DriverPhase.HANDSHAKE
    assert port.connected
    driver.tick()
    assert driver.phase is _DriverPhase.HANDSHAKE
    port.handshaked = True
    driver.tick()
    assert driver.phase is _DriverPhase.PREARM_SETTLE

    # PREARM_SETTLE clears immediately (prearm_settle_s=0); FETCH_TARGET holds
    # until a global position is available.
    driver.tick()
    assert driver.phase is _DriverPhase.FETCH_TARGET
    driver.tick()
    assert driver.phase is _DriverPhase.FETCH_TARGET
    port.global_position = (47.0, 8.0, 488.0)
    driver.tick()
    assert driver.phase is _DriverPhase.BUILD

    # BUILD assembles the routine/commander; then DRIVE.
    driver.tick()
    assert driver.phase is _DriverPhase.DRIVE

    # DRIVE: _arm SEND, then the _arm ack advances to the _takeoff step.
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.0))
    driver.tick()
    assert [c.label for c in port.sent] == ["arm"]
    port.feedback = [CommandAck.from_result(_MAV_CMD_COMPONENT_ARM_DISARM, 0)]
    driver.tick()
    assert env.advanced_steps == ["arm"]

    # Takeoff SEND, then a real climb advances past it.
    driver.tick()
    assert [c.label for c in port.sent] == ["arm", "takeoff"]
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.2))
    driver.tick()  # not climbed enough -> still _takeoff
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 2.0))
    driver.tick()
    assert env.advanced_steps == ["arm", "takeoff"]

    # Hover: command-less; holds a stable hover for hold_s (sim time) then done.
    env.observation = make_routine_observation(
        position_enu=(0.0, 0.0, 2.0), linear_velocity_enu=(0.0, 0.0, 0.0)
    )
    env.sim_us = 100_000
    driver.tick()  # hover begins
    driver.tick()  # stable; hold window starts
    env.sim_us = 100_000 + 1_000_000  # hold_s=1.0 elapsed
    driver.tick()
    assert env.advanced_steps == ["arm", "takeoff", "hover"]
    assert driver.is_done
    assert env.failures == []


def _drive_to_takeoff(driver, port, env):
    """Advance a fresh driver to the _takeoff SEND in the DRIVE phase."""
    env.connected = True
    port.handshaked = True
    port.global_position = (47.0, 8.0, 488.0)
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.0))
    for _ in range(20):
        driver.tick()
        if driver.phase is _DriverPhase.DRIVE and any(c.label == "arm" for c in port.sent):
            break
    # advance the _arm step
    port.feedback = [CommandAck.from_result(_MAV_CMD_COMPONENT_ARM_DISARM, 0)]
    driver.tick()
    driver.tick()  # _takeoff SEND


def test_routine_driver_envelope_breach_fails():
    driver, port, env = _make_driver()[:3]
    _drive_to_takeoff(driver, port, env)
    # A hard tilt during the _takeoff step is an ENVELOPE breach.
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.5), tilt_deg=50.0)
    driver.tick()
    assert driver.is_done
    assert len(env.failures) == 1
    kind, _message = env.failures[0]
    assert kind is FailureKind.ENVELOPE


def test_routine_driver_failsafe_fails_distinctly():
    driver, port, env = _make_driver()[:3]
    _drive_to_takeoff(driver, port, env)
    port.failsafe_flag = True
    port.system_status = 6
    driver.tick()
    assert driver.is_done
    assert env.failures and env.failures[0][0] == "failsafe"


def test_routine_driver_phase_timeout_fails():
    driver, port, env, clock = _make_driver()
    # Never connect the bridge; push the wall clock past the WAIT_BRIDGE deadline.
    clock["now"] = 10_000.0
    driver.tick()
    assert driver.is_done
    assert env.failures and env.failures[0][0] == "timeout"


def test_routine_driver_strict_env_raises_on_failure():
    driver, port, env = _make_driver(raise_on_failure=True)[:3]
    _drive_to_takeoff(driver, port, env)
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.5), tilt_deg=50.0)
    try:
        driver.tick()
    except AssertionError as exc:
        assert "ENVELOPE" in str(exc)
    else:  # pragma: no cover - the strict env must raise
        raise AssertionError("strict env did not raise on an envelope breach")

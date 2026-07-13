"""Pure per-drone routine-driver tests: the phase machine over fake seams.

Drive a frozen driver state with a :class:`FakeCommandPort` and a
:class:`FakeDriverEnvironment` through ``WAIT_BRIDGE..DONE`` without Docker,
Isaac, or pymavlink, and exercise the failure reactions (envelope / command /
failsafe / timeout) and the per-step advance callback.
"""

from robotics_experiments.commanding import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_SET_MODE,
    CommandAck,
    DriverParams,
    DriverPhase,
    EnvelopeGates,
    FailureKind,
    VelocitySetpoint,
    create_routine_driver,
    routine_driver_is_done,
    tick_routine_driver,
)
from tests.support.commanding_fakes import (
    FakeCommandPort,
    FakeDriverEnvironment,
    make_routine_observation,
)

_GATES = EnvelopeGates(
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
        goto_position_tol_m=0.5,
        goto_heading_tol_deg=25.0,
        goto_cross_track_m=2.0,
        goto_offboard_prime_ticks=3,
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
    driver = create_routine_driver(
        params=_params(_STEPS),
        port=port,
        env=env,
        label="d0",
        monotonic=lambda: clock["now"],
    )
    return driver, port, env, clock


def test_routine_driver_runs_arm_takeoff_hover_to_done():
    driver, port, env, _clock = _make_driver()

    # WAIT_BRIDGE: holds until the bridge connects.
    driver = tick_routine_driver(state=driver)
    assert driver.phase is DriverPhase.WAIT_BRIDGE
    env.connected = True
    waiting = driver
    driver = tick_routine_driver(state=driver)
    assert driver is not waiting
    assert waiting.phase is DriverPhase.WAIT_BRIDGE
    assert driver.phase is DriverPhase.CONNECT

    # CONNECT opens the port; HANDSHAKE holds until the port handshakes.
    driver = tick_routine_driver(state=driver)
    assert driver.phase is DriverPhase.HANDSHAKE
    assert port.connected
    driver = tick_routine_driver(state=driver)
    assert driver.phase is DriverPhase.HANDSHAKE
    port.set_handshaked()
    driver = tick_routine_driver(state=driver)
    assert driver.phase is DriverPhase.PREARM_SETTLE

    # PREARM_SETTLE clears immediately (prearm_settle_s=0); FETCH_TARGET holds
    # until a global position is available.
    driver = tick_routine_driver(state=driver)
    assert driver.phase is DriverPhase.FETCH_TARGET
    driver = tick_routine_driver(state=driver)
    assert driver.phase is DriverPhase.FETCH_TARGET
    port.set_global_position((47.0, 8.0, 488.0))
    driver = tick_routine_driver(state=driver)
    assert driver.phase is DriverPhase.BUILD

    # BUILD assembles the routine/commander; then DRIVE.
    driver = tick_routine_driver(state=driver)
    assert driver.phase is DriverPhase.DRIVE

    # DRIVE: arm SEND, then the arm ack advances to the takeoff step.
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.0))
    driver = tick_routine_driver(state=driver)
    assert [c.label for c in port.sent] == ["arm"]
    port.feedback = [CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, 0)]
    driver = tick_routine_driver(state=driver)
    assert env.advanced_steps == ["arm"]

    # Takeoff SEND, then a real climb advances past it.
    driver = tick_routine_driver(state=driver)
    assert [c.label for c in port.sent] == ["arm", "takeoff"]
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.2))
    driver = tick_routine_driver(state=driver)  # not climbed enough -> still takeoff
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 2.0))
    driver = tick_routine_driver(state=driver)
    assert env.advanced_steps == ["arm", "takeoff"]

    # Hover: command-less; holds a stable hover for hold_s (sim time) then done.
    env.observation = make_routine_observation(
        position_enu=(0.0, 0.0, 2.0), linear_velocity_enu=(0.0, 0.0, 0.0)
    )
    env.sim_us = 100_000
    driver = tick_routine_driver(state=driver)  # hover begins
    driver = tick_routine_driver(state=driver)  # stable; hold window starts
    env.sim_us = 100_000 + 1_000_000  # hold_s=1.0 elapsed
    driver = tick_routine_driver(state=driver)
    assert env.advanced_steps == ["arm", "takeoff", "hover"]
    assert routine_driver_is_done(state=driver)
    assert env.failures == []


def test_empty_routine_runs_generic_lifecycle_and_services_terminal_gcs():
    port = FakeCommandPort()
    env = FakeDriverEnvironment()
    env.connected = True
    port.set_handshaked()
    port.set_global_position((47.0, 8.0, 488.0))
    driver = create_routine_driver(
        params=_params([]), port=port, env=env, label="d0", monotonic=lambda: 0.0
    )

    phases = []
    for _ in range(7):
        driver = tick_routine_driver(state=driver)
        phases.append(driver.phase)

    assert phases == [
        DriverPhase.CONNECT,
        DriverPhase.HANDSHAKE,
        DriverPhase.PREARM_SETTLE,
        DriverPhase.FETCH_TARGET,
        DriverPhase.BUILD,
        DriverPhase.DRIVE,
        DriverPhase.DONE,
    ]
    assert port.sent == port.setpoints == port.velocity_setpoints == port.attitude_targets == []
    servicing = (port.heartbeats, port.feedback_polls)
    terminal = tick_routine_driver(state=driver)
    assert terminal is driver
    assert (port.heartbeats, port.feedback_polls) == (servicing[0] + 1, servicing[1] + 1)


def _drive_to_takeoff(driver, port, env):
    """Advance a fresh driver to the takeoff SEND in the DRIVE phase."""
    env.connected = True
    port.set_handshaked()
    port.set_global_position((47.0, 8.0, 488.0))
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.0))
    for _ in range(20):
        driver = tick_routine_driver(state=driver)
        if driver.phase is DriverPhase.DRIVE and any(c.label == "arm" for c in port.sent):
            break
    # advance the arm step
    port.feedback = [CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, 0)]
    driver = tick_routine_driver(state=driver)
    driver = tick_routine_driver(state=driver)  # takeoff SEND
    return driver


def test_routine_driver_envelope_breach_fails():
    driver, port, env = _make_driver()[:3]
    driver = _drive_to_takeoff(driver, port, env)
    # A hard tilt during the takeoff step is an ENVELOPE breach.
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.5), tilt_deg=50.0)
    driver = tick_routine_driver(state=driver)
    assert routine_driver_is_done(state=driver)
    assert len(env.failures) == 1
    kind, _message = env.failures[0]
    assert kind is FailureKind.ENVELOPE


def test_routine_driver_failsafe_fails_distinctly():
    driver, port, env = _make_driver()[:3]
    driver = _drive_to_takeoff(driver, port, env)
    port.set_failsafe(active=True, system_status=6)
    driver = tick_routine_driver(state=driver)
    assert routine_driver_is_done(state=driver)
    assert env.failures and env.failures[0][0] is FailureKind.FAILSAFE


def test_routine_driver_phase_timeout_fails():
    driver, port, env, clock = _make_driver()
    # Never connect the bridge; push the wall clock past the WAIT_BRIDGE deadline.
    clock["now"] = 10_000.0
    driver = tick_routine_driver(state=driver)
    assert routine_driver_is_done(state=driver)
    assert env.failures and env.failures[0][0] is FailureKind.TIMEOUT


def test_routine_driver_strict_env_raises_on_failure():
    driver, port, env = _make_driver(raise_on_failure=True)[:3]
    driver = _drive_to_takeoff(driver, port, env)
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 0.5), tilt_deg=50.0)
    try:
        tick_routine_driver(state=driver)
    except AssertionError as exc:
        assert "ENVELOPE" in str(exc)
    else:  # pragma: no cover - the strict env must raise
        raise AssertionError("strict env did not raise on an envelope breach")


_GOTO_STEPS = [{"kind": "goto", "x": 3.0, "y": 0.0, "z": 2.0, "timeout_s": 20.0}]


def test_routine_driver_goto_streams_setpoints_and_engages_offboard():
    port = FakeCommandPort()
    env = FakeDriverEnvironment()
    driver = create_routine_driver(
        params=_params(_GOTO_STEPS), port=port, env=env, label="d0", monotonic=lambda: 0.0
    )

    env.connected = True
    port.set_handshaked()
    port.set_global_position((47.0, 8.0, 488.0))
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 2.0))
    for _ in range(20):
        driver = tick_routine_driver(state=driver)
        if driver.phase is DriverPhase.DRIVE:
            break
    assert driver.phase is DriverPhase.DRIVE

    # prime_ticks=3: setpoints stream, then a single OFFBOARD-mode command, then
    # the stream continues -- the driver routes STREAM to port.send_setpoint.
    for _ in range(6):
        driver = tick_routine_driver(state=driver)
    assert len(port.setpoints) >= 3
    assert [c.label for c in port.sent] == ["set_offboard_mode"]

    # Reach the waypoint facing the leg direction (+east) -> routine completes.
    env.observation = make_routine_observation(position_enu=(3.0, 0.0, 2.0), yaw_deg=0.0)
    driver = tick_routine_driver(state=driver)
    assert env.advanced_steps == ["goto[3.0,0.0,2.0]"]
    assert routine_driver_is_done(state=driver)
    assert env.failures == []


_POLICY_STEPS = [
    {"kind": "policy", "artifact": "does/not/matter", "hold_s": 1.0, "timeout_s": 10.0}
]


def _fake_policy_fn(vehicle_state, target_enu):
    return VelocitySetpoint(forward_mps=0.2, right_mps=0.0, down_mps=0.0, yaw_rate_radps=0.0)


def test_routine_driver_policy_streams_velocity_setpoints_and_holds_to_done():
    port = FakeCommandPort()
    env = FakeDriverEnvironment()
    params = DriverParams(
        steps=_POLICY_STEPS,
        gates=_GATES,
        command_ack_timeout_s=10.0,
        bridge_heartbeat_timeout_s=12.0,
        handshake_timeout_s=12.0,
        prearm_settle_s=0.0,
        takeoff_target_timeout_s=30.0,
        takeoff_timeout_s=30.0,
        hover_settle_s=2.0,
        goto_position_tol_m=0.5,
        goto_heading_tol_deg=25.0,
        goto_cross_track_m=2.0,
        goto_offboard_prime_ticks=3,
        policy_fn_factory=lambda descriptor: _fake_policy_fn,
    )
    driver = create_routine_driver(
        params=params, port=port, env=env, label="d0", monotonic=lambda: 0.0
    )

    env.connected = True
    port.set_handshaked()
    port.set_global_position((47.0, 8.0, 488.0))
    env.observation = make_routine_observation(sim_time_us=0, position_enu=(0.0, 0.0, 2.0))
    for _ in range(20):
        driver = tick_routine_driver(state=driver)
        if driver.phase is DriverPhase.DRIVE:
            break
    assert driver.phase is DriverPhase.DRIVE

    # prime_ticks=3: the policy's velocity payloads stream (dispatched to
    # port.send_velocity_setpoint), then the single OFFBOARD-mode command.
    for _ in range(6):
        driver = tick_routine_driver(state=driver)
    assert len(port.velocity_setpoints) >= 3
    assert port.setpoints == []
    assert [c.label for c in port.sent] == ["set_offboard_mode"]

    # It does not complete before PX4 accepts OFFBOARD, even after hold_s elapses.
    env.sim_us = 2_000_000
    driver = tick_routine_driver(state=driver)
    assert env.advanced_steps == []
    assert not routine_driver_is_done(state=driver)

    # hold_s of calm in-envelope sim time after OFFBOARD acceptance -> complete.
    port.feedback.append(CommandAck.from_result(MAV_CMD_DO_SET_MODE, 0))
    driver = tick_routine_driver(state=driver)
    env.sim_us = 4_000_000
    driver = tick_routine_driver(state=driver)
    assert env.advanced_steps == ["policy"]
    assert routine_driver_is_done(state=driver)
    assert env.failures == []


def test_routine_driver_policy_step_without_factory_fails_at_build():
    port = FakeCommandPort()
    env = FakeDriverEnvironment()
    driver = create_routine_driver(
        params=_params(_POLICY_STEPS),
        port=port,
        env=env,
        label="d0",
        monotonic=lambda: 0.0,
    )

    env.connected = True
    port.set_handshaked()
    port.set_global_position((47.0, 8.0, 488.0))
    env.observation = make_routine_observation(position_enu=(0.0, 0.0, 2.0))
    for _ in range(20):
        driver = tick_routine_driver(state=driver)
        if driver.phase is DriverPhase.DONE:
            break
    assert env.failures
    kind, message = env.failures[0]
    assert "policy_fn_factory" in message

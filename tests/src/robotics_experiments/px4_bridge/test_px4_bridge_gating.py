import sys
from dataclasses import replace

import numpy as np
import pytest

from robotics_experiments.px4_bridge import (
    VehicleState,
    create_pegasus_thrust_model,
    create_px4_bridge,
    recv_and_apply,
    send_sensors,
    start_px4_bridge,
    stop_px4_bridge,
    tick_px4_bridge,
    tilt_angle_deg,
)
from tests.support.bridge_fakes import (
    _CapturingThrustModel,
    _FakeArticulationIO,
    _FakeFlightStack,
)
from tests.support.fakes import (
    _quat_pitch_deg,
    _quat_yaw_deg,
)


class _LinearThrustModel:
    def __init__(self, *, num_rotors: int, max_force_n: float):
        self.num_rotors = int(num_rotors)
        self._max_force_n = float(max_force_n)

    def commands_to_forces(self, commands):
        controls = np.clip(np.asarray(commands.controls, dtype=np.float64), 0.0, 1.0)
        return controls * self._max_force_n

    def commands_to_dynamics(self, commands, *, linear_body_velocity_flu=None):
        from robotics_experiments.px4_bridge import RotorDynamics

        return RotorDynamics(
            forces_n=self.commands_to_forces(commands),
            body_torque_n_m=np.zeros(3, dtype=np.float64),
            body_drag_n=np.zeros(3, dtype=np.float64),
            rotor_omega_rad_s=np.zeros(self.num_rotors, dtype=np.float64),
        )


def test_px4_bridge_tick_reads_state_and_applies_forces():
    state = VehicleState.hover_at(sim_time_us=0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
    )

    created = bridge
    bridge = start_px4_bridge(state=bridge)
    assert bridge is not created
    started = bridge
    bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    assert bridge is not started
    assert started.tick_count == 0 and started.pending is None
    assert io.is_initialized() is True
    bridge = stop_px4_bridge(state=bridge)

    assert port.started is True
    assert port.stopped is True
    assert io.is_initialized() is False
    assert io.shutdown_called is True
    assert port.states[0].sim_time_us == 4_000
    assert np.allclose(io.applied_forces[0], [0.0, 2.0, 4.0, 8.0])


def test_recv_and_apply_rejects_state_without_paired_send():
    io = _FakeArticulationIO(VehicleState.hover_at(sim_time_us=0))
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=_FakeFlightStack([0.0, 0.25, 0.5, 1.0]),
        articulation_io=io,
    )

    with pytest.raises(RuntimeError, match="without a paired send_sensors"):
        recv_and_apply(state=bridge)

    assert io.applied_dynamics == []


def test_receive_exception_leaves_frozen_input_for_caller_to_discard(monkeypatch):
    io = _FakeArticulationIO(VehicleState.hover_at(sim_time_us=0))
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
    )
    bridge = start_px4_bridge(state=bridge)
    sent = send_sensors(state=bridge, dt=0.004, sim_time_us=4_000)

    def fail_poll():
        raise RuntimeError("injected poll failure")

    monkeypatch.setattr(port, "poll", fail_poll)

    with pytest.raises(RuntimeError, match="injected poll failure"):
        recv_and_apply(state=sent)

    assert sent.pending is not None
    assert io.applied_dynamics == []


def test_px4_bridge_derives_drag_velocity_from_world_velocity():
    yaw_90_wxyz = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float64)
    state = replace(
        VehicleState.hover_at(sim_time_us=0),
        orientation_flu_enu_wxyz=yaw_90_wxyz,
        linear_velocity_enu=np.array([0.0, 3.0, 0.0], dtype=np.float64),
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.0, 0.0, 0.0])
    thrust_model = _CapturingThrustModel(num_rotors=4)
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=thrust_model,
        port=port,
        articulation_io=io,
    )

    bridge = start_px4_bridge(state=bridge)
    bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    bridge = stop_px4_bridge(state=bridge)

    np.testing.assert_allclose(
        thrust_model.captured_body_velocity_flu,
        [3.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_px4_bridge_validates_rotor_count_against_port():
    with pytest.raises(ValueError, match="rotor_link_paths has 3 entries"):
        create_px4_bridge(
            articulation_path="/World/Drone",
            rotor_link_paths=["m1", "m2", "m3"],
            thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
            port=_FakeFlightStack([0.0, 0.0, 0.0, 0.0]),
            articulation_io=_FakeArticulationIO(VehicleState.hover_at()),
        )


def test_px4_bridge_passes_visual_spin_signs_to_default_isaac_io(monkeypatch):
    captured = {}

    class FakeIsaacArticulationIO:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        sys.modules["robotics_experiments.px4_bridge.isaac_articulation"],
        "IsaacArticulationIO",
        FakeIsaacArticulationIO,
    )

    create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=_FakeFlightStack([0.0, 0.0, 0.0, 0.0]),
        rotor_spin_signs=[-1.0, 1.0, -1.0, 1.0],
    )

    assert captured["rotor_spin_signs"] == [-1.0, 1.0, -1.0, 1.0]


def test_tilt_angle_deg_level():
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert tilt_angle_deg(q) == pytest.approx(0.0, abs=1e-10)


def test_tilt_angle_deg_45_pitch():
    q = _quat_pitch_deg(45.0)
    assert tilt_angle_deg(q) == pytest.approx(45.0, abs=1e-10)


def test_tilt_angle_deg_inverted():
    q = _quat_pitch_deg(180.0)
    assert tilt_angle_deg(q) == pytest.approx(180.0, abs=1e-10)


def test_tilt_angle_deg_pure_yaw_is_zero():
    q = _quat_yaw_deg(90.0)
    assert tilt_angle_deg(q) == pytest.approx(0.0, abs=1e-10)


def test_bridge_tilt_cutoff_zeros_forces():
    state = replace(
        VehicleState.hover_at(sim_time_us=0), orientation_flu_enu_wxyz=_quat_pitch_deg(80.0)
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.5, 0.5, 0.5, 0.5])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge = start_px4_bridge(state=bridge)
    bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    bridge = stop_px4_bridge(state=bridge)

    # Forces are zeroed, but the MAVLink exchange must still run so PX4
    # lockstep keeps advancing (the sensor tick is not dropped).
    assert len(port.states) == 1
    assert len(io.applied_forces) == 1
    # Forces zeroed is the observable gating effect of the tilt cutoff firing.
    assert np.allclose(io.applied_forces[0], [0.0, 0.0, 0.0, 0.0])
    cutoff_dynamics = io.applied_dynamics[0]
    for array in (
        cutoff_dynamics.forces_n,
        cutoff_dynamics.body_torque_n_m,
        cutoff_dynamics.body_drag_n,
    ):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 1.0


def test_bridge_tilt_cutoff_sustained_preserves_lockstep():
    state = replace(
        VehicleState.hover_at(sim_time_us=0), orientation_flu_enu_wxyz=_quat_pitch_deg(80.0)
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.5, 0.5, 0.5, 0.5])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge = start_px4_bridge(state=bridge)
    sim_times = [4_000, 8_000, 12_000, 16_000, 20_000]
    for t in sim_times:
        bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=t)
    bridge = stop_px4_bridge(state=bridge)

    # Every tick exchanged sensors with strictly increasing sim time, while all
    # applied forces stayed zero throughout the sustained tilt excursion.
    assert port.sim_times == sim_times
    assert all(b > a for a, b in zip(port.sim_times, port.sim_times[1:], strict=False))
    assert len(io.applied_forces) == len(sim_times)
    # Every tick's forces stayed zeroed: the cutoff fired on all of them.
    for forces in io.applied_forces:
        assert np.allclose(forces, [0.0, 0.0, 0.0, 0.0])


def test_bridge_nonfinite_state_raises_without_exchange():
    state = replace(
        VehicleState.hover_at(sim_time_us=0),
        position_enu=np.array([0.0, np.nan, 0.0], dtype=np.float64),
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.5, 0.5, 0.5, 0.5])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge = start_px4_bridge(state=bridge)
    with pytest.raises(RuntimeError):
        tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    bridge = stop_px4_bridge(state=bridge)

    # No NaN forwarded into the lockstep exchange, and no forces applied.
    assert port.states == []
    assert io.applied_forces == []


def test_bridge_tilt_cutoff_passes_through_within_limit():
    state = replace(
        VehicleState.hover_at(sim_time_us=0), orientation_flu_enu_wxyz=_quat_pitch_deg(30.0)
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge = start_px4_bridge(state=bridge)
    bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    bridge = stop_px4_bridge(state=bridge)

    assert len(io.applied_forces) == 1
    # Within the limit the cutoff does not fire: forces pass through unchanged.
    assert np.allclose(io.applied_forces[0], [0.0, 2.0, 4.0, 8.0])


def test_bridge_tilt_cutoff_disabled_by_default():
    # The default max_tilt_angle_deg of 180.0 is the maximum possible tilt, so
    # the cutoff never fires: a 170 deg pitch must still apply forces.
    state = replace(
        VehicleState.hover_at(sim_time_us=0), orientation_flu_enu_wxyz=_quat_pitch_deg(170.0)
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
    )

    bridge = start_px4_bridge(state=bridge)
    bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    bridge = stop_px4_bridge(state=bridge)

    assert len(io.applied_forces) == 1
    # The cutoff is effectively disabled: a 170 deg pitch still applies forces.
    assert np.allclose(io.applied_forces[0], [0.0, 2.0, 4.0, 8.0])


def test_bridge_forwards_rotor_omega_to_visual_spin_seam():
    state = VehicleState.hover_at(sim_time_us=0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([100.0, 500.0, 1100.0, 1500.0])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=create_pegasus_thrust_model(num_rotors=4),
        port=port,
        articulation_io=io,
    )

    bridge = start_px4_bridge(state=bridge)
    bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    bridge = stop_px4_bridge(state=bridge)

    # The cosmetic spin seam is fed the clipped per-rotor omega and the tick dt.
    assert len(io.spin_updates) == 1
    omega, dt = io.spin_updates[0]
    np.testing.assert_allclose(omega, np.clip([100.0, 500.0, 1100.0, 1500.0], 0.0, 1100.0))
    assert dt == pytest.approx(0.004)


def test_bridge_visual_spin_uses_airframe_max_rotor_velocity():
    state = VehicleState.hover_at(sim_time_us=0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([1100.0, 1100.0, 1100.0, 1100.0])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=create_pegasus_thrust_model(
            num_rotors=4,
            max_rotor_velocity=[525.0, 525.0, 525.0, 525.0],
        ),
        port=port,
        articulation_io=io,
    )

    bridge = start_px4_bridge(state=bridge)
    bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    bridge = stop_px4_bridge(state=bridge)

    omega, _dt = io.spin_updates[0]
    np.testing.assert_allclose(omega, [525.0, 525.0, 525.0, 525.0])
    assert np.all(omega < 1100.0)


def test_bridge_tilt_cutoff_still_spins_rotors_with_real_omega():
    # Forces are gated to zero on a tilt cutoff, but the rotors keep spinning at
    # the real commanded omega: visual spin is decoupled from the force gate.
    state = replace(
        VehicleState.hover_at(sim_time_us=0), orientation_flu_enu_wxyz=_quat_pitch_deg(80.0)
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([700.0, 700.0, 700.0, 700.0])
    bridge = create_px4_bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=create_pegasus_thrust_model(num_rotors=4),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge = start_px4_bridge(state=bridge)
    bridge = tick_px4_bridge(state=bridge, dt=0.004, sim_time_us=4_000)
    bridge = stop_px4_bridge(state=bridge)

    # Forces gated to zero (cutoff fired), but the rotors still spin (next assert).
    assert np.allclose(io.applied_forces[0], [0.0, 0.0, 0.0, 0.0])
    assert len(io.spin_updates) == 1
    omega, _dt = io.spin_updates[0]
    np.testing.assert_allclose(omega, [700.0, 700.0, 700.0, 700.0])

from dataclasses import replace

import numpy as np
import pytest
from _fakes import (
    _CapturingThrustModel,
    _FakeArticulationIO,
    _FakeFlightStack,
    _quat_pitch_deg,
    _quat_yaw_deg,
)

from robotics_experiments.px4_bridge import (
    PegasusQuadraticThrustModel,
    Px4Bridge,
)
from robotics_experiments.px4_bridge.thrust_model import _LinearThrustModel
from robotics_experiments.px4_bridge.vehicle_state import VehicleState


def test_px4_bridge_tick_reads_state_and_applies_forces():
    state = VehicleState.hover_at(sim_time_us=0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    assert port.started is True
    assert port.stopped is True
    assert io.initialized is True
    assert io.shutdown_called is True
    assert port.states[0].sim_time_us == 4_000
    assert np.allclose(io.applied_forces[0], [0.0, 2.0, 4.0, 8.0])


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
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=thrust_model,
        port=port,
        articulation_io=io,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    np.testing.assert_allclose(
        thrust_model.captured_body_velocity_flu,
        [3.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_px4_bridge_validates_rotor_count_against_port():
    with pytest.raises(ValueError, match="rotor_link_paths has 3 entries"):
        Px4Bridge(
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

    import robotics_experiments.px4_bridge.isaac_articulation as isaac_articulation

    monkeypatch.setattr(isaac_articulation, "IsaacArticulationIO", FakeIsaacArticulationIO)

    Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=_FakeFlightStack([0.0, 0.0, 0.0, 0.0]),
        rotor_spin_signs=[-1.0, 1.0, -1.0, 1.0],
    )

    assert captured["rotor_spin_signs"] == [-1.0, 1.0, -1.0, 1.0]


def test_tilt_angle_deg_level():
    from robotics_experiments.px4_bridge.frames import tilt_angle_deg

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert tilt_angle_deg(q) == pytest.approx(0.0, abs=1e-10)


def test_tilt_angle_deg_45_pitch():
    from robotics_experiments.px4_bridge.frames import tilt_angle_deg

    q = _quat_pitch_deg(45.0)
    assert tilt_angle_deg(q) == pytest.approx(45.0, abs=1e-10)


def test_tilt_angle_deg_inverted():
    from robotics_experiments.px4_bridge.frames import tilt_angle_deg

    q = _quat_pitch_deg(180.0)
    assert tilt_angle_deg(q) == pytest.approx(180.0, abs=1e-10)


def test_tilt_angle_deg_pure_yaw_is_zero():
    from robotics_experiments.px4_bridge.frames import tilt_angle_deg

    q = _quat_yaw_deg(90.0)
    assert tilt_angle_deg(q) == pytest.approx(0.0, abs=1e-10)


def test_bridge_tilt_cutoff_zeros_forces():
    state = replace(
        VehicleState.hover_at(sim_time_us=0), orientation_flu_enu_wxyz=_quat_pitch_deg(80.0)
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.5, 0.5, 0.5, 0.5])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    # Forces are zeroed, but the MAVLink exchange must still run so PX4
    # lockstep keeps advancing (the sensor tick is not dropped).
    assert len(port.states) == 1
    assert len(io.applied_forces) == 1
    # Forces zeroed is the observable gating effect of the tilt cutoff firing.
    assert np.allclose(io.applied_forces[0], [0.0, 0.0, 0.0, 0.0])


def test_bridge_tilt_cutoff_sustained_preserves_lockstep():
    state = replace(
        VehicleState.hover_at(sim_time_us=0), orientation_flu_enu_wxyz=_quat_pitch_deg(80.0)
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.5, 0.5, 0.5, 0.5])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge.start()
    sim_times = [4_000, 8_000, 12_000, 16_000, 20_000]
    for t in sim_times:
        bridge.on_physics_step(0.004, t)
    bridge.stop()

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
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge.start()
    with pytest.raises(RuntimeError):
        bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    # No NaN forwarded into the lockstep exchange, and no forces applied.
    assert port.states == []
    assert io.applied_forces == []


def test_bridge_tilt_cutoff_passes_through_within_limit():
    state = replace(
        VehicleState.hover_at(sim_time_us=0), orientation_flu_enu_wxyz=_quat_pitch_deg(30.0)
    )
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([0.0, 0.25, 0.5, 1.0])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

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
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=_LinearThrustModel(num_rotors=4, max_force_n=8.0),
        port=port,
        articulation_io=io,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    assert len(io.applied_forces) == 1
    # The cutoff is effectively disabled: a 170 deg pitch still applies forces.
    assert np.allclose(io.applied_forces[0], [0.0, 2.0, 4.0, 8.0])


def test_bridge_forwards_rotor_omega_to_visual_spin_seam():
    state = VehicleState.hover_at(sim_time_us=0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([100.0, 500.0, 1100.0, 1500.0])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=PegasusQuadraticThrustModel(num_rotors=4),
        port=port,
        articulation_io=io,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    # The cosmetic spin seam is fed the clipped per-rotor omega and the tick dt.
    assert len(io.spin_updates) == 1
    omega, dt = io.spin_updates[0]
    np.testing.assert_allclose(omega, np.clip([100.0, 500.0, 1100.0, 1500.0], 0.0, 1100.0))
    assert dt == pytest.approx(0.004)


def test_bridge_visual_spin_uses_airframe_max_rotor_velocity():
    state = VehicleState.hover_at(sim_time_us=0)
    io = _FakeArticulationIO(state)
    port = _FakeFlightStack([1100.0, 1100.0, 1100.0, 1100.0])
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=PegasusQuadraticThrustModel(
            num_rotors=4,
            max_rotor_velocity=[525.0, 525.0, 525.0, 525.0],
        ),
        port=port,
        articulation_io=io,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

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
    bridge = Px4Bridge(
        articulation_path="/World/Drone",
        rotor_link_paths=["m1", "m2", "m3", "m4"],
        thrust_model=PegasusQuadraticThrustModel(num_rotors=4),
        port=port,
        articulation_io=io,
        max_tilt_angle_deg=60.0,
    )

    bridge.start()
    bridge.on_physics_step(0.004, 4_000)
    bridge.stop()

    # Forces gated to zero (cutoff fired), but the rotors still spin (next assert).
    assert np.allclose(io.applied_forces[0], [0.0, 0.0, 0.0, 0.0])
    assert len(io.spin_updates) == 1
    omega, _dt = io.spin_updates[0]
    np.testing.assert_allclose(omega, [700.0, 700.0, 700.0, 700.0])

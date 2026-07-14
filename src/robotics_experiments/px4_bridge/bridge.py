"""Frozen PX4 bridge state and stateless lifecycle/tick transitions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

import robotics_experiments.px4_bridge.frames as frames
from robotics_experiments.px4_bridge._simulator import _ArticulationIO
from robotics_experiments.px4_bridge.contract import ActuatorCommands
from robotics_experiments.px4_bridge.mavlink import Px4MavlinkPort
from robotics_experiments.px4_bridge.thrust_model import RotorDynamics, RotorThrustModel
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = [
    "Px4BridgeState",
    "create_px4_bridge",
    "recv_and_apply",
    "send_sensors",
    "start_px4_bridge",
    "stop_px4_bridge",
    "tick_px4_bridge",
]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PendingTick:
    state: VehicleState
    send_result: Optional[ActuatorCommands]
    force_cutoff: bool
    sim_time_us: int
    dt: float


@dataclass(frozen=True)
class Px4BridgeState:
    """Immutable bridge progress over mutable MAVLink and articulation adapters."""

    port: Px4MavlinkPort
    thrust_model: RotorThrustModel
    articulation_io: _ArticulationIO
    max_tilt_angle_deg: float
    tick_count: int = 0
    dynamics_debug_interval: int = 50
    tilt_cutoff_count: int = 0
    pending: Optional[_PendingTick] = None


def create_px4_bridge(
    *,
    articulation_path: str,
    rotor_link_paths: Sequence[str],
    thrust_model: RotorThrustModel,
    port: Px4MavlinkPort,
    articulation_io: Optional[_ArticulationIO] = None,
    max_tilt_angle_deg: float = 180.0,
    rotor_spin_visual_scale: float = 0.03,
    rotor_spin_signs: Sequence[float] | None = None,
) -> Px4BridgeState:
    """Create bridge state without opening sockets or articulation handles."""
    if thrust_model.num_rotors != port.num_rotors:
        raise ValueError(
            f"thrust_model num_rotors={thrust_model.num_rotors} "
            f"!= port num_rotors={port.num_rotors}"
        )
    if len(rotor_link_paths) != port.num_rotors:
        raise ValueError(
            f"rotor_link_paths has {len(rotor_link_paths)} entries; expected {port.num_rotors}"
        )
    if articulation_io is None:
        from robotics_experiments.px4_bridge.isaac_articulation import IsaacArticulationIO

        articulation_io = IsaacArticulationIO(
            articulation_path=articulation_path,
            rotor_link_paths=rotor_link_paths,
            rotor_spin_visual_scale=rotor_spin_visual_scale,
            rotor_spin_signs=rotor_spin_signs,
        )
    return Px4BridgeState(
        port=port,
        thrust_model=thrust_model,
        articulation_io=articulation_io,
        max_tilt_angle_deg=max_tilt_angle_deg,
    )


def start_px4_bridge(*, state: Px4BridgeState) -> Px4BridgeState:
    """Open the flight-stack resource and return reset tick progress."""
    state.port.start()
    return replace(state, tick_count=0, tilt_cutoff_count=0, pending=None)


def stop_px4_bridge(*, state: Px4BridgeState) -> Px4BridgeState:
    """Close bridge resources and clear any unpaired send."""
    state.port.stop()
    state.articulation_io.shutdown()
    return replace(state, pending=None)


def send_sensors(*, state: Px4BridgeState, dt: float, sim_time_us: int) -> Px4BridgeState:
    """Run the non-blocking sensor leg and return its pending poll state."""
    io = state.articulation_io
    if not io.is_initialized():
        io.initialize()
    vehicle_state = io.read_state(int(sim_time_us))
    if not np.all(np.isfinite(vehicle_state.position_enu)) or not np.all(
        np.isfinite(vehicle_state.orientation_flu_enu_wxyz)
    ):
        _LOGGER.error(
            "PX4 bridge non-finite vehicle state at sim_time_us=%d: "
            "position_enu=%s orientation_flu_enu_wxyz=%s",
            int(sim_time_us),
            np.asarray(vehicle_state.position_enu, dtype=np.float64).tolist(),
            np.asarray(vehicle_state.orientation_flu_enu_wxyz, dtype=np.float64).tolist(),
        )
        raise RuntimeError(
            f"PX4 bridge read a non-finite vehicle state at "
            f"sim_time_us={int(sim_time_us)}; refusing to forward NaN/Inf "
            f"into the PX4 lockstep exchange"
        )

    tilt = frames.tilt_angle_deg(vehicle_state.orientation_flu_enu_wxyz)
    # The cutoff gates only applied dynamics; the sensor/actuator exchange must
    # continue every tick so HIL_SENSOR.time_usec and PX4 lockstep keep advancing.
    force_cutoff = tilt > state.max_tilt_angle_deg
    cutoff_count = state.tilt_cutoff_count + int(force_cutoff)
    if force_cutoff and (cutoff_count <= 10 or cutoff_count % 250 == 0):
        _LOGGER.warning(
            "PX4 bridge tilt cutoff: count=%d tilt=%.1f deg limit=%.1f deg; "
            "zeroing rotor forces",
            cutoff_count,
            tilt,
            state.max_tilt_angle_deg,
        )
    pending = _PendingTick(
        state=vehicle_state,
        send_result=state.port.send(vehicle_state),
        force_cutoff=force_cutoff,
        sim_time_us=int(sim_time_us),
        dt=float(dt),
    )
    return replace(state, tilt_cutoff_count=cutoff_count, pending=pending)


def recv_and_apply(*, state: Px4BridgeState) -> Px4BridgeState:
    """Run the paired actuator leg and return state with no pending poll.

    An exception returns no successor, so a stateful split-leg caller must
    discard the input state's pending tick before continuing.
    """
    pending = state.pending
    if pending is None:
        raise RuntimeError(
            "PX4 bridge recv_and_apply called without a paired send_sensors; "
            "the lockstep contract requires the send leg to run first"
        )
    commands = state.port.poll() if pending.send_result is None else pending.send_result
    body_velocity_flu = frames.body_velocity_flu_from_state(pending.state)
    dynamics = state.thrust_model.commands_to_dynamics(
        commands,
        linear_body_velocity_flu=body_velocity_flu,
    )
    state.articulation_io.update_rotor_spin(dynamics.rotor_omega_rad_s, pending.dt)
    consumed = replace(state, pending=None)
    if pending.force_cutoff:
        zero_forces = np.zeros(state.thrust_model.num_rotors, dtype=np.float64)
        zero_torque = np.zeros(3, dtype=np.float64)
        zero_drag = np.zeros(3, dtype=np.float64)
        for array in (zero_forces, zero_torque, zero_drag):
            array.setflags(write=False)
        state.articulation_io.apply_dynamics(
            RotorDynamics(
                forces_n=zero_forces,
                body_torque_n_m=zero_torque,
                body_drag_n=zero_drag,
                rotor_omega_rad_s=dynamics.rotor_omega_rad_s,
            )
        )
        return consumed

    tick_count = state.tick_count + 1
    if tick_count <= 10 or tick_count % state.dynamics_debug_interval == 0:
        status = state.port.read_status()
        _LOGGER.debug(
            "PX4 bridge dynamics: tick=%d sim_time_us=%d connected=%s armed=%s "
            "commands=%s forces_n=%s body_torque_n_m=%s body_drag_n=%s "
            "position_enu=%s quat_wxyz=%s velocity_enu=%s body_velocity_flu=%s",
            tick_count,
            pending.sim_time_us,
            status.is_connected,
            status.is_armed,
            np.asarray(commands.controls, dtype=np.float64).tolist(),
            np.asarray(dynamics.forces_n, dtype=np.float64).tolist(),
            np.asarray(dynamics.body_torque_n_m, dtype=np.float64).tolist(),
            np.asarray(dynamics.body_drag_n, dtype=np.float64).tolist(),
            np.asarray(pending.state.position_enu, dtype=np.float64).tolist(),
            np.asarray(pending.state.orientation_flu_enu_wxyz, dtype=np.float64).tolist(),
            np.asarray(pending.state.linear_velocity_enu, dtype=np.float64).tolist(),
            np.asarray(body_velocity_flu, dtype=np.float64).tolist(),
        )
    state.articulation_io.apply_dynamics(dynamics)
    return replace(consumed, tick_count=tick_count)


def tick_px4_bridge(*, state: Px4BridgeState, dt: float, sim_time_us: int) -> Px4BridgeState:
    """Compose one vehicle's send and receive legs."""
    sent = send_sensors(state=state, dt=dt, sim_time_us=sim_time_us)
    return recv_and_apply(state=sent)

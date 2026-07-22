"""Shared Isaac-only DirectRLEnv flight runtime for hover and press tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.envs import DirectRLEnv

from robotics_experiments._cadence import _CadenceState, _next_interval_steps
from robotics_experiments.config import ExperimentConfig
from robotics_experiments.workflows.isaac_lab.tasks.flight import (
    action_to_rotor_omega,
    apply_body_wrench,
    build_flight_actuation_params,
    rotor_omega_to_body_wrench,
)

__all__: list[str] = []

_DRONE_PRIM_PATH = "/World/envs/env_.*/Drone"


def _episode_step_count(*, episode_length_s: float, policy_rate_hz: float) -> int:
    return math.ceil(episode_length_s * policy_rate_hz)


def _training_vehicle(config: ExperimentConfig, /):
    vehicle_id = config.rl.training.vehicle_id
    try:
        return next(vehicle for vehicle in config.vehicles.values() if vehicle.id == vehicle_id)
    except StopIteration as exc:
        raise ValueError(f"rl.training.vehicle_id={vehicle_id} is not in the experiment") from exc


class _FlightDirectRLEnv(DirectRLEnv):
    """Cadence, action buffering, rotor actuation, and drone reset behavior."""

    def __init__(
        self,
        *,
        cfg,
        experiment_config: ExperimentConfig,
        render_mode: str | None = None,
        **kwargs,
    ):
        self._experiment_config = experiment_config
        self._nominal_step_dt = 1.0 / cfg.policy_rate_hz
        self._physics_period_ns = int(round(cfg.sim.dt * 1_000_000_000))
        self._policy_cadence = _CadenceState(previous_time_ns=0)
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._setup_actuation()

    @property
    def step_dt(self) -> float:
        return self._nominal_step_dt

    @property
    def max_episode_length(self) -> int:
        return _episode_step_count(
            episode_length_s=self.cfg.episode_length_s,
            policy_rate_hz=self.cfg.policy_rate_hz,
        )

    def step(self, action: torch.Tensor) -> tuple[Any, ...]:
        self._policy_cadence, self.cfg.decimation = _next_interval_steps(
            state=self._policy_cadence,
            source_period_ns=self._physics_period_ns,
            rate_hz=self.cfg.policy_rate_hz,
        )
        return super().step(action)

    def _setup_actuation(self) -> None:
        import omni.usd

        from robotics_experiments.px4_bridge import discover_px4_rotor_mapping

        stage = omni.usd.get_context().get_stage()
        env0_path = _DRONE_PRIM_PATH.replace("env_.*", "env_0")
        rotors = discover_px4_rotor_mapping(stage, env0_path)
        bridge = _training_vehicle(self._experiment_config).bridge
        masses = self.drone.root_physx_view.get_masses()
        total_mass = float(masses[0].sum().item())
        self._body_id = int(self.drone.find_bodies("body")[0][0])
        self._linear_drag = torch.as_tensor(
            bridge.linear_drag, dtype=torch.float32, device=self.device
        )
        action_cfg = self._experiment_config.rl.action
        self.actuation_params = build_flight_actuation_params(
            kind=action_cfg.kind,
            positions_body=rotors.positions_body,
            rot_dir=rotors.rot_dir,
            rotor_constant=bridge.rotor_constant,
            rolling_moment_coefficient=bridge.rolling_moment_coefficient,
            min_rotor_velocity=bridge.min_rotor_velocity,
            max_rotor_velocity=bridge.max_rotor_velocity,
            action_cfg=action_cfg,
            mass=total_mass,
            device=self.device,
        )
        pusher_joint_ids, _ = self.drone.find_joints("pusher_joint")
        if len(pusher_joint_ids) != 1:
            raise RuntimeError(
                "Lab flight vehicle requires exactly one pusher_joint, "
                f"found {len(pusher_joint_ids)}"
            )
        self._pusher_joint_id = int(pusher_joint_ids[0])
        extension = self.drone.data.joint_pos_limits[:, self._pusher_joint_id, 1]
        self._pusher_extension_target = extension.unsqueeze(-1)
        self.drone.data.default_joint_pos[:, self._pusher_joint_id] = extension

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self.drone.set_joint_position_target(
            self._pusher_extension_target,
            joint_ids=[self._pusher_joint_id],
        )
        omega = action_to_rotor_omega(
            self.actuation_params,
            self.actions,
            self.drone.data.root_quat_w,
            self.drone.data.root_lin_vel_w,
            self.drone.data.root_ang_vel_b,
        )
        force_body, torque_body = rotor_omega_to_body_wrench(
            self.actuation_params,
            omega,
            self.drone.data.root_lin_vel_b,
            self._linear_drag,
        )
        apply_body_wrench(self.drone, self._body_id, force_body, torque_body)

    def _reset_drone(
        self,
        *,
        env_ids: Sequence[int] | torch.Tensor,
        root_state: torch.Tensor,
    ) -> None:
        self.drone.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.drone.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        joint_position = self.drone.data.default_joint_pos[env_ids].clone()
        joint_velocity = self.drone.data.default_joint_vel[env_ids].clone()
        self.drone.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
        self.actions[env_ids] = 0.0

"""Shared Isaac-only DirectRLEnv flight runtime for hover and press tasks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab_contrib.actuators import Thruster, ThrusterCfg
from isaaclab_contrib.utils.types import MultiRotorActions

from robotics_experiments._cadence import _CadenceState, _next_interval_steps
from robotics_experiments.alignment import ControllerSettings
from robotics_experiments.config import ExperimentConfig
from robotics_experiments.workflows.isaac_lab.tasks.flight import (
    _latch_velocity_action_world,
    _replace_controller_settings_batch,
    _world_velocity_action_to_body,
    action_to_rotor_omega,
    apply_body_wrench,
    build_flight_actuation_params,
    initial_flight_controller_state,
    reset_flight_controller_state,
    rotor_thrust_to_body_wrench,
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
        controller_settings: ControllerSettings,
        policy_contract=None,
        render_mode: str | None = None,
        **kwargs,
    ):
        self._experiment_config = experiment_config
        self._controller_settings = controller_settings
        self._policy_contract = policy_contract
        self._nominal_step_dt = 1.0 / cfg.policy_rate_hz
        self._physics_period_ns = int(round(cfg.sim.dt * 1_000_000_000))
        self._policy_cadence = _CadenceState(previous_time_ns=0)
        self._setpoint_stream_cadence = _CadenceState(previous_time_ns=0)
        self._setpoint_stream_steps_remaining = 0
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._controller_actions = self.actions.clone()
        self._applied_actions = self.actions.clone()
        self._command_delay_steps: torch.Tensor
        self._command_delay_index: int
        self._command_delay_buffer: torch.Tensor
        self._set_command_delay_steps(
            delay_steps=torch.full(
                (self.num_envs,),
                round(controller_settings.command_delay_s * experiment_config.sim.physics_hz),
                dtype=torch.long,
                device=self.device,
            )
        )
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
        self._body_com_position_body = self.drone.data.body_com_pos_b[:, self._body_id].clone()
        self._linear_drag = torch.as_tensor(
            bridge.linear_drag, dtype=torch.float32, device=self.device
        )
        action_cfg = self._experiment_config.rl.action
        action_kind = (
            action_cfg.kind if self._policy_contract is None else self._policy_contract.action_kind
        )
        self.actuation_params = build_flight_actuation_params(
            kind=action_kind,
            positions_body=rotors.positions_body,
            rot_dir=rotors.rot_dir,
            rotor_constant=bridge.rotor_constant,
            rolling_moment_coefficient=bridge.rolling_moment_coefficient,
            min_rotor_velocity=bridge.min_rotor_velocity,
            max_rotor_velocity=bridge.max_rotor_velocity,
            input_offset=bridge.input_offset,
            input_scaling=bridge.input_scaling,
            zero_position_armed=bridge.zero_position_armed,
            action_cfg=action_cfg,
            controller_settings=self._controller_settings,
            policy_contract=self._policy_contract,
            mass=total_mass,
            controller_dt_s=1.0 / self._experiment_config.sim.physics_hz,
            device=self.device,
        )
        self._controller_state = initial_flight_controller_state(
            num_envs=self.num_envs,
            device=self.device,
        )
        rotor_count = len(rotors.rot_dir)
        thrust_max = max(
            float(constant) * float(max_omega) ** 2
            for constant, max_omega in zip(
                bridge.rotor_constant,
                bridge.max_rotor_velocity,
                strict=True,
            )
        )
        thrust_constant_min = min(float(value) for value in bridge.rotor_constant)
        thrust_constant_max = max(float(value) for value in bridge.rotor_constant)
        initial_omega = torch.zeros(
            (self.num_envs, rotor_count),
            dtype=torch.float32,
            device=self.device,
        )
        self._thruster = Thruster(
            cfg=ThrusterCfg(
                dt=1.0 / self._experiment_config.sim.physics_hz,
                thrust_range=(0.0, thrust_max),
                max_thrust_rate=self._controller_settings.thruster_max_thrust_rate_nps,
                thrust_const_range=(thrust_constant_min, thrust_constant_max),
                tau_inc_range=(
                    self._controller_settings.thruster_tau_inc_s,
                    self._controller_settings.thruster_tau_inc_s,
                ),
                tau_dec_range=(
                    self._controller_settings.thruster_tau_dec_s,
                    self._controller_settings.thruster_tau_dec_s,
                ),
                torque_to_thrust_ratio=0.0,
                use_discrete_approximation=True,
                integration_scheme="euler",
                thruster_names_expr=[f"rotor_{index}" for index in range(rotor_count)],
            ),
            thruster_names=[f"rotor_{index}" for index in range(rotor_count)],
            thruster_ids=slice(None),
            num_envs=self.num_envs,
            device=str(self.device),
            init_thruster_rps=initial_omega,
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
        if self._setpoint_stream_steps_remaining == 0:
            if self.actuation_params.kind == "velocity":
                self._controller_actions = _latch_velocity_action_world(
                    self.actions,
                    self.drone.data.root_quat_w,
                )
            else:
                self._controller_actions = self.actions.clone()
            (
                self._setpoint_stream_cadence,
                self._setpoint_stream_steps_remaining,
            ) = _next_interval_steps(
                state=self._setpoint_stream_cadence,
                source_period_ns=self._physics_period_ns,
                rate_hz=self._experiment_config.driver.offboard_setpoint_rate_hz,
            )
        self._command_delay_buffer[self._command_delay_index].copy_(self._controller_actions)
        delayed_indices = (
            self._command_delay_index - self._command_delay_steps
        ) % self._command_delay_buffer.shape[0]
        environment_indices = torch.arange(self.num_envs, device=self.device)
        delayed_actions = self._command_delay_buffer[
            delayed_indices,
            environment_indices,
        ].clone()
        if self.actuation_params.kind == "velocity":
            self._applied_actions = _world_velocity_action_to_body(
                delayed_actions,
                self.drone.data.root_quat_w,
            )
        else:
            self._applied_actions = delayed_actions
        self._command_delay_index = (
            self._command_delay_index + 1
        ) % self._command_delay_buffer.shape[0]
        self._setpoint_stream_steps_remaining -= 1
        self.drone.set_joint_position_target(
            self._pusher_extension_target,
            joint_ids=[self._pusher_joint_id],
        )
        self._controller_state, omega = action_to_rotor_omega(
            self.actuation_params,
            self._controller_state,
            self._applied_actions,
            self.drone.data.root_quat_w,
            self.drone.data.root_lin_vel_w,
            self.drone.data.root_ang_vel_b,
            self.actuation_params.controller_dt_s,
        )
        desired_thrust = self.actuation_params.rotor_constant * omega.square()
        realized = self._thruster.compute(
            MultiRotorActions(
                thrusts=desired_thrust,
                thruster_indices=slice(None),
            )
        ).thrusts
        if realized is None:
            raise RuntimeError("Isaac Lab Thruster returned no realized thrust")
        force_body, torque_body = rotor_thrust_to_body_wrench(
            self.actuation_params,
            realized,
            self.drone.data.root_lin_vel_b,
            self._linear_drag,
        )
        apply_body_wrench(
            self.drone,
            self._body_id,
            self._body_com_position_body,
            force_body,
            torque_body,
        )

    def _set_command_delay_steps(self, *, delay_steps: torch.Tensor) -> None:
        if delay_steps.shape != (self.num_envs,):
            raise ValueError(
                f"command delays have shape {tuple(delay_steps.shape)}, "
                f"expected {(self.num_envs,)}"
            )
        self._command_delay_steps = delay_steps
        self._command_delay_index = 0
        self._command_delay_buffer = torch.zeros(
            (
                int(delay_steps.max().item()) + 1,
                self.num_envs,
                self.cfg.action_space,
            ),
            dtype=self.actions.dtype,
            device=self.device,
        )

    def _set_controller_settings_batch(
        self,
        *,
        controller_settings: Sequence[ControllerSettings],
        environments_per_setting: int,
    ) -> None:
        if len(controller_settings) * environments_per_setting != self.num_envs:
            raise ValueError("controller batch does not cover every environment")
        self.actuation_params = _replace_controller_settings_batch(
            params=self.actuation_params,
            controller_settings=controller_settings,
            environments_per_setting=environments_per_setting,
        )
        delay_steps = [
            round(settings.command_delay_s * self._experiment_config.sim.physics_hz)
            for settings in controller_settings
            for _environment in range(environments_per_setting)
        ]
        self._set_command_delay_steps(
            delay_steps=torch.as_tensor(
                delay_steps,
                dtype=torch.long,
                device=self.device,
            )
        )
        self._setpoint_stream_cadence = _CadenceState(previous_time_ns=0)
        self._setpoint_stream_steps_remaining = 0
        self._applied_actions.zero_()
        self._controller_actions.zero_()
        self._controller_state = initial_flight_controller_state(
            num_envs=self.num_envs,
            device=self.device,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._thruster.reset_idx(env_ids)
        tau_inc = [
            settings.thruster_tau_inc_s
            for settings in controller_settings
            for _environment in range(environments_per_setting)
        ]
        tau_dec = [
            settings.thruster_tau_dec_s
            for settings in controller_settings
            for _environment in range(environments_per_setting)
        ]
        max_rate = [
            settings.thruster_max_thrust_rate_nps
            for settings in controller_settings
            for _environment in range(environments_per_setting)
        ]
        self._thruster.tau_inc_s[:] = torch.as_tensor(
            tau_inc, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        self._thruster.tau_dec_s[:] = torch.as_tensor(
            tau_dec, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        self._thruster.max_rate[:] = torch.as_tensor(
            max_rate, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

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
        self._controller_actions[env_ids] = 0.0
        self._applied_actions[env_ids] = 0.0
        self._command_delay_buffer[:, env_ids] = 0.0
        self._controller_state = reset_flight_controller_state(
            state=self._controller_state,
            env_ids=env_ids,
        )
        self._thruster.reset_idx(env_ids)

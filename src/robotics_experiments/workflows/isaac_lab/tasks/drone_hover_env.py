"""Flyable Isaac Lab DirectRLEnv for drone hover.

Maps a normalized RL action to per-rotor forces through the pluggable in-Lab
actuation seam (the ``flight`` params + pure functions, the batched twin of
the host-tested ``robotics_experiments.rl``), with no PX4 lockstep
per env. The action level is ``rl.action.kind`` (``velocity`` / ``ctbr``).
The env class is thin wiring: the frozen ``actuation_params`` plus the pure
per-step math (``_hover_observation`` / ``_hover_reward`` /
``_hover_terminated``) do the work, so external composition (rollout/eval
scripts, the standalone gate) can run the same function chain. The reward here
is a temporary hover-tracking shaping term so the env is trainable end-to-end
for the F1 validation gate; F3 replaces it with the switch-press reward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any, cast

import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.array import convert_to_torch

from robotics_experiments.config import ExperimentConfig
from robotics_experiments.workflows.isaac_lab.tasks._drone_cfg import _make_drone_articulation_cfg
from robotics_experiments.workflows.isaac_lab.tasks.flight import (
    action_to_rotor_omega,
    apply_body_wrench,
    build_flight_actuation_params,
    rotor_omega_to_body_wrench,
)

__all__ = ["DroneHoverEnvCfg", "DroneHoverEnv", "make_drone_hover_env_cfg"]

_DRONE_PRIM_PATH = "/World/envs/env_.*/Drone"


def _xyzw_to_wxyz(quat_xyzw: torch.Tensor, /) -> torch.Tensor:
    """Convert Lab 3's simulator quaternion order at a project WXYZ boundary."""
    return torch.cat((quat_xyzw[..., 3:], quat_xyzw[..., :3]), dim=-1)


def _hover_observation(
    position_w: torch.Tensor,
    target_w: torch.Tensor,
    quat_wxyz: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_b: torch.Tensor,
) -> torch.Tensor:
    """Pack the 13-dim hover observation ``[pos_err, quat, lin_vel_w, ang_vel_b]``.

    The batched twin of the pure ``rl.press.pack_hover_observation``
    layout (`HOVER_OBSERVATION_SCHEMA_VERSION`): the angular velocity is
    body-frame, matching ``VehicleState.angular_velocity_flu`` so the same
    vector is assemblable at deploy time.
    """
    return torch.cat((position_w - target_w, quat_wxyz, lin_vel_w, ang_vel_b), dim=-1)


def _hover_reward(
    task,
    position_w: torch.Tensor,
    target_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_w: torch.Tensor,
) -> torch.Tensor:
    position_error = torch.linalg.norm(position_w - target_w, dim=-1)
    linear_speed = torch.linalg.norm(lin_vel_w, dim=-1)
    angular_speed = torch.linalg.norm(ang_vel_w, dim=-1)
    return (
        task.reward_alive
        - task.reward_pos_weight * position_error
        - task.reward_linvel_weight * linear_speed
        - task.reward_angvel_weight * angular_speed
    )


def _hover_terminated(
    task,
    position_w: torch.Tensor,
    target_w: torch.Tensor,
    quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    # Body-Z projected onto world-Z: the (0,0,1) column of the body->world
    # rotation is 1 - 2(x^2 + y^2) for an xyzw quaternion.
    body_z_world_z = 1.0 - 2.0 * (quat_xyzw[:, 0] ** 2 + quat_xyzw[:, 1] ** 2)
    tilted = body_z_world_z < task.crash_tilt_cos
    too_low = position_w[:, 2] < task.min_altitude_m
    position_error = torch.linalg.norm(position_w - target_w, dim=-1)
    out_of_bounds = position_error > task.max_position_error_m
    return tilted | too_low | out_of_bounds


def _training_vehicle(config, /):
    vehicle_id = config.rl.training.vehicle_id
    try:
        return next(vehicle for vehicle in config.vehicles.values() if vehicle.id == vehicle_id)
    except StopIteration as exc:
        raise ValueError(f"rl.training.vehicle_id={vehicle_id} is not in the experiment") from exc


@configclass
class DroneHoverEnvCfg(DirectRLEnvCfg):
    """Configuration for the flyable drone-hover DirectRLEnv."""

    # Isaac Lab's configclass requires the dataclasses.MISSING sentinel for
    # explicitly supplied fields; the cast keeps their consumer-facing types.
    decimation: int = cast(Any, MISSING)
    episode_length_s: float = cast(Any, MISSING)
    action_space: int = cast(Any, MISSING)
    observation_space = 13
    state_space = 0
    sim: SimulationCfg = cast(Any, MISSING)
    drone_cfg: ArticulationCfg = cast(Any, MISSING)
    scene: InteractiveSceneCfg = cast(Any, MISSING)


def make_drone_hover_env_cfg(*, experiment_config) -> DroneHoverEnvCfg:
    """Build the Isaac Lab task config from one resolved experiment."""
    vehicle = _training_vehicle(experiment_config)
    decimation = max(
        1,
        round(experiment_config.sim.physics_hz / experiment_config.rl.policy_rate_hz),
    )
    return DroneHoverEnvCfg(
        decimation=decimation,
        episode_length_s=experiment_config.rl.episode_length_s,
        action_space=experiment_config.rl.action.action_dim,
        observation_space=13,
        state_space=0,
        sim=SimulationCfg(
            dt=1.0 / experiment_config.sim.physics_hz,
            render_interval=decimation,
        ),
        drone_cfg=_make_drone_articulation_cfg(vehicle.spawn.usd_path, _DRONE_PRIM_PATH),
        scene=InteractiveSceneCfg(
            num_envs=experiment_config.rl.num_envs,
            env_spacing=experiment_config.rl.env_spacing,
            replicate_physics=True,
            clone_in_fabric=True,
        ),
    )


class DroneHoverEnv(DirectRLEnv):
    """Drone-hover task that actually flies N parallel drones.

    ``_apply_action`` closes the actuation loop every physics substep: the stored
    normalized action -> per-rotor omega (``action_to_rotor_omega`` over the
    frozen ``actuation_params``) -> the realized body wrench (thrust at the
    rotor offsets + yaw reaction + drag) -> the vectorized force sink.
    ``actuation_params`` is public: it is the composition seam external
    consumers (e.g. the standalone hover+goto gate) run the same pure functions
    against.
    """

    cfg: DroneHoverEnvCfg

    def __init__(
        self,
        *,
        cfg: DroneHoverEnvCfg,
        experiment_config: ExperimentConfig,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._experiment_config = experiment_config
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._hover_target_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._task = self._experiment_config.rl.press.hover_task
        self._setup_actuation()

    def _setup_scene(self) -> None:
        self.drone = Articulation(self.cfg.drone_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["drone"] = self.drone

    def _setup_actuation(self) -> None:
        """Build the batched actuation from the resolved airframe + spawned mass.

        Rotor geometry comes from the spawned USD metadata (env 0), the thrust
        constants from the per-asset airframe bridge config, and the vehicle mass
        from the spawned articulation, so the in-Lab model matches the asset.
        """
        import omni.usd

        from robotics_experiments.px4_bridge import discover_px4_rotor_mapping

        stage = omni.usd.get_context().get_stage()
        env0_path = _DRONE_PRIM_PATH.replace("env_.*", "env_0")
        rotors = discover_px4_rotor_mapping(stage, env0_path)
        bridge = _training_vehicle(self._experiment_config).bridge

        masses = convert_to_torch(self.drone.root_view.get_masses())
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        quat_wxyz = _xyzw_to_wxyz(self.drone.data.root_quat_w.torch)
        omega = action_to_rotor_omega(
            self.actuation_params,
            self.actions,
            quat_wxyz,
            self.drone.data.root_lin_vel_w.torch,
            self.drone.data.root_ang_vel_b.torch,
        )
        force_body, torque_body = rotor_omega_to_body_wrench(
            self.actuation_params,
            omega,
            self.drone.data.root_lin_vel_b.torch,
            self._linear_drag,
        )
        apply_body_wrench(self.drone, self._body_id, force_body, torque_body)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = _hover_observation(
            self.drone.data.root_pos_w.torch,
            self._hover_target_w,
            _xyzw_to_wxyz(self.drone.data.root_quat_w.torch),
            self.drone.data.root_lin_vel_w.torch,
            self.drone.data.root_ang_vel_b.torch,
        )
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        return _hover_reward(
            self._task,
            self.drone.data.root_pos_w.torch,
            self._hover_target_w,
            self.drone.data.root_lin_vel_w.torch,
            self.drone.data.root_ang_vel_w.torch,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = _hover_terminated(
            self._task,
            self.drone.data.root_pos_w.torch,
            self._hover_target_w,
            self.drone.data.root_quat_w.torch,
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.drone.data.default_root_state.torch[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        # The hover target is the (un-randomized) spawn pose; the start pose is
        # randomized within a small box around it so the policy learns to recover.
        self._hover_target_w[env_ids] = default_root_state[:, :3]
        offset = torch.zeros_like(default_root_state[:, :3])
        offset[:, :2].uniform_(-self._task.reset_xy_half_m, self._task.reset_xy_half_m)
        offset[:, 2].uniform_(-self._task.reset_z_half_m, self._task.reset_z_half_m)
        default_root_state[:, :3] += offset

        self.drone.write_root_pose_to_sim_index(
            root_pose=default_root_state[:, :7], env_ids=env_ids
        )
        self.drone.write_root_velocity_to_sim_index(
            root_velocity=default_root_state[:, 7:], env_ids=env_ids
        )

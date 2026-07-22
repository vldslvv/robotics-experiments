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

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any, cast

import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from robotics_experiments.config import ExperimentConfig
from robotics_experiments.rl.press import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
)
from robotics_experiments.workflows.isaac_lab.tasks._drone_cfg import _make_drone_articulation_cfg
from robotics_experiments.workflows.isaac_lab.tasks._flight_env import (
    _DRONE_PRIM_PATH,
    _FlightDirectRLEnv,
    _training_vehicle,
)

__all__ = ["DroneHoverEnvCfg", "DroneHoverEnv", "make_drone_hover_env_cfg"]


def _hover_observation(
    position_w: torch.Tensor,
    target_w: torch.Tensor,
    quat_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_b: torch.Tensor,
) -> torch.Tensor:
    """Pack the 13-dim hover observation ``[pos_err, quat, lin_vel_w, ang_vel_b]``.

    The batched twin of the pure ``rl.press.pack_hover_observation``
    layout (`HOVER_OBSERVATION_SCHEMA_VERSION`): the angular velocity is
    body-frame, matching ``VehicleState.angular_velocity_flu`` so the same
    vector is assemblable at deploy time.
    """
    return torch.cat((position_w - target_w, quat_w, lin_vel_w, ang_vel_b), dim=-1)


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
    quat_w: torch.Tensor,
) -> torch.Tensor:
    # Body-Z projected onto world-Z: the (0,0,1) column of the body->world
    # rotation is 1 - 2(x^2 + y^2) for a wxyz quaternion.
    body_z_world_z = 1.0 - 2.0 * (quat_w[:, 1] ** 2 + quat_w[:, 2] ** 2)
    tilted = body_z_world_z < task.crash_tilt_cos
    too_low = position_w[:, 2] < task.min_altitude_m
    position_error = torch.linalg.norm(position_w - target_w, dim=-1)
    out_of_bounds = position_error > task.max_position_error_m
    return tilted | too_low | out_of_bounds


@configclass
class DroneHoverEnvCfg(DirectRLEnvCfg):
    """Configuration for the flyable drone-hover DirectRLEnv."""

    # Isaac Lab's configclass requires the dataclasses.MISSING sentinel for
    # explicitly supplied fields; the cast keeps their consumer-facing types.
    decimation: int = cast(Any, MISSING)
    policy_rate_hz: float = cast(Any, MISSING)
    episode_length_s: float = cast(Any, MISSING)
    action_space: int = cast(Any, MISSING)
    observation_schema_version: int = cast(Any, MISSING)
    observation_space: int = cast(Any, MISSING)
    state_space = 0
    sim: SimulationCfg = cast(Any, MISSING)
    drone_cfg: ArticulationCfg = cast(Any, MISSING)
    scene: InteractiveSceneCfg = cast(Any, MISSING)


def make_drone_hover_env_cfg(*, experiment_config) -> DroneHoverEnvCfg:
    """Build the Isaac Lab task config from one resolved experiment."""
    vehicle = _training_vehicle(experiment_config)
    decimation = math.ceil(experiment_config.sim.physics_hz / experiment_config.rl.policy_rate_hz)
    return DroneHoverEnvCfg(
        decimation=decimation,
        policy_rate_hz=experiment_config.rl.policy_rate_hz,
        episode_length_s=experiment_config.rl.episode_length_s,
        action_space=experiment_config.rl.action.action_dim,
        observation_schema_version=HOVER_OBSERVATION_SCHEMA_VERSION,
        observation_space=HOVER_OBSERVATION_DIM,
        state_space=0,
        sim=SimulationCfg(
            dt=1.0 / experiment_config.sim.physics_hz,
            render_interval=decimation,
        ),
        drone_cfg=_make_drone_articulation_cfg(
            usd_path=vehicle.spawn.usd_path,
            prim_path=_DRONE_PRIM_PATH,
            lab_vehicle=experiment_config.rl.lab_vehicle,
        ),
        scene=InteractiveSceneCfg(
            num_envs=experiment_config.rl.num_envs,
            env_spacing=experiment_config.rl.env_spacing,
            replicate_physics=True,
            clone_in_fabric=True,
        ),
    )


class DroneHoverEnv(_FlightDirectRLEnv):
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
        super().__init__(
            cfg=cfg,
            experiment_config=experiment_config,
            render_mode=render_mode,
            **kwargs,
        )
        self._hover_target_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._task = self._experiment_config.rl.press.hover_task

    def _setup_scene(self) -> None:
        self.drone = Articulation(self.cfg.drone_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["drone"] = self.drone

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = _hover_observation(
            self.drone.data.root_pos_w,
            self._hover_target_w,
            self.drone.data.root_quat_w,
            self.drone.data.root_lin_vel_w,
            self.drone.data.root_ang_vel_b,
        )
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        return _hover_reward(
            self._task,
            self.drone.data.root_pos_w,
            self._hover_target_w,
            self.drone.data.root_lin_vel_w,
            self.drone.data.root_ang_vel_w,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = _hover_terminated(
            self._task,
            self.drone.data.root_pos_w,
            self._hover_target_w,
            self.drone.data.root_quat_w,
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.drone.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        # The hover target is the (un-randomized) spawn pose; the start pose is
        # randomized within a small box around it so the policy learns to recover.
        self._hover_target_w[env_ids] = default_root_state[:, :3]
        offset = torch.zeros_like(default_root_state[:, :3])
        offset[:, :2].uniform_(-self._task.reset_xy_half_m, self._task.reset_xy_half_m)
        offset[:, 2].uniform_(-self._task.reset_z_half_m, self._task.reset_z_half_m)
        default_root_state[:, :3] += offset

        self._reset_drone(env_ids=env_ids, root_state=default_root_state)

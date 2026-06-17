"""Minimal Isaac Lab DirectRLEnv scaffold for drone hover work."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

__all__ = ["DroneHoverEnvCfg", "DroneHoverEnv"]


def _experiment_config():
    from robotics_experiments.config import get_config

    return get_config()


def _physics_dt() -> float:
    return 1.0 / _experiment_config().sim.physics_hz


def _make_drone_cfg() -> ArticulationCfg:
    cfg = _experiment_config()
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Drone",
        spawn=UsdFileCfg(usd_path=cfg.drone.usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(value) for value in cfg.drone.spawn_volume_center),
        ),
    )


@configclass
class DroneHoverEnvCfg(DirectRLEnvCfg):
    """Configuration for the minimal drone-hover DirectRLEnv skeleton."""

    decimation = 1
    episode_length_s = _experiment_config().isaac_lab.drone_hover_episode_length_s
    action_scale = _experiment_config().isaac_lab.drone_hover_action_scale
    action_space = 4
    observation_space = 13
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=_physics_dt(), render_interval=decimation)
    drone_cfg: ArticulationCfg = _make_drone_cfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_experiment_config().isaac_lab.drone_hover_num_envs,
        env_spacing=_experiment_config().isaac_lab.drone_hover_env_spacing,
        replicate_physics=True,
        clone_in_fabric=True,
    )


class DroneHoverEnv(DirectRLEnv):
    """No-op drone-hover task scaffold.

    The task defines the DirectRLEnv lifecycle and data surfaces, but intentionally
    does not map actions to rotor forces yet. That keeps this step focused on
    Isaac Lab layout alignment without duplicating PX4 bridge thrust policy.
    """

    cfg: DroneHoverEnvCfg

    def __init__(self, cfg: DroneHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    def _setup_scene(self) -> None:
        self.drone = Articulation(self.cfg.drone_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["drone"] = self.drone

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.cfg.action_scale * actions.clone()

    def _apply_action(self) -> None:
        return None

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_pos = self.drone.data.root_pos_w
        root_quat = self.drone.data.root_quat_w
        root_lin_vel = self.drone.data.root_lin_vel_w
        root_ang_vel = self.drone.data.root_ang_vel_w
        observations = torch.cat((root_pos, root_quat, root_lin_vel, root_ang_vel), dim=-1)
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.drone.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.drone.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.drone.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

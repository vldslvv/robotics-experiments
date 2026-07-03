"""Flyable Isaac Lab DirectRLEnv for drone hover.

Maps a normalized RL action to per-rotor forces through the pluggable in-Lab
actuation seam (the ``_flight`` params + pure functions, the batched twin of
the host-tested ``robotics_experiments.press.actuation``), with no PX4 lockstep
per env. The action level is ``press.action.kind`` (``velocity`` / ``ctbr``).
The env class is thin wiring: the frozen ``actuation_params`` plus the pure
per-step math (``_hover_observation`` / ``_hover_reward`` /
``_hover_terminated``) do the work, so external composition (rollout/eval
scripts, the standalone gate) can run the same function chain. The reward here
is a temporary hover-tracking shaping term so the env is trainable end-to-end
for the F1 validation gate; F3 replaces it with the switch-press reward.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from ._drone_cfg import make_drone_articulation_cfg
from ._flight import (
    action_to_rotor_omega,
    apply_body_wrench,
    build_flight_actuation_params,
    rotor_omega_to_body_wrench,
)

__all__ = ["DroneHoverEnvCfg", "DroneHoverEnv"]

_DRONE_PRIM_PATH = "/World/envs/env_.*/Drone"

# Temporary hover-reward shaping weights (F1 validation gate only; F3's press
# reward replaces this whole term). Not a tuned config surface.
_REWARD_POS_WEIGHT = 1.0
_REWARD_LINVEL_WEIGHT = 0.1
_REWARD_ANGVEL_WEIGHT = 0.05
_REWARD_ALIVE = 0.5

# Crash / out-of-bounds termination thresholds (placeholder envelope for the
# hover gate; F3 owns the task termination).
_CRASH_TILT_COS = 0.0  # body-Z world-Z alignment below this (> 90 deg tilt) = crashed
_MIN_ALTITUDE_M = 0.15
_MAX_POSITION_ERROR_M = 5.0

# Reset start-pose randomization box around the hover target (meters).
_RESET_XY_HALF_M = 0.5
_RESET_Z_HALF_M = 0.3


def _hover_observation(
    position_w: torch.Tensor,
    target_w: torch.Tensor,
    quat_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_w: torch.Tensor,
) -> torch.Tensor:
    """Pack the 13-dim hover observation ``[pos_err, quat, lin_vel, ang_vel]``.

    Placeholder F1 observation: carries the world-frame angular velocity even
    though the actuation controller consumes the body-frame one; F2's
    ``PressObservation`` replaces this vector and reconciles the frames.
    """
    return torch.cat((position_w - target_w, quat_w, lin_vel_w, ang_vel_w), dim=-1)


def _hover_reward(
    position_w: torch.Tensor,
    target_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_w: torch.Tensor,
) -> torch.Tensor:
    position_error = torch.linalg.norm(position_w - target_w, dim=-1)
    linear_speed = torch.linalg.norm(lin_vel_w, dim=-1)
    angular_speed = torch.linalg.norm(ang_vel_w, dim=-1)
    return (
        _REWARD_ALIVE
        - _REWARD_POS_WEIGHT * position_error
        - _REWARD_LINVEL_WEIGHT * linear_speed
        - _REWARD_ANGVEL_WEIGHT * angular_speed
    )


def _hover_terminated(
    position_w: torch.Tensor,
    target_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> torch.Tensor:
    # Body-Z projected onto world-Z: the (0,0,1) column of the body->world
    # rotation is 1 - 2(x^2 + y^2) for a wxyz quaternion.
    body_z_world_z = 1.0 - 2.0 * (quat_w[:, 1] ** 2 + quat_w[:, 2] ** 2)
    tilted = body_z_world_z < _CRASH_TILT_COS
    too_low = position_w[:, 2] < _MIN_ALTITUDE_M
    position_error = torch.linalg.norm(position_w - target_w, dim=-1)
    out_of_bounds = position_error > _MAX_POSITION_ERROR_M
    return tilted | too_low | out_of_bounds


def _experiment_config():
    from robotics_experiments.config import get_config

    return get_config()


def _physics_dt() -> float:
    return 1.0 / _experiment_config().sim.physics_hz


def _decimation() -> int:
    cfg = _experiment_config()
    return max(1, round(cfg.sim.physics_hz / cfg.press.policy_rate_hz))


@configclass
class DroneHoverEnvCfg(DirectRLEnvCfg):
    """Configuration for the flyable drone-hover DirectRLEnv."""

    decimation = _decimation()
    episode_length_s = _experiment_config().press.episode_length_s
    action_space = _experiment_config().press.action.action_dim
    observation_space = 13
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=_physics_dt(), render_interval=decimation)
    drone_cfg: ArticulationCfg = make_drone_articulation_cfg(
        _experiment_config().drone.usd_path, _DRONE_PRIM_PATH
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_experiment_config().press.num_envs,
        env_spacing=_experiment_config().press.env_spacing,
        replicate_physics=True,
        clone_in_fabric=True,
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

    def __init__(self, cfg: DroneHoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._hover_target_w = torch.zeros((self.num_envs, 3), device=self.device)
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

        from robotics_experiments.config import resolve_drone_bridge_config
        from robotics_experiments.px4_bridge.rotor_discovery import discover_px4_rotor_mapping

        stage = omni.usd.get_context().get_stage()
        env0_path = _DRONE_PRIM_PATH.replace("env_.*", "env_0")
        rotors = discover_px4_rotor_mapping(stage, env0_path)
        bridge = resolve_drone_bridge_config(_experiment_config().drone.usd_path)

        masses = self.drone.root_physx_view.get_masses()
        total_mass = float(masses[0].sum().item())
        self._body_id = int(self.drone.find_bodies("body")[0][0])
        self._linear_drag = torch.as_tensor(
            bridge.linear_drag, dtype=torch.float32, device=self.device
        )
        action_cfg = _experiment_config().press.action
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
        omega = action_to_rotor_omega(
            self.actuation_params,
            self.actions,
            self.drone.data.root_quat_w,
            self.drone.data.root_lin_vel_w,
            self.drone.data.root_ang_vel_b,
        )
        force_body, torque_body = rotor_omega_to_body_wrench(
            self.actuation_params, omega, self.drone.data.root_lin_vel_b, self._linear_drag
        )
        apply_body_wrench(self.drone, self._body_id, force_body, torque_body)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = _hover_observation(
            self.drone.data.root_pos_w,
            self._hover_target_w,
            self.drone.data.root_quat_w,
            self.drone.data.root_lin_vel_w,
            self.drone.data.root_ang_vel_w,
        )
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        return _hover_reward(
            self.drone.data.root_pos_w,
            self._hover_target_w,
            self.drone.data.root_lin_vel_w,
            self.drone.data.root_ang_vel_w,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = _hover_terminated(
            self.drone.data.root_pos_w, self._hover_target_w, self.drone.data.root_quat_w
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.drone._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.drone.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        # The hover target is the (un-randomized) spawn pose; the start pose is
        # randomized within a small box around it so the policy learns to recover.
        self._hover_target_w[env_ids] = default_root_state[:, :3]
        offset = torch.zeros_like(default_root_state[:, :3])
        offset[:, :2].uniform_(-_RESET_XY_HALF_M, _RESET_XY_HALF_M)
        offset[:, 2].uniform_(-_RESET_Z_HALF_M, _RESET_Z_HALF_M)
        default_root_state[:, :3] += offset

        self.drone.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.drone.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

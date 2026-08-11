"""Ground-truth Isaac Lab DirectRLEnv for the press approach curriculum stage."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import cast

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from robotics_experiments.config import ExperimentConfig, RlPolicyConfig
from robotics_experiments.rl import PolicyContract
from robotics_experiments.rl.press import (
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    HandoffGate,
    PressApproachInput,
    PressApproachResult,
    PressObservationMask,
    initial_press_approach_state,
    pack_press_observation,
    reset_press_approach_state,
    step_press_approach,
)
from robotics_experiments.rl.press._switch_asset import ASSET_TO_CANONICAL_WXYZ, SQRT_HALF
from robotics_experiments.workflows.isaac_lab.tasks._drone_cfg import _make_drone_articulation_cfg
from robotics_experiments.workflows.isaac_lab.tasks._flight_env import (
    _DRONE_PRIM_PATH,
    _FlightDirectRLEnv,
    _training_vehicle,
)
from robotics_experiments.workflows.isaac_lab.tasks._press_reset import (
    _press_entry_state_from_unit_samples,
    _press_training_entry_state_from_unit_samples,
)
from robotics_experiments.workflows.isaac_lab.tasks.flight import (
    _normalized_action_to_controller_command,
)

__all__ = ["SwitchPressEnv", "SwitchPressEnvCfg", "make_switch_press_env_cfg"]

_SWITCH_PRIM_PATH = "/World/envs/env_.*/Switch"
_WALL_PRIM_PATH = "/World/envs/env_.*/Wall"
_BODY_PRIM_PATH = "/World/envs/env_.*/Drone/body"
_PAD_PRIM_PATH = "/World/envs/env_.*/Drone/pusher/pad"
# The source asset's outward plate normal is local -Y, while the policy
# contract defines canonical switch +X as outward. Spawning the asset at -pi/2
# about Z faces the physical button toward world -X; applying this same fixed
# correction to its root pose yields the canonical pi-about-Z provider pose.
_SWITCH_ASSET_ROT_WXYZ = (SQRT_HALF, 0.0, 0.0, -SQRT_HALF)
_TRAINING_REWARD_TERMS = (
    "range_progress",
    "range_barrier",
    "alignment_error",
    "velocity_tracking",
    "wall_normal_braking_command",
    "upright_error",
    "action_delta",
    "success",
    "collision",
    "flight_failure",
    "timeout",
    "total",
)


@configclass
class SwitchPressEnvCfg(DirectRLEnvCfg):
    """Configuration for the ground-truth switch-press DirectRLEnv."""

    decimation: int = cast(int, MISSING)
    policy_rate_hz: float = cast(float, MISSING)
    episode_length_s: float = cast(float, MISSING)
    action_space: int = cast(int, MISSING)
    observation_schema_version: int = cast(int, MISSING)
    observation_space: int = cast(int, MISSING)
    policy_interface: str = cast(str, MISSING)
    policy_config: RlPolicyConfig = cast(RlPolicyConfig, MISSING)
    use_training_reset_distribution: bool = cast(bool, MISSING)
    state_space = 0
    sim: SimulationCfg = cast(SimulationCfg, MISSING)
    drone_cfg: ArticulationCfg = cast(ArticulationCfg, MISSING)
    switch_cfg: ArticulationCfg = cast(ArticulationCfg, MISSING)
    wall_cfg: RigidObjectCfg = cast(RigidObjectCfg, MISSING)
    body_contact_sensor_cfg: ContactSensorCfg = cast(ContactSensorCfg, MISSING)
    pusher_contact_sensor_cfg: ContactSensorCfg = cast(ContactSensorCfg, MISSING)
    scene: InteractiveSceneCfg = cast(InteractiveSceneCfg, MISSING)


def make_switch_press_env_cfg(
    *,
    experiment_config: ExperimentConfig,
    policy_contract: PolicyContract | None = None,
    use_training_reset_distribution: bool = False,
) -> SwitchPressEnvCfg:
    """Build the press task from one immutable resolved experiment config."""
    vehicle = _training_vehicle(experiment_config)
    rl = experiment_config.rl
    reset = rl.press.reset
    if (
        policy_contract is not None
        and policy_contract.curriculum_stage != rl.press.curriculum_stage
    ):
        raise ValueError(
            f"press artifact stage {policy_contract.curriculum_stage!r} does not match "
            f"configured stage {rl.press.curriculum_stage!r}"
        )
    policy_rate_hz = (
        rl.policy_rate_hz if policy_contract is None else policy_contract.policy_rate_hz
    )
    action_dim = rl.action.action_dim if policy_contract is None else policy_contract.action_dim
    decimation = math.ceil(experiment_config.sim.physics_hz / policy_rate_hz)
    switch_position = (
        reset.switch_distance_from_env_origin_m,
        0.0,
        reset.switch_height_m,
    )
    wall_center = (
        switch_position[0] + reset.switch_wall_offset_m + reset.wall_half_extents_m[0],
        0.0,
        reset.switch_height_m,
    )
    return SwitchPressEnvCfg(
        decimation=decimation,
        policy_rate_hz=policy_rate_hz,
        episode_length_s=rl.episode_length_s,
        action_space=action_dim,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        observation_space=PRESS_OBSERVATION_DIM,
        policy_interface="press-approach",
        policy_config=rl.press.policy,
        use_training_reset_distribution=use_training_reset_distribution,
        state_space=0,
        sim=SimulationCfg(
            dt=1.0 / experiment_config.sim.physics_hz,
            render_interval=decimation,
            physx=PhysxCfg(
                enable_enhanced_determinism=experiment_config.sim.enhanced_determinism,
            ),
        ),
        drone_cfg=_make_drone_articulation_cfg(
            usd_path=vehicle.spawn.usd_path,
            prim_path=_DRONE_PRIM_PATH,
            lab_vehicle=rl.lab_vehicle,
        ),
        switch_cfg=ArticulationCfg(
            prim_path=_SWITCH_PRIM_PATH,
            spawn=UsdFileCfg(
                usd_path=experiment_config.switches.switch_usd_path,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=switch_position,
                rot=_SWITCH_ASSET_ROT_WXYZ,
            ),
            actuators={
                "button": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=None,
                    damping=None,
                )
            },
        ),
        wall_cfg=RigidObjectCfg(
            prim_path=_WALL_PRIM_PATH,
            spawn=sim_utils.CuboidCfg(
                size=tuple(2.0 * value for value in reset.wall_half_extents_m),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=reset.physical_contact_enabled
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=wall_center),
        ),
        body_contact_sensor_cfg=ContactSensorCfg(
            prim_path=_BODY_PRIM_PATH,
            update_period=0.0,
            history_length=0,
        ),
        pusher_contact_sensor_cfg=ContactSensorCfg(
            prim_path=_PAD_PRIM_PATH,
            update_period=0.0,
            history_length=0,
        ),
        scene=InteractiveSceneCfg(
            num_envs=rl.num_envs,
            env_spacing=rl.env_spacing,
            replicate_physics=True,
            clone_in_fabric=False,
        ),
    )


class SwitchPressEnv(_FlightDirectRLEnv):
    """One real switch articulation and pusher contact chain per cloned env."""

    cfg: SwitchPressEnvCfg

    def __init__(
        self,
        *,
        cfg: SwitchPressEnvCfg,
        experiment_config: ExperimentConfig,
        policy_contract=None,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(
            cfg=cfg,
            experiment_config=experiment_config,
            policy_contract=policy_contract,
            render_mode=render_mode,
            **kwargs,
        )
        switch_joint_ids, _ = self.switch.find_joints(".*")
        if len(switch_joint_ids) != 1:
            raise RuntimeError(
                f"press task requires exactly one switch joint, found {len(switch_joint_ids)}"
            )
        self._switch_joint_id = int(switch_joint_ids[0])
        stiffness = (
            self.switch.data.default_joint_stiffness[:, switch_joint_ids]
            * self._experiment_config.rl.press.reset.button_stiffness_scale
        )
        self.switch.write_joint_stiffness_to_sim(stiffness, joint_ids=switch_joint_ids)
        relative_position, _ = self._relative_switch_pose()
        self._task_state = initial_press_approach_state(
            relative_switch_range_m=relative_position[:, 0],
            action=self.actions,
        )
        self._task_result: PressApproachResult[torch.Tensor] | None = None
        self._training_hard_case_fraction = float(
            self._experiment_config.rl.press.training_reset.hard_case_fraction
        )
        self._physics_substep_index = 0
        self._body_collision_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pusher_collision_latch = torch.zeros_like(self._body_collision_latch)
        self._completed_episode_count = torch.zeros((), device=self.device)
        self._successful_episode_count = torch.zeros((), device=self.device)
        self._collision_episode_count = torch.zeros((), device=self.device)
        self._body_collision_episode_count = torch.zeros((), device=self.device)
        self._pusher_collision_episode_count = torch.zeros((), device=self.device)
        self._flight_failure_episode_count = torch.zeros((), device=self.device)
        self._timeout_episode_count = torch.zeros((), device=self.device)

    def set_training_hard_case_fraction(self, fraction: float, /) -> None:
        """Change the reset mixture at an in-run curriculum boundary."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("training hard-case fraction must be between 0 and 1")
        self._training_hard_case_fraction = float(fraction)

    @property
    def press_approach_result(self):
        """Most recent transition result for rollout evaluation."""
        return self._task_result

    def _setup_scene(self) -> None:
        self.drone = Articulation(self.cfg.drone_cfg)
        self.switch = Articulation(self.cfg.switch_cfg)
        self._scale_switch_button()
        self.wall = RigidObject(self.cfg.wall_cfg)
        self.body_contact_sensor = ContactSensor(self.cfg.body_contact_sensor_cfg)
        self.contact_sensor = ContactSensor(self.cfg.pusher_contact_sensor_cfg)
        self.scene.articulations["drone"] = self.drone
        self.scene.articulations["switch"] = self.switch
        self.scene.rigid_objects["wall"] = self.wall
        self.scene.sensors["body_contact"] = self.body_contact_sensor
        self.scene.sensors["pusher_contact"] = self.contact_sensor
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _scale_switch_button(self) -> None:
        """Enlarge the pressable target without enlarging the boom-blocking plate."""
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = sim_utils.get_current_stage()
        switch_root = stage.GetPrimAtPath("/World/envs/env_0/Switch")
        if not self._experiment_config.rl.press.reset.physical_contact_enabled:
            for prim in stage.Traverse():
                if prim.GetPath().HasPrefix(switch_root.GetPath()) and prim.HasAPI(
                    UsdPhysics.CollisionAPI
                ):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        buttons = [
            prim
            for prim in stage.Traverse()
            if prim.GetPath().HasPrefix(switch_root.GetPath()) and prim.GetName() == "ButtonMesh"
        ]
        if len(buttons) != 1:
            raise RuntimeError(f"switch requires exactly one ButtonMesh prim, found {len(buttons)}")
        button = buttons[0]
        scale = self._experiment_config.rl.press.reset.button_geometry_scale
        if button.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(button)
            points = mesh.GetPointsAttr().Get()
            mesh.GetPointsAttr().Set(
                [Gf.Vec3f(point[0] * scale, point[1] * scale, point[2] * scale) for point in points]
            )
        elif button.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(button)
            cube.GetSizeAttr().Set(float(cube.GetSizeAttr().Get()) * scale)
        else:
            raise RuntimeError(f"unsupported ButtonMesh type {button.GetTypeName()!r}")

    def _contact_frame_quat(self) -> torch.Tensor:
        switch_quat = self.switch.data.root_quat_w
        asset_to_canonical = torch.zeros_like(switch_quat)
        asset_to_canonical[:, :] = torch.as_tensor(
            ASSET_TO_CANONICAL_WXYZ,
            dtype=switch_quat.dtype,
            device=switch_quat.device,
        )
        canonical_switch_quat = math_utils.quat_mul(switch_quat, asset_to_canonical)
        half_turn_z = torch.zeros_like(switch_quat)
        half_turn_z[:, 3] = 1.0
        return math_utils.quat_mul(canonical_switch_quat, half_turn_z)

    def _relative_switch_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_quat = self._contact_frame_quat()
        offset_world = self.switch.data.root_pos_w - self.drone.data.root_pos_w
        relative_position = math_utils.quat_apply_inverse(contact_quat, offset_world)
        relative_quat_wxyz = math_utils.quat_mul(
            math_utils.quat_conjugate(contact_quat),
            self.drone.data.root_quat_w,
        )
        relative_quat_wxyz = torch.where(
            (relative_quat_wxyz[:, :1] < 0.0),
            -relative_quat_wxyz,
            relative_quat_wxyz,
        )
        relative_quat_xyzw = relative_quat_wxyz[:, [1, 2, 3, 0]]
        return relative_position, relative_quat_xyzw

    def _collision_flag(self, *, sensor: ContactSensor) -> torch.Tensor:
        force_norm = torch.linalg.norm(sensor.data.net_forces_w, dim=-1)
        threshold = self._experiment_config.rl.press.termination.contact_force_threshold_n
        return force_norm.amax(dim=-1) >= threshold

    def _contact_flag(self) -> torch.Tensor:
        """Return instantaneous pusher/pad contact for the policy observation."""
        return self._collision_flag(sensor=self.contact_sensor)

    def _latch_collision_sample(self) -> None:
        self._body_collision_latch |= self._collision_flag(sensor=self.body_contact_sensor)
        self._pusher_collision_latch |= self._collision_flag(sensor=self.contact_sensor)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self._physics_substep_index = 0
        self._body_collision_latch.zero_()
        self._pusher_collision_latch.zero_()

    def _apply_action(self) -> None:
        # At substep N, sensor buffers contain the result of substep N-1. Skip
        # the first read because it belongs to the preceding policy interval;
        # _sample() captures the final substep after the decimation loop.
        if self._physics_substep_index > 0:
            self._latch_collision_sample()
        self._physics_substep_index += 1
        super()._apply_action()

    def _sample(self, *, time_out: torch.Tensor) -> PressApproachInput[torch.Tensor]:
        self._latch_collision_sample()
        relative_position, relative_quat_xyzw = self._relative_switch_pose()
        velocity_command_world = _normalized_action_to_controller_command(
            self.actuation_params,
            self.actions,
            self.drone.data.root_quat_w,
        )[:, :3]
        quat = self.drone.data.root_quat_w
        body_z_world_z = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
        relative_quat_wxyz = relative_quat_xyzw[:, [3, 0, 1, 2]]
        return PressApproachInput(
            relative_switch_pos_m=relative_position,
            linear_velocity_mps=math_utils.quat_apply(
                relative_quat_wxyz,
                self.drone.data.root_lin_vel_b,
            ),
            wall_normal_velocity_command_mps=math_utils.quat_apply_inverse(
                self._contact_frame_quat(), velocity_command_world
            )[:, 0],
            angular_velocity_radps=self.drone.data.root_ang_vel_b,
            body_z_world_z=body_z_world_z,
            altitude_m=self.drone.data.root_pos_w[:, 2],
            body_collision_flag=self._body_collision_latch.clone(),
            pusher_collision_flag=self._pusher_collision_latch.clone(),
            action=self.actions,
            time_out=time_out,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        press = self._experiment_config.rl.press
        self._task_result = step_press_approach(
            state=self._task_state,
            sample=self._sample(time_out=time_out),
            goal_cfg=press.goal,
            reward_cfg=press.reward,
            termination_cfg=press.termination,
        )
        result = self._task_result
        self._task_state = result.state
        return result.termination.terminated, result.termination.time_out

    def _get_rewards(self) -> torch.Tensor:
        if self._task_result is None:
            raise RuntimeError("press transition must advance in _get_dones before rewards")
        result = self._task_result
        ended = self.reset_buf
        collision = result.termination.collision
        self._completed_episode_count += ended.count_nonzero()
        self._successful_episode_count += (ended & result.termination.success).count_nonzero()
        self._collision_episode_count += (ended & collision).count_nonzero()
        self._body_collision_episode_count += (
            ended & result.termination.body_collision
        ).count_nonzero()
        self._pusher_collision_episode_count += (
            ended & result.termination.pusher_collision
        ).count_nonzero()
        self._flight_failure_episode_count += (
            ended & result.termination.flight_failure & ~collision
        ).count_nonzero()
        self._timeout_episode_count += (ended & result.termination.time_out).count_nonzero()
        episode_denominator = self._completed_episode_count.clamp_min(1.0)
        self.extras["log"] = {
            "Press/completed_episodes": self._completed_episode_count,
            "Press/success_percent": 100.0 * self._successful_episode_count / episode_denominator,
            "Press/collision_percent": 100.0 * self._collision_episode_count / episode_denominator,
            "Press/body_collision_percent": 100.0
            * self._body_collision_episode_count
            / episode_denominator,
            "Press/pusher_collision_percent": 100.0
            * self._pusher_collision_episode_count
            / episode_denominator,
            "Press/flight_failure_percent": 100.0
            * self._flight_failure_episode_count
            / episode_denominator,
            "Press/timeout_percent": 100.0 * self._timeout_episode_count / episode_denominator,
        }
        self.extras["log"].update(
            {
                f"Reward/{name}": getattr(result.rewards, name).mean()
                for name in _TRAINING_REWARD_TERMS
            }
        )
        return result.rewards.total

    def _get_observations(self) -> dict[str, torch.Tensor]:
        relative_position, relative_quat_xyzw = self._relative_switch_pose()
        quat_wxyz = self.drone.data.root_quat_w
        attitude_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        ones = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        zeros = torch.zeros_like(ones)
        mask = float(
            PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
        )
        observations = pack_press_observation(
            relative_switch_pos_m=relative_position,
            relative_switch_quat_xyzw=relative_quat_xyzw,
            pose_confidence=ones,
            pose_age_s=zeros,
            detection_valid=ones,
            body_lin_vel_mps=self.drone.data.root_lin_vel_b,
            body_ang_vel_radps=self.drone.data.root_ang_vel_b,
            attitude_quat_xyzw=attitude_xyzw,
            pusher_compression_m=(
                self._pusher_extension_target[:, 0]
                - self.drone.data.joint_pos[:, self._pusher_joint_id]
            ),
            pusher_compression_rate_mps=-self.drone.data.joint_vel[:, self._pusher_joint_id],
            contact_valid=ones,
            contact_flag=self._contact_flag(),
            lamp_state=zeros,
            lamp_confidence=ones,
            lamp_age_s=zeros,
            previous_action=self._task_state.previous_action,
            handoff_mode=ones,
            estimator_health=ones,
            obs_mask=torch.full_like(ones, mask),
        )
        return {"policy": observations}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        handoff_gate = HandoffGate.from_config(self._experiment_config.rl.press.handoff)
        switch_position = self.switch.data.default_root_state[env_ids, :3]
        switch_position = switch_position + self.scene.env_origins[env_ids]
        count = len(env_ids)
        if self.cfg.use_training_reset_distribution:
            wall_normal, lateral_vertical, linear_velocity_contact, angular_velocity = (
                _press_training_entry_state_from_unit_samples(
                    unit_samples=torch.rand((count, 10), device=self.device),
                    gate=handoff_gate,
                    training_reset=self._experiment_config.rl.press.training_reset,
                    hard_case_fraction=self._training_hard_case_fraction,
                )
            )
        else:
            wall_normal, lateral_vertical, linear_velocity_contact, angular_velocity = (
                _press_entry_state_from_unit_samples(
                    unit_samples=torch.rand((count, 9), device=self.device),
                    gate=handoff_gate,
                )
            )
        root_state = self.drone.data.default_root_state[env_ids].clone()
        root_state[:, :3] = switch_position - torch.cat(
            (wall_normal[:, None], lateral_vertical),
            dim=-1,
        )
        root_state[:, 7:10] = math_utils.quat_apply(
            self._contact_frame_quat()[env_ids],
            linear_velocity_contact,
        )
        root_state[:, 10:13] = angular_velocity
        self._reset_drone(env_ids=env_ids, root_state=root_state)
        switch_joint_position = self.switch.data.default_joint_pos[env_ids].clone()
        switch_joint_velocity = self.switch.data.default_joint_vel[env_ids].clone()
        self.switch.write_joint_state_to_sim(
            switch_joint_position,
            switch_joint_velocity,
            env_ids=env_ids,
        )
        if hasattr(self, "_task_state"):
            self._task_state = reset_press_approach_state(
                state=self._task_state,
                env_ids=env_ids,
                relative_switch_range_m=wall_normal,
                action=torch.zeros((count, self.cfg.action_space), device=self.device),
            )
            self._body_collision_latch[env_ids] = False
            self._pusher_collision_latch[env_ids] = False

"""Ground-truth Isaac Lab hover, approach, contact and press tasks."""

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
from robotics_experiments.flight_parameters import FlightParameters
from robotics_experiments.rl import HoverResetSamples, PolicyContract
from robotics_experiments.rl.press import (
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    HandoffGate,
    PressApproachInput,
    PressApproachResult,
    PressObservationMask,
    initial_press_approach_state,
    pack_press_observation,
    quaternion_geodesic_error_rad,
    reset_press_approach_state,
    step_press_approach,
)
from robotics_experiments.rl.press._switch_asset import ASSET_TO_CANONICAL_WXYZ, SQRT_HALF
from robotics_experiments.rl.press.contact import (
    PressContactResult,
    classify_press_contacts,
    contact_force_totals,
    initial_press_contact_state,
    reset_press_contact_state,
    step_press_contact,
)
from robotics_experiments.switches import (
    ButtonLatchState,
    SwitchAsset,
    load_switch_asset,
    step_button_latch,
)
from robotics_experiments.workflows.isaac_lab.tasks._drone_cfg import _make_drone_articulation_cfg
from robotics_experiments.workflows.isaac_lab.tasks._flight_env import (
    _DRONE_PRIM_PATH,
    _FlightDirectRLEnv,
    _training_vehicle,
)
from robotics_experiments.workflows.isaac_lab.tasks._press_reset import (
    _press_entry_state_from_unit_samples,
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
_TRAINING_REWARD_TERMS = {
    "position_progress": "Position correction reward per step",
    "stabilization": "Pose stabilization reward per step",
    "predictive_barrier": "Predicted range violation cost per step",
    "action_effort": "Action effort cost per step",
    "hold_progress": "New longest hover reward per step",
    "terminal_outcome": "Episode outcome reward per step",
}


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
    switch_asset: SwitchAsset,
    policy_contract: PolicyContract | None = None,
) -> SwitchPressEnvCfg:
    """Build the press task from one immutable resolved experiment config."""
    vehicle = _training_vehicle(experiment_config)
    rl = experiment_config.rl
    reset = rl.press.reset
    if policy_contract is not None and policy_contract.curriculum_stage != rl.press.task:
        raise ValueError(
            f"press artifact stage {policy_contract.curriculum_stage!r} does not match "
            f"configured stage {rl.press.task!r}"
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
        policy_interface="press",
        policy_config=rl.press.policy,
        state_space=0,
        is_finite_horizon=False,
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
                usd_path=str(
                    load_switch_asset(
                        model=experiment_config.switches.model, expected=switch_asset
                    ).path
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=reset.physical_contact_enabled
                ),
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
            max_contact_data_count_per_prim=rl.press.termination.contact_data_capacity_per_body,
            filter_prim_paths_expr=[
                f"{_SWITCH_PRIM_PATH}/LightSwitch/Button",
                f"{_SWITCH_PRIM_PATH}/LightSwitch/BaseMesh",
                _WALL_PRIM_PATH,
                _PAD_PRIM_PATH,
            ],
        ),
        pusher_contact_sensor_cfg=ContactSensorCfg(
            prim_path=_PAD_PRIM_PATH,
            update_period=0.0,
            history_length=0,
            max_contact_data_count_per_prim=rl.press.termination.contact_data_capacity_per_body,
            filter_prim_paths_expr=[
                f"{_SWITCH_PRIM_PATH}/LightSwitch/Button",
                f"{_SWITCH_PRIM_PATH}/LightSwitch/BaseMesh",
                _WALL_PRIM_PATH,
                _BODY_PRIM_PATH,
            ],
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
        flight_parameters: FlightParameters,
        hover_reset_samples: HoverResetSamples,
        training: bool,
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
        self._training = training
        self._flight_parameters = flight_parameters
        self._hover_reset_position_errors = torch.as_tensor(
            hover_reset_samples.position_error_enu_m,
            dtype=torch.float32,
            device=self.device,
        )
        self._hover_reset_attitude_errors = torch.as_tensor(
            hover_reset_samples.attitude_error_flu_enu_wxyz,
            dtype=torch.float32,
            device=self.device,
        )
        self._hover_reset_linear_velocities = torch.as_tensor(
            hover_reset_samples.linear_velocity_enu_mps,
            dtype=torch.float32,
            device=self.device,
        )
        self._hover_reset_angular_velocities = torch.as_tensor(
            hover_reset_samples.angular_velocity_enu_radps,
            dtype=torch.float32,
            device=self.device,
        )
        relative_position, relative_quat_xyzw = self._relative_switch_pose()
        position_error_enu_m, attitude_error_wxyz = self._hover_pose_errors(
            relative_position=relative_position,
            relative_quat_xyzw=relative_quat_xyzw,
        )
        self._task_state = initial_press_approach_state(
            relative_switch_pos_m=relative_position,
            relative_switch_quat_xyzw=relative_quat_xyzw,
            goal_cfg=self._experiment_config.rl.press.goal,
            flight_parameters=self._flight_parameters,
            position_error_enu_m=position_error_enu_m,
            attitude_error_flu_enu_wxyz=attitude_error_wxyz,
        )
        self._task_result: PressApproachResult[torch.Tensor] | None = None
        self._task_sample: PressApproachInput[torch.Tensor] | None = None
        self._physics_substep_index = 0
        self._body_collision_latch = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pusher_collision_latch = torch.zeros_like(self._body_collision_latch)
        self._completed_episode_count = torch.zeros((), device=self.device)
        self._contact_task = self._experiment_config.rl.press.task in {"contact", "press"}
        self._button_latch = ButtonLatchState(
            torch.zeros_like(relative_position[:, 0]), torch.zeros_like(self._body_collision_latch)
        )
        self._button_toggle_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._contact_categories = torch.zeros(
            (self.num_envs, 5), dtype=torch.bool, device=self.device
        )
        self._simultaneous_contact = torch.zeros_like(self._body_collision_latch)
        self._contact_state = initial_press_contact_state(ranges=relative_position[:, 0])
        self._contact_result: PressContactResult | None = None
        self._intended_contact = torch.zeros_like(self._body_collision_latch)
        self._compression_peak_m = torch.zeros_like(relative_position[:, 0])
        self._contact_seen = torch.zeros_like(self._body_collision_latch)
        self._first_contact_stable = torch.zeros_like(self._body_collision_latch)
        self._previous_contact_quality = torch.zeros_like(self._body_collision_latch)
        if self._contact_task:
            stroke = (
                self.drone.data.joint_pos_limits[:, self._pusher_joint_id, 1]
                - (self.drone.data.joint_pos_limits[:, self._pusher_joint_id, 0])
            )
            if bool(torch.any(stroke < self._experiment_config.rl.press.contact.compression_max_m)):
                raise ValueError("contact compression limit exceeds the authored pusher stroke")

    @property
    def press_contact_result(self):
        """Contact evidence from the completed step, retained across auto-reset."""
        return self._contact_result

    @property
    def press_approach_result(self):
        """Most recent transition result for rollout evaluation."""
        return self._task_result

    def _setup_scene(self) -> None:
        self.drone = Articulation(self.cfg.drone_cfg)
        self.switch = Articulation(self.cfg.switch_cfg)
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

    def _hover_pose_errors(
        self,
        *,
        relative_position: torch.Tensor,
        relative_quat_xyzw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_range = self._experiment_config.rl.press.goal.velocity_reference.range_center_m
        target_relative = torch.zeros_like(relative_position)
        target_relative[:, 0] = target_range
        position_error = math_utils.quat_apply(
            self._contact_frame_quat(), target_relative - relative_position
        )
        return position_error, relative_quat_xyzw[:, [3, 0, 1, 2]]

    def _contact_pair_forces(self, *, sensor: ContactSensor) -> torch.Tensor:
        # Sum positive normal force magnitudes per filtered pair. Opposing
        # contacts cannot cancel, including a button and housing hit together.
        forces, _points, _normals, _distances, counts, starts = (
            sensor.contact_physx_view.get_contact_data(dt=self.physics_dt)
        )
        sums = contact_force_totals(forces=forces, counts=counts, starts=starts)
        pairs = sums.reshape(self.num_envs, -1)
        # GPU PhysX filters rigid actors only. The sole unfiltered collider in
        # this scene is the static ground plane, whose normals are parallel.
        # Remove the explicit actor vectors before measuring its remaining force.
        view = sensor.contact_physx_view
        vectors = view.get_contact_force_matrix(dt=self.physics_dt).reshape(self.num_envs, -1, 3)
        net = view.get_net_contact_forces(dt=self.physics_dt).reshape(self.num_envs, 3)
        ground = torch.linalg.vector_norm(net - vectors.sum(dim=1), dim=-1)
        return torch.cat((pairs, ground[:, None]), dim=-1)

    def _contact_flag(self) -> torch.Tensor:
        threshold = self._experiment_config.rl.press.termination.contact_force_threshold_n
        return self._contact_pair_forces(sensor=self.contact_sensor).amax(dim=-1) >= threshold

    def _latch_collision_sample(self) -> None:
        self._button_latch, toggled = step_button_latch(
            state=self._button_latch,
            displacement_m=self.switch.data.joint_pos[:, self._switch_joint_id],
            dt_s=self.physics_dt,
            model=self._experiment_config.switches.model,
        )
        self._button_toggle_count += toggled.long()
        threshold = self._experiment_config.rl.press.termination.contact_force_threshold_n
        body_force = self._contact_pair_forces(sensor=self.body_contact_sensor).amax(dim=-1)
        forces = self._contact_pair_forces(sensor=self.contact_sensor)
        categories = classify_press_contacts(
            pad_forces_n=forces, body_force_n=body_force, threshold_n=threshold
        )
        body = categories[:, 4]
        self._contact_categories |= categories
        self._simultaneous_contact |= categories.sum(dim=-1) > 1
        self._body_collision_latch |= body
        if not self._contact_task:
            self._pusher_collision_latch |= categories[:, :4].any(dim=-1)
            return
        self._compression_peak_m = torch.maximum(
            self._compression_peak_m,
            self._pusher_extension_target[:, 0]
            - self.drone.data.joint_pos[:, self._pusher_joint_id],
        )
        intended = categories[:, 0]
        self._pusher_collision_latch |= categories[:, 1:4].any(dim=-1)
        self._intended_contact |= intended
        first = intended & ~self._contact_seen
        self._first_contact_stable |= first & self._previous_contact_quality
        self._contact_seen |= intended
        relative_position, relative_quat = self._relative_switch_pose()
        goal = self._experiment_config.rl.press.goal
        self._previous_contact_quality = (
            (
                torch.linalg.vector_norm(relative_position[:, 1:3], dim=-1)
                <= goal.alignment_error_max_m
            )
            & (
                quaternion_geodesic_error_rad(quaternion_xyzw=relative_quat)
                <= goal.relative_attitude_error_max_rad
            )
            & (
                torch.linalg.vector_norm(self.drone.data.root_lin_vel_b, dim=-1)
                <= goal.linear_speed_max_mps
            )
            & (
                torch.linalg.vector_norm(self.drone.data.root_ang_vel_b, dim=-1)
                <= goal.angular_speed_max_radps
            )
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self._physics_substep_index = 0
        self._body_collision_latch.zero_()
        self._pusher_collision_latch.zero_()
        self._intended_contact.zero_()
        self._compression_peak_m.zero_()
        self._contact_categories.zero_()
        self._simultaneous_contact.zero_()

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
        position_error_enu_m, attitude_error_wxyz = self._hover_pose_errors(
            relative_position=relative_position,
            relative_quat_xyzw=relative_quat_xyzw,
        )
        quat = self.drone.data.root_quat_w
        body_z_world_z = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
        relative_quat_wxyz = relative_quat_xyzw[:, [3, 0, 1, 2]]
        return PressApproachInput(
            relative_switch_pos_m=relative_position,
            relative_switch_quat_xyzw=relative_quat_xyzw,
            position_error_enu_m=position_error_enu_m,
            attitude_error_flu_enu_wxyz=attitude_error_wxyz,
            linear_velocity_mps=math_utils.quat_apply(
                relative_quat_wxyz,
                self.drone.data.root_lin_vel_b,
            ),
            angular_velocity_radps=self.drone.data.root_ang_vel_b,
            normalized_action=self.actions,
            body_z_world_z=body_z_world_z,
            altitude_m=self.drone.data.root_pos_w[:, 2],
            body_collision_flag=self._body_collision_latch.clone(),
            pusher_collision_flag=self._pusher_collision_latch.clone(),
            time_out=time_out,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # DirectRLEnv increments the clock before checking completion.
        horizon = self.max_episode_length
        if self._training and self._experiment_config.rl.press.task == "hover":
            time_out = self.episode_length_buf >= horizon
        else:
            time_out = self.episode_length_buf >= horizon - 1
        press = self._experiment_config.rl.press
        self._task_sample = self._sample(time_out=time_out)
        self._task_result = step_press_approach(
            state=self._task_state,
            sample=self._task_sample,
            goal_cfg=press.goal,
            reward_cfg=press.reward,
            termination_cfg=press.termination,
            minimum_range_m=press.acceptance.minimum_range_m,
            flight_parameters=self._flight_parameters,
            hold_steps=self._experiment_config.rl.press.goal.stable_steps,
            approach_enabled=self._experiment_config.rl.press.task == "approach",
            training=self._training or self._contact_task,
        )
        self._task_state = self._task_result.state
        if self._contact_task:
            self._contact_result = step_press_contact(
                state=self._contact_state,
                flight=self._task_result,
                button_position_m=self.switch.data.joint_pos[:, self._switch_joint_id],
                compression_m=self._compression_peak_m,
                intended_contact=self._intended_contact,
                contact_categories=self._contact_categories.clone(),
                simultaneous_contact=self._simultaneous_contact.clone(),
                contact_seen=self._contact_seen.clone(),
                first_contact_stable=self._first_contact_stable.clone(),
                action=self.actions,
                config=press,
                press_threshold_m=self._experiment_config.switches.model.activation_displacement_m,
                toggle_count=self._button_toggle_count.clone(),
            )
            self._contact_state = self._contact_result.state
            return (
                self._contact_result.success | self._contact_result.failure,
                self._contact_result.timeout,
            )
        result = self._task_result
        return result.termination.terminated, result.termination.time_out

    def _get_rewards(self) -> torch.Tensor:
        if self._contact_task:
            if self._contact_result is None:
                raise RuntimeError("contact transition must advance before rewards")
            self.extras["log"] = {
                f"training/reward/{name}": value.mean()
                for name, value in self._contact_result.rewards.items()
            }
            self.extras["log"]["training/Contact task progress"] = (
                self._contact_result.progress.mean()
            )
            return self._contact_result.reward
        if self._task_result is None:
            raise RuntimeError("press transition must advance in _get_dones before rewards")
        result = self._task_result
        ended = self.reset_buf
        self._completed_episode_count += ended.count_nonzero()
        self.extras["log"] = {
            "training/Completed training episodes (count)": self._completed_episode_count,
            "training/Fraction of samples inside hover limits": (result.strict_goal.float().mean()),
            "training/Mean longest hover so far in active episodes (seconds)": (
                result.state.longest_stable_goal_steps.float().mean() * self.step_dt
            ),
            "training/Mean worst pose error relative to hover limit (1 = limit)": (
                result.state.normalized_pose_error.mean()
            ),
            "training/Mean position error relative to hover limit (1 = limit)": (
                result.hover_evaluation.position_utilization.mean()
            ),
            "training/Mean attitude error relative to hover limit (1 = limit)": (
                result.hover_evaluation.attitude_utilization.mean()
            ),
        }
        self.extras["log"].update(
            {
                f"training/reward/{label}": getattr(result.rewards, name).mean()
                for name, label in _TRAINING_REWARD_TERMS.items()
            }
        )
        self.extras["log"].update(
            {
                "training/reward/Total reward per step": result.rewards.total.mean(),
            }
        )
        return result.rewards.total

    def _get_observations(self) -> dict[str, torch.Tensor]:
        relative_position, relative_quat_xyzw = self._relative_switch_pose()
        position_error_enu_m, _attitude_error_wxyz = self._hover_pose_errors(
            relative_position=relative_position,
            relative_quat_xyzw=relative_quat_xyzw,
        )
        quat_wxyz = self.drone.data.root_quat_w
        attitude_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        ones = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        zeros = torch.zeros_like(ones)
        mask = float(
            PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
        )
        observations = pack_press_observation(
            position_error_enu_m=position_error_enu_m,
            goal_quat_world_wxyz=self._contact_frame_quat(),
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
            lamp_state=(self._button_toggle_count.remainder(2).float()),
            lamp_confidence=ones,
            lamp_age_s=zeros,
            previous_action=self.actions,
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
        navigation = _press_entry_state_from_unit_samples(
            unit_samples=torch.rand((count, 12), device=self.device),
            gate=handoff_gate,
        )
        hover_indices = torch.randint(
            self._hover_reset_position_errors.shape[0],
            (count,),
            device=self.device,
        )
        position_error_enu_m = self._hover_reset_position_errors[hover_indices]
        hover_quat_wxyz = self._hover_reset_attitude_errors[hover_indices]
        hover_linear_velocity = self._hover_reset_linear_velocities[hover_indices]
        hover_angular_velocity = self._hover_reset_angular_velocities[hover_indices]
        target_range = self._experiment_config.rl.press.goal.velocity_reference.range_center_m
        contact_quat = self._contact_frame_quat()[env_ids]
        position_error_contact = math_utils.quat_apply_inverse(contact_quat, position_error_enu_m)
        hover_wall_normal = target_range - position_error_contact[:, 0]
        hover_lateral_vertical = -position_error_contact[:, 1:3]
        reset_weights = {
            source.source: source.weight for source in self._experiment_config.rl.press.reset.mix
        }
        unknown_sources = reset_weights.keys() - {
            "measured_navigation_handoff",
            "pinned_flight_parameters",
        }
        if unknown_sources:
            raise ValueError(f"unsupported press reset sources: {sorted(unknown_sources)}")
        navigation_mask = torch.rand(count, device=self.device) < reset_weights.get(
            "measured_navigation_handoff", 0.0
        )
        wall_normal = torch.where(navigation_mask, navigation[0], hover_wall_normal)
        lateral_vertical = torch.where(
            navigation_mask[:, None], navigation[1], hover_lateral_vertical
        )
        relative_quat_wxyz = torch.where(navigation_mask[:, None], navigation[4], hover_quat_wxyz)
        root_state = self.drone.data.default_root_state[env_ids].clone()
        root_state[:, :3] = switch_position - torch.cat(
            (wall_normal[:, None], lateral_vertical),
            dim=-1,
        )
        root_state[:, 3:7] = math_utils.quat_mul(contact_quat, relative_quat_wxyz)
        root_state[:, 7:10] = torch.where(
            navigation_mask[:, None],
            math_utils.quat_apply(contact_quat, navigation[2]),
            hover_linear_velocity,
        )
        root_state[:, 10:13] = torch.where(
            navigation_mask[:, None],
            navigation[3],
            hover_angular_velocity,
        )
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
                relative_switch_pos_m=torch.cat((wall_normal[:, None], lateral_vertical), dim=-1),
                relative_switch_quat_xyzw=relative_quat_wxyz[:, [1, 2, 3, 0]],
                goal_cfg=self._experiment_config.rl.press.goal,
                flight_parameters=self._flight_parameters,
                position_error_enu_m=math_utils.quat_apply(
                    contact_quat,
                    torch.cat(
                        (
                            torch.full_like(
                                wall_normal[:, None],
                                self._experiment_config.rl.press.goal.velocity_reference.range_center_m,
                            )
                            - wall_normal[:, None],
                            -lateral_vertical,
                        ),
                        dim=-1,
                    ),
                ),
                attitude_error_flu_enu_wxyz=relative_quat_wxyz,
            )
            self._body_collision_latch[env_ids] = False
            self._pusher_collision_latch[env_ids] = False
            self._contact_state = reset_press_contact_state(
                state=self._contact_state,
                env_ids=torch.as_tensor(env_ids, device=self.device),
                ranges=wall_normal,
            )
            self._intended_contact[env_ids] = False
            self._compression_peak_m[env_ids] = 0.0
            self._button_latch.duration_s[env_ids] = 0.0
            self._button_latch.latched[env_ids] = False
            self._button_toggle_count[env_ids] = 0
            self._contact_categories[env_ids] = False
            self._simultaneous_contact[env_ids] = False
            self._contact_seen[env_ids] = False
            self._first_contact_stable[env_ids] = False
            self._previous_contact_quality[env_ids] = False

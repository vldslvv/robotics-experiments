from dataclasses import dataclass
from typing import List, Tuple

from robotics_experiments.config._immutability import _FreezableConfig

__all__ = [
    "HoverTaskConfig",
    "IsaacLabVehicleConfig",
    "PressActionConfig",
    "PressApproachEvaluationConfig",
    "PressApproachGoalConfig",
    "PressApproachRewardConfig",
    "PressApproachTerminationConfig",
    "PressHandoffConfig",
    "PressResetConfig",
    "PressTaskConfig",
    "RlConfig",
    "RlPolicyActorConfig",
    "RlPolicyConfig",
    "RlPolicyCriticConfig",
    "RlTrainingConfig",
]


@dataclass
class IsaacLabVehicleConfig(_FreezableConfig):
    """Isaac-Lab-only vehicle spawn and solver limits.

    Pusher stiffness and damping deliberately do not live here: the generated
    USD is authoritative for those physical values and Lab inherits them.
    """

    spawn_height_m: float
    max_linear_velocity_mps: float
    max_angular_velocity_radps: float
    max_depenetration_velocity_mps: float
    pusher_effort_limit_n: float
    pusher_velocity_limit_mps: float


@dataclass
class PressActionConfig(_FreezableConfig):
    """RL action-level (``kind``) selection plus its controller tunables.

    The ``kind`` picks the in-Lab actuation model (see
    :mod:`robotics_experiments.rl`): ``velocity`` (cascaded PD:
    body-frame velocity + yaw rate) or ``ctbr`` (collective thrust + body-rate
    setpoint). The gains/limits below are a superset covering both; the selected
    ``kind`` reads the subset it needs, so switching is a one-line ``kind`` edit.

    Attributes:
        kind: Action level selecting the actuation model (``velocity`` /
            ``ctbr``; other levels are reserved and raise if selected).
        action_dim: Length of the normalized policy action vector (4 for both
            ``velocity`` = ``[vx, vy, vz, yaw_rate]`` and ``ctbr`` =
            ``[thrust, wx, wy, wz]``).
        max_body_velocity_mps: Body-frame linear-velocity command scale (m/s) a
            normalized ``velocity`` action of magnitude 1 maps to.
        max_yaw_rate_radps: Yaw-rate command scale (rad/s) for a normalized
            ``velocity`` action.
        max_body_rate_radps: Body-rate command scale (rad/s) for a normalized
            ``ctbr`` action.
        max_tilt_cmd_deg: Saturation on the commanded tilt of the desired thrust
            axis in the ``velocity`` cascade, degrees.
        hover_thrust_fraction: Collective thrust at hover as a fraction of the
            summed per-rotor static max thrust; centers the ``ctbr`` collective
            command and the ``velocity`` thrust feed-forward.
        controller_kp_vel: Per-axis P gain on world-frame velocity error
            (``velocity`` cascade), length 3.
        controller_kd_vel: Per-axis damping gain on world-frame velocity
            (``velocity`` cascade), length 3.
        controller_kp_att: Per-axis P gain mapping geometric attitude error to a
            body-rate setpoint (``velocity`` cascade), length 3.
        controller_kp_rate: Per-axis P gain mapping body-rate error to desired
            body torque (both models), length 3.
    """

    kind: str
    action_dim: int
    max_body_velocity_mps: float
    max_yaw_rate_radps: float
    max_body_rate_radps: float
    max_tilt_cmd_deg: float
    hover_thrust_fraction: float
    controller_kp_vel: List[float]
    controller_kd_vel: List[float]
    controller_kp_att: List[float]
    controller_kp_rate: List[float]


@dataclass
class RlPolicyActorConfig(_FreezableConfig):
    """Actor topology and exploration initialization."""

    hidden_dims: List[int]
    initial_noise_std: float


@dataclass
class RlPolicyCriticConfig(_FreezableConfig):
    """Critic topology."""

    hidden_dims: List[int]


@dataclass
class RlPolicyConfig(_FreezableConfig):
    """Task-owned rsl_rl ``ActorCritic`` topology."""

    activation: str
    actor: RlPolicyActorConfig
    critic: RlPolicyCriticConfig


@dataclass
class HoverTaskConfig(_FreezableConfig):
    """The hover task definition: reward weights, termination, and reset ranges.

    The task half of the ``task definition -> policy`` contract for the hover
    task (`Isaac-Drone-Hover-Direct-v0`): authoring these values IS defining the
    task; the env module carries no task-tuning literals.

    Attributes:
        reward_pos_weight: Penalty weight on the position error norm, per meter.
        reward_linvel_weight: Penalty weight on linear speed, per m/s.
        reward_angvel_weight: Penalty weight on angular speed, per rad/s.
        reward_alive: Constant per-step alive bonus.
        crash_tilt_cos: Terminate when the body-Z world-Z alignment (cosine)
            drops below this (0.0 = more than 90 deg tilt).
        min_altitude_m: Terminate below this altitude, meters.
        max_position_error_m: Terminate beyond this distance from the hover
            target, meters.
        reset_xy_half_m: Half-extent of the horizontal start-pose randomization
            box around the hover target, meters.
        reset_z_half_m: Half-extent of the vertical start-pose randomization box
            around the hover target, meters.
    """

    policy: RlPolicyConfig
    reward_pos_weight: float
    reward_linvel_weight: float
    reward_angvel_weight: float
    reward_alive: float
    crash_tilt_cos: float
    min_altitude_m: float
    max_position_error_m: float
    reset_xy_half_m: float
    reset_z_half_m: float


@dataclass
class PressHandoffConfig(_FreezableConfig):
    """Thresholds for the nav -> contact handoff gate (``rl.press.handoff_ready``).

    All checks are over the task-relative :class:`PressObservation` (contact
    frame); the gate decides whether the contact behavior may start.

    Attributes:
        pose_confidence_min: Minimum pose confidence, [0, 1].
        pose_age_max_s: Maximum pose-estimate age, seconds.
        switch_range_max_m: Maximum wall-normal range to the switch, meters.
        lateral_align_max_m: Maximum lateral (in-plate) alignment error, meters.
        wall_normal_speed_max_mps: Maximum speed along the press axis, m/s.
        lateral_speed_max_mps: Maximum in-plate speed, m/s.
        angular_speed_max_radps: Maximum body angular speed, rad/s.
        estimator_health_min: Minimum aggregate estimator health, [0, 1].
    """

    pose_confidence_min: float
    pose_age_max_s: float
    switch_range_max_m: float
    lateral_align_max_m: float
    wall_normal_speed_max_mps: float
    lateral_speed_max_mps: float
    angular_speed_max_radps: float
    estimator_health_min: float


@dataclass
class PressApproachRewardConfig(_FreezableConfig):
    """Weights for the nine signed approach-stage reward contributions."""

    range_progress_weight: float
    alignment_error_weight: float
    closing_overspeed_weight: float
    upright_error_weight: float
    action_delta_weight: float
    success_weight: float
    collision_weight: float
    flight_failure_weight: float
    timeout_weight: float


@dataclass
class PressApproachGoalConfig(_FreezableConfig):
    """Stable approach target and wall-normal braking schedule."""

    range_min_m: float
    range_max_m: float
    alignment_error_max_m: float
    linear_speed_max_mps: float
    angular_speed_max_radps: float
    body_z_alignment_min: float
    stable_steps: int
    closing_speed_far_mps: float
    closing_speed_far_range_m: float
    closing_speed_zero_range_m: float


@dataclass
class PressResetConfig(_FreezableConfig):
    """Easy-task geometry and initial-state ranges for switch pressing."""

    button_geometry_scale: float
    button_stiffness_scale: float
    button_threshold_scale: float
    switch_distance_from_env_origin_m: float
    switch_wall_offset_m: float
    switch_height_m: float
    wall_half_extents_m: Tuple[float, float, float]
    wall_normal_min_m: float
    wall_normal_max_m: float
    lateral_error_min_m: float
    lateral_error_max_m: float
    vertical_error_min_m: float
    vertical_error_max_m: float
    initial_linear_velocity_max_mps: float
    initial_angular_velocity_max_radps: float


@dataclass
class PressApproachTerminationConfig(_FreezableConfig):
    """Contact and flight-envelope failure conditions for approach."""

    crash_tilt_cos: float
    min_altitude_m: float
    max_wall_normal_distance_m: float
    max_lateral_error_m: float
    max_vertical_error_m: float
    contact_force_threshold_n: float


@dataclass
class PressApproachEvaluationConfig(_FreezableConfig):
    """Deterministic approach-policy workload and publication gates."""

    seeds: List[int]
    episodes_per_seed: int
    success_rate_min: float
    collision_rate_max: float


@dataclass
class RlTrainingConfig(_FreezableConfig):
    """PPO training hyperparameters + task selection for ``make train-rl``.

    Consumed by the ``rl.training`` adapter (rsl_rl today); no hyperparameter
    lives in Python.

    Attributes:
        task_id: Gymnasium task id the entrypoint trains
            (``Isaac-Drone-Hover-Direct-v0`` for the F2b pipeline; F3 points it
            at the press task).
        policy_name: Required artifact namespace under ``.build/training``;
            selection publishes accepted artifacts to that namespace's
            ``latest`` symlink.
        max_iterations: Training iterations for a full run.
        num_steps_per_env: On-policy rollout length per env per iteration.
        save_interval: Checkpoint save interval, iterations.
        learning_rate: PPO learning rate.
        schedule: PPO learning-rate schedule (``adaptive`` or ``fixed``).
        gamma: Discount factor.
        lam: GAE lambda.
        entropy_coef: Entropy bonus coefficient.
        clip_param: PPO clip parameter.
        value_loss_coef: Value-loss coefficient.
        num_learning_epochs: PPO epochs per iteration.
        num_mini_batches: Minibatches per epoch.
        desired_kl: Adaptive-schedule KL target.
        max_grad_norm: Gradient clipping norm.
        seed: Training seed (env reset + torch).
    """

    task_id: str
    vehicle_id: int
    policy_name: str
    max_iterations: int
    num_steps_per_env: int
    save_interval: int
    learning_rate: float
    schedule: str
    gamma: float
    lam: float
    entropy_coef: float
    clip_param: float
    value_loss_coef: float
    num_learning_epochs: int
    num_mini_batches: int
    desired_kl: float
    max_grad_norm: float
    seed: int


@dataclass
class PressTaskConfig(_FreezableConfig):
    """Named press curriculum-stage config.

    Attributes:
        curriculum_stage: Required stage name written to artifacts.
        policy: Task-owned actor/critic topology.
        handoff: Thresholds for the nav -> policy handoff gate.
        goal: Stable approach target and braking schedule.
        reward: Approach-stage reward weights.
        reset: Physical scene and initial-state ranges.
        termination: Contact and flight-envelope failure conditions.
        evaluation: Deterministic approach workload and acceptance targets.
    """

    curriculum_stage: str
    policy: RlPolicyConfig
    handoff: PressHandoffConfig
    goal: PressApproachGoalConfig
    reward: PressApproachRewardConfig
    reset: PressResetConfig
    termination: PressApproachTerminationConfig
    evaluation: PressApproachEvaluationConfig


@dataclass
class RlConfig(_FreezableConfig):
    """Shared Isaac Lab RL scale + the pluggable action level.

    Replaces the retired ``isaac_lab`` group as the single source of truth for
    the DirectRLEnv Lab-scale tunables. ``decimation`` is not authored here: the
    env schedules it from ``sim.physics_hz`` and ``policy_rate_hz`` so the
    physics cadence stays single-sourced on ``sim.physics_hz``.

    Attributes:
        action: The RL action-level selection and controller tunables.
        training: PPO hyperparameters + task selection for training runs.
        hover: Hover-task-specific config and policy topology.
        press: Press-task-specific config and policy topology.
        num_envs: Number of parallel cloned envs.
        env_spacing: Distance between cloned env origins, meters.
        episode_length_s: Episode timeout, seconds.
        policy_rate_hz: Exact-average RL policy decision rate.
    """

    action: PressActionConfig
    lab_vehicle: IsaacLabVehicleConfig
    training: RlTrainingConfig
    hover: HoverTaskConfig
    press: PressTaskConfig
    num_envs: int
    env_spacing: float
    episode_length_s: float
    policy_rate_hz: float

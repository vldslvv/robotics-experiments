from dataclasses import dataclass
from typing import List, Tuple

from robotics_experiments.config._immutability import _FreezableConfig

__all__ = [
    "HoverTaskConfig",
    "IsaacLabVehicleConfig",
    "RlActionConfig",
    "RlInputsConfig",
    "PressApproachAcceptanceConfig",
    "PressApproachEvaluationConfig",
    "PressApproachGoalConfig",
    "PressApproachRewardConfig",
    "PressApproachTerminationConfig",
    "PressApproachVelocityReferenceConfig",
    "PressHandoffConfig",
    "PressNavigationNoiseConfig",
    "PressCurriculumConfig",
    "PressCurriculumResetSourceConfig",
    "PressCurriculumStageConfig",
    "PressResetConfig",
    "PressTaskConfig",
    "RlConfig",
    "RlPolicyActorConfig",
    "RlPolicyConfig",
    "RlPolicyCriticConfig",
    "RlPlateauCriteriaConfig",
    "RlTrainingPhaseConfig",
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
    enable_gyroscopic_forces: bool
    pusher_effort_limit_n: float
    pusher_velocity_limit_mps: float


@dataclass
class RlActionConfig(_FreezableConfig):
    """Normalized policy-action interpretation, independent of its controller.

    The ``kind`` picks the in-Lab actuation model (see
    :mod:`robotics_experiments.rl`): ``velocity`` (cascaded PD:
    body-frame velocity + yaw rate) or ``ctbr`` (collective thrust + body-rate
    setpoint). Controller gains, command delay, tilt limit, and actuator response
    are owned separately by the selected alignment artifact.

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
    """

    kind: str
    action_dim: int
    max_body_velocity_mps: float
    max_yaw_rate_radps: float
    max_body_rate_radps: float


@dataclass
class RlInputsConfig(_FreezableConfig):
    """Producer artifacts selected for controller-backed policy training."""

    controller_artifact: str
    flight_parameters_artifact: str


@dataclass
class RlPolicyActorConfig(_FreezableConfig):
    """Actor topology and exploration initialization."""

    hidden_dims: List[int]
    initial_noise_std: float
    output_initialization: str


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
    task (`Isaac-Drone-Hover-Direct-v1`): authoring these values IS defining the
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
        switch_range_min_m: Minimum wall-normal range to the switch, meters.
        switch_range_max_m: Maximum wall-normal range to the switch, meters.
        lateral_align_max_m: Maximum lateral (in-plate) alignment error, meters.
        wall_normal_speed_max_mps: Maximum speed along the press axis, m/s.
        lateral_speed_max_mps: Maximum in-plate speed, m/s.
        angular_speed_max_radps: Maximum body angular speed, rad/s.
        body_z_alignment_min: Minimum body-Z/world-Z alignment cosine.
        estimator_health_min: Minimum aggregate estimator health, [0, 1].
    """

    pose_confidence_min: float
    pose_age_max_s: float
    switch_range_min_m: float
    switch_range_max_m: float
    lateral_align_max_m: float
    wall_normal_speed_max_mps: float
    lateral_speed_max_mps: float
    angular_speed_max_radps: float
    relative_attitude_error_max_rad: float
    body_z_alignment_min: float
    estimator_health_min: float


@dataclass
class PressNavigationNoiseConfig(_FreezableConfig):
    """Measured final-waypoint pose variation covered by actor entry.

    Attributes:
        source: Repo-relative accepted measurement-summary path.
        sample_count: Number of independent production-route arrivals measured.
        position_margin_m: Extra coverage beyond every observed position bound.
        rotation_margin_rad: Extra coverage beyond the observed attitude error.
        switch_range_min_m: Minimum observed switch range, meters.
        switch_range_max_m: Maximum observed switch range, meters.
        lateral_align_max_m: Maximum observed in-plate position error, meters.
        relative_attitude_error_max_rad: Maximum observed full relative-attitude
            error, radians.
    """

    source: str
    sample_count: int
    position_margin_m: float
    rotation_margin_rad: float
    switch_range_min_m: float
    switch_range_max_m: float
    lateral_align_max_m: float
    relative_attitude_error_max_rad: float


@dataclass
class PressApproachRewardConfig(_FreezableConfig):
    """Magnitudes for the signed approach-stage reward contributions."""

    position_progress_weight: float
    stabilization_weight: float
    predictive_barrier_weight: float
    strict_hold_weight: float
    terminal_success_weight: float
    terminal_failure_weight: float


@dataclass
class PressApproachVelocityReferenceConfig(_FreezableConfig):
    """Dense contact-frame velocity reference for the approach stage."""

    range_center_m: float
    wall_normal_prediction_horizon_s: float
    wall_normal_alignment_gate_m: float
    wall_normal_gain_per_s: float
    wall_normal_min_mps: float
    wall_normal_max_mps: float
    tangential_gain_per_s: float
    tangential_speed_max_mps: float


@dataclass
class PressApproachGoalConfig(_FreezableConfig):
    """Stable approach target and dense velocity reference."""

    range_min_m: float
    range_max_m: float
    alignment_error_max_m: float
    relative_attitude_error_max_rad: float
    linear_speed_max_mps: float
    angular_speed_max_radps: float
    body_z_alignment_min: float
    stable_steps: int
    velocity_reference: PressApproachVelocityReferenceConfig


@dataclass
class PressResetConfig(_FreezableConfig):
    """Easy-task physical-scene parameters for switch pressing."""

    physical_contact_enabled: bool
    button_geometry_scale: float
    button_stiffness_scale: float
    button_threshold_scale: float
    switch_distance_from_env_origin_m: float
    switch_wall_offset_m: float
    switch_height_m: float
    wall_half_extents_m: Tuple[float, float, float]


@dataclass
class PressApproachTerminationConfig(_FreezableConfig):
    """Contact and flight-parameters failure conditions for approach."""

    crash_tilt_cos: float
    min_altitude_m: float
    max_wall_normal_distance_m: float
    max_lateral_error_m: float
    max_vertical_error_m: float
    contact_force_threshold_n: float


@dataclass
class PressApproachEvaluationConfig(_FreezableConfig):
    """Disjoint deterministic validation/test workloads and ranking costs."""

    validation_seeds: List[int]
    test_seeds: List[int]
    episodes_per_seed: int
    collision_penalty: float
    flight_failure_penalty: float
    timeout_penalty: float
    non_success_penalty: float
    dwell_progress_weight: float
    terminal_range_weight: float
    trajectory_alignment_weight: float
    relative_attitude_error_weight: float
    velocity_tracking_weight: float
    angular_velocity_weight: float
    upright_error_weight: float
    action_delta_weight: float
    completion_time_weight: float


@dataclass
class PressApproachAcceptanceConfig(_FreezableConfig):
    """Task-owned requirements applied to validation and the final test."""

    success_rate_min: float
    collision_rate_max: float
    minimum_range_m: float


@dataclass
class RlPlateauCriteriaConfig(_FreezableConfig):
    """Phase-local progress definition for deterministic policy evaluation."""

    metric_name: str
    direction: str
    minimum_delta: float
    patience_evaluations: int


@dataclass
class RlTrainingPhaseConfig(_FreezableConfig):
    """One phase in an attempt's validation-selected PPO branch."""

    max_iterations: int
    evaluation_interval: int
    learning_rate: float
    schedule: str
    entropy_coef: float
    range_undershoot_min_m: float | None


@dataclass
class PressCurriculumResetSourceConfig(_FreezableConfig):
    """One typed reset source in a press curriculum-stage mixture."""

    source: str
    weight: float


@dataclass
class PressCurriculumStageConfig(_FreezableConfig):
    """The task distribution and training schedule for one ordered press stage."""

    name: str
    reset_mix: List[PressCurriculumResetSourceConfig]
    hold_s: float
    training_hold_s: List[float]
    episode_length_s: float
    phases: List[RlTrainingPhaseConfig]


@dataclass
class PressCurriculumConfig(_FreezableConfig):
    """A linear, validation-gated press-policy curriculum."""

    active_stage: str
    stages: List[PressCurriculumStageConfig]


@dataclass
class RlTrainingConfig(_FreezableConfig):
    """PPO training hyperparameters + task selection for ``make train-rl``.

    Consumed by the ``rl.training`` adapter (rsl_rl today); no hyperparameter
    lives in Python.

    Attributes:
        task_id: Gymnasium task id the entrypoint trains.
        policy_name: Required artifact namespace under ``.build/training``;
            completed runs publish their selected artifacts to that namespace's
            ``latest`` symlink.
        phases: Ordered curriculum phases executed by each fresh attempt.
        telemetry_sample_interval_s: Diagnostic hardware sampling cadence.
        gpu_utilization_target_rate: Diagnostic GPU-utilization target in [0, 1].
        num_steps_per_env: On-policy rollout length per env per iteration.
        gamma: Discount factor.
        lam: GAE lambda.
        clip_param: PPO clip parameter.
        value_loss_coef: Value-loss coefficient.
        num_learning_epochs: PPO epochs per iteration.
        num_mini_batches: Minibatches per epoch.
        desired_kl: Adaptive-schedule KL target.
        max_grad_norm: Gradient clipping norm.
        attempt_seeds: Independent training seeds (env reset + torch).
    """

    task_id: str
    vehicle_id: int
    policy_name: str
    phases: List[RlTrainingPhaseConfig]
    plateau: RlPlateauCriteriaConfig
    evaluation_worker_timeout_s: float
    telemetry_sample_interval_s: float
    gpu_utilization_target_rate: float
    num_steps_per_env: int
    gamma: float
    lam: float
    clip_param: float
    value_loss_coef: float
    num_learning_epochs: int
    num_mini_batches: int
    desired_kl: float
    max_grad_norm: float
    attempt_seeds: List[int]


@dataclass
class PressTaskConfig(_FreezableConfig):
    """Named press curriculum-stage config.

    Attributes:
        curriculum: Ordered typed stages and the stage selected by this config.
        policy: Task-owned actor/critic topology.
        handoff: Thresholds for the nav -> policy handoff gate.
        navigation_noise: Measured final-waypoint position variation that the
            handoff and training-reset envelope must cover.
        goal: Shared target pose and velocity reference.
        reward: Shared reward weights; the stage decides which terms apply.
        reset: Physical scene and initial-state ranges.
        termination: Shared safety termination conditions.
        evaluation: Shared deterministic validation/test workloads and costs.
        acceptance: Requirements applied to validation and the final test.
    """

    curriculum: PressCurriculumConfig
    policy: RlPolicyConfig
    handoff: PressHandoffConfig
    navigation_noise: PressNavigationNoiseConfig
    goal: PressApproachGoalConfig
    reward: PressApproachRewardConfig
    reset: PressResetConfig
    termination: PressApproachTerminationConfig
    evaluation: PressApproachEvaluationConfig
    acceptance: PressApproachAcceptanceConfig

    def selected_stage(self) -> PressCurriculumStageConfig:
        """Return the configured stage after loader validation."""

        matches = [
            stage for stage in self.curriculum.stages if stage.name == self.curriculum.active_stage
        ]
        if len(matches) != 1:
            raise ValueError(
                "rl.press.curriculum.active_stage must select exactly one ordered stage"
            )
        return matches[0]


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

    action: RlActionConfig
    inputs: RlInputsConfig
    lab_vehicle: IsaacLabVehicleConfig
    training: RlTrainingConfig
    hover: HoverTaskConfig
    press: PressTaskConfig
    num_envs: int
    env_spacing: float
    episode_length_s: float
    policy_rate_hz: float

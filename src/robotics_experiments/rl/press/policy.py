"""Press policy contract derived from prepared campaign inputs."""

from robotics_experiments.config import ExperimentConfig
from robotics_experiments.controller import ControllerArtifact
from robotics_experiments.rl.policy import PolicyContract, PolicyInterface
from robotics_experiments.rl.press.observation import (
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
)

__all__ = ["press_policy_contract"]


def press_policy_contract(
    *,
    config: ExperimentConfig,
    controller: ControllerArtifact,
) -> PolicyContract:
    """Derive the press actor contract before an Isaac environment exists."""

    action = config.rl.action
    return PolicyContract(
        policy_interface=PolicyInterface.PRESS,
        obs_schema_version=OBSERVATION_SCHEMA_VERSION,
        obs_dim=PRESS_OBSERVATION_DIM,
        policy_rate_hz=config.rl.policy_rate_hz,
        action_kind=action.kind,
        action_dim=action.action_dim,
        max_body_velocity_mps=action.max_body_velocity_mps,
        max_yaw_rate_radps=action.max_yaw_rate_radps,
        max_body_rate_radps=action.max_body_rate_radps,
        hover_thrust_normalized=controller.selected_settings.hover_thrust_normalized,
        curriculum_stage=config.rl.press.selected_stage().name,
    )

"""Curriculum composition; task and PPO settings belong to stage experiments."""

from dataclasses import dataclass

from robotics_experiments.config._immutability import _FreezableConfig

__all__ = ["CurriculumConfig", "CurriculumStageConfig"]


@dataclass
class CurriculumStageConfig(_FreezableConfig):
    name: str
    config_path: str


@dataclass
class CurriculumConfig(_FreezableConfig):
    policy_name: str
    initial_learner: str | None
    stages: list[CurriculumStageConfig]

from hydra.core.config_store import ConfigStore

from robotics_experiments.config._commanding import DriverConfig, RoutineConfig
from robotics_experiments.config._curriculum import CurriculumConfig
from robotics_experiments.config._environment import ExperimentConfig, VehicleConfig
from robotics_experiments.config._px4 import Px4BridgeConfig
from robotics_experiments.config._rl import PressTaskConfig, RlConfig
from robotics_experiments.config._scene import LightSwitchConfig, SwitchModelConfig
from robotics_experiments.config._simulation import SimSettings

__all__: list[str] = []


def _register_schemas() -> None:
    """Register the structured-config schemas with Hydra's ``ConfigStore``.

    Idempotent: safe to call on every config composition. The group schemas
    let YAML files under ``configs/experiment`` be validated
    against their dataclass shape.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_experiment", node=ExperimentConfig)
    cs.store(name="base_curriculum", node=CurriculumConfig)
    # Group schemas are stored under the experiment/ subdir where their YAML
    # building blocks live, so each group file's ``defaults: - base_<group>``
    # resolves within its own group; the top-level entry configs at configs/
    # select them via ``experiment/<group>@<group>``.
    cs.store(group="experiment/sim", name="base_sim", node=SimSettings)
    cs.store(group="experiment/switches", name="base_switches", node=LightSwitchConfig)
    cs.store(group="experiment/switches/model", name="base_model", node=SwitchModelConfig)
    cs.store(group="experiment/px4", name="base_px4", node=Px4BridgeConfig)
    cs.store(group="experiment/rl", name="base_rl", node=RlConfig)
    cs.store(group="experiment/rl/press", name="base_press_task", node=PressTaskConfig)
    cs.store(group="experiment/routine", name="base_routine", node=RoutineConfig)
    cs.store(group="experiment/driver", name="base_driver", node=DriverConfig)
    cs.store(group="experiment/vehicle", name="base_vehicle", node=VehicleConfig)

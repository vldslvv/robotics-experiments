#! /bin/sh

# This script runs inside isaac sim docker, therefore
# scene path depends on the mounts used
EXT_FOLDER="/isaac-sim/proj/isaac/runtime/extensions"
SCENE_PATH="/isaac-sim/proj/isaac/assets/LightSwitchWarehouse/Warehouse.usd"
OPEN_STAGE_SCRIPT="open_stage.py"
ROBOTICS_EXPERIMENTS_CORE_EXT="robotics_experiments_core"
SIM_SETTINGS_EXT="sim_settings"
LIGHTSWITCH_EXT="lightswitch_spawner"
DRONE_SPAWNER_EXT="drone_spawner"
PX4_SITL_BRIDGE_EXT="px4_sitl_brigde"

# Hydra-config overrides are picked up by extensions that read
# robotics_experiments.config.
# The default tree is /isaac-sim/proj/configs/experiment.
# Set EXPERIMENT_CONFIG_DIR to point at an alternative config tree;
# EXPERIMENT_CONFIG_NAME selects which top-level yaml to compose;
# EXPERIMENT_CONFIG_OVERRIDES is a space-separated list of Hydra dot-list
# overrides (e.g. "switches.num_switches=10 sim.physics_hz=200").
export EXPERIMENT_CONFIG_DIR="${EXPERIMENT_CONFIG_DIR:-/isaac-sim/proj/configs/experiment}"
export EXPERIMENT_CONFIG_NAME="${EXPERIMENT_CONFIG_NAME:-config}"
export EXPERIMENT_CONFIG_OVERRIDES="${EXPERIMENT_CONFIG_OVERRIDES-}"

# For log level changes, use:
# LOG_LEVEL="info"
  # --/log/level="$LOG_LEVEL" \
  # --/log/outputStreamLevel="$LOG_LEVEL" \

/isaac-sim/license.sh && \
/isaac-sim/privacy.sh && \
/isaac-sim/isaac-sim.sh \
  --ext-folder "$EXT_FOLDER" \
  --enable "$ROBOTICS_EXPERIMENTS_CORE_EXT" \
  --enable "$SIM_SETTINGS_EXT" \
  --enable "$LIGHTSWITCH_EXT" \
  --enable "$DRONE_SPAWNER_EXT" \
  --enable "$PX4_SITL_BRIDGE_EXT" \
  --/persistent/isaac/asset_root/default="$OMNI_SERVER" \
  --merge-config="/isaac-sim/config/open_endpoint.toml" \
  --allow-root \
  ${SCENE_PATH:+--exec "$OPEN_STAGE_SCRIPT $SCENE_PATH"}

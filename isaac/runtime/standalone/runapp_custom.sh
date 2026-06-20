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

# Extensions read the active Hydra config from ROBOTICS_EXPERIMENT_CONFIG_PATH.

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

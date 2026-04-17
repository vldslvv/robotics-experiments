#! /bin/sh

# This script runs inside isaac sim docker, therefore
# scene path depends on the mounts used
EXT_FOLDER="/isaac-sim/proj/robotics-experiments/isaac_extensions"
SCENE_PATH="/isaac-sim/proj/assets/LightSwitchWarehouse/Warehouse.usd"
OPEN_STAGE_SCRIPT="open_stage.py"
LIGHTSWITCH_EXTENSION="lightswitch_example"
DRONE_SPAWNER_EXTENSION="drone_spawner"

# For log level changes, use:
# LOG_LEVEL="info"
  # --/log/level="$LOG_LEVEL" \
  # --/log/outputStreamLevel="$LOG_LEVEL" \

/isaac-sim/license.sh && \
/isaac-sim/privacy.sh && \
/isaac-sim/isaac-sim.sh \
  --ext-folder "$EXT_FOLDER" \
  --enable "$LIGHTSWITCH_EXTENSION" \
  --enable "$DRONE_SPAWNER_EXTENSION" \
  --/persistent/isaac/asset_root/default="$OMNI_SERVER" \
  --merge-config="/isaac-sim/config/open_endpoint.toml" \
  --allow-root \
  ${SCENE_PATH:+--exec "$OPEN_STAGE_SCRIPT $SCENE_PATH"}

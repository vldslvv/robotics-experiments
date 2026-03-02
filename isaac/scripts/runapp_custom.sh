#! /bin/sh

# This script runs inside isaac sim docker, therefore
# scene path depends on the mounts used
DEFAULT_EXT_FOLDER="/isaac-sim/proj/robotics-experiments/isaac_extensions"
SCENE_PATH="/isaac-sim/proj/assets/LightSwitchWarehouse/Warehouse.usd"
OPEN_STAGE_SCRIPT="open_stage.py"

EXT_FOLDER="$1"

if [ -z "$EXT_FOLDER" ]; then
  EXT_FOLDER="$DEFAULT_EXT_FOLDER"
else
  shift
fi

/isaac-sim/license.sh && \
/isaac-sim/privacy.sh && \
/isaac-sim/isaac-sim.sh \
  --ext-folder "$EXT_FOLDER" \
  --enable "lightswitch_example" \
  --/persistent/isaac/asset_root/default="$OMNI_SERVER" \
  --merge-config="/isaac-sim/config/open_endpoint.toml" \
  --allow-root \
  ${SCENE_PATH:+--exec "$OPEN_STAGE_SCRIPT $SCENE_PATH"} \
  "$@"

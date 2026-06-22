#! /bin/sh

# This script runs inside isaac sim docker, therefore paths depend on the mounts
# used. PROJECT_DIR is the repository root mounted into the container.
PROJECT_DIR="/isaac-sim/proj"
EXT_FOLDER="$PROJECT_DIR/isaac/runtime/extensions"
OPEN_STAGE_SCRIPT="open_stage.py"

# The scene comes from the active Hydra experiment's `scene` field (the same
# config the extensions read via ROBOTICS_EXPERIMENT_CONFIG_PATH) so the
# persistent runtime and the standalone tests share one source of truth. A
# leading '@' is repository-relative and maps to PROJECT_DIR in-container.
ISAAC_PYTHON="/isaac-sim/kit/python/bin/python3"
RAW_SCENE="$(PYTHONPATH="$PROJECT_DIR/src" "$ISAAC_PYTHON" -c \
  'from robotics_experiments.config import get_config; print(get_config().scene)')"
case "$RAW_SCENE" in
  @*) SCENE_PATH="$PROJECT_DIR/${RAW_SCENE#@}" ;;
  *) SCENE_PATH="$RAW_SCENE" ;;
esac
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

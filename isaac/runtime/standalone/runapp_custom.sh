#! /bin/sh

# This script runs inside isaac sim docker, therefore paths depend on the mounts
# used. PROJECT_DIR is the repository root mounted into the container.
PROJECT_DIR="/isaac-sim/proj"
EXT_FOLDER="$PROJECT_DIR/isaac/runtime/extensions"

# Scene-open is owned by the experiment_runtime extension's fleet lifecycle
# (which reads the active Hydra experiment's `scene` field), so there is no
# `--exec open_stage.py` here and no host-side scene resolution: the orchestrator
# is the single source of truth for scene + sim settings.
ROBOTICS_EXPERIMENTS_CORE_EXT="robotics_experiments_core"
LIGHTSWITCH_EXT="lightswitch_spawner"
EXPERIMENT_RUNTIME_EXT="experiment_runtime"
EXPERIMENT_RUNTIME_HUD_EXT="experiment_runtime_hud"
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
  --enable "$LIGHTSWITCH_EXT" \
  --enable "$EXPERIMENT_RUNTIME_EXT" \
  --enable "$EXPERIMENT_RUNTIME_HUD_EXT" \
  --enable "$PX4_SITL_BRIDGE_EXT" \
  --/persistent/isaac/asset_root/default="$OMNI_SERVER" \
  --merge-config="/isaac-sim/config/open_endpoint.toml" \
  --allow-root

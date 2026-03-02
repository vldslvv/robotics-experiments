#! /bin/sh

DEFAULT_EXT_FOLDER="/isaac-sim/proj/robotics-experiments/isaac_extensions"

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
  "$@"

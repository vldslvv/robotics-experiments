#!/usr/bin/env bash
set -euo pipefail

px4_dir="${PX4_DIR:-/opt/PX4-Autopilot}"
reset_helper="${PX4_RESET_HELPER:-/usr/local/lib/px4/reset_px4_state.sh}"

source "${reset_helper}"
reset_px4_state

cd "${px4_dir}"
set -o pipefail
make px4_sitl_default none_iris 2>&1 \
  | sed -u -E 's/\r/\n/g' \
  | sed -u -E 's/^pxh>[[:space:]]*//; /^$/d'

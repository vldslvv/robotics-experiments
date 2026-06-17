#!/usr/bin/env bash
set -euo pipefail

px4_dir="${PX4_DIR:-/opt/PX4-Autopilot}"
reset_helper="${PX4_RESET_HELPER:-/usr/local/lib/px4/reset_px4_state.sh}"

# shellcheck source-path=SCRIPTDIR
# shellcheck source=reset_px4_state.sh
source "${reset_helper}"
reset_px4_state

cd "${px4_dir}"
exec make px4_sitl_default none_iris

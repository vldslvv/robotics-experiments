#!/usr/bin/env bash
set -euo pipefail

px4_dir="${PX4_DIR:-/opt/PX4-Autopilot}"
reset_helper="${PX4_RESET_HELPER:-/usr/local/lib/px4/reset_px4_state.sh}"
common_helper="${PX4_COMMON_HELPER:-/usr/local/lib/px4/px4_iris_common.sh}"
# PX4 instance index. 0 is the single-vehicle baseline. Each additional instance
# (i>0) shares instance 0's network namespace, so PX4 adds the index to every
# port: simulator TCP 4560 + i, GCS UDP 14550 + i, offboard 14540 + i.

# shellcheck source-path=SCRIPTDIR
# shellcheck source=reset_px4_state.sh
source "${reset_helper}"
# shellcheck source-path=SCRIPTDIR
# shellcheck disable=SC1090
source "${common_helper}"
px4_iris_configure_instance
reset_px4_state

cd "${px4_dir}"
set -o pipefail
configure_gcs_remote_port

px4_filter() {
  sed -u -E 's/\r/\n/g' | sed -u -E 's/^pxh>[[:space:]]*//; /^$/d'
}

launch_px4_iris 2>&1 | px4_filter

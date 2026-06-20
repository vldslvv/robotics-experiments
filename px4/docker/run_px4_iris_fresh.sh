#!/usr/bin/env bash
set -euo pipefail

reset_helper="${PX4_RESET_HELPER:-/usr/local/lib/px4/reset_px4_state.sh}"
common_helper="${PX4_COMMON_HELPER:-/usr/local/lib/px4/px4_iris_common.sh}"

# shellcheck source-path=SCRIPTDIR
# shellcheck source=reset_px4_state.sh
source "${reset_helper}"
# shellcheck source-path=SCRIPTDIR
# shellcheck disable=SC1090
source "${common_helper}"
px4_iris_configure_instance
reset_px4_state

cd "${px4_dir}"
configure_gcs_remote_port
launch_px4_iris

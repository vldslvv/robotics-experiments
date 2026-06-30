#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
}

for name in HYDRA_CONFIG_HOST_PATH MAKE_CMD PX4_IMAGE; do
    require_env "${name}"
done

# shellcheck source-path=SCRIPTDIR
# shellcheck source=px4_airframe_manifest_check.sh
source "$(dirname "${BASH_SOURCE[0]}")/px4_airframe_manifest_check.sh"
# PX4_IMAGE is a required env var (asserted above), not the helper's local px4_image.
# shellcheck disable=SC2153
check_px4_airframe_manifest_current "${PX4_IMAGE}"

# The resolved fleet is the single source of truth: iterate the PX4-enabled
# vehicles' derived launch specs (instance index = PX4 -i / port offset =
# vehicle id, plus the per-asset PX4 airframe) instead of a blind 0..count loop,
# so non-contiguous ids keep their derived ports and each container launches the
# generated airframe matching its drone asset. Each
# line is "<instance> <sim_model> <sys_autostart>".
mapfile -t launch_specs < <(
    ROBOTICS_EXPERIMENT_CONFIG_PATH="${HYDRA_CONFIG_HOST_PATH}" \
        uv run python -c 'from robotics_experiments.config import get_enabled_px4_launch_configs
for c in get_enabled_px4_launch_configs():
    print(c.instance_index, c.sim_model, c.sys_autostart)'
)

count="${#launch_specs[@]}"

if [ "${count}" -eq 0 ]; then
    echo "No PX4-enabled vehicles in Hydra config; skipping PX4 SITL."
    exit 0
fi

# MAKE_CMD is provided by Make and may contain jobserver/flag arguments.
# shellcheck disable=SC2153,SC2206
make_cmd=(${MAKE_CMD})

for spec in "${launch_specs[@]}"; do
    read -r i sim_model sys_autostart <<<"${spec}"
    kitty --title "PX4[${i}]" --hold "${make_cmd[@]}" run-px4-instance \
        PX4_INSTANCE="${i}" \
        PX4_SIM_MODEL="${sim_model}" \
        PX4_SYS_AUTOSTART="${sys_autostart}" \
        PX4_TOTAL_INSTANCES="${count}" &
done

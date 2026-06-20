#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
}

for name in HYDRA_CONFIG_HOST_PATH MAKE_CMD; do
    require_env "${name}"
done

count="$(
    ROBOTICS_EXPERIMENT_CONFIG_PATH="${HYDRA_CONFIG_HOST_PATH}" \
        uv run python -c 'from robotics_experiments.config import get_enabled_px4_vehicle_count; print(get_enabled_px4_vehicle_count())'
)"

if [ "${count}" -eq 0 ]; then
    echo "No PX4-enabled vehicles in Hydra config; skipping PX4 SITL."
    exit 0
fi

# MAKE_CMD is provided by Make and may contain jobserver/flag arguments.
# shellcheck disable=SC2153,SC2206
make_cmd=(${MAKE_CMD})

i=0
while [ "${i}" -lt "${count}" ]; do
    kitty --title "PX4[${i}]" --hold "${make_cmd[@]}" run-px4-instance PX4_INSTANCE="${i}" PX4_TOTAL_INSTANCES="${count}" &
    i=$((i + 1))
done

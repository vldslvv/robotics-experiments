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

# The resolved fleet is the single source of truth: iterate the PX4-enabled
# vehicles' derived instance indices (PX4 -i / port offset = vehicle id) instead
# of a blind 0..count loop, so non-contiguous ids keep their derived ports.
indices="$(
    ROBOTICS_EXPERIMENT_CONFIG_PATH="${HYDRA_CONFIG_HOST_PATH}" \
        uv run python -c 'from robotics_experiments.config import get_enabled_px4_instance_indices; print(" ".join(str(i) for i in get_enabled_px4_instance_indices()))'
)"

# shellcheck disable=SC2206
indices=(${indices})
count="${#indices[@]}"

if [ "${count}" -eq 0 ]; then
    echo "No PX4-enabled vehicles in Hydra config; skipping PX4 SITL."
    exit 0
fi

# MAKE_CMD is provided by Make and may contain jobserver/flag arguments.
# shellcheck disable=SC2153,SC2206
make_cmd=(${MAKE_CMD})

for i in "${indices[@]}"; do
    kitty --title "PX4[${i}]" --hold "${make_cmd[@]}" run-px4-instance PX4_INSTANCE="${i}" PX4_TOTAL_INSTANCES="${count}" &
done

#!/usr/bin/env bash
set -euo pipefail

for name in MAKE_COMMAND PRESS_NAVIGATION_MEASUREMENT_ROOT PRESS_NAVIGATION_MEASUREMENT_RUNS PRESS_NAVIGATION_POSITION_MARGIN_M PRESS_NAVIGATION_ROTATION_MARGIN_RAD; do
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
done

if ! [[ "${PRESS_NAVIGATION_MEASUREMENT_RUNS}" =~ ^[0-9]+$ ]] || [ "${PRESS_NAVIGATION_MEASUREMENT_RUNS}" -lt 10 ]; then
    echo "PRESS_NAVIGATION_MEASUREMENT_RUNS must be an integer of at least 10" >&2
    exit 2
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ).$$"
run_dir="${PRESS_NAVIGATION_MEASUREMENT_ROOT}/${run_id}"
mkdir -p "${run_dir}"

for index in $(seq 1 "${PRESS_NAVIGATION_MEASUREMENT_RUNS}"); do
    sample_path="${run_dir}/sample_$(printf '%03d' "${index}").json"
    echo "Press navigation handoff measurement ${index}/${PRESS_NAVIGATION_MEASUREMENT_RUNS}"
    "${MAKE_COMMAND}" --no-print-directory test-integration-press-navigation-handoff-sample \
        PRESS_NAVIGATION_MEASUREMENT_OUTPUT="${sample_path}"
done

mapfile -t samples < <(find "${run_dir}" -maxdepth 1 -type f -name 'sample_*.json' -print | sort)
uv run python make/scripts/summarize_press_navigation_handoff.py \
    --output "${run_dir}/summary.json" \
    --position-margin-m "${PRESS_NAVIGATION_POSITION_MARGIN_M}" \
    --rotation-margin-rad "${PRESS_NAVIGATION_ROTATION_MARGIN_RAD}" \
    "${samples[@]}"

echo "Press navigation handoff summary: ${run_dir}/summary.json"

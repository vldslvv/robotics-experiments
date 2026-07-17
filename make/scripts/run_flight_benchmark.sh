#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
}

for name in \
    BENCHMARK_BASELINE \
    BENCHMARK_ROOT \
    ISAAC_IMAGE \
    MAKE_COMMAND \
    PX4_REF \
    REPORT_SCRIPT
do
    require_env "${name}"
done

run_id="$(date -u +%Y%m%dT%H%M%SZ).$$"
run_root="${BENCHMARK_ROOT}/flight.${run_id}"
raw_root="${run_root}/raw"
mkdir -p "${raw_root}"

echo "Flight benchmark artifacts: ${run_root}"
echo "Warming the shared Isaac shader and asset caches..."
"${MAKE_COMMAND}" test-integration-warm-isaac-cache \
    INTEGRATION_ROOT="${run_root}/warmup"

for repetition in 1 2 3; do
    echo "Physics-only benchmark repetition ${repetition}/3..."
    "${MAKE_COMMAND}" test-integration-drone-articulation \
        INTEGRATION_ROOT="${raw_root}" \
        ISAAC_TEST_EXTRA_ARGS=--benchmark

    echo "PX4 flight benchmark repetition ${repetition}/3..."
    "${MAKE_COMMAND}" test-integration-px4-takeoff \
        INTEGRATION_ROOT="${raw_root}" \
        ISAAC_TEST_EXTRA_ARGS=--benchmark
done

isaac_image_id="$(docker image inspect "${ISAAC_IMAGE}" --format '{{.Id}}')"
set +e
uv run python "${REPORT_SCRIPT}" build \
    --raw-root "${raw_root}" \
    --run-root "${run_root}" \
    --baseline "${BENCHMARK_BASELINE}" \
    --isaac-image "${ISAAC_IMAGE}" \
    --isaac-image-id "${isaac_image_id}" \
    --px4-ref "${PX4_REF}"
status=$?
set -e

echo "Flight benchmark JSON: ${run_root}/report.json"
echo "Flight benchmark Markdown: ${run_root}/report.md"
exit "${status}"

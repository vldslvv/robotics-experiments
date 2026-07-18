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
    REPORT_SCRIPT
do
    require_env "${name}"
done

run_id="$(date -u +%Y%m%dT%H%M%SZ).$$"
run_root="${BENCHMARK_ROOT}/isaac-core.${run_id}"
raw_root="${run_root}/raw"
profile_root="${run_root}/profile"
mkdir -p "${raw_root}" "${profile_root}"

echo "Isaac core benchmark artifacts: ${run_root}"
echo "Warming the shared Isaac shader and asset caches..."
"${MAKE_COMMAND}" test-integration-warm-isaac-cache \
    INTEGRATION_ROOT="${run_root}/warmup"

for repetition in 1 2 3; do
    echo "Isaac core benchmark repetition ${repetition}/3..."
    "${MAKE_COMMAND}" test-integration-drone-articulation \
        INTEGRATION_ROOT="${raw_root}" \
        ISAAC_TEST_EXTRA_ARGS=--benchmark
done

echo "Isaac core one-shot Tracy attribution run..."
"${MAKE_COMMAND}" test-integration-drone-articulation \
    INTEGRATION_ROOT="${profile_root}" \
    ISAAC_PROFILE_TRACE_NAME=isaac_core.tracy \
    ISAAC_TEST_EXTRA_ARGS="--benchmark --benchmark-profile"

isaac_image_id="$(docker image inspect "${ISAAC_IMAGE}" --format '{{.Id}}')"
set +e
uv run python "${REPORT_SCRIPT}" build \
    --raw-root "${raw_root}" \
    --profile-root "${profile_root}" \
    --run-root "${run_root}" \
    --baseline "${BENCHMARK_BASELINE}" \
    --isaac-image "${ISAAC_IMAGE}" \
    --isaac-image-id "${isaac_image_id}"
status=$?
set -e

echo "Isaac core benchmark JSON: ${run_root}/report.json"
echo "Isaac core benchmark Markdown: ${run_root}/report.md"
exit "${status}"

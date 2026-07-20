#!/usr/bin/env bash
set -euo pipefail

for name in \
    FLIGHT_BENCHMARK_CONFIG_SHA256 \
    FLIGHT_BENCHMARK_LOG \
    FLIGHT_BENCHMARK_OUTPUT_DIR
do
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
done

project_dir="/isaac-sim/proj"
extension_dir="${project_dir}/isaac/runtime/extensions"
trace_pid=""

cleanup() {
    if [ -n "${trace_pid}" ] && kill -0 "${trace_pid}" 2>/dev/null; then
        kill "${trace_pid}" 2>/dev/null || true
        wait "${trace_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_trace_capture_when_ready() {
    local trace_port_hex="1F96" # TCP 8086
    for _ in $(seq 1 1200); do
        if awk -v suffix=":${trace_port_hex}" \
            '$2 ~ suffix "$" && $4 == "0A" { found = 1 } END { exit !found }' \
            /proc/net/tcp /proc/net/tcp6; then
            exec "${trace_bins[0]}" -a 127.0.0.1 -o "${FLIGHT_BENCHMARK_TRACE}" -f
        fi
        sleep 0.25
    done
    echo "Tracy server did not listen on TCP 8086 within 300 seconds" >&2
    return 2
}

profile_args=()
if [ "${FLIGHT_BENCHMARK_RUN_KIND:-measured}" = "profile" ]; then
    if [ -z "${FLIGHT_BENCHMARK_TRACE:-}" ]; then
        echo "Profile run requires FLIGHT_BENCHMARK_TRACE" >&2
        exit 2
    fi
    trace_bins=(/isaac-sim/extscache/omni.kit.profiler.tracy-*/bin/capture)
    if [ "${#trace_bins[@]}" -ne 1 ] || [ ! -x "${trace_bins[0]}" ]; then
        echo "Expected exactly one Tracy capture binary, found ${#trace_bins[@]}" >&2
        exit 2
    fi
    mkdir -p "$(dirname "${FLIGHT_BENCHMARK_TRACE}")"
    start_trace_capture_when_ready &
    trace_pid=$!
    profile_args=(
        --enable omni.kit.profiler.tracy
        --/app/profilerBackend=tracy
        --/profiler/enabled=true
        --/app/profileFromStart=true
        --/profiler/gpu=true
        --/profiler/gpu/tracyInject/enabled=true
        --/profiler/gpu/tracyInject/msBetweenClockCalibration=0
        --/app/profilerMask=1
        --/profiler/channels/carb.tasking/enabled=false
        --/profiler/channels/carb.events/enabled=false
        --/plugins/carb.profiler-tracy.plugin/fibersAsThreads=false
    )
fi

mkdir -p "${FLIGHT_BENCHMARK_OUTPUT_DIR}" "$(dirname "${FLIGHT_BENCHMARK_LOG}")"

asset_root_url="${OMNI_SERVER:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0}"
asset_root_args=(--/persistent/isaac/asset_root/default="${asset_root_url}")

set +e
/isaac-sim/license.sh && /isaac-sim/privacy.sh && \
    /isaac-sim/isaac-sim.sh \
        --ext-folder "${extension_dir}" \
        --enable flight_benchmark \
        --renderer RaytracedLighting \
        --/app/window/width=1280 \
        --/app/window/height=720 \
        --/app/renderer/resolution/width=1280 \
        --/app/renderer/resolution/height=720 \
        --/app/file/ignoreUnsavedOnExit=true \
        "${asset_root_args[@]}" \
        --merge-config=/isaac-sim/config/open_endpoint.toml \
        --allow-root \
        "${profile_args[@]}" \
        2>&1 | tee "${FLIGHT_BENCHMARK_LOG}"
status=${PIPESTATUS[0]}
set -e

if [ -n "${trace_pid}" ]; then
    if ! wait "${trace_pid}"; then
        echo "Tracy capture failed" >&2
        exit 2
    fi
    trace_pid=""
    if [ ! -s "${FLIGHT_BENCHMARK_TRACE}" ]; then
        echo "Tracy trace was not written: ${FLIGHT_BENCHMARK_TRACE}" >&2
        exit 2
    fi
fi
if [ "${status}" -ne 0 ]; then
    exit "${status}"
fi
if [ ! -s "${FLIGHT_BENCHMARK_OUTPUT_DIR}/samples.json" ]; then
    echo "Flight benchmark samples were not written" >&2
    exit 2
fi

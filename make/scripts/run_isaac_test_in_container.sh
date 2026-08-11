#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
}

terminal_control_filter() {
    sed -u -E 's|\x1B\[[0-9;?]*[ -/]*[@-~]||g; s|\x1B[=>c]||g'
}

now_ms() {
    date +%s%3N
}

format_ms() {
    local ms="$1"
    printf "%s.%03d" "$((ms / 1000))" "$((ms % 1000))"
}

integration_timing() {
    local stage="$1"
    local start_ms="$2"
    local status="${3:-ok}"
    local end_ms
    local elapsed_ms
    local line
    local total_ms

    end_ms="$(now_ms)"
    elapsed_ms="$((end_ms - start_ms))"
    total_ms="$((end_ms - SCRIPT_START_MS))"
    line="$(printf "[integration-timing] target=%s stage=%s elapsed_s=%s total_s=%s status=%s" \
        "${TEST_TARGET}" \
        "${stage}" \
        "$(format_ms "${elapsed_ms}")" \
        "$(format_ms "${total_ms}")" \
        "${status}")"
    printf "%s\n" "${line}"
    printf "%s\n" "${line}" >>"${HOST_TIMING_LOG}" || true
}

for name in \
    HOST_TIMING_LOG \
    ISAAC_PYTHON \
    ISAAC_TEST_ERROR_FILTER \
    ISAAC_TEST_FAILURE_FILTER \
    ISAAC_TEST_LOG \
    ISAAC_TEST_LOG_FILTER \
    ISAAC_TEST_SCRIPT \
    SCENE \
    TEST_TARGET
do
    require_env "${name}"
done

# STRICT=1 runs the scenario as an asserting integration test (--strict); unset
# or empty runs it forgiving (a flight-envelope breach is logged, not fatal).
: "${STRICT:=}"
strict_args=()
if [ -n "${STRICT}" ]; then
    strict_args+=("--strict")
fi
# Optional extra CLI args for the test script (word-split on purpose), e.g.
# a training smoke run's "--max-iterations 5".
: "${ISAAC_TEST_EXTRA_ARGS:=}"
extra_args=()
if [ -n "${ISAAC_TEST_EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    extra_args=(${ISAAC_TEST_EXTRA_ARGS})
fi

trace_capture_pid=""
test_filter_pid=""
test_output_dir=""
test_pid=""
cleanup_trace_capture() {
    if [ -n "${trace_capture_pid}" ] && kill -0 "${trace_capture_pid}" 2>/dev/null; then
        kill "${trace_capture_pid}" 2>/dev/null || true
        wait "${trace_capture_pid}" 2>/dev/null || true
    fi
}

cleanup_test_processes() {
    if [ -n "${test_pid}" ] && kill -0 "${test_pid}" 2>/dev/null; then
        kill -TERM -- "-${test_pid}" 2>/dev/null || true
        wait "${test_pid}" 2>/dev/null || true
    fi
    if [ -n "${test_filter_pid}" ] && kill -0 "${test_filter_pid}" 2>/dev/null; then
        kill "${test_filter_pid}" 2>/dev/null || true
        wait "${test_filter_pid}" 2>/dev/null || true
    fi
    if [ -n "${test_output_dir}" ]; then
        rm -f "${test_output_dir}/stream"
        rmdir "${test_output_dir}" 2>/dev/null || true
    fi
}

cleanup_runtime() {
    cleanup_test_processes
    cleanup_trace_capture
}
trap cleanup_runtime EXIT

: "${ISAAC_PROFILE_TRACE:=}"
if [ -n "${ISAAC_PROFILE_TRACE}" ]; then
    trace_bins=(/isaac-sim/extscache/omni.kit.profiler.tracy-*/bin/capture)
    if [ "${#trace_bins[@]}" -ne 1 ] || [ ! -x "${trace_bins[0]}" ]; then
        echo "Expected exactly one executable Tracy capture binary, found ${#trace_bins[@]}" >&2
        exit 2
    fi
    mkdir -p "$(dirname "${ISAAC_PROFILE_TRACE}")"
    "${trace_bins[0]}" -a 127.0.0.1 -o "${ISAAC_PROFILE_TRACE}" -f &
    trace_capture_pid=$!
fi

SCRIPT_START_MS="$(now_ms)"
cd /isaac-sim
rm -f "${ISAAC_TEST_LOG}"

stage_start_ms="$(now_ms)"
interrupted_status=""
forward_test_interrupt() {
    interrupted_status="$1"
    trap - HUP INT TERM
    if [ -n "${test_pid}" ]; then
        kill -INT -- "-${test_pid}" 2>/dev/null || true
    fi
}
trap 'forward_test_interrupt 129' HUP
trap 'forward_test_interrupt 130' INT
trap 'forward_test_interrupt 143' TERM

test_output_dir="$(mktemp -d /tmp/isaac-test-output.XXXXXX)"
mkfifo "${test_output_dir}/stream"
tee -a "${ISAAC_TEST_LOG}" <"${test_output_dir}/stream" | \
    terminal_control_filter | \
    grep --line-buffered -E "${ISAAC_TEST_LOG_FILTER}" &
test_filter_pid=$!
setsid /bin/bash -c 'trap - HUP INT TERM; exec "$@"' isaac-test-python \
    "${ISAAC_PYTHON}" "${ISAAC_TEST_SCRIPT}" \
    --scene "${SCENE}" "${strict_args[@]}" "${extra_args[@]}" \
    >"${test_output_dir}/stream" 2>&1 &
test_pid=$!

set +e
wait "${test_pid}"
test_status=$?
if [ -n "${interrupted_status}" ]; then
    wait "${test_pid}" 2>/dev/null
    test_status=$?
fi
wait "${test_filter_pid}"
filter_status=$?
set -e
trap - HUP INT TERM
test_pid=""
test_filter_pid=""
rm -f "${test_output_dir}/stream"
rmdir "${test_output_dir}"
test_output_dir=""

if [ -n "${interrupted_status}" ]; then
    integration_timing "test_script" "${stage_start_ms}" "interrupted"
    exit "${interrupted_status}"
fi
if [ "${test_status}" -eq 0 ] && [ "${filter_status}" -eq 0 ]; then
    integration_timing "test_script" "${stage_start_ms}"
else
    status="${test_status}"
    if [ "${status}" -eq 0 ]; then
        status="${filter_status}"
    fi
    integration_timing "test_script" "${stage_start_ms}" "fail"
    exit "${status}"
fi

if [ -n "${trace_capture_pid}" ]; then
    if ! wait "${trace_capture_pid}"; then
        echo "Tracy capture failed" >&2
        exit 2
    fi
    trace_capture_pid=""
    if [ ! -s "${ISAAC_PROFILE_TRACE}" ]; then
        echo "Tracy capture was not written: ${ISAAC_PROFILE_TRACE}" >&2
        exit 2
    fi
fi

stage_start_ms="$(now_ms)"
if ! grep -q "${ISAAC_TEST_FAILURE_FILTER}" "${ISAAC_TEST_LOG}"; then
    integration_timing "failure_scan" "${stage_start_ms}"
else
    integration_timing "failure_scan" "${stage_start_ms}" "fail"
    exit 1
fi

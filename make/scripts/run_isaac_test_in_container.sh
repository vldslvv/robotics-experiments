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

: "${ISAAC_TEST_PIP_PACKAGES:=hydra-core omegaconf}"

SCRIPT_START_MS="$(now_ms)"
cd /isaac-sim
rm -f "${ISAAC_TEST_LOG}"

stage_start_ms="$(now_ms)"
# shellcheck disable=SC2086
if PIP_ROOT_USER_ACTION=ignore "${ISAAC_PYTHON}" -m pip install \
    --quiet --disable-pip-version-check ${ISAAC_TEST_PIP_PACKAGES} \
    >>"${ISAAC_TEST_LOG}" 2>&1; then
    integration_timing "pip_install" "${stage_start_ms}"
else
    integration_timing "pip_install" "${stage_start_ms}" "fail"
    grep -E "${ISAAC_TEST_ERROR_FILTER}" "${ISAAC_TEST_LOG}" || true
    exit 1
fi

stage_start_ms="$(now_ms)"
if "${ISAAC_PYTHON}" "${ISAAC_TEST_SCRIPT}" --scene "${SCENE}" \
    2>&1 | tee -a "${ISAAC_TEST_LOG}" | terminal_control_filter | \
        grep --line-buffered -E "${ISAAC_TEST_LOG_FILTER}"; then
    integration_timing "test_script" "${stage_start_ms}"
else
    status=$?
    integration_timing "test_script" "${stage_start_ms}" "fail"
    exit "${status}"
fi

stage_start_ms="$(now_ms)"
if ! grep -q "${ISAAC_TEST_FAILURE_FILTER}" "${ISAAC_TEST_LOG}"; then
    integration_timing "failure_scan" "${stage_start_ms}"
else
    integration_timing "failure_scan" "${stage_start_ms}" "fail"
    exit 1
fi

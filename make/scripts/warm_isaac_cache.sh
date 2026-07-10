#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
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

make_run_root() {
    printf "%s/%s.%s.%s" \
        "${INTEGRATION_ROOT}" \
        "${TEST_TARGET}" \
        "$(date -u +%Y%m%dT%H%M%SZ)" \
        "$$"
}

mkdir_colon_dirs() {
    local dirs="$1"
    local old_ifs="${IFS}"
    local dir

    IFS=:
    for dir in ${dirs}; do
        if [ -n "${dir}" ]; then
            mkdir -p "${dir}"
        fi
    done
    IFS="${old_ifs}"
}

for name in \
    ISAAC_PYTHON \
    ISAAC_TEST_ERROR_FILTER
do
    require_env "${name}"
done

: "${TEST_TARGET:=test-integration-warm-isaac-cache}"

SCRIPT_START_MS="$(now_ms)"

if [ "${WARM_ISAAC_CACHE_IN_CONTAINER:-0}" = "1" ]; then
    require_env "HOST_TIMING_LOG"
    mkdir -p "$(dirname "${HOST_TIMING_LOG}")"

    cd /isaac-sim
    warm_log="/isaac-sim/.nvidia-omniverse/logs/test_integration_warm_isaac_cache.log"
    rm -f "${warm_log}"

    stage_start_ms="$(now_ms)"
    if "${ISAAC_PYTHON}" -c \
        'from isaacsim import SimulationApp; app = SimulationApp({"headless": True, "hide_ui": True}); app.update(); app.close()' \
        >>"${warm_log}" 2>&1; then
        integration_timing "simulation_app_start" "${stage_start_ms}"
    else
        status=$?
        integration_timing "simulation_app_start" "${stage_start_ms}" "fail"
        grep -E "${ISAAC_TEST_ERROR_FILTER}" "${warm_log}" || true
        exit "${status}"
    fi
else
    for name in \
        ISAAC_DOCKER_ENV \
        ISAAC_DOCKER_VOLUMES \
        ISAAC_HOST_CACHE_DIRS \
        ISAAC_IMAGE \
        ISAAC_PROJECT_DIR \
        INTEGRATION_ROOT
    do
        require_env "${name}"
    done

    run_root="$(make_run_root)"
    ISAAC_RUN_ROOT="${run_root}/isaac"
    HOST_TIMING_LOG="${run_root}/integration-timing.log"
    CONTAINER_TIMING_LOG="${ISAAC_PROJECT_DIR}/.build/docker/integration/$(basename "${run_root}")/integration-timing.log"

    mkdir -p "${ISAAC_RUN_ROOT}/logs" "${ISAAC_RUN_ROOT}/data" "${ISAAC_RUN_ROOT}/documents"
    mkdir_colon_dirs "${ISAAC_HOST_CACHE_DIRS}"
    : >"${HOST_TIMING_LOG}"
    echo "Integration artifacts: ${run_root}"

    : "${ISAAC_DOCKER_RUNTIME:=}"
    ISAAC_DOCKER_VOLUMES="${ISAAC_DOCKER_VOLUMES//\$\{ISAAC_RUN_ROOT\}/${ISAAC_RUN_ROOT}}"

    stage_start_ms="$(now_ms)"
    # The Docker option variables are Make-provided argument fragments.
    # shellcheck disable=SC2086
    if docker run --rm \
        --network=host \
        ${ISAAC_DOCKER_RUNTIME} \
        ${ISAAC_DOCKER_ENV} \
        -e "HOST_TIMING_LOG=${CONTAINER_TIMING_LOG}" \
        -e "ISAAC_PYTHON=${ISAAC_PYTHON}" \
        -e "ISAAC_TEST_ERROR_FILTER=${ISAAC_TEST_ERROR_FILTER}" \
        -e "TEST_TARGET=${TEST_TARGET}" \
        -e "WARM_ISAAC_CACHE_IN_CONTAINER=1" \
        ${ISAAC_DOCKER_VOLUMES} \
        --entrypoint /bin/bash \
        "${ISAAC_IMAGE}" \
        /isaac-sim/proj/make/scripts/warm_isaac_cache.sh; then
        integration_timing "isaac_cache_warmup_container" "${stage_start_ms}"
    else
        status=$?
        integration_timing "isaac_cache_warmup_container" "${stage_start_ms}" "fail"
        exit "${status}"
    fi
fi

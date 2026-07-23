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
    HOST_PROJECT_DIR \
    INTEGRATION_ROOT \
    ISAAC_DOCKER_ENV \
    ISAAC_DOCKER_VOLUMES \
    ISAAC_HOST_CACHE_DIRS \
    ISAAC_IMAGE \
    ISAAC_PROJECT_DIR \
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

: "${ISAAC_DOCKER_RUNTIME:=}"
: "${ISAAC_TEST_EXTRA_ARGS:=}"
: "${ISAAC_PROFILE_TRACE_NAME:=}"
: "${RL_RUN_ID:=}"

SCRIPT_START_MS="$(now_ms)"
run_root="$(make_run_root)"
ISAAC_RUN_ROOT="${run_root}/isaac"
HOST_TIMING_LOG="${run_root}/integration-timing.log"
case "${run_root}" in
    "${HOST_PROJECT_DIR}"/*)
        container_run_root="${ISAAC_PROJECT_DIR}/${run_root#"${HOST_PROJECT_DIR}"/}"
        ;;
    *)
        echo "Integration root must be inside the project: ${run_root}" >&2
        exit 2
        ;;
esac
CONTAINER_TIMING_LOG="${container_run_root}/integration-timing.log"
profile_env=()
if [ -n "${ISAAC_PROFILE_TRACE_NAME}" ]; then
    profile_env+=(
        -e "ISAAC_PROFILE_TRACE=${container_run_root}/${ISAAC_PROFILE_TRACE_NAME}"
    )
fi

mkdir -p "${ISAAC_RUN_ROOT}/logs" "${ISAAC_RUN_ROOT}/data" "${ISAAC_RUN_ROOT}/documents"
mkdir_colon_dirs "${ISAAC_HOST_CACHE_DIRS}"
: >"${HOST_TIMING_LOG}"
echo "Integration artifacts: ${run_root}"

ISAAC_DOCKER_VOLUMES="${ISAAC_DOCKER_VOLUMES//\$\{ISAAC_RUN_ROOT\}/${ISAAC_RUN_ROOT}}"

stage_start_ms="$(now_ms)"
# The Docker option variables are Make-provided argument fragments.
# shellcheck disable=SC2086
if docker run --rm \
    --network=host \
    ${ISAAC_DOCKER_RUNTIME} \
    ${ISAAC_DOCKER_ENV} \
    -e "HOST_TIMING_LOG=${CONTAINER_TIMING_LOG}" \
    -e "ISAAC_BENCHMARK_IMAGE=${ISAAC_IMAGE}" \
    -e "ISAAC_PYTHON=${ISAAC_PYTHON}" \
    -e "ISAAC_TEST_ERROR_FILTER=${ISAAC_TEST_ERROR_FILTER}" \
    -e "ISAAC_TEST_FAILURE_FILTER=${ISAAC_TEST_FAILURE_FILTER}" \
    -e "ISAAC_TEST_LOG=${ISAAC_TEST_LOG}" \
    -e "ISAAC_TEST_LOG_FILTER=${ISAAC_TEST_LOG_FILTER}" \
    -e "RL_RUN_ID=${RL_RUN_ID}" \
    -e "ISAAC_TEST_SCRIPT=${ISAAC_TEST_SCRIPT}" \
    -e "ISAAC_TEST_EXTRA_ARGS=${ISAAC_TEST_EXTRA_ARGS}" \
    -e "SCENE=${SCENE}" \
    -e "TEST_TARGET=${TEST_TARGET}" \
    "${profile_env[@]}" \
    ${ISAAC_DOCKER_VOLUMES} \
    --entrypoint /bin/bash \
    "${ISAAC_IMAGE}" \
    "${ISAAC_PROJECT_DIR}/make/scripts/run_isaac_test_in_container.sh"; then
    integration_timing "isaac_container" "${stage_start_ms}"
else
    status=$?
    integration_timing "isaac_container" "${stage_start_ms}" "fail"
    exit "${status}"
fi

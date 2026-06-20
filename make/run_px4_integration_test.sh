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

px4_prompt_filter() {
    sed -u -E 's|\r|\n|g' | sed -u -E 's|^pxh>[[:space:]]*||; /^$/d'
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
    TEST_TARGET \
    CONTAINER_TIMING_LOG \
    HOST_ISAAC_LOG \
    HOST_PX4_LOG \
    HOST_TIMING_LOG \
    ISAAC_TEST_LOG \
    ISAAC_TEST_SCRIPT \
    ISAAC_DOCKER_ENV \
    ISAAC_DOCKER_VOLUMES \
    ISAAC_IMAGE \
    ISAAC_PYTHON \
    ISAAC_RUN_ROOT \
    ISAAC_TEST_ERROR_FILTER \
    ISAAC_TEST_FAILURE_FILTER \
    ISAAC_TEST_LOG_FILTER \
    PX4_EEPROM \
    PX4_IMAGE \
    PX4_LOGS \
    PX4_ROOTFS \
    PX4_TEST_CONTAINER_NAME \
    PX4_TEST_LOGS_DIR \
    PX4_TEST_READY_TIMEOUT_S \
    SCENE
do
    require_env "${name}"
done

: "${ISAAC_DOCKER_RUNTIME:=}"
: "${INTEGRATION_LOG_TAIL_LINES:=120}"
: "${INTEGRATION_VERBOSE:=0}"
PX4_TOTAL_INSTANCES="$(
    uv run python -c \
        'from robotics_experiments.config import get_enabled_px4_vehicle_count; print(get_enabled_px4_vehicle_count())'
)"
if [ "${PX4_TOTAL_INSTANCES}" -lt 1 ]; then
    echo "Active Hydra config has no PX4-enabled vehicles; PX4 integration requires at least one." >&2
    exit 2
fi

SCRIPT_START_MS="$(now_ms)"
mkdir -p "$(dirname "${HOST_TIMING_LOG}")"
: >"${HOST_TIMING_LOG}"
ISAAC_DOCKER_VOLUMES="${ISAAC_DOCKER_VOLUMES//\$\{ISAAC_RUN_ROOT\}/${ISAAC_RUN_ROOT}}"

# Container names and the primary container (whose network namespace Isaac and
# every additional PX4 instance share).
STARTED_CONTAINERS=()
PRIMARY_PX4_CONTAINER=""

px4_container_name() {
    local idx="$1"
    printf "%s.%s" "${PX4_TEST_CONTAINER_NAME}" "${idx}"
}

px4_host_log() {
    local idx="$1"
    if [ "${idx}" -eq 0 ]; then
        printf "%s" "${HOST_PX4_LOG}"
    else
        printf "%s.%s" "${HOST_PX4_LOG}" "${idx}"
    fi
}

wait_px4_ready() {
    local container="$1"
    local port="$2"
    local _
    for _ in $(seq 1 "${PX4_TEST_READY_TIMEOUT_S}"); do
        if docker logs "${container}" 2>&1 \
            | grep -q "Waiting for simulator to accept connection on TCP port ${port}"; then
            return 0
        fi
        if ! docker inspect --format "{{.State.Running}}" "${container}" 2>/dev/null \
            | grep -qx true; then
            echo "PX4 container ${container} exited before simulator port ${port} became ready."
            return 1
        fi
        sleep 1
    done
    echo "Timed out waiting for PX4 port ${port} readiness after ${PX4_TEST_READY_TIMEOUT_S}s."
    return 1
}

cleanup() {
    local status=$?
    local timing_status
    local stage_start_ms
    local idx

    stage_start_ms="$(now_ms)"
    timing_status="fail"
    if [ "${status}" -eq 0 ]; then
        timing_status="ok"
    fi

    if [ "${status}" -eq 0 ]; then
        echo "${TEST_TARGET}: PASS"
    else
        echo "${TEST_TARGET}: FAIL"
    fi

    echo "Detailed logs:"
    if [ -f "${HOST_ISAAC_LOG}" ]; then
        echo "  Isaac: ${HOST_ISAAC_LOG}"
    else
        echo "  Isaac: ${HOST_ISAAC_LOG} (not written)"
    fi

    idx=0
    for container in "${STARTED_CONTAINERS[@]}"; do
        local host_log
        host_log="$(px4_host_log "${idx}")"
        if docker inspect "${container}" >/dev/null 2>&1; then
            if docker logs "${container}" >"${host_log}" 2>&1; then
                echo "  PX4 SITL[${idx}]: ${host_log}"
            else
                echo "  PX4 SITL[${idx}]: ${host_log} (docker logs failed)"
            fi

            if [ "${INTEGRATION_VERBOSE}" = "1" ]; then
                terminal_control_filter <"${host_log}" | px4_prompt_filter || true
            elif [ "${status}" -ne 0 ]; then
                echo "PX4 SITL[${idx}] log tail:"
                tail -n "${INTEGRATION_LOG_TAIL_LINES}" "${host_log}" \
                    | terminal_control_filter \
                    | px4_prompt_filter \
                    || true
            fi
        else
            echo "  PX4 SITL[${idx}]: ${host_log} (not written; container was not started)"
        fi
        idx=$((idx + 1))
    done
    integration_timing "collect_logs" "${stage_start_ms}" "${timing_status}"

    stage_start_ms="$(now_ms)"
    for container in "${STARTED_CONTAINERS[@]}"; do
        docker rm -f "${container}" >/dev/null 2>&1 || true
    done
    integration_timing "remove_px4_containers" "${stage_start_ms}"
    integration_timing "total" "${SCRIPT_START_MS}" "${timing_status}"
    exit "${status}"
}

trap cleanup EXIT

stage_start_ms="$(now_ms)"
mkdir -p "$(dirname "${HOST_ISAAC_LOG}")" "$(dirname "${HOST_PX4_LOG}")"
rm -f "${HOST_ISAAC_LOG}" "${HOST_PX4_LOG}"
integration_timing "prepare_logs" "${stage_start_ms}"

# Start one PX4 SITL container per instance. Instance 0 owns the network
# namespace; additional instances join it (--network container:) and run with a
# distinct PX4 instance index so each listens on simulator TCP port 4560 + i and
# GCS UDP 14550 + i. Per-instance log/eeprom mounts keep their state isolated.
stage_start_ms="$(now_ms)"
for ((i = 0; i < PX4_TOTAL_INSTANCES; i++)); do
    container="$(px4_container_name "${i}")"
    docker rm -f "${container}" >/dev/null 2>&1 || true

    net_args=()
    log_mount="${PX4_LOGS}"
    eeprom_mount="${PX4_EEPROM}"
    rootfs_log="${PX4_ROOTFS}/log"
    rootfs_eeprom="${PX4_ROOTFS}/eeprom"
    if [ "${i}" -eq 0 ]; then
        PRIMARY_PX4_CONTAINER="${container}"
    else
        net_args=(--network "container:${PRIMARY_PX4_CONTAINER}")
        log_mount="${PX4_LOGS}/${i}"
        eeprom_mount="${PX4_EEPROM}/${i}"
        rootfs_log="${PX4_ROOTFS}/${i}/log"
        rootfs_eeprom="${PX4_ROOTFS}/${i}/eeprom"
        mkdir -p "${log_mount}" "${eeprom_mount}"
    fi

    echo "Starting temporary PX4 SITL container ${container} (instance ${i})..."
    docker run -d \
        --name "${container}" \
        "${net_args[@]}" \
        -e "PX4_INSTANCE=${i}" \
        -v "${log_mount}:${rootfs_log}" \
        -v "${eeprom_mount}:${rootfs_eeprom}" \
        "${PX4_IMAGE}" \
        /bin/bash -lc "exec /usr/local/bin/run_px4_iris_ci" \
        >/dev/null
    STARTED_CONTAINERS+=("${container}")

    echo "Waiting for PX4 instance ${i} simulator TCP port $((4560 + i)) readiness..."
    if ! wait_px4_ready "${container}" "$((4560 + i))"; then
        integration_timing "px4_readiness_wait" "${stage_start_ms}" "fail"
        exit 1
    fi
done
integration_timing "px4_readiness_wait" "${stage_start_ms}"
echo "All ${PX4_TOTAL_INSTANCES} PX4 instance(s) are waiting for Isaac."

stage_start_ms="$(now_ms)"
# The Docker option variables are Make-provided argument fragments.
# shellcheck disable=SC2086
if docker run --rm \
    --network "container:${PRIMARY_PX4_CONTAINER}" \
    ${ISAAC_DOCKER_RUNTIME} \
    ${ISAAC_DOCKER_ENV} \
    -e "HOST_TIMING_LOG=${CONTAINER_TIMING_LOG}" \
    -e "PX4_TEST_CONTAINER_NAME=${PRIMARY_PX4_CONTAINER}" \
    -e "PX4_TEST_LOGS_DIR=${PX4_TEST_LOGS_DIR}" \
    ${ISAAC_DOCKER_VOLUMES} \
    --entrypoint /bin/bash \
    "${ISAAC_IMAGE}" \
    -lc "set -o pipefail && \
        script_start_ms=\$(date +%s%3N); \
        fmt_ms() { ms=\"\$1\"; printf \"%s.%03d\" \"\$((ms / 1000))\" \"\$((ms % 1000))\"; }; \
        timing() { stage=\"\$1\"; start_ms=\"\$2\"; status=\"\${3:-ok}\"; end_ms=\$(date +%s%3N); line=\$(printf \"[integration-timing] target=%s stage=%s elapsed_s=%s total_s=%s status=%s\" \"${TEST_TARGET}\" \"\$stage\" \"\$(fmt_ms \"\$((end_ms - start_ms))\")\" \"\$(fmt_ms \"\$((end_ms - script_start_ms))\")\" \"\$status\"); printf \"%s\n\" \"\$line\"; printf \"%s\n\" \"\$line\" >>\"\$HOST_TIMING_LOG\" || true; }; \
        cd /isaac-sim && \
        rm -f \"${ISAAC_TEST_LOG}\" && \
        stage_start_ms=\$(date +%s%3N) && \
        { PIP_ROOT_USER_ACTION=ignore ${ISAAC_PYTHON} -m pip install \
            --quiet --disable-pip-version-check hydra-core omegaconf pymavlink \
            >>\"${ISAAC_TEST_LOG}\" 2>&1 || \
            { grep -E \"${ISAAC_TEST_ERROR_FILTER}\" \
                \"${ISAAC_TEST_LOG}\"; exit 1; }; } && \
        timing pip_install \"\$stage_start_ms\" && \
        stage_start_ms=\$(date +%s%3N) && \
        if ${ISAAC_PYTHON} ${ISAAC_TEST_SCRIPT} --scene \"${SCENE}\" \
            2>&1 | tee -a \"${ISAAC_TEST_LOG}\" | \
                sed -u -E 's|\x1B\[[0-9;?]*[ -/]*[@-~]||g; s|\x1B[=>c]||g' | \
                grep --line-buffered -E \"${ISAAC_TEST_LOG_FILTER}\"; then \
            timing test_script \"\$stage_start_ms\"; \
        else \
            status=\$?; timing test_script \"\$stage_start_ms\" fail; exit \"\$status\"; \
        fi && \
        stage_start_ms=\$(date +%s%3N) && \
        if ! grep -q \"${ISAAC_TEST_FAILURE_FILTER}\" \"${ISAAC_TEST_LOG}\"; then \
            timing failure_scan \"\$stage_start_ms\"; \
        else \
            timing failure_scan \"\$stage_start_ms\" fail; exit 1; \
        fi"; then
    integration_timing "isaac_container" "${stage_start_ms}"
else
    status=$?
    integration_timing "isaac_container" "${stage_start_ms}" "fail"
    exit "${status}"
fi

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
    TEST_TARGET \
    HOST_ISAAC_LOG_BASENAME \
    HOST_PX4_LOG_BASENAME \
    INTEGRATION_ROOT \
    ISAAC_TEST_LOG \
    ISAAC_TEST_SCRIPT \
    ISAAC_DOCKER_ENV \
    ISAAC_DOCKER_VOLUMES \
    ISAAC_HOST_CACHE_DIRS \
    ISAAC_IMAGE \
    ISAAC_PROJECT_DIR \
    ISAAC_PYTHON \
    ISAAC_TEST_ERROR_FILTER \
    ISAAC_TEST_FAILURE_FILTER \
    ISAAC_TEST_LOG_FILTER \
    PX4_IMAGE \
    PX4_ROOTFS \
    PX4_TEST_CONTAINER_NAME \
    PX4_TEST_READY_TIMEOUT_S \
    SCENE
do
    require_env "${name}"
done

: "${ISAAC_DOCKER_RUNTIME:=}"
: "${INTEGRATION_LOG_TAIL_LINES:=120}"
: "${INTEGRATION_VERBOSE:=0}"
: "${ISAAC_TEST_EXTRA_ARGS:=}"

# shellcheck source-path=SCRIPTDIR
# shellcheck source=px4_airframe_manifest_check.sh
# The hook may lint this changed runner without passing its unchanged helper.
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/px4_airframe_manifest_check.sh"
# PX4_IMAGE is a required env var (validated above), not the helper's local px4_image.
# shellcheck disable=SC2153
check_px4_airframe_manifest_current "${PX4_IMAGE}"

# Iterate the PX4-enabled vehicles' derived launch specs (instance index = PX4
# -i / port offset = vehicle id, plus the per-asset PX4 airframe) rather than a
# blind 0..count loop, so non-contiguous ids keep their derived ports and each
# container launches the airframe matching its drone asset. Each line is
# "<instance> <sim_model> <sys_autostart> <offboard-loss-action>
# <offboard-loss-timeout>"; split into parallel arrays.
mapfile -t PX4_LAUNCH_SPECS < <(
    uv run python -c \
        'import os
from robotics_experiments.config import load_config
for vehicle in load_config(os.environ["ROBOTICS_EXPERIMENT_CONFIG_PATH"]).vehicles.values():
    if vehicle.bridge.enabled:
        print(
            vehicle.id,
            vehicle.px4_sim_model,
            vehicle.px4_sys_autostart,
            vehicle.px4.failsafe.offboard_loss_action_param,
            vehicle.px4.failsafe.offboard_loss_timeout_s,
        )'
)
PX4_INSTANCE_INDICES=()
PX4_SIM_MODELS=()
PX4_SYS_AUTOSTARTS=()
PX4_OFFBOARD_LOSS_ACTIONS=()
PX4_OFFBOARD_LOSS_TIMEOUTS=()
for spec in "${PX4_LAUNCH_SPECS[@]}"; do
    read -r idx sim_model sys_autostart offboard_loss_action offboard_loss_timeout <<<"${spec}"
    PX4_INSTANCE_INDICES+=("${idx}")
    PX4_SIM_MODELS+=("${sim_model}")
    PX4_SYS_AUTOSTARTS+=("${sys_autostart}")
    PX4_OFFBOARD_LOSS_ACTIONS+=("${offboard_loss_action}")
    PX4_OFFBOARD_LOSS_TIMEOUTS+=("${offboard_loss_timeout}")
done
PX4_TOTAL_INSTANCES="${#PX4_INSTANCE_INDICES[@]}"
if [ "${PX4_TOTAL_INSTANCES}" -lt 1 ]; then
    echo "Active Hydra config has no PX4-enabled vehicles; PX4 integration requires at least one." >&2
    exit 2
fi

SCRIPT_START_MS="$(now_ms)"
run_root="$(make_run_root)"
ISAAC_RUN_ROOT="${run_root}/isaac"
PX4_RUN_ROOT="${run_root}/px4"
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
HOST_ISAAC_LOG="${ISAAC_RUN_ROOT}/logs/${HOST_ISAAC_LOG_BASENAME}"
HOST_PX4_LOG="${PX4_RUN_ROOT}/${HOST_PX4_LOG_BASENAME}"
PX4_TEST_LOGS_DIR="${container_run_root}/px4/logs"
PX4_EEPROM="${PX4_RUN_ROOT}/eeprom"
PX4_LOGS="${PX4_RUN_ROOT}/logs"

mkdir -p \
    "${ISAAC_RUN_ROOT}/logs" \
    "${ISAAC_RUN_ROOT}/data" \
    "${ISAAC_RUN_ROOT}/documents" \
    "${PX4_RUN_ROOT}/logs" \
    "${PX4_RUN_ROOT}/eeprom"
mkdir_colon_dirs "${ISAAC_HOST_CACHE_DIRS}"
: >"${HOST_TIMING_LOG}"
echo "Integration artifacts: ${run_root}"

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

    # Log-capture cost scales with the PX4 console scrollback, which is mostly
    # ANSI redraw noise (the real artifacts are the persisted ULog + bridge logs
    # under the run-id dir). On success capture only the tail; capture the full
    # scrollback only when it is actually needed for diagnosis (a failure or
    # explicit verbose), so a long passing flight does not pay to dump it all.
    local -a docker_logs_args=()
    if [ "${status}" -eq 0 ] && [ "${INTEGRATION_VERBOSE}" != "1" ]; then
        docker_logs_args+=(--tail "${INTEGRATION_LOG_TAIL_LINES}")
    fi

    idx=0
    for container in "${STARTED_CONTAINERS[@]}"; do
        local host_log
        host_log="$(px4_host_log "${idx}")"
        if docker inspect "${container}" >/dev/null 2>&1; then
            if docker logs "${docker_logs_args[@]}" "${container}" >"${host_log}" 2>&1; then
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

# Start one PX4 SITL container per resolved instance index. The first instance
# owns the network namespace; additional instances join it (--network container:)
# and run with their derived PX4 instance index so each listens on simulator TCP
# port 4560 + i and GCS UDP 14550 + i. Per-instance log/eeprom mounts keep their
# state isolated.
stage_start_ms="$(now_ms)"
for spec_idx in "${!PX4_INSTANCE_INDICES[@]}"; do
    i="${PX4_INSTANCE_INDICES[${spec_idx}]}"
    sim_model="${PX4_SIM_MODELS[${spec_idx}]}"
    sys_autostart="${PX4_SYS_AUTOSTARTS[${spec_idx}]}"
    offboard_loss_action="${PX4_OFFBOARD_LOSS_ACTIONS[${spec_idx}]}"
    offboard_loss_timeout="${PX4_OFFBOARD_LOSS_TIMEOUTS[${spec_idx}]}"
    container="$(px4_container_name "${i}")"
    docker rm -f "${container}" >/dev/null 2>&1 || true

    # Every instance runs from its own per-instance rootfs subdir, so log/eeprom
    # mounts are uniform (matching the persistent launcher and
    # reset_px4_state.sh). Only the network topology is special: the first
    # instance owns the shared namespace; additional instances join it.
    net_args=()
    log_mount="${PX4_LOGS}/${i}"
    eeprom_mount="${PX4_EEPROM}/${i}"
    rootfs_log="${PX4_ROOTFS}/${i}/log"
    rootfs_eeprom="${PX4_ROOTFS}/${i}/eeprom"
    mkdir -p "${log_mount}" "${eeprom_mount}"
    if [ -z "${PRIMARY_PX4_CONTAINER}" ]; then
        PRIMARY_PX4_CONTAINER="${container}"
    else
        net_args=(--network "container:${PRIMARY_PX4_CONTAINER}")
    fi

    echo "Starting temporary PX4 SITL container ${container} (instance ${i})..."
    docker run -d \
        --name "${container}" \
        "${net_args[@]}" \
        -e "PX4_INSTANCE=${i}" \
        -e "PX4_SIM_MODEL=${sim_model}" \
        -e "PX4_SYS_AUTOSTART=${sys_autostart}" \
        -e "PX4_PARAM_COM_OBL_RC_ACT=${offboard_loss_action}" \
        -e "PX4_PARAM_COM_OF_LOSS_T=${offboard_loss_timeout}" \
        -v "${log_mount}:${rootfs_log}" \
        -v "${eeprom_mount}:${rootfs_eeprom}" \
        "${PX4_IMAGE}" \
        /bin/bash -lc "exec /usr/local/bin/run_px4_sitl_ci" \
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
    -e "ISAAC_BENCHMARK_IMAGE=${ISAAC_IMAGE}" \
    -e "ISAAC_PYTHON=${ISAAC_PYTHON}" \
    -e "ISAAC_TEST_ERROR_FILTER=${ISAAC_TEST_ERROR_FILTER}" \
    -e "ISAAC_TEST_FAILURE_FILTER=${ISAAC_TEST_FAILURE_FILTER}" \
    -e "ISAAC_TEST_LOG=${ISAAC_TEST_LOG}" \
    -e "ISAAC_TEST_LOG_FILTER=${ISAAC_TEST_LOG_FILTER}" \
    -e "ISAAC_TEST_SCRIPT=${ISAAC_TEST_SCRIPT}" \
    -e "ISAAC_TEST_EXTRA_ARGS=${ISAAC_TEST_EXTRA_ARGS}" \
    -e "ALIGNMENT_ARTIFACT=${ALIGNMENT_ARTIFACT:-}" \
    -e "ALIGNMENT_REFERENCE=${ALIGNMENT_REFERENCE:-}" \
    -e "ALIGNMENT_TARGET=${ALIGNMENT_TARGET:-}" \
    -e "ROBOTICS_ALIGNMENT_RUN_DIR=${ROBOTICS_ALIGNMENT_RUN_DIR:-}" \
    -e "PX4_BENCHMARK_IMAGE=${PX4_IMAGE}" \
    -e "PX4_TEST_CONTAINER_NAME=${PRIMARY_PX4_CONTAINER}" \
    -e "PX4_TEST_LOGS_DIR=${PX4_TEST_LOGS_DIR}" \
    -e "SCENE=${SCENE}" \
    -e "STRICT=${STRICT:-}" \
    -e "TEST_TARGET=${TEST_TARGET}" \
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

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

for name in \
    TEST_TARGET \
    HOST_ISAAC_LOG \
    HOST_PX4_LOG \
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
: "${EXPERIMENT_CONFIG_OVERRIDES:=}"
: "${INTEGRATION_LOG_TAIL_LINES:=120}"
: "${INTEGRATION_VERBOSE:=0}"

px4_debug_log_path="${PX4_TEST_LOGS_DIR}/{date}/{time}_bridge.log"
config_overrides="${EXPERIMENT_CONFIG_OVERRIDES}"
if [ -n "${config_overrides}" ]; then
    config_overrides="${config_overrides} "
fi
config_overrides="${config_overrides}px4.debug_log_path=\\'${px4_debug_log_path}\\'"
ISAAC_DOCKER_VOLUMES="${ISAAC_DOCKER_VOLUMES//\$\{ISAAC_RUN_ROOT\}/${ISAAC_RUN_ROOT}}"

cleanup() {
    local status=$?

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

    if docker inspect "${PX4_TEST_CONTAINER_NAME}" >/dev/null 2>&1; then
        if docker logs "${PX4_TEST_CONTAINER_NAME}" >"${HOST_PX4_LOG}" 2>&1; then
            echo "  PX4 SITL: ${HOST_PX4_LOG}"
        else
            echo "  PX4 SITL: ${HOST_PX4_LOG} (docker logs failed)"
        fi

        if [ "${INTEGRATION_VERBOSE}" = "1" ]; then
            terminal_control_filter <"${HOST_PX4_LOG}" | px4_prompt_filter || true
        elif [ "${status}" -ne 0 ]; then
            echo "PX4 SITL log tail:"
            tail -n "${INTEGRATION_LOG_TAIL_LINES}" "${HOST_PX4_LOG}" \
                | terminal_control_filter \
                | px4_prompt_filter \
                || true
        fi
    else
        echo "  PX4 SITL: ${HOST_PX4_LOG} (not written; test container was not started)"
    fi

    docker rm -f "${PX4_TEST_CONTAINER_NAME}" >/dev/null 2>&1 || true
    exit "${status}"
}

trap cleanup EXIT

mkdir -p "$(dirname "${HOST_ISAAC_LOG}")" "$(dirname "${HOST_PX4_LOG}")"
rm -f "${HOST_ISAAC_LOG}" "${HOST_PX4_LOG}"

docker rm -f "${PX4_TEST_CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting temporary PX4 SITL container ${PX4_TEST_CONTAINER_NAME}..."
docker run -d \
    --name "${PX4_TEST_CONTAINER_NAME}" \
    -v "${PX4_LOGS}:${PX4_ROOTFS}/log" \
    -v "${PX4_EEPROM}:${PX4_ROOTFS}/eeprom" \
    "${PX4_IMAGE}" \
    /bin/bash -lc "exec /usr/local/bin/run_px4_iris_ci" \
    >/dev/null

echo "Waiting for PX4 simulator TCP port 4560 readiness..."
ready=0
for _ in $(seq 1 "${PX4_TEST_READY_TIMEOUT_S}"); do
    if docker logs "${PX4_TEST_CONTAINER_NAME}" 2>&1 \
        | grep -q "Waiting for simulator to accept connection on TCP port 4560"; then
        ready=1
        break
    fi

    if ! docker inspect --format "{{.State.Running}}" "${PX4_TEST_CONTAINER_NAME}" 2>/dev/null | grep -qx true; then
        echo "PX4 test container exited before simulator port became ready."
        exit 1
    fi

    sleep 1
done

if [ "${ready}" -ne 1 ]; then
    echo "Timed out waiting for PX4 simulator port readiness after ${PX4_TEST_READY_TIMEOUT_S}s."
    exit 1
fi

echo "PX4 simulator is waiting for Isaac on TCP port 4560."

# The Docker option variables are Make-provided argument fragments.
# shellcheck disable=SC2086
docker run --rm \
    --network "container:${PX4_TEST_CONTAINER_NAME}" \
    ${ISAAC_DOCKER_RUNTIME} \
    ${ISAAC_DOCKER_ENV} \
    -e "PX4_TEST_CONTAINER_NAME=${PX4_TEST_CONTAINER_NAME}" \
    -e "PX4_TEST_LOGS_DIR=${PX4_TEST_LOGS_DIR}" \
    -e "EXPERIMENT_CONFIG_OVERRIDES=${config_overrides}" \
    ${ISAAC_DOCKER_VOLUMES} \
    --entrypoint /bin/bash \
    "${ISAAC_IMAGE}" \
    -lc "set -o pipefail && \
        cd /isaac-sim && \
        rm -f \"${ISAAC_TEST_LOG}\" && \
        { PIP_ROOT_USER_ACTION=ignore ${ISAAC_PYTHON} -m pip install \
            --quiet --disable-pip-version-check hydra-core omegaconf pymavlink \
            >>\"${ISAAC_TEST_LOG}\" 2>&1 || \
            { grep -E \"${ISAAC_TEST_ERROR_FILTER}\" \
                \"${ISAAC_TEST_LOG}\"; exit 1; }; } && \
        ${ISAAC_PYTHON} ${ISAAC_TEST_SCRIPT} --scene \"${SCENE}\" \
            2>&1 | tee -a \"${ISAAC_TEST_LOG}\" | \
                sed -u -E 's|\x1B\[[0-9;?]*[ -/]*[@-~]||g; s|\x1B[=>c]||g' | \
                grep --line-buffered -E \"${ISAAC_TEST_LOG_FILTER}\" && \
        ! grep -q \"${ISAAC_TEST_FAILURE_FILTER}\" \"${ISAAC_TEST_LOG}\""

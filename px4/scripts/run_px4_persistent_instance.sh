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
    PX4_CONTAINER_NAME \
    PX4_DOCKER_RUNTIME \
    PX4_EEPROM \
    PX4_IMAGE \
    PX4_INSTANCE \
    PX4_LOGS \
    PX4_ROOTFS \
    PX4_TOTAL_INSTANCES
do
    require_env "${name}"
done

if ! [[ "${PX4_INSTANCE}" =~ ^[0-9]+$ ]]; then
    echo "PX4_INSTANCE must be a non-negative integer, got ${PX4_INSTANCE}" >&2
    exit 2
fi
if ! [[ "${PX4_TOTAL_INSTANCES}" =~ ^[0-9]+$ ]]; then
    echo "PX4_TOTAL_INSTANCES must be a non-negative integer, got ${PX4_TOTAL_INSTANCES}" >&2
    exit 2
fi
if [ "${PX4_INSTANCE}" -ge "${PX4_TOTAL_INSTANCES}" ]; then
    echo "PX4_INSTANCE ${PX4_INSTANCE} must be less than ${PX4_TOTAL_INSTANCES}" >&2
    exit 2
fi

container="${PX4_CONTAINER_NAME}.${PX4_INSTANCE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
    container_cmd="$(docker inspect --format '{{json .Config.Cmd}}' "${container}")"
    if [[ "${container_cmd}" != *start_px4_tmux* ]]; then
        echo "Recreating PX4 container ${container} to restore interactive tmux launch..."
        docker rm -f "${container}" >/dev/null
    else
        echo "Starting existing PX4 container ${container}..."
        exec docker start -ai "${container}"
    fi
fi

log_mount="${PX4_LOGS}/${PX4_INSTANCE}"
eeprom_mount="${PX4_EEPROM}/${PX4_INSTANCE}"
rootfs_log="${PX4_ROOTFS}/log"
rootfs_eeprom="${PX4_ROOTFS}/eeprom"

# Every persistent PX4 instance runs on host networking (the same as Isaac), so
# they share the host namespace and PX4 -i port-shifts each instance to a
# distinct simulator TCP 4560 + i / GCS UDP 14550 + i. (The integration runner's
# private shared-namespace "--network container:" model is specific to its
# ephemeral, host-network-free Isaac container; do not reuse it here.)
# PX4_DOCKER_RUNTIME is a Make-provided argument fragment.
# shellcheck disable=SC2206
net_args=(${PX4_DOCKER_RUNTIME})

if [ "${PX4_INSTANCE}" -ne 0 ]; then
    rootfs_log="${PX4_ROOTFS}/${PX4_INSTANCE}/log"
    rootfs_eeprom="${PX4_ROOTFS}/${PX4_INSTANCE}/eeprom"
fi

mkdir -p "${log_mount}" "${eeprom_mount}"

echo "Creating and starting PX4 container ${container} (instance ${PX4_INSTANCE})..."
exec docker run -it \
    --name "${container}" \
    "${net_args[@]}" \
    -e "PX4_INSTANCE=${PX4_INSTANCE}" \
    -e "PX4_ROOTFS=${PX4_ROOTFS}" \
    -v "${log_mount}:${rootfs_log}" \
    -v "${eeprom_mount}:${rootfs_eeprom}" \
    "${PX4_IMAGE}" \
    /bin/bash -lc "exec /usr/local/bin/start_px4_tmux"

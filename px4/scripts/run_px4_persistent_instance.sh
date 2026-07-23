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
    PX4_PARAM_COM_OBL_RC_ACT \
    PX4_PARAM_COM_OF_LOSS_T \
    PX4_ROOTFS \
    PX4_SIM_MODEL \
    PX4_SYS_AUTOSTART \
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

# Every instance (including 0) runs from its own per-instance rootfs subdir, so
# its log/eeprom mounts are uniform: ${PX4_ROOTFS}/${PX4_INSTANCE}/{log,eeprom}.
# This matches px4_sitl_common.sh::launch_px4_sitl and reset_px4_state.sh.
log_mount="${PX4_LOGS}/${PX4_INSTANCE}"
eeprom_mount="${PX4_EEPROM}/${PX4_INSTANCE}"
rootfs_log="${PX4_ROOTFS}/${PX4_INSTANCE}/log"
rootfs_eeprom="${PX4_ROOTFS}/${PX4_INSTANCE}/eeprom"

if ! desired_image_id="$(docker image inspect --format '{{.Id}}' "${PX4_IMAGE}")"; then
    echo "PX4 image ${PX4_IMAGE} is not available; run make build-px4." >&2
    exit 1
fi

container_env_value() {
    local key="$1"
    docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${container}" \
        | awk -v key="${key}" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }'
}

container_has_mount() {
    local source="$1"
    local destination="$2"
    docker inspect --format '{{range .Mounts}}{{println .Source .Destination}}{{end}}' "${container}" \
        | grep -Fxq "${source} ${destination}"
}

if docker ps -a --format '{{.Names}}' | grep -qx "${container}"; then
    recreate_reason=""
    container_cmd="$(docker inspect --format '{{json .Config.Cmd}}' "${container}")"
    container_image_id="$(docker inspect --format '{{.Image}}' "${container}")"
    if [[ "${container_cmd}" != *start_px4_tmux* ]]; then
        recreate_reason="command is not the interactive tmux launcher"
    elif [ "${container_image_id}" != "${desired_image_id}" ]; then
        recreate_reason="image changed"
    elif [ "$(container_env_value PX4_INSTANCE)" != "${PX4_INSTANCE}" ]; then
        recreate_reason="PX4_INSTANCE changed"
    elif [ "$(container_env_value PX4_TOTAL_INSTANCES)" != "${PX4_TOTAL_INSTANCES}" ]; then
        recreate_reason="PX4_TOTAL_INSTANCES changed"
    elif [ "$(container_env_value PX4_SIM_MODEL)" != "${PX4_SIM_MODEL}" ]; then
        recreate_reason="PX4_SIM_MODEL changed"
    elif [ "$(container_env_value PX4_SYS_AUTOSTART)" != "${PX4_SYS_AUTOSTART}" ]; then
        recreate_reason="PX4_SYS_AUTOSTART changed"
    elif [ "$(container_env_value PX4_PARAM_COM_OBL_RC_ACT)" != "${PX4_PARAM_COM_OBL_RC_ACT}" ]; then
        recreate_reason="PX4_PARAM_COM_OBL_RC_ACT changed"
    elif [ "$(container_env_value PX4_PARAM_COM_OF_LOSS_T)" != "${PX4_PARAM_COM_OF_LOSS_T}" ]; then
        recreate_reason="PX4_PARAM_COM_OF_LOSS_T changed"
    elif [ "$(container_env_value PX4_ROOTFS)" != "${PX4_ROOTFS}" ]; then
        recreate_reason="PX4_ROOTFS changed"
    elif ! container_has_mount "${log_mount}" "${rootfs_log}"; then
        recreate_reason="log mount changed"
    elif ! container_has_mount "${eeprom_mount}" "${rootfs_eeprom}"; then
        recreate_reason="EEPROM mount changed"
    fi

    if [ -n "${recreate_reason}" ]; then
        echo "Recreating PX4 container ${container}: ${recreate_reason}."
        docker rm -f "${container}" >/dev/null
    else
        echo "Starting existing PX4 container ${container}..."
        exec docker start -ai "${container}"
    fi
fi

# Every persistent PX4 instance runs on host networking (the same as Isaac), so
# they share the host namespace and PX4 -i port-shifts each instance to a
# distinct simulator TCP 4560 + i / GCS UDP 14550 + i. (The integration runner's
# private shared-namespace "--network container:" model is specific to its
# ephemeral, host-network-free Isaac container; do not reuse it here.)
# PX4_DOCKER_RUNTIME is a Make-provided argument fragment.
# shellcheck disable=SC2206
net_args=(${PX4_DOCKER_RUNTIME})

mkdir -p "${log_mount}" "${eeprom_mount}"

echo "Creating and starting PX4 container ${container} (instance ${PX4_INSTANCE})..."
exec docker run -it \
    --name "${container}" \
    "${net_args[@]}" \
    -e "PX4_INSTANCE=${PX4_INSTANCE}" \
    -e "PX4_TOTAL_INSTANCES=${PX4_TOTAL_INSTANCES}" \
    -e "PX4_SIM_MODEL=${PX4_SIM_MODEL}" \
    -e "PX4_SYS_AUTOSTART=${PX4_SYS_AUTOSTART}" \
    -e "PX4_PARAM_COM_OBL_RC_ACT=${PX4_PARAM_COM_OBL_RC_ACT}" \
    -e "PX4_PARAM_COM_OF_LOSS_T=${PX4_PARAM_COM_OF_LOSS_T}" \
    -e "PX4_ROOTFS=${PX4_ROOTFS}" \
    -v "${log_mount}:${rootfs_log}" \
    -v "${eeprom_mount}:${rootfs_eeprom}" \
    "${PX4_IMAGE}" \
    /bin/bash -lc "exec /usr/local/bin/start_px4_tmux"

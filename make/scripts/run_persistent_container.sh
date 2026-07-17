#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ] || [ "$5" != "--" ]; then
    echo "Usage: $0 DISPLAY_NAME CONTAINER_NAME IMAGE EXPECTED_ENV -- [DOCKER_ARGS...]" >&2
    exit 2
fi

display_name="$1"
container_name="$2"
image="$3"
expected_env="$4"
shift 5
docker_flags=("$@")

# Optional startup command override (image CMD args). EXPECTED_CMD is a trusted,
# internal command line supplied by the Makefile; eval lets its quoting (a single
# tmux argument that itself contains spaces) survive into discrete argv elements.
cmd_argv=()
if [ -n "${EXPECTED_CMD:-}" ]; then
    # shellcheck disable=SC2294 # trusted internal value; eval preserves quoting
    eval "cmd_argv=(${EXPECTED_CMD})"
fi

expected_binds=()
for ((index = 0; index < ${#docker_flags[@]}; index++)); do
    case "${docker_flags[$index]}" in
        -v|--volume)
            ((index += 1))
            expected_binds+=("${docker_flags[$index]}")
            ;;
        --volume=*)
            expected_binds+=("${docker_flags[$index]#--volume=}")
            ;;
    esac
done

if docker ps -a --format '{{.Names}}' | grep -qx "${container_name}"; then
    recreate_reason=""
    expected_image_id="$(docker image inspect --format '{{.Id}}' "${image}")"
    existing_image_id="$(docker inspect --format '{{.Image}}' "${container_name}")"
    if [ "${existing_image_id}" != "${expected_image_id}" ]; then
        recreate_reason="updated image"
    elif [ -n "${expected_env}" ] && ! docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${container_name}" | grep -qx "${expected_env}"; then
        recreate_reason="updated runtime environment"
    fi
    if [ -z "${recreate_reason}" ] && [ "${#expected_binds[@]}" -gt 0 ]; then
        existing_binds="$(docker inspect --format '{{range .HostConfig.Binds}}{{println .}}{{end}}' "${container_name}" | sort)"
        expected_bind_list="$(printf '%s\n' "${expected_binds[@]}" | sort)"
        if [ "${existing_binds}" != "${expected_bind_list}" ]; then
            recreate_reason="updated volume mounts"
        fi
    fi
    if [ -z "${recreate_reason}" ] && [ "${#cmd_argv[@]}" -gt 0 ]; then
        existing_cmd="$(docker inspect --format '{{join .Config.Cmd " "}}' "${container_name}")"
        if [ "${cmd_argv[*]}" != "${existing_cmd}" ]; then
            recreate_reason="updated startup command"
        fi
    fi
    if [ -n "${recreate_reason}" ]; then
        echo "Recreating ${display_name} container ${container_name} for ${recreate_reason}..."
        docker rm -f "${container_name}" >/dev/null
    else
        echo "Starting existing ${display_name} container ${container_name}..."
        exec docker start -ai "${container_name}"
    fi
fi

echo "Creating and starting ${display_name} container ${container_name}..."
exec docker run -it \
    --name "${container_name}" \
    "${docker_flags[@]}" \
    "${image}" \
    "${cmd_argv[@]}"

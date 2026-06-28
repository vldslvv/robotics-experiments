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

if docker ps -a --format '{{.Names}}' | grep -qx "${container_name}"; then
    recreate_reason=""
    if [ -n "${expected_env}" ] && ! docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${container_name}" | grep -qx "${expected_env}"; then
        recreate_reason="updated runtime environment"
    elif [ "${#cmd_argv[@]}" -gt 0 ]; then
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

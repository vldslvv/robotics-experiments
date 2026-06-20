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

if docker ps -a --format '{{.Names}}' | grep -qx "${container_name}"; then
    if [ -n "${expected_env}" ] && ! docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${container_name}" | grep -qx "${expected_env}"; then
        echo "Recreating ${display_name} container ${container_name} for updated runtime environment..."
        docker rm -f "${container_name}" >/dev/null
    else
        echo "Starting existing ${display_name} container ${container_name}..."
        exec docker start -ai "${container_name}"
    fi
fi

echo "Creating and starting ${display_name} container ${container_name}..."
exec docker run -it \
    --name "${container_name}" \
    "$@" \
    "${image}"

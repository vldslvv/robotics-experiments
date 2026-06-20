#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 stop|remove BASE_CONTAINER_NAME" >&2
    exit 2
fi

action="$1"
base_container_name="$2"

case "${action}" in
    stop)
        docker_action=(docker stop)
        ;;
    remove)
        docker_action=(docker rm -f)
        ;;
    *)
        echo "Unknown PX4 container action: ${action}" >&2
        exit 2
        ;;
esac

containers="$(
    docker ps -a --format '{{.Names}}' \
        | grep -E "^${base_container_name}(\.[0-9]+)?$" \
        || true
)"

if [ -n "${containers}" ]; then
    echo "${containers}" | xargs "${docker_action[@]}" >/dev/null 2>&1 || true
fi

#!/usr/bin/env bash

_px4_airframe_check_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_px4_airframe_manifest_path="/usr/local/share/robotics-experiments/px4-airframes.json"

check_px4_airframe_manifest_current() {
    local px4_image="$1"
    local expected_manifest
    local image_manifest

    expected_manifest="$(
        cd "${_px4_airframe_check_repo_root}" && \
            uv run python "${_px4_airframe_check_repo_root}/px4/scripts/generate_px4_airframes.py" \
                --config-dir "${_px4_airframe_check_repo_root}/configs/airframes" \
                --print-manifest
    )"
    if ! docker image inspect "${px4_image}" >/dev/null 2>&1; then
        echo "PX4 image ${px4_image} is not available; run make build-px4." >&2
        return 1
    fi
    if ! image_manifest="$(
        docker run --rm --entrypoint /bin/cat \
            "${px4_image}" \
            "${_px4_airframe_manifest_path}" \
            2>/dev/null
    )"; then
        echo \
            "PX4 image ${px4_image} does not contain ${_px4_airframe_manifest_path}; run make build-px4." \
            >&2
        return 1
    fi
    if [ "${expected_manifest}" != "${image_manifest}" ]; then
        echo "PX4 image ${px4_image} has stale generated airframes; run make build-px4." >&2
        return 1
    fi
}

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
    BENCHMARK_ROOT \
    FLIGHT_BENCHMARK_BUILD_FRESH \
    FLIGHT_BENCHMARK_MEASUREMENT_LEVEL \
    HOST_PROJECT_DIR \
    HYDRA_CONFIG_HOST_PATH \
    ISAAC_DOCKER_ENV \
    ISAAC_DOCKER_RUNTIME \
    ISAAC_DOCKER_VOLUMES \
    ISAAC_IMAGE \
    ISAAC_INTERACTIVE_ENV \
    ISAAC_INTERACTIVE_VOLUMES \
    MAKE_COMMAND \
    PX4_IMAGE \
    PX4_REF \
    PX4_ROOTFS \
    REPORT_SCRIPT
do
    require_env "${name}"
done

case "${FLIGHT_BENCHMARK_MEASUREMENT_LEVEL}" in
    lifecycle | nvidia | full) ;;
    *)
        echo "FLIGHT_BENCHMARK_MEASUREMENT_LEVEL must be lifecycle, nvidia, or full" >&2
        exit 2
        ;;
esac

run_id="$(date -u +%Y%m%dT%H%M%SZ).$$"
run_root="${BENCHMARK_ROOT}/flight.${run_id}"
container_prefix="flight-benchmark-${run_id//./-}"
mkdir -p "${run_root}"

# MAKE_COMMAND may contain jobserver flags. The benchmark calls each target
# serially so branch-local image builds and flight repetitions never overlap.
# shellcheck disable=SC2153,SC2206
make_command=(${MAKE_COMMAND})

echo "Stopping persistent simulator/PX4 stacks before the isolated benchmark..."
"${make_command[@]}" stop
echo "Building benchmark images serially (fresh=${FLIGHT_BENCHMARK_BUILD_FRESH})..."
"${make_command[@]}" build-isaac \
    ISAAC_DOCKER_BUILD_FRESH="${FLIGHT_BENCHMARK_BUILD_FRESH}"
"${make_command[@]}" build-px4 \
    PX4_DOCKER_BUILD_FRESH="${FLIGHT_BENCHMARK_BUILD_FRESH}"

isaac_image_id="$(docker image inspect "${ISAAC_IMAGE}" --format '{{.Id}}')"
px4_image_id="$(docker image inspect "${PX4_IMAGE}" --format '{{.Id}}')"
source_commit="$(git -C "${HOST_PROJECT_DIR}" rev-parse HEAD)"
read -r px4_ready_timeout config_sha256 <<<"$(
    PYTHONPATH="${HOST_PROJECT_DIR}/src" uv run python "${REPORT_SCRIPT}" fingerprint \
        --config "${HYDRA_CONFIG_HOST_PATH}" \
        --output "${run_root}/fingerprint.json" \
        --isaac-image "${ISAAC_IMAGE}" \
        --isaac-image-id "${isaac_image_id}" \
        --px4-image "${PX4_IMAGE}" \
        --px4-image-id "${px4_image_id}" \
        --px4-revision "${PX4_REF}" \
        --source-commit "${source_commit}"
)"

mapfile -t launch_specs < <(
    PYTHONPATH="${HOST_PROJECT_DIR}/src" \
        ROBOTICS_EXPERIMENT_CONFIG_PATH="${HYDRA_CONFIG_HOST_PATH}" uv run python -c \
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
            vehicle.px4.logging_profile,
        )'
)
if [ "${#launch_specs[@]}" -eq 0 ]; then
    echo "Flight benchmark requires at least one PX4-enabled vehicle" >&2
    exit 2
fi

started_containers=()
sampler_pid=""

cleanup_containers() {
    if [ -n "${sampler_pid}" ] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    sampler_pid=""
    for container in "${started_containers[@]}"; do
        docker rm -f "${container}" >/dev/null 2>&1 || true
    done
    started_containers=()
}
trap cleanup_containers EXIT

wait_px4_ready() {
    local container="$1"
    local port="$2"
    for _ in $(seq 1 "${px4_ready_timeout}"); do
        if docker logs "${container}" 2>&1 | grep -q \
            "Waiting for simulator to accept connection on TCP port ${port}"; then
            return 0
        fi
        if [ "$(docker inspect --format '{{.State.Running}}' "${container}" 2>/dev/null)" != true ]; then
            return 1
        fi
        sleep 1
    done
    return 1
}

sample_resources() {
    local stats_path="$1"
    shift
    while true; do
        docker stats --no-stream --format '{{json .}}' "$@" >>"${stats_path}" 2>/dev/null || true
        sleep 1
    done
}

run_repetition() {
    local kind="$1"
    local repetition="$2"
    local output_dir="$3"
    local label="${kind}-${repetition}"
    local isaac_container="${container_prefix}-${label}-isaac"
    local -a px4_containers=()
    local spec
    local i
    local sim_model
    local sys_autostart
    local offboard_loss_action
    local offboard_loss_timeout
    local logging_profile

    mkdir -p "${output_dir}/logs" "${output_dir}/px4/logs" "${output_dir}/px4/eeprom"
    for spec in "${launch_specs[@]}"; do
        read -r i sim_model sys_autostart offboard_loss_action offboard_loss_timeout logging_profile <<<"${spec}"
        local container="${container_prefix}-${label}-px4-${i}"
        local log_mount="${output_dir}/px4/logs/${i}"
        local eeprom_mount="${output_dir}/px4/eeprom/${i}"
        mkdir -p "${log_mount}" "${eeprom_mount}"
        docker run -d \
            --name "${container}" \
            --network=host \
            -e "PX4_INSTANCE=${i}" \
            -e "PX4_SIM_MODEL=${sim_model}" \
            -e "PX4_SYS_AUTOSTART=${sys_autostart}" \
            -e "PX4_PARAM_COM_OBL_RC_ACT=${offboard_loss_action}" \
            -e "PX4_PARAM_COM_OF_LOSS_T=${offboard_loss_timeout}" \
            -e "PX4_PARAM_SDLOG_PROFILE=${logging_profile}" \
            -v "${log_mount}:${PX4_ROOTFS}/${i}/log" \
            -v "${eeprom_mount}:${PX4_ROOTFS}/${i}/eeprom" \
            "${PX4_IMAGE}" \
            /bin/bash -lc 'exec /usr/local/bin/run_px4_sitl_ci' \
            >/dev/null
        px4_containers+=("${container}")
        started_containers+=("${container}")
        if ! wait_px4_ready "${container}" "$((4560 + i))"; then
            echo "PX4 instance ${i} did not become ready" >&2
            return 1
        fi
    done

    case "${output_dir}" in
        "${HOST_PROJECT_DIR}"/*)
            container_output="/isaac-sim/proj/${output_dir#"${HOST_PROJECT_DIR}"/}"
            ;;
        *)
            echo "Benchmark output must be inside the project" >&2
            return 2
            ;;
    esac
    trace_env=()
    if [ "${kind}" = profile ]; then
        trace_env=(-e "FLIGHT_BENCHMARK_TRACE=${container_output}/flight.tracy")
    fi

    # Docker option fragments are prepared by Make and intentionally split.
    # shellcheck disable=SC2086
    docker run -d \
        --name "${isaac_container}" \
        ${ISAAC_DOCKER_RUNTIME} \
        ${ISAAC_DOCKER_ENV} \
        ${ISAAC_INTERACTIVE_ENV} \
        -e "FLIGHT_BENCHMARK_OUTPUT_DIR=${container_output}" \
        -e "FLIGHT_BENCHMARK_CONFIG_SHA256=${config_sha256}" \
        -e "FLIGHT_BENCHMARK_MEASUREMENT_LEVEL=${FLIGHT_BENCHMARK_MEASUREMENT_LEVEL}" \
        -e "FLIGHT_BENCHMARK_LOG=${container_output}/logs/isaac-console.log" \
        -e "FLIGHT_BENCHMARK_RUN_KIND=${kind}" \
        -e "FLIGHT_BENCHMARK_REPETITION=${repetition}" \
        -e "FLIGHT_BENCHMARK_SOURCE_COMMIT=${source_commit}" \
        -e "ISAAC_BENCHMARK_IMAGE=${ISAAC_IMAGE}" \
        -e "ISAAC_BENCHMARK_IMAGE_ID=${isaac_image_id}" \
        -e "PX4_BENCHMARK_IMAGE=${PX4_IMAGE}" \
        -e "PX4_BENCHMARK_IMAGE_ID=${px4_image_id}" \
        "${trace_env[@]}" \
        ${ISAAC_DOCKER_VOLUMES} \
        ${ISAAC_INTERACTIVE_VOLUMES} \
        --entrypoint /bin/bash \
        "${ISAAC_IMAGE}" \
        /isaac-sim/proj/make/scripts/run_flight_benchmark_in_container.sh \
        >/dev/null
    started_containers+=("${isaac_container}")

    sample_resources "${output_dir}/container-stats.jsonl" \
        "${isaac_container}" "${px4_containers[@]}" &
    sampler_pid=$!
    echo "Running ${label}; the application will Play, capture, and close automatically..."
    status="$(docker wait "${isaac_container}")"
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true
    sampler_pid=""

    for container in "${px4_containers[@]}"; do
        docker logs "${container}" >"${output_dir}/logs/${container}.log" 2>&1 || true
    done
    docker cp "${isaac_container}:/root/.nvidia-omniverse/logs/Kit/." \
        "${output_dir}/logs/kit" >/dev/null 2>&1 || true
    if [ "${status}" -ne 0 ]; then
        docker logs "${isaac_container}" >&2 || true
        return "${status}"
    fi
    if [ ! -s "${output_dir}/samples.json" ]; then
        echo "${label} did not produce samples.json" >&2
        return 2
    fi
    cleanup_containers
}

echo "Flight benchmark artifacts: ${run_root}"
run_repetition warmup 0 "${run_root}/warmup"
for repetition in 1 2 3; do
    run_repetition measured "${repetition}" "${run_root}/raw/repeat-${repetition}"
done
run_repetition profile 0 "${run_root}/profile"

uv run python "${REPORT_SCRIPT}" build \
    --raw-root "${run_root}/raw" \
    --profile-root "${run_root}/profile" \
    --run-root "${run_root}" \
    --fingerprint "${run_root}/fingerprint.json"

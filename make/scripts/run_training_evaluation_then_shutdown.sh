#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  make/scripts/run_training_evaluation_then_shutdown.sh \
    --hydra-config PATH \
    --eval-target TARGET \
    [--train-target TARGET] \
    [--train-args ARGS] \
    [--eval-args ARGS] \
    [--label LABEL] \
    [--log-dir PATH] \
    [--shutdown]

Run Make training/publication and independent-evaluation targets in that order,
preserving console and status logs. The wrapper also serves the
training logs through TensorBoard for the lifetime of the workflow and prints
its URL. A failed stage skips every later stage. Power-off after success,
failure, or a handled interrupt is disabled by default and must be requested
with --shutdown.

Defaults:
  --train-target train-rl
  --train-args   ""
  --eval-args    ""
  --label        rl-training-evaluation
  --log-dir      .build/overnight
  --shutdown     disabled
EOF
}

require_value() {
    local option="$1"
    local value="${2:-}"

    if [ -z "${value}" ]; then
        echo "${option} requires a non-empty value" >&2
        usage >&2
        exit 2
    fi
}

validate_name() {
    local label="$1"
    local value="$2"

    if [[ ! "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        echo "${label} must contain only letters, digits, '.', '_', or '-': ${value}" >&2
        exit 2
    fi
}

hydra_config=""
train_target="train-rl"
eval_target=""
train_args=""
eval_args=""
run_label="rl-training-evaluation"
log_dir=".build/overnight"
shutdown_requested=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --hydra-config)
            require_value "$1" "${2:-}"
            hydra_config="$2"
            shift 2
            ;;
        --train-target)
            require_value "$1" "${2:-}"
            train_target="$2"
            shift 2
            ;;
        --eval-target)
            require_value "$1" "${2:-}"
            eval_target="$2"
            shift 2
            ;;
        --train-args)
            train_args="${2-}"
            shift 2
            ;;
        --eval-args)
            eval_args="${2-}"
            shift 2
            ;;
        --label)
            require_value "$1" "${2:-}"
            run_label="$2"
            shift 2
            ;;
        --log-dir)
            require_value "$1" "${2:-}"
            log_dir="$2"
            shift 2
            ;;
        --shutdown)
            shutdown_requested=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

require_value "--hydra-config" "${hydra_config}"
require_value "--eval-target" "${eval_target}"
validate_name "training target" "${train_target}"
validate_name "evaluation target" "${eval_target}"
validate_name "label" "${run_label}"

if [ ! -f "${hydra_config}" ]; then
    echo "Hydra config does not exist: ${hydra_config}" >&2
    exit 2
fi

policy_name="$(
    ROBOTICS_EXPERIMENT_CONFIG_PATH="${hydra_config}" \
        PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" \
        uv run python -c \
        'import os; from robotics_experiments.config import load_config; print(load_config(os.environ["ROBOTICS_EXPERIMENT_CONFIG_PATH"]).rl.training.policy_name)'
)"
validate_name "policy name" "${policy_name}"

mkdir -p "${log_dir}"
artifact_run_id="$(date -u +%Y%m%dT%H%M%SZ).$$"
run_id="${run_label}.${artifact_run_id}"
console_log="${log_dir}/${run_id}.console.log"
status_log="${log_dir}/${run_id}.status"
train_status="not-run"
eval_status="not-run"
sudo_keeper_pid=""
tensorboard_monitor_pid=""
sudo_validated=false
tensorboard_host="127.0.0.1"
tensorboard_port="${RL_TENSORBOARD_PORT:-6006}"
tensorboard_base_url="http://${tensorboard_host}:${tensorboard_port}/"
tensorboard_primary_filter='^(train/01_reset/(sample_count|position_envelope_utilization_(mean|p95|max)|attitude_envelope_utilization_(mean|p95|max))|train/02_progress/(recent_100_episode_return_mean|hover_envelope_occupancy_rate|episode_longest_hover_duration_mean_s|hover_envelope_utilization_mean)|train/03_reward/per_step_mean|train/05_schedule/recovery_boundary_m|val/01_acceptance/(all_checks_passed|success_rate|collision_rate|flight_failure_rate|minimum_closest_range_m|all_traces_finite)|val/02_hover/(envelope_occupancy_rate|episode_longest_duration_mean_s|episode_longest_duration_max_s|envelope_utilization_mean|position_envelope_utilization_mean|attitude_envelope_utilization_mean)|val/03_outcome/range_undershoot_rate)$'
tensorboard_url="${tensorboard_base_url}?smoothing=0&tagFilter=${tensorboard_primary_filter}"
tensorboard_container_name="vv-rl-tensorboard.${artifact_run_id}"
tensorboard_logdir=".build/training/${policy_name}/${artifact_run_id}/logs"
tensorboard_ready_file="${log_dir}/${run_id}.tensorboard-ready"
shutdown_mode="disabled"
if [ "${shutdown_requested}" = true ]; then
    shutdown_mode="enabled"
fi

keep_sudo_alive() {
    # shellcheck disable=SC2154 # `read` assigns job inside the single-quoted trap.
    trap 'jobs -pr | while read -r job; do kill "${job}" 2>/dev/null; done; exit 0' TERM
    while sudo -n true; do
        sleep 60 &
        wait "$!"
    done
}

# Called only by the EXIT-trap handler below.
# shellcheck disable=SC2329
write_status() {
    local overall_status="$1"
    local shutdown_status="$2"
    local status_line

    status_line="$(printf \
        'run_id=%s artifact_run_id=%s train_target=%s eval_target=%s train_exit=%s eval_exit=%s overall_exit=%s shutdown=%s shutdown_exit=%s' \
        "${run_id}" \
        "${artifact_run_id}" \
        "${train_target}" \
        "${eval_target}" \
        "${train_status}" \
        "${eval_status}" \
        "${overall_status}" \
        "${shutdown_mode}" \
        "${shutdown_status}")"
    printf '%s\n' "${status_line}" | tee -a "${console_log}" >"${status_log}"
}

# Called indirectly by the EXIT trap installed before the workflow starts.
# shellcheck disable=SC2329
finish_run() {
    local workflow_status=$?
    local overall_status="${workflow_status}"
    local shutdown_status="not-requested"

    set +e
    trap - EXIT HUP INT TERM
    if [ -n "${tensorboard_monitor_pid}" ]; then
        kill "${tensorboard_monitor_pid}" 2>/dev/null
        docker stop "${tensorboard_container_name}" >/dev/null 2>&1
        wait "${tensorboard_monitor_pid}" 2>/dev/null
    fi
    rm -f "${tensorboard_ready_file}"
    if [ -n "${sudo_keeper_pid}" ]; then
        kill "${sudo_keeper_pid}" 2>/dev/null
        wait "${sudo_keeper_pid}" 2>/dev/null
    fi
    if [ "${shutdown_requested}" = true ]; then
        shutdown_status="not-run"
    fi
    write_status "${overall_status}" "${shutdown_status}"
    if [ "${shutdown_requested}" = false ] || [ "${sudo_validated}" = false ]; then
        exit "${workflow_status}"
    fi
    printf '[%s] Power-off requested after workflow exit %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${workflow_status}" |
        tee -a "${console_log}"
    sync
    sudo -n shutdown -P now
    shutdown_status=$?
    if [ "${shutdown_status}" -ne 0 ]; then
        overall_status="${shutdown_status}"
    fi
    write_status "${overall_status}" "${shutdown_status}"
    if [ "${shutdown_status}" -ne 0 ]; then
        printf 'ERROR: automatic power-off failed with exit %s\n' \
            "${shutdown_status}" |
            tee -a "${console_log}" >&2
        exit "${shutdown_status}"
    fi
    exit "${workflow_status}"
}

trap finish_run EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

serve_tensorboard_when_ready() {
    local environment=""
    local server_pid=""
    local scalar_tags=""

    trap 'if [ -n "${server_pid}" ]; then kill "${server_pid}" 2>/dev/null; fi; exit 0' HUP INT TERM
    until find "${tensorboard_logdir}" \
        -maxdepth 2 \
        -type f \
        -name 'events.out.tfevents.*' \
        -size +0c \
        -print \
        -quit \
        2>/dev/null |
        grep -q .; do
        sleep 1
    done
    make tensorboard-rl \
        "RL_TENSORBOARD_CONTAINER_NAME=${tensorboard_container_name}" \
        "RL_TENSORBOARD_HOST=${tensorboard_host}" \
        "RL_TENSORBOARD_LOGDIR=${tensorboard_logdir}" \
        "RL_TENSORBOARD_PORT=${tensorboard_port}" \
        >>"${console_log}" 2>&1 &
    server_pid=$!
    while true; do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            printf 'ERROR: TensorBoard exited before becoming ready; see %s\n' \
                "${console_log}" |
                tee -a "${console_log}" >&2
            return 1
        fi
        if curl --fail --silent --output /dev/null "${tensorboard_base_url}"; then
            environment="$(
                curl --fail --silent \
                    "${tensorboard_base_url}data/environment" ||
                    true
            )"
            scalar_tags="$(
                curl --fail --silent \
                    "${tensorboard_base_url}data/plugin/scalars/tags" ||
                    true
            )"
            if [[ "${environment}" == *'"data_location":'*"${tensorboard_logdir}"'"'* ]] &&
                [[ "${scalar_tags}" == *\"* ]]; then
                printf '[%s] TensorBoard URL: %s (run: %s/%s; scalar data loaded)\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    "${tensorboard_url}" \
                    "${policy_name}" \
                    "${artifact_run_id}" |
                    tee -a "${console_log}"
                printf '[%s] TensorBoard opens the primary learning and validation charts with Smoothing=0; exact selected-step values are in %s/training_report.json\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    ".build/training/${policy_name}/${artifact_run_id}" |
                    tee -a "${console_log}"
                : >"${tensorboard_ready_file}"
                wait "${server_pid}"
                return $?
            fi
        fi
        sleep 1
    done
}

run_stage() {
    local target="$1"
    local argument_name="$2"
    local argument_value="$3"
    local -a pipeline_status

    set +e
    make "${target}" \
        "HYDRA_CONFIG=${hydra_config}" \
        "RL_RUN_ID=${artifact_run_id}" \
        "${argument_name}=${argument_value}" \
        2>&1 | tee -a "${console_log}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    stage_status="${pipeline_status[0]}"
    if [ "${stage_status}" -eq 0 ] && [ "${pipeline_status[1]}" -ne 0 ]; then
        stage_status="${pipeline_status[1]}"
    fi
}

printf '[%s] run_id=%s artifact_run_id=%s config=%s train_target=%s eval_target=%s shutdown=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${run_id}" \
    "${artifact_run_id}" \
    "${hydra_config}" \
    "${train_target}" \
    "${eval_target}" \
    "${shutdown_mode}" |
    tee -a "${console_log}"

serve_tensorboard_when_ready &
tensorboard_monitor_pid=$!

if [ "${shutdown_requested}" = true ]; then
    sudo -v
    sudo_validated=true
    keep_sudo_alive &
    sudo_keeper_pid=$!
fi

run_stage "${train_target}" "TRAIN_ARGS" "${train_args}"
train_status="${stage_status}"

if [ "${train_status}" -ne 0 ]; then
    exit "${train_status}"
fi
while [ ! -f "${tensorboard_ready_file}" ]; do
    if ! kill -0 "${tensorboard_monitor_pid}" 2>/dev/null; then
        set +e
        wait "${tensorboard_monitor_pid}"
        tensorboard_status=$?
        set -e
        tensorboard_monitor_pid=""
        if [ "${tensorboard_status}" -ne 0 ]; then
            exit "${tensorboard_status}"
        fi
        break
    fi
    sleep 0.1
done

run_stage "${eval_target}" "EVAL_ARGS" "${eval_args}"
eval_status="${stage_status}"

exit "${eval_status}"

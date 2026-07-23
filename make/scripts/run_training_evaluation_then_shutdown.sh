#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  make/scripts/run_training_evaluation_then_shutdown.sh \
    --hydra-config PATH \
    --eval-target TARGET \
    [--train-target TARGET] \
    [--select-target TARGET] \
    [--train-args ARGS] \
    [--select-args ARGS] \
    [--eval-args ARGS] \
    [--label LABEL] \
    [--log-dir PATH] \
    [--shutdown]

Run Make training, checkpoint-selection, and independent-evaluation targets in
that order, preserving console and status logs. The wrapper also serves the
training logs through TensorBoard for the lifetime of the workflow and prints
its URL. A failed stage skips every later stage. Power-off after success,
failure, or a handled interrupt is disabled by default and must be requested
with --shutdown.

Defaults:
  --train-target train-rl
  --select-target select-rl-checkpoint
  --train-args   ""
  --select-args  ""
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
select_target="select-rl-checkpoint"
eval_target=""
train_args=""
select_args=""
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
        --select-target)
            require_value "$1" "${2:-}"
            select_target="$2"
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
        --select-args)
            select_args="${2-}"
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
validate_name "selection target" "${select_target}"
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
select_status="not-run"
eval_status="not-run"
sudo_keeper_pid=""
tensorboard_pid=""
sudo_validated=false
tensorboard_host="127.0.0.1"
tensorboard_port="6006"
tensorboard_url="http://${tensorboard_host}:${tensorboard_port}/"
tensorboard_container_name="vv-rl-tensorboard.${artifact_run_id}"
tensorboard_logdir=".build/training/${policy_name}/${artifact_run_id}/logs"
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
        'run_id=%s artifact_run_id=%s train_target=%s select_target=%s eval_target=%s train_exit=%s select_exit=%s eval_exit=%s overall_exit=%s shutdown=%s shutdown_exit=%s' \
        "${run_id}" \
        "${artifact_run_id}" \
        "${train_target}" \
        "${select_target}" \
        "${eval_target}" \
        "${train_status}" \
        "${select_status}" \
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
    if [ -n "${tensorboard_pid}" ]; then
        kill "${tensorboard_pid}" 2>/dev/null
        docker stop "${tensorboard_container_name}" >/dev/null 2>&1
        wait "${tensorboard_pid}" 2>/dev/null
    fi
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

start_tensorboard() {
    make tensorboard-rl \
        "RL_TENSORBOARD_CONTAINER_NAME=${tensorboard_container_name}" \
        "RL_TENSORBOARD_HOST=${tensorboard_host}" \
        "RL_TENSORBOARD_LOGDIR=${tensorboard_logdir}" \
        "RL_TENSORBOARD_PORT=${tensorboard_port}" \
        >>"${console_log}" 2>&1 &
    tensorboard_pid=$!
    for _ in {1..30}; do
        if ! kill -0 "${tensorboard_pid}" 2>/dev/null; then
            printf 'ERROR: TensorBoard exited before becoming ready; see %s\n' \
                "${console_log}" >&2
            return 1
        fi
        if curl --fail --silent --output /dev/null "${tensorboard_url}"; then
            printf '[%s] TensorBoard URL: %s (run: %s/%s)\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                "${tensorboard_url}" \
                "${policy_name}" \
                "${artifact_run_id}" |
                tee -a "${console_log}"
            return 0
        fi
        sleep 1
    done
    printf 'ERROR: TensorBoard did not become ready at %s; see %s\n' \
        "${tensorboard_url}" \
        "${console_log}" >&2
    return 1
}

printf '[%s] run_id=%s artifact_run_id=%s config=%s train_target=%s select_target=%s eval_target=%s shutdown=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${run_id}" \
    "${artifact_run_id}" \
    "${hydra_config}" \
    "${train_target}" \
    "${select_target}" \
    "${eval_target}" \
    "${shutdown_mode}" |
    tee -a "${console_log}"

start_tensorboard

if [ "${shutdown_requested}" = true ]; then
    sudo -v
    sudo_validated=true
    keep_sudo_alive &
    sudo_keeper_pid=$!
fi

set +e
make "${train_target}" \
    "HYDRA_CONFIG=${hydra_config}" \
    "RL_RUN_ID=${artifact_run_id}" \
    "TRAIN_ARGS=${train_args}" \
    2>&1 | tee -a "${console_log}"
train_pipeline=("${PIPESTATUS[@]}")
set -e
train_status="${train_pipeline[0]}"
if [ "${train_status}" -eq 0 ] && [ "${train_pipeline[1]}" -ne 0 ]; then
    train_status="${train_pipeline[1]}"
fi

if [ "${train_status}" -ne 0 ]; then
    exit "${train_status}"
fi

set +e
make "${select_target}" \
    "HYDRA_CONFIG=${hydra_config}" \
    "RL_RUN_ID=${artifact_run_id}" \
    "SELECT_ARGS=${select_args}" \
    2>&1 | tee -a "${console_log}"
select_pipeline=("${PIPESTATUS[@]}")
set -e
select_status="${select_pipeline[0]}"
if [ "${select_status}" -eq 0 ] && [ "${select_pipeline[1]}" -ne 0 ]; then
    select_status="${select_pipeline[1]}"
fi

if [ "${select_status}" -ne 0 ]; then
    exit "${select_status}"
fi

set +e
make "${eval_target}" \
    "HYDRA_CONFIG=${hydra_config}" \
    "RL_RUN_ID=${artifact_run_id}" \
    "EVAL_ARGS=${eval_args}" \
    2>&1 | tee -a "${console_log}"
eval_pipeline=("${PIPESTATUS[@]}")
set -e
eval_status="${eval_pipeline[0]}"
if [ "${eval_status}" -eq 0 ] && [ "${eval_pipeline[1]}" -ne 0 ]; then
    eval_status="${eval_pipeline[1]}"
fi

exit "${eval_status}"

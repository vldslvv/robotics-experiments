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
    [--log-dir PATH]

Run one Make training target followed by one Make evaluation target, preserving
console and status logs and powering off after success, failure, or a handled
interrupt. Evaluation is skipped when training fails.

Defaults:
  --train-target train-rl
  --train-args   ""
  --eval-args    ""
  --label        rl-training-evaluation
  --log-dir      .build/overnight
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

mkdir -p "${log_dir}"
run_id="${run_label}.$(date -u +%Y%m%dT%H%M%SZ).$$"
console_log="${log_dir}/${run_id}.console.log"
status_log="${log_dir}/${run_id}.status"
train_status="not-run"
eval_status="not-run"
sudo_keeper_pid=""

keep_sudo_alive() {
    local sleep_pid=""

    trap 'kill "${sleep_pid}" 2>/dev/null; exit 0' TERM
    while sudo -n true; do
        sleep 60 &
        sleep_pid=$!
        wait "${sleep_pid}"
    done
}

# Called indirectly by the EXIT trap installed after sudo validation.
# shellcheck disable=SC2317
finish_and_power_off() {
    local workflow_status=$?
    local shutdown_status
    local status_line

    set +e
    trap - EXIT HUP INT TERM
    if [ -n "${sudo_keeper_pid}" ]; then
        kill "${sudo_keeper_pid}" 2>/dev/null
        wait "${sudo_keeper_pid}" 2>/dev/null
    fi
    status_line="$(printf \
        'run_id=%s train_exit=%s eval_exit=%s overall_exit=%s' \
        "${run_id}" \
        "${train_status}" \
        "${eval_status}" \
        "${workflow_status}")"
    printf '%s\n' "${status_line}" | tee -a "${console_log}" >"${status_log}"
    printf '[%s] Power-off requested after workflow exit %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "${workflow_status}" |
        tee -a "${console_log}"
    sync
    sudo -n shutdown -P now
    shutdown_status=$?
    if [ "${shutdown_status}" -ne 0 ]; then
        printf 'ERROR: automatic power-off failed with exit %s\n' \
            "${shutdown_status}" |
            tee -a "${console_log}" >&2
        exit "${shutdown_status}"
    fi
    exit "${workflow_status}"
}

sudo -v
keep_sudo_alive &
sudo_keeper_pid=$!
trap finish_and_power_off EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

printf '[%s] run_id=%s config=%s train_target=%s eval_target=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${run_id}" \
    "${hydra_config}" \
    "${train_target}" \
    "${eval_target}" |
    tee -a "${console_log}"

set +e
make "${train_target}" \
    "HYDRA_CONFIG=${hydra_config}" \
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
make "${eval_target}" \
    "HYDRA_CONFIG=${hydra_config}" \
    "EVAL_ARGS=${eval_args}" \
    2>&1 | tee -a "${console_log}"
eval_pipeline=("${PIPESTATUS[@]}")
set -e
eval_status="${eval_pipeline[0]}"
if [ "${eval_status}" -eq 0 ] && [ "${eval_pipeline[1]}" -ne 0 ]; then
    eval_status="${eval_pipeline[1]}"
fi

exit "${eval_status}"

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
    BUILD_ROOT \
    INTEGRATION_ROOT \
    INTEGRATION_TEST_JOBS \
    INTEGRATION_TEST_TARGETS
do
    require_env "${name}"
done

: "${MAKE_COMMAND:=make}"

mkdir -p "${INTEGRATION_ROOT}"
results_file="${BUILD_ROOT}/integration-test-results.$$"
: >"${results_file}"
trap 'rm -f "${results_file}"' EXIT

# shellcheck disable=SC2086
printf "%s\n" ${INTEGRATION_TEST_TARGETS} | \
    xargs -P "${INTEGRATION_TEST_JOBS}" -I {} bash -c '
        target="$1"
        results_file="$2"
        make_command="$3"

        printf "\n==> %s\n" "${target}"
        if "${make_command}" --no-print-directory "${target}"; then
            printf "%s PASS\n" "${target}" >>"${results_file}"
        else
            printf "%s FAIL\n" "${target}" >>"${results_file}"
        fi
    ' bash {} "${results_file}" "${MAKE_COMMAND}"

printf "\nIntegration test summary\n"
printf "%-45s %s\n" "Target" "Result"
printf "%-45s %s\n" "------" "------"
passed=0
failed=0
while read -r target result; do
    printf "%-45s %s\n" "${target}" "${result}"
    if [ "${result}" = "PASS" ]; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
done <"${results_file}"
printf "\nPassed: %s  Failed: %s\n" "${passed}" "${failed}"
if [ "${failed}" -ne 0 ]; then
    exit 1
fi

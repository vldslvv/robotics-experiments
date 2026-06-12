########################################
# Container-backed integration tests
########################################

.PHONY: test-integration
test-integration: test-integration-isaac test-integration-px4

INTEGRATION_TEST_TARGETS = \
    test-integration-isaac \
    test-integration-drone-articulation \
    test-integration-px4 \
    test-integration-px4-takeoff \
    test-integration-px4-takeoff-instability

.PHONY: test-integration-all
test-integration-all:
	@mkdir -p "$(BUILD_ROOT)"; \
	results_file="$(BUILD_ROOT)/integration-test-results.$$$$"; \
	: >"$$results_file"; \
	trap 'rm -f "$$results_file"' EXIT; \
	passed=0; \
	failed=0; \
	for target in $(INTEGRATION_TEST_TARGETS); do \
		printf '\n==> %s\n' "$$target"; \
		if $(MAKE) --no-print-directory "$$target"; then \
			printf '%s PASS\n' "$$target" >>"$$results_file"; \
			passed=$$((passed + 1)); \
		else \
			printf '%s FAIL\n' "$$target" >>"$$results_file"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	printf '\nIntegration test summary\n'; \
	printf '%-45s %s\n' 'Target' 'Result'; \
	printf '%-45s %s\n' '------' '------'; \
	while read -r target result; do \
		printf '%-45s %s\n' "$$target" "$$result"; \
	done <"$$results_file"; \
	printf '\nPassed: %s  Failed: %s\n' "$$passed" "$$failed"; \
	if [ "$$failed" -ne 0 ]; then exit 1; fi

.PHONY: test-integration-isaac
test-integration-isaac: ; @mkdir -p $(ISAAC_RUNTIME_DIRS) && \
    docker run --rm \
        $(ISAAC_DOCKER_RUNTIME) \
        $(ISAAC_DOCKER_ENV) \
        $(ISAAC_DOCKER_VOLUMES) \
        --entrypoint /bin/bash \
        $(ISAAC_IMAGE) \
        -lc 'set -o pipefail && \
            cd /isaac-sim && \
            rm -f "$(ISAAC_SIM_TEST_LOG)" && \
            { PIP_ROOT_USER_ACTION=ignore $(ISAAC_PYTHON) -m pip install \
                --quiet --disable-pip-version-check hydra-core omegaconf \
                >>"$(ISAAC_SIM_TEST_LOG)" 2>&1 || \
                { grep -E "$(ISAAC_TEST_ERROR_FILTER)" \
                    "$(ISAAC_SIM_TEST_LOG)"; exit 1; }; } && \
            $(ISAAC_PYTHON) $(ISAAC_SIM_TEST) --scene "$(SCENE)" \
                2>&1 | tee -a "$(ISAAC_SIM_TEST_LOG)" | $(TERMINAL_CONTROL_FILTER) | \
                    grep --line-buffered -E "$(ISAAC_TEST_LOG_FILTER)" && \
            ! grep -q "$(ISAAC_TEST_FAILURE_FILTER)" "$(ISAAC_SIM_TEST_LOG)"'

.PHONY: test-integration-drone-articulation
test-integration-drone-articulation: ; @mkdir -p $(ISAAC_RUNTIME_DIRS) && \
    docker run --rm \
        $(ISAAC_DOCKER_RUNTIME) \
        $(ISAAC_DOCKER_ENV) \
        $(ISAAC_DOCKER_VOLUMES) \
        --entrypoint /bin/bash \
        $(ISAAC_IMAGE) \
        -lc 'set -o pipefail && \
            cd /isaac-sim && \
            rm -f "$(ISAAC_DRONE_ARTICULATION_TEST_LOG)" && \
            { PIP_ROOT_USER_ACTION=ignore $(ISAAC_PYTHON) -m pip install \
                --quiet --disable-pip-version-check hydra-core omegaconf \
                >>"$(ISAAC_DRONE_ARTICULATION_TEST_LOG)" 2>&1 || \
                { grep -E "$(ISAAC_TEST_ERROR_FILTER)" \
                    "$(ISAAC_DRONE_ARTICULATION_TEST_LOG)"; exit 1; }; } && \
            $(ISAAC_PYTHON) $(ISAAC_DRONE_ARTICULATION_TEST) --scene "$(SCENE)" \
                2>&1 | tee -a "$(ISAAC_DRONE_ARTICULATION_TEST_LOG)" | \
                    $(TERMINAL_CONTROL_FILTER) | \
                    grep --line-buffered -E "$(ISAAC_TEST_LOG_FILTER)" && \
            ! grep -q "$(ISAAC_TEST_FAILURE_FILTER)" "$(ISAAC_DRONE_ARTICULATION_TEST_LOG)"'

PX4_INTEGRATION_RUNNER = make/run_px4_integration_test.sh

PX4_INTEGRATION_COMMON_ENV = \
    ISAAC_DOCKER_RUNTIME='$(ISAAC_DOCKER_RUNTIME)' \
    ISAAC_DOCKER_ENV='$(ISAAC_DOCKER_ENV)' \
    ISAAC_DOCKER_VOLUMES='$(ISAAC_DOCKER_VOLUMES)' \
    ISAAC_IMAGE='$(ISAAC_IMAGE)' \
    ISAAC_PYTHON='$(ISAAC_PYTHON)' \
    ISAAC_TEST_ERROR_FILTER='$(ISAAC_TEST_ERROR_FILTER)' \
    ISAAC_TEST_FAILURE_FILTER='$(ISAAC_TEST_FAILURE_FILTER)' \
    ISAAC_TEST_LOG_FILTER='$(ISAAC_TEST_LOG_FILTER)' \
    INTEGRATION_LOG_TAIL_LINES='$(INTEGRATION_LOG_TAIL_LINES)' \
    INTEGRATION_VERBOSE='$(INTEGRATION_VERBOSE)' \
    PX4_CONTAINER_NAME='$(PX4_CONTAINER_NAME)' \
    PX4_EEPROM='$(PX4_EEPROM)' \
    PX4_IMAGE='$(PX4_IMAGE)' \
    PX4_LOGS='$(PX4_LOGS)' \
    PX4_ROOTFS='$(PX4_ROOTFS)' \
    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME)' \
    PX4_TEST_READY_TIMEOUT_S='$(PX4_TEST_READY_TIMEOUT_S)' \
    SCENE='$(SCENE)'

.PHONY: test-integration-px4
test-integration-px4: ; @mkdir -p $(ISAAC_RUNTIME_DIRS) "$(PX4_LOGS)" "$(PX4_EEPROM)" && \
    TEST_TARGET='test-integration-px4' \
        HOST_ISAAC_LOG='$(ISAAC_LOGS)/test_px4_bridge_host.log' \
        HOST_PX4_LOG='$(PX4_TEST_LOG)' \
        ISAAC_TEST_LOG='$(ISAAC_PX4_BRIDGE_TEST_LOG)' \
        ISAAC_TEST_SCRIPT='$(ISAAC_PX4_BRIDGE_TEST)' \
        $(PX4_INTEGRATION_COMMON_ENV) \
        bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-integration-px4-takeoff
test-integration-px4-takeoff: ; @mkdir -p $(ISAAC_RUNTIME_DIRS) "$(PX4_LOGS)" "$(PX4_EEPROM)" && \
    TEST_TARGET='test-integration-px4-takeoff' \
        HOST_ISAAC_LOG='$(ISAAC_LOGS)/test_px4_takeoff_stability.log' \
        HOST_PX4_LOG='$(PX4_TAKEOFF_TEST_LOG)' \
        ISAAC_TEST_LOG='$(ISAAC_PX4_TAKEOFF_TEST_LOG)' \
        ISAAC_TEST_SCRIPT='$(ISAAC_PX4_TAKEOFF_TEST)' \
        $(PX4_INTEGRATION_COMMON_ENV) \
        bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-integration-px4-takeoff-instability
test-integration-px4-takeoff-instability: ; @mkdir -p $(ISAAC_RUNTIME_DIRS) "$(PX4_LOGS)" "$(PX4_EEPROM)" && \
    TEST_TARGET='test-integration-px4-takeoff-instability' \
        HOST_ISAAC_LOG='$(ISAAC_LOGS)/test_px4_takeoff_instability_detection.log' \
        HOST_PX4_LOG='$(PX4_TAKEOFF_INSTABILITY_TEST_LOG)' \
        ISAAC_TEST_LOG='$(ISAAC_PX4_TAKEOFF_INSTABILITY_TEST_LOG)' \
        ISAAC_TEST_SCRIPT='$(ISAAC_PX4_TAKEOFF_INSTABILITY_TEST)' \
        $(PX4_INTEGRATION_COMMON_ENV) \
        bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-isaac-simulation
test-isaac-simulation: test-integration-isaac

.PHONY: test-isaac-drone-articulation
test-isaac-drone-articulation: test-integration-drone-articulation

.PHONY: test-isaac-px4-bridge
test-isaac-px4-bridge: test-integration-px4

.PHONY: test-isaac-px4-takeoff
test-isaac-px4-takeoff: test-integration-px4-takeoff

.PHONY: test-isaac-px4-takeoff-instability
test-isaac-px4-takeoff-instability: test-integration-px4-takeoff-instability

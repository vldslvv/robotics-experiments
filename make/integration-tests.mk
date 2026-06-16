########################################
# Container-backed integration tests
########################################

.PHONY: test-integration
test-integration: test-integration-isaac test-integration-px4-takeoff

INTEGRATION_TEST_TARGETS = \
    test-integration-isaac \
    test-integration-drone-articulation \
    test-integration-px4-takeoff \
    test-integration-px4-takeoff-instability

INTEGRATION_RUN_ROOT = $(INTEGRATION_ROOT)/$(1).$$(date -u +%Y%m%dT%H%M%SZ).$$$$

define RUN_ISAAC_INTEGRATION_TEST
@run_root="$(call INTEGRATION_RUN_ROOT,$(1))"; \
    ISAAC_RUN_ROOT="$$run_root/isaac"; \
    mkdir -p "$$ISAAC_RUN_ROOT"; \
    echo "Integration artifacts: $$run_root"; \
    docker run --rm \
        --network=host \
        $(ISAAC_INTEGRATION_DOCKER_RUNTIME) \
        $(ISAAC_DOCKER_ENV) \
        $(ISAAC_INTEGRATION_DOCKER_VOLUMES) \
        --entrypoint /bin/bash \
        $(ISAAC_IMAGE) \
        -lc 'set -o pipefail && \
            cd /isaac-sim && \
            rm -f "$(3)" && \
            { PIP_ROOT_USER_ACTION=ignore $(ISAAC_PYTHON) -m pip install \
                --quiet --disable-pip-version-check hydra-core omegaconf \
                >>"$(3)" 2>&1 || \
                { grep -E "$(ISAAC_TEST_ERROR_FILTER)" \
                    "$(3)"; exit 1; }; } && \
            $(ISAAC_PYTHON) $(2) --scene "$(SCENE)" \
                2>&1 | tee -a "$(3)" | $(TERMINAL_CONTROL_FILTER) | \
                    grep --line-buffered -E "$(ISAAC_TEST_LOG_FILTER)" && \
            ! grep -q "$(ISAAC_TEST_FAILURE_FILTER)" "$(3)"'
endef

define RUN_PX4_INTEGRATION_TEST
@run_root="$(call INTEGRATION_RUN_ROOT,$(1))"; \
    ISAAC_RUN_ROOT="$$run_root/isaac"; \
    PX4_RUN_ROOT="$$run_root/px4"; \
    mkdir -p "$$ISAAC_RUN_ROOT" "$$PX4_RUN_ROOT/logs" "$$PX4_RUN_ROOT/eeprom"; \
    echo "Integration artifacts: $$run_root"; \
    TEST_TARGET='$(1)' \
        HOST_ISAAC_LOG="$$ISAAC_RUN_ROOT/logs/$(4)" \
        HOST_PX4_LOG="$$PX4_RUN_ROOT/$(5)" \
        ISAAC_TEST_LOG='$(3)' \
        ISAAC_TEST_SCRIPT='$(2)' \
        ISAAC_RUN_ROOT="$$ISAAC_RUN_ROOT" \
        PX4_TEST_CONTAINER_NAME="$(PX4_TEST_CONTAINER_NAME).$(1).$$$$" \
        PX4_TEST_LOGS_DIR="$(ISAAC_PROJECT_DIR)/.build/docker/integration/$$(basename "$$run_root")/px4/logs" \
        PX4_EEPROM="$$PX4_RUN_ROOT/eeprom" \
        PX4_LOGS="$$PX4_RUN_ROOT/logs" \
        $(PX4_INTEGRATION_COMMON_ENV) \
        bash "$(PX4_INTEGRATION_RUNNER)"
endef

.PHONY: test-integration-all
test-integration-all:
	@mkdir -p "$(INTEGRATION_ROOT)"; \
	results_file="$(BUILD_ROOT)/integration-test-results.$$$$"; \
	: >"$$results_file"; \
	trap 'rm -f "$$results_file"' EXIT; \
	printf '%s\n' $(INTEGRATION_TEST_TARGETS) | \
		xargs -P "$(INTEGRATION_TEST_JOBS)" -I {} sh -c '\
			target="$$1"; \
			printf "\n==> %s\n" "$$target"; \
			if $(MAKE) --no-print-directory "$$target"; then \
				printf "%s PASS\n" "$$target" >>"$$2"; \
			else \
				printf "%s FAIL\n" "$$target" >>"$$2"; \
			fi' sh {} "$$results_file"; \
	printf '\nIntegration test summary\n'; \
	printf '%-45s %s\n' 'Target' 'Result'; \
	printf '%-45s %s\n' '------' '------'; \
	passed=0; \
	failed=0; \
	while read -r target result; do \
		printf '%-45s %s\n' "$$target" "$$result"; \
		if [ "$$result" = "PASS" ]; then \
			passed=$$((passed + 1)); \
		else \
			failed=$$((failed + 1)); \
		fi; \
	done <"$$results_file"; \
	printf '\nPassed: %s  Failed: %s\n' "$$passed" "$$failed"; \
	if [ "$$failed" -ne 0 ]; then exit 1; fi

.PHONY: test-integration-isaac
test-integration-isaac:
	$(call RUN_ISAAC_INTEGRATION_TEST,test-integration-isaac,$(ISAAC_SIM_TEST),$(ISAAC_SIM_TEST_LOG))

.PHONY: test-integration-drone-articulation
test-integration-drone-articulation:
	$(call RUN_ISAAC_INTEGRATION_TEST,test-integration-drone-articulation,$(ISAAC_DRONE_ARTICULATION_TEST),$(ISAAC_DRONE_ARTICULATION_TEST_LOG))

PX4_INTEGRATION_RUNNER = make/run_px4_integration_test.sh

PX4_INTEGRATION_COMMON_ENV = \
    ISAAC_DOCKER_RUNTIME='$(ISAAC_INTEGRATION_DOCKER_RUNTIME)' \
    ISAAC_DOCKER_ENV='$(ISAAC_DOCKER_ENV)' \
    ISAAC_DOCKER_VOLUMES='$(ISAAC_INTEGRATION_DOCKER_VOLUMES)' \
    ISAAC_IMAGE='$(ISAAC_IMAGE)' \
    ISAAC_PYTHON='$(ISAAC_PYTHON)' \
    ISAAC_TEST_ERROR_FILTER='$(ISAAC_TEST_ERROR_FILTER)' \
    ISAAC_TEST_FAILURE_FILTER='$(ISAAC_TEST_FAILURE_FILTER)' \
    ISAAC_TEST_LOG_FILTER='$(ISAAC_TEST_LOG_FILTER)' \
    INTEGRATION_LOG_TAIL_LINES='$(INTEGRATION_LOG_TAIL_LINES)' \
    INTEGRATION_VERBOSE='$(INTEGRATION_VERBOSE)' \
    PX4_IMAGE='$(PX4_IMAGE)' \
    PX4_ROOTFS='$(PX4_ROOTFS)' \
    PX4_TEST_READY_TIMEOUT_S='$(PX4_TEST_READY_TIMEOUT_S)' \
    SCENE='$(SCENE)'

.PHONY: test-integration-px4-takeoff
test-integration-px4-takeoff:
	$(call RUN_PX4_INTEGRATION_TEST,test-integration-px4-takeoff,$(ISAAC_PX4_TAKEOFF_TEST),$(ISAAC_PX4_TAKEOFF_TEST_LOG),test_px4_takeoff_stability.log,test_px4_takeoff_stability_px4.log)

.PHONY: test-integration-px4-takeoff-instability
test-integration-px4-takeoff-instability:
	$(call RUN_PX4_INTEGRATION_TEST,test-integration-px4-takeoff-instability,$(ISAAC_PX4_TAKEOFF_INSTABILITY_TEST),$(ISAAC_PX4_TAKEOFF_INSTABILITY_TEST_LOG),test_px4_takeoff_instability_detection.log,test_px4_takeoff_instability_px4.log)

.PHONY: test-isaac-simulation
test-isaac-simulation: test-integration-isaac

.PHONY: test-isaac-drone-articulation
test-isaac-drone-articulation: test-integration-drone-articulation

.PHONY: test-isaac-px4-takeoff
test-isaac-px4-takeoff: test-integration-px4-takeoff

.PHONY: test-isaac-px4-takeoff-instability
test-isaac-px4-takeoff-instability: test-integration-px4-takeoff-instability

########################################
# Container-backed integration tests
########################################

.PHONY: test-integration
test-integration: test-integration-isaac test-integration-px4

.PHONY: test-integration-isaac
test-integration-isaac:
	@mkdir -p $(ISAAC_RUNTIME_DIRS)
	@docker run --rm \
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
test-integration-drone-articulation:
	@mkdir -p $(ISAAC_RUNTIME_DIRS)
	@docker run --rm \
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

.PHONY: test-integration-px4
test-integration-px4:
	@mkdir -p $(ISAAC_RUNTIME_DIRS) "$(PX4_LOGS)" "$(PX4_EEPROM)"
	@bash -lc 'set -euo pipefail; \
		if docker inspect --format "{{.State.Running}}" "$(PX4_CONTAINER_NAME)" 2>/dev/null | grep -qx true; then \
			echo "PX4 Docker container $(PX4_CONTAINER_NAME) is already running; stop it before running this automated test."; \
			exit 1; \
		fi; \
		cleanup() { \
			status=$$?; \
			if docker logs "$(PX4_TEST_CONTAINER_NAME)" >"$(PX4_TEST_LOG)" 2>&1; then \
				echo "PX4 SITL log saved to $(PX4_TEST_LOG):"; \
				$(TERMINAL_CONTROL_FILTER) <"$(PX4_TEST_LOG)" || true; \
			fi; \
			docker rm -f "$(PX4_TEST_CONTAINER_NAME)" >/dev/null 2>&1 || true; \
			exit $$status; \
		}; \
		trap cleanup EXIT; \
		docker rm -f "$(PX4_TEST_CONTAINER_NAME)" >/dev/null 2>&1 || true; \
		echo "Starting temporary PX4 SITL container $(PX4_TEST_CONTAINER_NAME)..."; \
		docker run -d \
			--name "$(PX4_TEST_CONTAINER_NAME)" \
			--network=host \
			-v "$(PX4_LOGS):$(PX4_ROOTFS)/log" \
			-v "$(PX4_EEPROM):$(PX4_ROOTFS)/eeprom" \
			"$(PX4_IMAGE)" \
			/bin/bash -lc "cd /opt/PX4-Autopilot && exec make px4_sitl_default none_iris" \
			>/dev/null; \
		echo "Waiting for PX4 simulator TCP port 4560 readiness..."; \
		ready=0; \
		for _ in $$(seq 1 "$(PX4_TEST_READY_TIMEOUT_S)"); do \
			if docker logs "$(PX4_TEST_CONTAINER_NAME)" 2>&1 | grep -q "Waiting for simulator to accept connection on TCP port 4560"; then \
				ready=1; \
				break; \
			fi; \
			if ! docker inspect --format "{{.State.Running}}" "$(PX4_TEST_CONTAINER_NAME)" 2>/dev/null | grep -qx true; then \
				echo "PX4 test container exited before simulator port became ready."; \
				docker logs "$(PX4_TEST_CONTAINER_NAME)" 2>&1 | $(TERMINAL_CONTROL_FILTER) || true; \
				exit 1; \
			fi; \
			sleep 1; \
		done; \
		if [ "$$ready" -ne 1 ]; then \
			echo "Timed out waiting for PX4 simulator port readiness after $(PX4_TEST_READY_TIMEOUT_S)s."; \
			docker logs "$(PX4_TEST_CONTAINER_NAME)" 2>&1 | $(TERMINAL_CONTROL_FILTER) || true; \
			exit 1; \
		fi; \
		echo "PX4 simulator is waiting for Isaac on TCP port 4560."; \
		docker run --rm \
			$(ISAAC_DOCKER_RUNTIME) \
			$(ISAAC_DOCKER_ENV) \
			$(ISAAC_DOCKER_VOLUMES) \
			--entrypoint /bin/bash \
			$(ISAAC_IMAGE) \
			-lc '"'"'set -o pipefail && \
				cd /isaac-sim && \
				rm -f "$(ISAAC_PX4_BRIDGE_TEST_LOG)" && \
				{ PIP_ROOT_USER_ACTION=ignore $(ISAAC_PYTHON) -m pip install \
					--quiet --disable-pip-version-check hydra-core omegaconf pymavlink \
					>>"$(ISAAC_PX4_BRIDGE_TEST_LOG)" 2>&1 || \
					{ grep -E "$(ISAAC_TEST_ERROR_FILTER)" \
						"$(ISAAC_PX4_BRIDGE_TEST_LOG)"; exit 1; }; } && \
				$(ISAAC_PYTHON) $(ISAAC_PX4_BRIDGE_TEST) --scene "$(SCENE)" \
					2>&1 | tee -a "$(ISAAC_PX4_BRIDGE_TEST_LOG)" | $(TERMINAL_CONTROL_FILTER) | \
						grep --line-buffered -E "$(ISAAC_TEST_LOG_FILTER)" && \
				! grep -q "$(ISAAC_TEST_FAILURE_FILTER)" "$(ISAAC_PX4_BRIDGE_TEST_LOG)"'"'"''

.PHONY: test-isaac-simulation
test-isaac-simulation: test-integration-isaac

.PHONY: test-isaac-drone-articulation
test-isaac-drone-articulation: test-integration-drone-articulation

.PHONY: test-isaac-px4-bridge
test-isaac-px4-bridge: test-integration-px4

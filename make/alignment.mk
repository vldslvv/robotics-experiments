########################################
# Controller alignment
########################################

ALIGNMENT_ID ?= x500-px4-v1
ALIGNMENT_TARGET ?= configs/alignment/controllers/isaac_lab_x500_velocity.yaml
ALIGNMENT_REFERENCE ?= configs/alignment/controllers/px4_sitl_x500_velocity.yaml
FLIGHT_PARAMETERS_ARTIFACT ?= .build/flight-parameters/holybro-x500-v2-hover-v1/latest
ALIGNMENT_RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
ALIGNMENT_RUN_ID := $(ALIGNMENT_RUN_ID)
ALIGNMENT_RUN_DIR = .build/alignment/$(ALIGNMENT_ID)/$(ALIGNMENT_RUN_ID)
ALIGNMENT_RUN_DIR_CONTAINER = $(ISAAC_PROJECT_DIR)/$(ALIGNMENT_RUN_DIR)
ALIGNMENT_CAPACITY_BENCHMARK ?= 0
ALIGNMENT_TARGET_CONTAINER = $(ISAAC_PROJECT_DIR)/$(ALIGNMENT_TARGET)
ALIGNMENT_REFERENCE_CONTAINER = $(ISAAC_PROJECT_DIR)/$(ALIGNMENT_REFERENCE)
ALIGNMENT_SWEEP_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/alignment_sweep_target.py
ALIGNMENT_FIT_SCRIPT = isaac/runtime/standalone/alignment_fit.py
ALIGNMENT_AUDIT_SCRIPT = flight_analysis/tools/controller_state_replay.py
ALIGNMENT_PREPARE_SCRIPT = make/scripts/prepare_alignment_reference.py

.PHONY: alignment-prepare-reference
alignment-prepare-reference:
	@PYTHONPATH=src uv run python '$(ALIGNMENT_PREPARE_SCRIPT)' \
	    --flight-parameters '$(FLIGHT_PARAMETERS_ARTIFACT)' \
	    --definition '$(ALIGNMENT_REFERENCE)' \
	    --run-dir '$(ALIGNMENT_RUN_DIR)'

.PHONY: alignment-audit-actuators
alignment-audit-actuators:
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.json'
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.ulg'
	@test -f '$(ALIGNMENT_RUN_DIR)/bridge.log'
	@for set_name in fit validation acceptance; do \
	    directory='$(ALIGNMENT_RUN_DIR)'; \
	    if [ "$$set_name" != fit ]; then directory="$$directory/$$set_name"; fi; \
	    PYTHONPATH=src uv run --with pyulog python '$(ALIGNMENT_AUDIT_SCRIPT)' \
	        --config configs/alignment_px4_x500_velocity.yaml \
	        --audit-reference "$$directory/reference.json" \
	        --bridge-log "$$directory/bridge.log" --ulog "$$directory/reference.ulg" \
	        --json "$$directory/actuator_audit.json" || exit $$?; \
	done

.PHONY: alignment-sweep-target
alignment-sweep-target: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
alignment-sweep-target: HYDRA_CONFIG = configs/alignment_px4_x500_velocity.yaml
alignment-sweep-target: generate-drone-usd
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.json'
	@TEST_TARGET='alignment-sweep-target' \
	    ISAAC_TEST_LOG='/isaac-sim/.nvidia-omniverse/logs/alignment_sweep_target.log' \
	    ISAAC_TEST_SCRIPT='$(ALIGNMENT_SWEEP_SCRIPT)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

.PHONY: alignment-benchmark-capacity
alignment-benchmark-capacity: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
alignment-benchmark-capacity: HYDRA_CONFIG = configs/alignment_px4_x500_velocity.yaml
alignment-benchmark-capacity: ALIGNMENT_CAPACITY_BENCHMARK = 1
alignment-benchmark-capacity: generate-drone-usd
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.json'
	@TEST_TARGET='alignment-benchmark-capacity' \
	    ISAAC_TEST_LOG='/isaac-sim/.nvidia-omniverse/logs/alignment_benchmark_capacity.log' \
	    ISAAC_TEST_SCRIPT='$(ALIGNMENT_SWEEP_SCRIPT)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

.PHONY: alignment-fit
alignment-fit:
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.json'
	@test -f '$(ALIGNMENT_RUN_DIR)/target_candidates.json'
	@PYTHONPATH=src uv run python '$(ALIGNMENT_FIT_SCRIPT)' \
	    --target '$(ALIGNMENT_TARGET)' \
	    --reference '$(ALIGNMENT_REFERENCE)' \
	    --run-dir '$(ALIGNMENT_RUN_DIR)' \
	    --run-id '$(ALIGNMENT_RUN_ID)'

.PHONY: alignment-replay-controller
alignment-replay-controller:
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.json'
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.ulg'
	@test -f '$(ALIGNMENT_RUN_DIR)/controller.json'
	@for set_name in fit validation acceptance; do \
	    directory='$(ALIGNMENT_RUN_DIR)'; \
	    if [ "$$set_name" != fit ]; then directory="$$directory/$$set_name"; fi; \
	    PYTHONPATH=src uv run --with pyulog python '$(ALIGNMENT_AUDIT_SCRIPT)' \
	        --config configs/alignment_px4_x500_velocity.yaml \
	        --controller-artifact '$(ALIGNMENT_RUN_DIR)/controller.json' \
	        --ulog "$$directory/reference.ulg" --profile-reference "$$directory/reference.json" \
	        --bridge-log "$$directory/bridge.log" \
	        --json "$$directory/controller_boundary_replay.json" || exit $$?; \
	done

.PHONY: align-controllers
align-controllers:
	@$(MAKE) alignment-prepare-reference ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)' \
	    ALIGNMENT_TARGET='$(ALIGNMENT_TARGET)' ALIGNMENT_REFERENCE='$(ALIGNMENT_REFERENCE)'
	@$(MAKE) alignment-audit-actuators ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)'
	@$(MAKE) alignment-sweep-target ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)' \
	    ALIGNMENT_TARGET='$(ALIGNMENT_TARGET)' ALIGNMENT_REFERENCE='$(ALIGNMENT_REFERENCE)'
	@$(MAKE) alignment-fit ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)' \
	    ALIGNMENT_TARGET='$(ALIGNMENT_TARGET)' ALIGNMENT_REFERENCE='$(ALIGNMENT_REFERENCE)'
	@$(MAKE) alignment-replay-controller ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)'
	@$(MAKE) test-integration-alignment-target \
	    CONTROLLER_ARTIFACT='$(ALIGNMENT_RUN_DIR)/controller.json' \
	    ISAAC_TEST_EXTRA_ARGS='--alignment-report $(ALIGNMENT_RUN_DIR_CONTAINER)/integration.json'
	@PYTHONPATH=src uv run python '$(ALIGNMENT_FIT_SCRIPT)' \
	    --run-dir '$(ALIGNMENT_RUN_DIR)' --finalize

.PHONY: test-integration-alignment-target
test-integration-alignment-target: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-alignment-target: CONTROLLER_ARTIFACT = .build/alignment/$(ALIGNMENT_ID)/best
test-integration-alignment-target: generate-drone-usd
	@TEST_TARGET='test-integration-alignment-target' \
	    ISAAC_TEST_LOG='$(ISAAC_PRESS_HOVER_GOTO_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PRESS_HOVER_GOTO_TEST)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

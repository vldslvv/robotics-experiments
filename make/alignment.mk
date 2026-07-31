########################################
# Controller alignment
########################################

ALIGNMENT_ID ?= x500-velocity-px4-v5
ALIGNMENT_TARGET ?= configs/alignment/controllers/isaac_lab_x500_velocity.yaml
ALIGNMENT_REFERENCE ?= configs/alignment/controllers/px4_sitl_x500_velocity.yaml
ALIGNMENT_RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
ALIGNMENT_RUN_ID := $(ALIGNMENT_RUN_ID)
ALIGNMENT_RUN_DIR = .build/alignment/$(ALIGNMENT_ID)/$(ALIGNMENT_RUN_ID)
ALIGNMENT_RUN_DIR_CONTAINER = $(ISAAC_PROJECT_DIR)/$(ALIGNMENT_RUN_DIR)
ALIGNMENT_TARGET_CONTAINER = $(ISAAC_PROJECT_DIR)/$(ALIGNMENT_TARGET)
ALIGNMENT_REFERENCE_CONTAINER = $(ISAAC_PROJECT_DIR)/$(ALIGNMENT_REFERENCE)
ALIGNMENT_CAPTURE_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/alignment_capture_reference.py
ALIGNMENT_SWEEP_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/alignment_sweep_target.py
ALIGNMENT_FIT_SCRIPT = isaac/runtime/standalone/alignment_fit.py
ALIGNMENT_AUDIT_SCRIPT = flight_analysis/tools/controller_state_replay.py

.PHONY: alignment-capture-reference
alignment-capture-reference: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
alignment-capture-reference: HYDRA_CONFIG = configs/alignment_px4_x500_velocity.yaml
alignment-capture-reference: SCENE = @config
alignment-capture-reference: generate-drone-usd
	@mkdir -p '$(ALIGNMENT_RUN_DIR)'
	@TEST_TARGET='alignment-capture-reference' \
	    ALIGNMENT_CAPTURE_RUN_DIR='$(ALIGNMENT_RUN_DIR)' \
	    HOST_ISAAC_LOG_BASENAME='alignment_capture_reference.log' \
	    HOST_PX4_LOG_BASENAME='alignment_capture_reference_px4.log' \
	    ISAAC_TEST_LOG='/isaac-sim/.nvidia-omniverse/logs/alignment_capture_reference.log' \
	    ISAAC_TEST_SCRIPT='$(ALIGNMENT_CAPTURE_SCRIPT)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).alignment-capture-reference' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"
	@PYTHONPATH=src uv run --with pyulog python '$(ALIGNMENT_AUDIT_SCRIPT)' \
	    --config configs/alignment_px4_x500_velocity.yaml \
	    --ulog '$(ALIGNMENT_RUN_DIR)/reference.ulg' \
	    --augment-reference '$(ALIGNMENT_RUN_DIR)/reference.json'

.PHONY: alignment-audit-actuators
alignment-audit-actuators:
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.json'
	@test -f '$(ALIGNMENT_RUN_DIR)/reference.ulg'
	@test -f '$(ALIGNMENT_RUN_DIR)/bridge.log'
	@PYTHONPATH=src uv run --with pyulog python '$(ALIGNMENT_AUDIT_SCRIPT)' \
	    --config configs/alignment_px4_x500_velocity.yaml \
	    --audit-reference '$(ALIGNMENT_RUN_DIR)/reference.json' \
	    --bridge-log '$(ALIGNMENT_RUN_DIR)/bridge.log' \
	    --ulog '$(ALIGNMENT_RUN_DIR)/reference.ulg' \
	    --json '$(ALIGNMENT_RUN_DIR)/actuator_audit.json'

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
	@test -f '$(ALIGNMENT_RUN_DIR)/alignment.json'
	@PYTHONPATH=src uv run --with pyulog python '$(ALIGNMENT_AUDIT_SCRIPT)' \
	    --config configs/alignment_px4_x500_velocity.yaml \
	    --ulog '$(ALIGNMENT_RUN_DIR)/reference.ulg' \
	    --alignment-artifact '$(ALIGNMENT_RUN_DIR)/alignment.json' \
	    --profile-reference '$(ALIGNMENT_RUN_DIR)/reference.json' \
	    --json '$(ALIGNMENT_RUN_DIR)/controller_boundary_replay.json'

.PHONY: align-controllers
align-controllers:
	@$(MAKE) alignment-capture-reference ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)' \
	    ALIGNMENT_TARGET='$(ALIGNMENT_TARGET)' ALIGNMENT_REFERENCE='$(ALIGNMENT_REFERENCE)'
	@$(MAKE) alignment-audit-actuators ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)'
	@$(MAKE) alignment-sweep-target ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)' \
	    ALIGNMENT_TARGET='$(ALIGNMENT_TARGET)' ALIGNMENT_REFERENCE='$(ALIGNMENT_REFERENCE)'
	@$(MAKE) alignment-fit ALIGNMENT_RUN_ID='$(ALIGNMENT_RUN_ID)' \
	    ALIGNMENT_TARGET='$(ALIGNMENT_TARGET)' ALIGNMENT_REFERENCE='$(ALIGNMENT_REFERENCE)'

.PHONY: test-integration-alignment-target
test-integration-alignment-target: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-alignment-target: ALIGNMENT_ARTIFACT = .build/alignment/$(ALIGNMENT_ID)/best
test-integration-alignment-target: generate-drone-usd
	@TEST_TARGET='test-integration-alignment-target' \
	    ISAAC_TEST_LOG='$(ISAAC_PRESS_HOVER_GOTO_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PRESS_HOVER_GOTO_TEST)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

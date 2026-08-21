########################################
# Task-neutral flight parameters
########################################

FLIGHT_CONFIG ?= configs/flight/holybro_x500_v2_hover.yaml
FLIGHT_PARAMETERS_OUTPUT_ROOT ?= .build/flight-parameters
FLIGHT_PARAMETERS_RUN_ID ?=
FLIGHT_PARAMETERS_TRIAL_INDEX ?= 1
FLIGHT_PARAMETERS_TRIAL_DIR ?= $(FLIGHT_PARAMETERS_OUTPUT_ROOT)/sample
FLIGHT_PARAMETERS_CONFIG_HOST = $(abspath $(FLIGHT_CONFIG))
FLIGHT_PARAMETERS_CONFIG_CONTAINER = $(patsubst $(CURDIR)%,$(ISAAC_PROJECT_DIR)%,$(FLIGHT_PARAMETERS_CONFIG_HOST))
FLIGHT_PARAMETERS_TRIAL_DIR_HOST = $(abspath $(FLIGHT_PARAMETERS_TRIAL_DIR))
FLIGHT_PARAMETERS_TRIAL_DIR_CONTAINER = $(patsubst $(CURDIR)%,$(ISAAC_PROJECT_DIR)%,$(FLIGHT_PARAMETERS_TRIAL_DIR_HOST))
FLIGHT_PARAMETERS_CAPTURE_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/flight_parameters_capture.py
FLIGHT_PARAMETERS_RESPONSE_DEFINITION ?= configs/alignment/controllers/px4_sitl_x500_velocity.yaml
FLIGHT_PARAMETERS_RESPONSE_DIR ?= $(FLIGHT_PARAMETERS_OUTPUT_ROOT)/response-sample
FLIGHT_PARAMETERS_RESPONSE_DIR_HOST = $(abspath $(FLIGHT_PARAMETERS_RESPONSE_DIR))
FLIGHT_PARAMETERS_RESPONSE_DIR_CONTAINER = $(patsubst $(CURDIR)%,$(ISAAC_PROJECT_DIR)%,$(FLIGHT_PARAMETERS_RESPONSE_DIR_HOST))
FLIGHT_PARAMETERS_RESPONSE_SCRIPT = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/flight_parameters_capture_response.py

.PHONY: test-integration-flight-parameters-sample
test-integration-flight-parameters-sample: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-flight-parameters-sample: SCENE = @config
test-integration-flight-parameters-sample: generate-drone-usd
	@mkdir -p '$(FLIGHT_PARAMETERS_TRIAL_DIR_HOST)'
	@TEST_TARGET='test-integration-flight-parameters-sample' \
	    HOST_ISAAC_LOG_BASENAME='flight_parameters_capture.log' \
	    HOST_PX4_LOG_BASENAME='flight_parameters_capture_px4.log' \
	    ISAAC_TEST_LOG='/isaac-sim/.nvidia-omniverse/logs/flight_parameters_capture.log' \
	    ISAAC_TEST_SCRIPT='$(FLIGHT_PARAMETERS_CAPTURE_SCRIPT)' \
	    FLIGHT_PARAMETERS_CONFIG='$(FLIGHT_PARAMETERS_CONFIG_CONTAINER)' \
	    FLIGHT_PARAMETERS_RUN_ID='$(FLIGHT_PARAMETERS_RUN_ID)' \
	    FLIGHT_PARAMETERS_TRIAL_INDEX='$(FLIGHT_PARAMETERS_TRIAL_INDEX)' \
	    FLIGHT_PARAMETERS_TRIAL_DIR='$(FLIGHT_PARAMETERS_TRIAL_DIR_HOST)' \
	    FLIGHT_PARAMETERS_TRIAL_DIR_CONTAINER='$(FLIGHT_PARAMETERS_TRIAL_DIR_CONTAINER)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).flight-parameters' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-integration-flight-parameters-response-sample
test-integration-flight-parameters-response-sample: ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)
test-integration-flight-parameters-response-sample: SCENE = @config
test-integration-flight-parameters-response-sample: generate-drone-usd
	@mkdir -p '$(FLIGHT_PARAMETERS_RESPONSE_DIR_HOST)'
	@TEST_TARGET='test-integration-flight-parameters-response-sample' \
	    HOST_ISAAC_LOG_BASENAME='flight_parameters_capture_response.log' \
	    HOST_PX4_LOG_BASENAME='flight_parameters_capture_response_px4.log' \
	    ISAAC_TEST_LOG='/isaac-sim/.nvidia-omniverse/logs/flight_parameters_capture_response.log' \
	    ISAAC_TEST_SCRIPT='$(FLIGHT_PARAMETERS_RESPONSE_SCRIPT)' \
	    FLIGHT_PARAMETERS_RESPONSE_DEFINITION='$(FLIGHT_PARAMETERS_RESPONSE_DEFINITION)' \
	    FLIGHT_PARAMETERS_RESPONSE_DIR='$(FLIGHT_PARAMETERS_RESPONSE_DIR_HOST)' \
	    FLIGHT_PARAMETERS_RESPONSE_DIR_CONTAINER='$(FLIGHT_PARAMETERS_RESPONSE_DIR_CONTAINER)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).flight-parameters-response' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: measure-flight-parameters
measure-flight-parameters:
	@PYTHONPATH=src uv run --with pyulog python make/scripts/run_flight_parameters_campaign.py \
	    --config '$(FLIGHT_CONFIG)' \
	    --output-root '$(FLIGHT_PARAMETERS_OUTPUT_ROOT)' \
	    $(if $(strip $(FLIGHT_PARAMETERS_RUN_ID)),--run-id '$(FLIGHT_PARAMETERS_RUN_ID)')

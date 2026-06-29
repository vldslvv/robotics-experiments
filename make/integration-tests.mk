########################################
# Container-backed integration tests
########################################

.PHONY: test-integration
test-integration: test-integration-isaac test-integration-px4-takeoff

INTEGRATION_TEST_TARGETS = \
    test-integration-isaac \
    test-integration-drone-articulation \
    test-integration-px4-takeoff \
    test-integration-px4-navigation \
    test-integration-px4-takeoff-instability \
    test-integration-px4-multi-drone-takeoff

ISAAC_CACHE_WARMUP_RUNNER = make/scripts/warm_isaac_cache.sh
ISAAC_INTEGRATION_RUNNER = make/scripts/run_isaac_integration_test.sh
INTEGRATION_SUITE_RUNNER = make/scripts/run_integration_suite.sh

ISAAC_HOST_CACHE_DIRS = $(ISAAC_KIT_CACHE):$(ISAAC_OV_CACHE):$(ISAAC_PIP_CACHE):$(ISAAC_GLCACHE):$(ISAAC_COMPUTEC):$(ISAAC_LOGS):$(ISAAC_DATA):$(ISAAC_DOCS)

ISAAC_INTEGRATION_COMMON_ENV = \
    INTEGRATION_ROOT='$(INTEGRATION_ROOT)' \
    ISAAC_DOCKER_RUNTIME='$(ISAAC_INTEGRATION_DOCKER_RUNTIME)' \
    ISAAC_DOCKER_ENV='$(ISAAC_DOCKER_ENV)' \
    ISAAC_DOCKER_VOLUMES='$(ISAAC_INTEGRATION_DOCKER_VOLUMES)' \
    ISAAC_HOST_CACHE_DIRS='$(ISAAC_HOST_CACHE_DIRS)' \
    ISAAC_IMAGE='$(ISAAC_IMAGE)' \
    ISAAC_PROJECT_DIR='$(ISAAC_PROJECT_DIR)' \
    ISAAC_PYTHON='$(ISAAC_PYTHON)' \
    ISAAC_TEST_ERROR_FILTER='$(ISAAC_TEST_ERROR_FILTER)' \
    ISAAC_TEST_FAILURE_FILTER='$(ISAAC_TEST_FAILURE_FILTER)' \
    ISAAC_TEST_LOG_FILTER='$(ISAAC_TEST_LOG_FILTER)' \
    SCENE='$(SCENE)'

.PHONY: test-integration-warm-isaac-cache
test-integration-warm-isaac-cache:
	@TEST_TARGET='test-integration-warm-isaac-cache' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_CACHE_WARMUP_RUNNER)"

.PHONY: test-integration-all
test-integration-all: test-integration-warm-isaac-cache
	@BUILD_ROOT='$(BUILD_ROOT)' \
	    INTEGRATION_ROOT='$(INTEGRATION_ROOT)' \
	    INTEGRATION_TEST_JOBS='$(INTEGRATION_TEST_JOBS)' \
	    INTEGRATION_TEST_TARGETS='$(INTEGRATION_TEST_TARGETS)' \
	    MAKE_COMMAND='make' \
	    bash "$(INTEGRATION_SUITE_RUNNER)"

.PHONY: test-integration-isaac
test-integration-isaac:
	@TEST_TARGET='test-integration-isaac' \
	    ISAAC_TEST_LOG='$(ISAAC_SIM_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_SIM_TEST)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

.PHONY: test-integration-drone-articulation
test-integration-drone-articulation:
	@TEST_TARGET='test-integration-drone-articulation' \
	    ISAAC_TEST_LOG='$(ISAAC_DRONE_ARTICULATION_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_DRONE_ARTICULATION_TEST)' \
	    $(ISAAC_INTEGRATION_COMMON_ENV) \
	    bash "$(ISAAC_INTEGRATION_RUNNER)"

PX4_INTEGRATION_RUNNER = make/scripts/run_px4_integration_test.sh

PX4_INTEGRATION_COMMON_ENV = \
    $(ISAAC_INTEGRATION_COMMON_ENV) \
    INTEGRATION_LOG_TAIL_LINES='$(INTEGRATION_LOG_TAIL_LINES)' \
    INTEGRATION_VERBOSE='$(INTEGRATION_VERBOSE)' \
    PX4_IMAGE='$(PX4_IMAGE)' \
    PX4_ROOTFS='$(PX4_ROOTFS)' \
    PX4_TEST_READY_TIMEOUT_S='$(PX4_TEST_READY_TIMEOUT_S)' \
    ROBOTICS_EXPERIMENT_CONFIG_PATH='$(HYDRA_CONFIG_HOST_PATH)'

.PHONY: test-integration-px4-takeoff
test-integration-px4-takeoff: HYDRA_CONFIG = configs/integration_test_takeoff.yaml
test-integration-px4-takeoff:
	@TEST_TARGET='test-integration-px4-takeoff' \
	    HOST_ISAAC_LOG_BASENAME='test_px4_takeoff_stability.log' \
	    HOST_PX4_LOG_BASENAME='test_px4_takeoff_stability_px4.log' \
	    ISAAC_TEST_LOG='$(ISAAC_PX4_TAKEOFF_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PX4_TAKEOFF_TEST)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).test-integration-px4-takeoff' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-integration-px4-navigation
test-integration-px4-navigation: HYDRA_CONFIG = configs/runtime_drone_navigate.yaml
test-integration-px4-navigation: SCENE = @config
test-integration-px4-navigation:
	@TEST_TARGET='test-integration-px4-navigation' \
	    HOST_ISAAC_LOG_BASENAME='test_px4_navigation_routine.log' \
	    HOST_PX4_LOG_BASENAME='test_px4_navigation_routine_px4.log' \
	    ISAAC_TEST_LOG='$(ISAAC_PX4_NAVIGATION_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PX4_NAVIGATION_TEST)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).test-integration-px4-navigation' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-integration-px4-takeoff-instability
test-integration-px4-takeoff-instability: HYDRA_CONFIG = configs/integration_test_takeoff.yaml
test-integration-px4-takeoff-instability:
	@TEST_TARGET='test-integration-px4-takeoff-instability' \
	    HOST_ISAAC_LOG_BASENAME='test_px4_takeoff_instability_detection.log' \
	    HOST_PX4_LOG_BASENAME='test_px4_takeoff_instability_px4.log' \
	    ISAAC_TEST_LOG='$(ISAAC_PX4_TAKEOFF_INSTABILITY_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PX4_TAKEOFF_INSTABILITY_TEST)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).test-integration-px4-takeoff-instability' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

# Multi-drone takeoff. Starts one PX4 SITL instance per PX4-enabled vehicle in
# the integration_test_multi_drone_takeoff experiment, uses its scene config via
# the @config sentinel, and requires every drone to hold its takeoff/hover
# envelope.
.PHONY: test-integration-px4-multi-drone-takeoff
test-integration-px4-multi-drone-takeoff: HYDRA_CONFIG = configs/integration_test_multi_drone_takeoff.yaml
test-integration-px4-multi-drone-takeoff: SCENE = @config
test-integration-px4-multi-drone-takeoff:
	@TEST_TARGET='test-integration-px4-multi-drone-takeoff' \
	    HOST_ISAAC_LOG_BASENAME='test_px4_multi_drone_takeoff.log' \
	    HOST_PX4_LOG_BASENAME='test_px4_multi_drone_takeoff_px4.log' \
	    ISAAC_TEST_LOG='$(ISAAC_PX4_MULTI_DRONE_TEST_LOG)' \
	    ISAAC_TEST_SCRIPT='$(ISAAC_PX4_MULTI_DRONE_TEST)' \
	    PX4_TEST_CONTAINER_NAME='$(PX4_TEST_CONTAINER_NAME).test-integration-px4-multi-drone-takeoff' \
	    STRICT='1' \
	    $(PX4_INTEGRATION_COMMON_ENV) \
	    bash "$(PX4_INTEGRATION_RUNNER)"

.PHONY: test-isaac-simulation
test-isaac-simulation: test-integration-isaac

.PHONY: test-isaac-drone-articulation
test-isaac-drone-articulation: test-integration-drone-articulation

.PHONY: test-isaac-px4-takeoff
test-isaac-px4-takeoff: test-integration-px4-takeoff

.PHONY: test-isaac-px4-takeoff-instability
test-isaac-px4-takeoff-instability: test-integration-px4-takeoff-instability

.PHONY: test-isaac-px4-multi-drone-takeoff
test-isaac-px4-multi-drone-takeoff: test-integration-px4-multi-drone-takeoff

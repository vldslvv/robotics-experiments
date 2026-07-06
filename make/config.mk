########################################
# Shared configuration
########################################

BUILD_ROOT             = $(CURDIR)/.build/docker
SRC_DIR                = $(CURDIR)/src
VENV_DIR               = $(CURDIR)/.venv
VENV_ACTIVATE          = $(VENV_DIR)/bin/activate

# Isaac custom image
ISAAC_BASE_IMAGE ?= nvcr.io/nvidia/isaac-sim
ISAAC_BASE_TAG   ?= 5.1.0

ISAAC_IMAGE_NAME     ?= vv-isaac-sim-custom
ISAAC_IMAGE_TAG      ?= $(ISAAC_BASE_TAG)
ISAAC_IMAGE          = $(ISAAC_IMAGE_NAME):$(ISAAC_IMAGE_TAG)
ISAAC_CACHE_IMAGE    = $(ISAAC_IMAGE_NAME):$(ISAAC_IMAGE_TAG)-buildcache
ISAAC_CONTAINER_NAME ?= vv-isaac-sim-custom
ISAAC_PROJECT_DIR    = /isaac-sim/proj

# Isaac Lab image (Dockerfile `isaac-lab` stage, FROM isaac-sim). Only the Phase F
# RL workflow uses it (the test-integration-press-hover-goto gate; future F2/F3
# training). ISAACLAB_REF pins the Isaac Lab release: v2.3.2 is built on Isaac
# Sim 5.1. Keep it in sync with the Dockerfile ARG default.
ISAACLAB_REF         ?= v2.3.2
ISAAC_LAB_IMAGE_NAME ?= vv-isaac-lab-custom
ISAAC_LAB_IMAGE      = $(ISAAC_LAB_IMAGE_NAME):$(ISAAC_IMAGE_TAG)
ISAAC_LAB_CACHE_IMAGE = $(ISAAC_LAB_IMAGE_NAME):$(ISAAC_IMAGE_TAG)-buildcache

# Auto-launch of the custom Isaac runtime (runapp_custom.sh) on container start.
# ISAAC_AUTORUN=0 (default) keeps the manual two-pane shell; ISAAC_AUTORUN=1
# overrides the container tmux CMD so the first pane runs the runtime then drops
# to a login shell (the second pane stays a plain shell). The `run-all-autorun`
# target is the convenience entry that sets ISAAC_AUTORUN=1. The selected command
# is passed to run_persistent_container.sh as EXPECTED_CMD and is also used to
# recreate the persistent container when the desired startup command changes (so
# toggling the flag takes effect).
# The CMD strings are the container's tmux session description (passed to the
# image ENTRYPOINT `tmux "$@"`). They are eval-parsed by the script, so the tmux
# command separators are escaped (`\;`) to survive eval as literal arguments.
ISAAC_AUTORUN       ?= 0
ISAAC_RUNAPP         = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/runapp_custom.sh
ISAAC_MANUAL_CMD     = new-session /usr/bin/zsh -l \; split-window -h /usr/bin/zsh -l \; attach
ISAAC_AUTORUN_CMD    = new-session /usr/bin/zsh -lc '$(ISAAC_RUNAPP); exec /usr/bin/zsh -l' \; split-window -h /usr/bin/zsh -l \; attach
ISAAC_CONTAINER_CMD  = $(if $(filter 1,$(ISAAC_AUTORUN)),$(ISAAC_AUTORUN_CMD),$(ISAAC_MANUAL_CMD))
HYDRA_CONFIG_HOST_PATH      = $(abspath $(HYDRA_CONFIG))
HYDRA_CONFIG_CONTAINER_PATH = $(if $(filter $(CURDIR)/%,$(HYDRA_CONFIG_HOST_PATH)),$(patsubst $(CURDIR)%,$(ISAAC_PROJECT_DIR)%,$(HYDRA_CONFIG_HOST_PATH)),$(HYDRA_CONFIG_HOST_PATH))
ISAAC_PYTHON         = /isaac-sim/python.sh
ISAAC_SIM_TEST       = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_sim_settings.py
ISAAC_SIM_TEST_LOG   = /isaac-sim/.nvidia-omniverse/logs/test_simulation_settings.log
ISAAC_DRONE_ARTICULATION_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_drone_articulation_io.py
ISAAC_DRONE_ARTICULATION_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_drone_articulation_io.log
ISAAC_PX4_TAKEOFF_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_px4_takeoff_stability.py
ISAAC_PX4_TAKEOFF_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_px4_takeoff_stability.log
ISAAC_PX4_TAKEOFF_INSTABILITY_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_px4_takeoff_instability_detection.py
ISAAC_PX4_TAKEOFF_INSTABILITY_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_px4_takeoff_instability_detection.log
ISAAC_PX4_MULTI_DRONE_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_px4_multi_drone_takeoff.py
ISAAC_PX4_MULTI_DRONE_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_px4_multi_drone_takeoff.log
ISAAC_PX4_NAVIGATION_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_px4_navigation_routine.py
ISAAC_PX4_NAVIGATION_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_px4_navigation_routine.log
ISAAC_PRESS_HOVER_GOTO_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_press_hover_goto.py
ISAAC_PRESS_HOVER_GOTO_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_press_hover_goto.log
ISAAC_PRESS_POLICY_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_press_policy_hover.log
ISAAC_TEST_ERROR_FILTER = ERROR|Error|Exception|Traceback|Could not|No matching|Failed|failed
ISAAC_TEST_LOG_FILTER = ^\[isaac-standalone-test\]|^Traceback|^[[:space:]]+File |^[A-Za-z_]+(Error|Exception):
ISAAC_TEST_FAILURE_FILTER = ^\[isaac-standalone-test\] ERROR:
TERMINAL_CONTROL_FILTER = sed -u -E "s|\x1B\[[0-9;?]*[ -/]*[@-~]||g; s|\x1B[=>c]||g"
PX4_PROMPT_FILTER = sed -u -E "s|\r|\n|g" | sed -u -E "s|^pxh>[[:space:]]*||; /^$$/d"
INTEGRATION_VERBOSE ?= 0
INTEGRATION_LOG_TAIL_LINES ?= 120
INTEGRATION_ROOT ?= $(BUILD_ROOT)/integration
INTEGRATION_TEST_JOBS ?= 1
SCENE                ?= @isaac/assets/TestScenes/MinimalIntegration.usda
HYDRA_CONFIG         ?= configs/takeoff_quad.yaml

# Host paths for persistence
ISAAC_HOME             = $(CURDIR)/isaac
ISAAC_KIT_CACHE        = $(ISAAC_HOME)/isaac-sim/cache/kit
ISAAC_OV_CACHE         = $(ISAAC_HOME)/isaac-sim/cache/ov
ISAAC_PIP_CACHE        = $(ISAAC_HOME)/isaac-sim/cache/pip
ISAAC_GLCACHE          = $(ISAAC_HOME)/isaac-sim/cache/glcache
ISAAC_COMPUTEC         = $(ISAAC_HOME)/isaac-sim/cache/computecache
ISAAC_LOGS             = $(ISAAC_HOME)/isaac-sim/logs
ISAAC_DATA             = $(ISAAC_HOME)/isaac-sim/data
ISAAC_DOCS             = $(ISAAC_HOME)/isaac-sim/documents
ISAAC_ZSH_CACHE        = $(ISAAC_HOME)/isaac-sim/cache/zsh

# These paths have to be relative to be inside docker build context.
ISAAC_ASSETS_DIR       = isaac/assets
ISAAC_DOCKER_DIR       = isaac/docker

ISAAC_RUNTIME_DIRS = \
    "$(ISAAC_KIT_CACHE)" "$(ISAAC_OV_CACHE)" "$(ISAAC_PIP_CACHE)" \
    "$(ISAAC_GLCACHE)" "$(ISAAC_COMPUTEC)" "$(ISAAC_LOGS)" \
    "$(ISAAC_DATA)" "$(ISAAC_DOCS)"

# ROS 2 image / container (humble is the other option)
ROS_DISTRO         ?= jazzy
ROS_BASE_IMAGE     ?= ros
ROS_BASE_TAG       ?= $(ROS_DISTRO)
ROS_IMAGE_NAME     ?= vv-ros-$(ROS_DISTRO)-custom
ROS_IMAGE_TAG      ?= $(ROS_DISTRO)
ROS_IMAGE          = $(ROS_IMAGE_NAME):$(ROS_IMAGE_TAG)
ROS_CACHE_IMAGE    = $(ROS_IMAGE_NAME):$(ROS_IMAGE_TAG)-buildcache
ROS_CONTAINER_NAME ?= vv-ros-$(ROS_DISTRO)-custom

# These paths have to be relative to be inside docker build context.
ROS_WS_DIR         = ros/ros2_ws
ROS_CONFIG_DIR     = ros/config

# This dir is mounted at runtime.
ROS_ZSH_CACHE      = $(CURDIR)/ros/cache/zsh

# ROS 2 middleware config
ROS_DOMAIN_ID                  ?= 42
RMW_IMPLEMENTATION             ?= rmw_fastrtps_cpp
FASTRTPS_DEFAULT_PROFILES_FILE ?= /etc/ros/fastdds.xml

# PX4 SITL image / container (stands in for the flight controller)
PX4_TAG            ?= v1.14.3
PX4_IMAGE_NAME     ?= vv-px4-sitl
PX4_IMAGE_TAG      ?= $(PX4_TAG)
PX4_IMAGE          = $(PX4_IMAGE_NAME):$(PX4_IMAGE_TAG)
PX4_CACHE_IMAGE    = $(PX4_IMAGE_NAME):$(PX4_IMAGE_TAG)-buildcache
PX4_CONTAINER_NAME ?= vv-px4-sitl
PX4_TEST_CONTAINER_NAME ?= vv-px4-sitl-test
PX4_TAKEOFF_TEST_LOG = $(PX4_HOME)/logs/test_px4_takeoff_stability_px4.log
PX4_TAKEOFF_INSTABILITY_TEST_LOG = $(PX4_HOME)/logs/test_px4_takeoff_instability_px4.log
PX4_NAVIGATION_TEST_LOG = $(PX4_HOME)/logs/test_px4_navigation_routine_px4.log
PX4_TEST_READY_TIMEOUT_S ?= 60

# Host-side persistence for PX4 SITL (ULog flight logs + simulated EEPROM params).
PX4_HOME           = $(CURDIR)/px4
PX4_LOGS           = $(PX4_HOME)/logs
PX4_EEPROM         = $(PX4_HOME)/eeprom

# Paths inside the container (under PX4 SITL rootfs).
PX4_ROOTFS         = /opt/PX4-Autopilot/build/px4_sitl_default/rootfs

# Make neovim version same a host's.
NVIM_VERSION := $(shell nvim --version | awk 'NR==1 {print $$2}')

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
ISAAC_CONTAINER_NAME ?= vv-isaac-sim-custom
ISAAC_PROJECT_DIR    = /isaac-sim/proj
ISAAC_PYTHON         = /isaac-sim/python.sh
ISAAC_SIM_TEST       = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_sim_settings.py
ISAAC_SIM_TEST_LOG   = /isaac-sim/.nvidia-omniverse/logs/test_simulation_settings.log
ISAAC_DRONE_ARTICULATION_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_drone_articulation_io.py
ISAAC_DRONE_ARTICULATION_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_drone_articulation_io.log
ISAAC_PX4_BRIDGE_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_px4_bridge_host.py
ISAAC_PX4_BRIDGE_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_px4_bridge_host.log
ISAAC_PX4_TAKEOFF_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_px4_takeoff_stability.py
ISAAC_PX4_TAKEOFF_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_px4_takeoff_stability.log
ISAAC_PX4_TAKEOFF_INSTABILITY_TEST     = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_px4_takeoff_instability_detection.py
ISAAC_PX4_TAKEOFF_INSTABILITY_TEST_LOG = /isaac-sim/.nvidia-omniverse/logs/test_px4_takeoff_instability_detection.log
ISAAC_TEST_ERROR_FILTER = ERROR|Error|Exception|Traceback|Could not|No matching|Failed|failed
ISAAC_TEST_LOG_FILTER = ^\[isaac-standalone-test\]|^Traceback|^[[:space:]]+File |^[A-Za-z_]+(Error|Exception):
ISAAC_TEST_FAILURE_FILTER = ^\[isaac-standalone-test\] ERROR:
TERMINAL_CONTROL_FILTER = sed -u -E "s|\x1B\[[0-9;?]*[ -/]*[@-~]||g; s|\x1B[=>c]||g"
PX4_PROMPT_FILTER = sed -u -E "s|\r|\n|g" | sed -u -E "s|^pxh>[[:space:]]*||; /^$$/d"
INTEGRATION_VERBOSE ?= 0
INTEGRATION_LOG_TAIL_LINES ?= 120
SCENE                ?= @isaac/assets/LightSwitchWarehouse/Warehouse.usd

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
PX4_CONTAINER_NAME ?= vv-px4-sitl
PX4_TEST_CONTAINER_NAME ?= vv-px4-sitl-test
PX4_TEST_LOG       = $(PX4_HOME)/logs/test_px4_bridge_host_px4.log
PX4_TAKEOFF_TEST_LOG = $(PX4_HOME)/logs/test_px4_takeoff_stability_px4.log
PX4_TAKEOFF_INSTABILITY_TEST_LOG = $(PX4_HOME)/logs/test_px4_takeoff_instability_px4.log
PX4_TEST_READY_TIMEOUT_S ?= 60

# Host-side persistence for PX4 SITL (ULog flight logs + simulated EEPROM params).
PX4_HOME           = $(CURDIR)/px4
PX4_LOGS           = $(PX4_HOME)/logs
PX4_EEPROM         = $(PX4_HOME)/eeprom

# Paths inside the container (under PX4 SITL rootfs).
PX4_ROOTFS         = /opt/PX4-Autopilot/build/px4_sitl_default/rootfs

# Make neovim version same a host's.
NVIM_VERSION := $(shell nvim --version | awk 'NR==1 {print $$2}')

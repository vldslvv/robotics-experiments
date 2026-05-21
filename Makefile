########################################
# Images / containers
########################################

BUILD_ROOT             = $(CURDIR)/.build/docker
SRC_DIR                = $(CURDIR)/src
VENV_DIR               = $(CURDIR)/.venv
VENV_ACTIVATE          = $(VENV_DIR)/bin/activate

# Isaac custom image
ISAAC_BASE_IMAGE = nvcr.io/nvidia/isaac-sim
ISAAC_BASE_TAG   = 5.1.0

ISAAC_IMAGE_NAME     = vv-isaac-sim-custom
ISAAC_IMAGE_TAG      = $(ISAAC_BASE_TAG)
ISAAC_IMAGE          = $(ISAAC_IMAGE_NAME):$(ISAAC_IMAGE_TAG)
ISAAC_CONTAINER_NAME = vv-isaac-sim-custom
ISAAC_PROJECT_DIR    = /isaac-sim/proj
ISAAC_PYTHON         = /isaac-sim/python.sh
ISAAC_SIM_TEST       = $(ISAAC_PROJECT_DIR)/isaac/runtime/standalone/tests/test_sim_settings.py
ISAAC_SIM_TEST_LOG   = /isaac-sim/.nvidia-omniverse/logs/test_simulation_settings.log
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

# These paths has to be relative to be inside docker build context
ISAAC_ASSETS_DIR       = isaac/assets
ISAAC_DOCKER_DIR       = isaac/docker

ISAAC_RUNTIME_DIRS = \
	"$(ISAAC_KIT_CACHE)" "$(ISAAC_OV_CACHE)" "$(ISAAC_PIP_CACHE)" \
	"$(ISAAC_GLCACHE)" "$(ISAAC_COMPUTEC)" "$(ISAAC_LOGS)" \
	"$(ISAAC_DATA)" "$(ISAAC_DOCS)"

ISAAC_DOCKER_RUNTIME = \
	--network=host \
	--user 0:0 \
	--gpus all \
	--device=/dev/dri

ISAAC_DOCKER_ENV = \
	-e ACCEPT_EULA=Y \
	-e OMNI_KIT_DISABLE_TELEMETRY=1 \
	-e PYTHONPATH=$(ISAAC_PROJECT_DIR)/src \
	-e EXPERIMENT_CONFIG_DIR=$(ISAAC_PROJECT_DIR)/configs/experiment \
	-e isaac_sim_package_path=/isaac-sim

ISAAC_DOCKER_VOLUMES = \
	-v $(ISAAC_KIT_CACHE):/isaac-sim/kit/cache \
	-v $(ISAAC_OV_CACHE):/isaac-sim/.cache/ov \
	-v $(ISAAC_PIP_CACHE):/isaac-sim/.cache/pip \
	-v $(ISAAC_GLCACHE):/isaac-sim/.cache/nvidia/GLCache \
	-v $(ISAAC_COMPUTEC):/isaac-sim/.nv/ComputeCache \
	-v $(ISAAC_LOGS):/isaac-sim/.nvidia-omniverse/logs \
	-v $(ISAAC_DATA):/isaac-sim/.local/share/ov/data \
	-v $(ISAAC_DOCS):/isaac-sim/Documents \
	-v $(CURDIR):$(ISAAC_PROJECT_DIR) \
	-v $(CURDIR)/$(ROS_CONFIG_DIR):/etc/ros:ro

ISAAC_INTERACTIVE_ENV = \
	-e DISPLAY=$(DISPLAY) \
	-e XDG_RUNTIME_DIR=$(XDG_RUNTIME_DIR) \
	-e QT_X11_NO_MITSHM=1 \
	-e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) \
	-e RMW_IMPLEMENTATION=$(RMW_IMPLEMENTATION) \
	-e ROS_DISTRO=$(ROS_DISTRO) \
	-e LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/$(ROS_DISTRO)/lib:$$LD_LIBRARY_PATH \
	-e FASTRTPS_DEFAULT_PROFILES_FILE=$(FASTRTPS_DEFAULT_PROFILES_FILE)

ISAAC_INTERACTIVE_VOLUMES = \
	-v $(XDG_RUNTIME_DIR):$(XDG_RUNTIME_DIR) \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $(ISAAC_ZSH_CACHE):/cache/zsh \
	-v $(BUILD_ROOT)/.config/lazyvim-starter:/root/.config/nvim \
	-v $(BUILD_ROOT)/.config/github-copilot:/root/.config/github-copilot \
	-v $(BUILD_ROOT)/.local/share/nvim:/root/.local/share/nvim \
	-v $(BUILD_ROOT)/.local/state/nvim:/root/.local/state/nvim \
	-v $(BUILD_ROOT)/.cache/nvim:/root/.cache/nvim

# ROS 2 image / container (humble is the other option)
ROS_DISTRO         = jazzy
ROS_BASE_IMAGE     = ros
ROS_BASE_TAG       = $(ROS_DISTRO)
ROS_IMAGE_NAME     = vv-ros-$(ROS_DISTRO)-custom
ROS_IMAGE_TAG      = $(ROS_DISTRO)
ROS_IMAGE          = $(ROS_IMAGE_NAME):$(ROS_IMAGE_TAG)
ROS_CONTAINER_NAME = vv-ros-$(ROS_DISTRO)-custom

# PX4 SITL image / container (stands in for the flight controller)
PX4_TAG            = v1.14.3
PX4_IMAGE_NAME     = vv-px4-sitl
PX4_IMAGE_TAG      = $(PX4_TAG)
PX4_IMAGE          = $(PX4_IMAGE_NAME):$(PX4_IMAGE_TAG)
PX4_CONTAINER_NAME = vv-px4-sitl

# Host-side persistence for PX4 SITL (ULog flight logs + simulated EEPROM params).
PX4_HOME           = $(CURDIR)/px4
PX4_LOGS           = $(PX4_HOME)/logs
PX4_EEPROM         = $(PX4_HOME)/eeprom
# Paths inside the container (under PX4 SITL rootfs).
PX4_ROOTFS         = /opt/PX4-Autopilot/build/px4_sitl_default/rootfs

# These paths has to be relative to be inside docker build context
ROS_WS_DIR         = ros/ros2_ws
ROS_CONFIG_DIR     = ros/config

# This dir is mounted at runtime
ROS_ZSH_CACHE      = $(CURDIR)/ros/cache/zsh

# ROS 2 middleware config
ROS_DOMAIN_ID                  = 42
RMW_IMPLEMENTATION             = rmw_fastrtps_cpp
FASTRTPS_DEFAULT_PROFILES_FILE = /etc/ros/fastdds.xml

# Make neovim version same a host's
NVIM_VERSION := $(shell nvim --version | awk 'NR==1 {print $$2}')

########################################
# X11 helper
########################################

.PHONY: xhost
xhost:
	@xhost +local:root >/dev/null 2>&1 || true

########################################
# Host tests
########################################

.PHONY: require-venv
require-venv:
	@test -f "$(VENV_ACTIVATE)" || { \
		echo "Missing virtual environment at $(VENV_DIR). Create it before running tests."; \
		exit 1; \
	}

.PHONY: test
test: require-venv
	@. "$(VENV_ACTIVATE)" && python -m pytest

########################################
# Build images
########################################

.PHONY: build-isaac
build-isaac:
	@echo "Building custom Isaac image: $(ISAAC_IMAGE)"
	docker buildx build \
		--progress=plain \
		--build-arg DOCKER_REPO="$(ISAAC_BASE_IMAGE)" \
		--build-arg DOCKER_TAG="$(ISAAC_BASE_TAG)" \
		--build-arg NVIM_TAG="$(NVIM_VERSION)" \
		--build-arg SCRIPTS_DIR="$(ISAAC_DOCKER_DIR)" \
		--target isaac-sim \
		-t $(ISAAC_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

.PHONY: build-ros
build-ros:
	@echo "Pulling ROS image: $(ROS_IMAGE)"
	docker buildx build \
		--progress=plain \
		--build-arg DOCKER_REPO="$(ROS_BASE_IMAGE)" \
		--build-arg DOCKER_TAG="$(ROS_BASE_TAG)" \
		--build-arg NVIM_TAG="$(NVIM_VERSION)" \
		--target ros \
		-t $(ROS_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

.PHONY: build-px4
build-px4:
	@echo "Building PX4 SITL image: $(PX4_IMAGE)"
	docker buildx build \
		--progress=plain \
		--build-arg PX4_TAG="$(PX4_TAG)" \
		--target px4 \
		-t $(PX4_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

.PHONY: build-all
build-all: build-isaac build-ros build-px4
	@echo "All images built/pulled."

########################################
# Interactive runs (persistent containers)
########################################

.PHONY: sync-nvim-caches
sync-nvim-caches:
	mkdir -p \
		$(BUILD_ROOT)/.config \
		$(BUILD_ROOT)/.local/share \
		$(BUILD_ROOT)/.local/state \
		$(BUILD_ROOT)/.cache
	rsync -a --exclude='**/.git/**' "${HOME}/code/lazyvim-starter" "$(BUILD_ROOT)/.config/"
	rsync -a --exclude='**/.git/**' "${HOME}/.config/github-copilot" "$(BUILD_ROOT)/.config/"
	rsync -a --exclude='**/.git/**' "${HOME}/.local/share/nvim" "$(BUILD_ROOT)/.local/share/"
	rsync -a --exclude='**/.git/**' "${HOME}/.local/state/nvim" "$(BUILD_ROOT)/.local/state/"
	rsync -a --exclude='**/.git/**' "${HOME}/.cache/nvim" "$(BUILD_ROOT)/.cache/"
	echo "Neovim caches synchronized to build root."

.PHONY: run-isaac
run-isaac: xhost sync-nvim-caches
	@mkdir -p $(ISAAC_RUNTIME_DIRS) "$(ISAAC_ZSH_CACHE)"
	@if docker ps -a --format '{{.Names}}' | grep -q '^$(ISAAC_CONTAINER_NAME)$$'; then \
		echo "Starting existing Isaac container $(ISAAC_CONTAINER_NAME)..."; \
		docker start -ai $(ISAAC_CONTAINER_NAME); \
	else \
		echo "Creating and starting Isaac container $(ISAAC_CONTAINER_NAME)..."; \
		docker run -it \
			--name $(ISAAC_CONTAINER_NAME) \
			$(ISAAC_DOCKER_RUNTIME) \
			$(ISAAC_DOCKER_ENV) \
			$(ISAAC_INTERACTIVE_ENV) \
			$(ISAAC_DOCKER_VOLUMES) \
			$(ISAAC_INTERACTIVE_VOLUMES) \
			$(ISAAC_IMAGE); \
	fi

.PHONY: test-isaac-simulation
test-isaac-simulation:
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
				{ grep -E "ERROR|Error|Exception|Traceback|Could not|No matching|Failed|failed" \
					"$(ISAAC_SIM_TEST_LOG)"; exit 1; }; } && \
			$(ISAAC_PYTHON) $(ISAAC_SIM_TEST) --scene "$(SCENE)" \
				2>&1 | tee -a "$(ISAAC_SIM_TEST_LOG)" | grep --line-buffered -E \
					"^\[isaac-standalone-test\]|^Traceback|^[[:space:]]+File \"|^[A-Za-z_]+(Error|Exception):"'

.PHONY: run-ros
run-ros: xhost sync-nvim-caches
	@if docker ps -a --format '{{.Names}}' | grep -q '^$(ROS_CONTAINER_NAME)$$'; then \
		echo "Starting existing ROS2 container $(ROS_CONTAINER_NAME)..."; \
		docker start -ai $(ROS_CONTAINER_NAME); \
	else \
		echo "Creating and starting ROS2 container $(ROS_CONTAINER_NAME)..."; \
		docker run -it \
			--name "$(ROS_CONTAINER_NAME)" \
			--network=host \
			--gpus all \
			--device=/dev/dri \
			-e NVIDIA_DRIVER_CAPABILITIES=all \
			-e NVIDIA_VISIBLE_DEVICES=all \
			-e DISPLAY="$(DISPLAY)" \
			-e XDG_RUNTIME_DIR="$(XDG_RUNTIME_DIR)" \
			-e QT_X11_NO_MITSHM=1 \
			-e QT_QPA_PLATFORM=xcb \
			-e ROS_DOMAIN_ID="$(ROS_DOMAIN_ID)" \
			-e RMW_IMPLEMENTATION="$(RMW_IMPLEMENTATION)" \
			-e FASTRTPS_DEFAULT_PROFILES_FILE=$(FASTRTPS_DEFAULT_PROFILES_FILE) \
			-v "$(XDG_RUNTIME_DIR)":"$(XDG_RUNTIME_DIR)" \
			-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
			-v $(CURDIR)/$(ROS_CONFIG_DIR):/etc/ros:ro \
			-v $(CURDIR)/$(ROS_WS_DIR):/ros2_ws \
			-v $(ROS_ZSH_CACHE):/cache/zsh \
			-v $(BUILD_ROOT)/.config/lazyvim-starter:/root/.config/nvim \
			-v $(BUILD_ROOT)/.config/github-copilot:/root/.config/github-copilot \
			-v $(BUILD_ROOT)/.local/share/nvim:/root/.local/share/nvim \
			-v $(BUILD_ROOT)/.local/state/nvim:/root/.local/state/nvim \
			-v $(BUILD_ROOT)/.cache/nvim:/root/.cache/nvim \
			"$(ROS_IMAGE)"; \
	fi

.PHONY: run-px4
run-px4:
	@mkdir -p "$(PX4_LOGS)" "$(PX4_EEPROM)"
	@if docker ps -a --format '{{.Names}}' | grep -q '^$(PX4_CONTAINER_NAME)$$'; then \
		echo "Starting existing PX4 container $(PX4_CONTAINER_NAME)..."; \
		docker start -ai $(PX4_CONTAINER_NAME); \
	else \
		echo "Creating and starting PX4 container $(PX4_CONTAINER_NAME)..."; \
		docker run -it \
			--name "$(PX4_CONTAINER_NAME)" \
			--network=host \
			-v $(PX4_LOGS):$(PX4_ROOTFS)/log \
			-v $(PX4_EEPROM):$(PX4_ROOTFS)/eeprom \
			"$(PX4_IMAGE)"; \
	fi

.PHONY: run-all
run-all:
	kitty --title ROS2 --hold make run-ros &
	kitty --title IsaacSim --hold make run-isaac &
	kitty --title PX4 --hold make run-px4 &

########################################
# Stop / clean
########################################

.PHONY: stop-isaac
stop-isaac:
	@docker stop $(ISAAC_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop-ros
stop-ros:
	@docker stop $(ROS_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop-px4
stop-px4:
	@docker stop $(PX4_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop-all
stop-all: stop-isaac stop-ros stop-px4
	@echo "Stopped Isaac, ROS, and PX4 containers (if running)."

.PHONY: clean-isaac-container
clean-isaac-container:
	@docker rm -f $(ISAAC_CONTAINER_NAME) 2>/dev/null || true

.PHONY: clean-ros-container
clean-ros-container:
	@docker rm -f $(ROS_CONTAINER_NAME) 2>/dev/null || true

.PHONY: clean-px4-container
clean-px4-container:
	@docker rm -f $(PX4_CONTAINER_NAME) 2>/dev/null || true

.PHONY: clean-container
clean-container: clean-isaac-container clean-ros-container clean-px4-container
	@echo "Removed Isaac, ROS, and PX4 containers (if they existed)."

.PHONY: clean-isaac-image
clean-isaac-image:
	@docker rmi $(ISAAC_IMAGE) 2>/dev/null || true

.PHONY: clean-ros-image
clean-ros-image:
	@docker rmi $(ROS_IMAGE) 2>/dev/null || true

.PHONY: clean-px4-image
clean-px4-image:
	@docker rmi $(PX4_IMAGE) 2>/dev/null || true

.PHONY: clean-images
clean-images: clean-isaac-image clean-ros-image clean-px4-image
	@echo "Removed Isaac, ROS, and PX4 images (if they existed)."

.PHONY: clean-all
clean-all: stop-all clean-container clean-images
	@echo "All containers and images removed."

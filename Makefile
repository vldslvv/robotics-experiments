########################################
# Images / containers
########################################

BUILD_ROOT             = $(CURDIR)/.build/docker
SRC_DIR                = $(CURDIR)/src

# Isaac custom image
ISAAC_BASE_IMAGE = nvcr.io/nvidia/isaac-sim
ISAAC_BASE_TAG   = 5.1.0

ISAAC_IMAGE_NAME     = vv-isaac-sim-custom
ISAAC_IMAGE_TAG      = $(ISAAC_BASE_TAG)
ISAAC_IMAGE          = $(ISAAC_IMAGE_NAME):$(ISAAC_IMAGE_TAG)
ISAAC_CONTAINER_NAME = vv-isaac-sim-custom

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

# These paths has to be relative to be inside docker build context
ISAAC_ASSETS_DIR       = isaac/assets
ISAAC_SCRIPTS_DIR      = isaac/scripts

# ROS 2 image / container (humble is the other option)
ROS_DISTRO         = jazzy
ROS_BASE_IMAGE     = ros
ROS_BASE_TAG       = $(ROS_DISTRO)
ROS_IMAGE_NAME     = vv-ros-$(ROS_DISTRO)-custom
ROS_IMAGE_TAG      = $(ROS_DISTRO)
ROS_IMAGE          = $(ROS_IMAGE_NAME):$(ROS_IMAGE_TAG)
ROS_CONTAINER_NAME = vv-ros-$(ROS_DISTRO)-custom

# These paths has to be relative to be inside docker build context
ROS_WS_DIR         = ros/ros2_ws
ROS_DOT_DIR        = ros/.ros

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
		--build-arg SCRIPTS_DIR="$(ISAAC_SCRIPTS_DIR)" \
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

.PHONY: build-all
build-all: build-isaac build-ros
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
	echo "$(ISAAC_PIP_CACHE)"
	@mkdir -p \
		"$(ISAAC_KIT_CACHE)" "$(ISAAC_OV_CACHE)" "$(ISAAC_PIP_CACHE)" \
		"$(ISAAC_GLCACHE)" "$(ISAAC_COMPUTEC)" "$(ISAAC_LOGS)" \
		"$(ISAAC_DATA)" "$(ISAAC_DOCS)"
	@if docker ps -a --format '{{.Names}}' | grep -q '^$(ISAAC_CONTAINER_NAME)$$'; then \
		echo "Starting existing Isaac container $(ISAAC_CONTAINER_NAME)..."; \
		docker start -ai $(ISAAC_CONTAINER_NAME); \
	else \
		echo "Creating and starting Isaac container $(ISAAC_CONTAINER_NAME)..."; \
		docker run -it \
			--name $(ISAAC_CONTAINER_NAME) \
			--network=host \
			--user 0:0 \
			--gpus all \
			-e ACCEPT_EULA=Y \
			-e OMNI_KIT_DISABLE_TELEMETRY=1 \
			-e DISPLAY=$(DISPLAY) \
			-e XDG_RUNTIME_DIR=$(XDG_RUNTIME_DIR) \
			-e QT_X11_NO_MITSHM=1 \
			-e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) \
			-e RMW_IMPLEMENTATION=$(RMW_IMPLEMENTATION) \
			-e ROS_DISTRO=$(ROS_DISTRO) \
			-e isaac_sim_package_path=/isaac-sim \
			-e LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/$(ROS_DISTRO)/lib:$$LD_LIBRARY_PATH \
			-e FASTRTPS_DEFAULT_PROFILES_FILE=/isaac-sim/.ros/fastdds.xml \
			--device=/dev/dri \
			-v $(XDG_RUNTIME_DIR):$(XDG_RUNTIME_DIR) \
			-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
			-v $(ISAAC_KIT_CACHE):/isaac-sim/kit/cache \
			-v $(ISAAC_OV_CACHE):/isaac-sim/.cache/ov \
			-v $(ISAAC_PIP_CACHE):/isaac-sim/.cache/pip \
			-v $(ISAAC_GLCACHE):/isaac-sim/.cache/nvidia/GLCache \
			-v $(ISAAC_COMPUTEC):/isaac-sim/.nv/ComputeCache \
			-v $(ISAAC_LOGS):/isaac-sim/.nvidia-omniverse/logs \
			-v $(ISAAC_DATA):/isaac-sim/.local/share/ov/data \
			-v $(ISAAC_DOCS):/isaac-sim/Documents \
			-v $(SRC_DIR)/robotics-experiments:/isaac-sim/proj/robotics-experiments \
			-v $(CURDIR)/$(ISAAC_SCRIPTS_DIR):/isaac-sim/proj/scripts \
			-v $(CURDIR)/$(ISAAC_ASSETS_DIR):/isaac-sim/proj/assets \
			-v $(CURDIR)/$(ROS_WS_DIR):/isaac-sim/proj/ros2_ws \
			-v $(CURDIR)/$(ROS_DOT_DIR):/isaac-sim/.ros \
			-v $(BUILD_ROOT)/.config/lazyvim-starter:/root/.config/nvim \
			-v $(BUILD_ROOT)/.config/github-copilot:/root/.config/github-copilot \
			-v $(BUILD_ROOT)/.local/share/nvim:/root/.local/share/nvim \
			-v $(BUILD_ROOT)/.local/state/nvim:/root/.local/state/nvim \
			-v $(BUILD_ROOT)/.cache/nvim:/root/.cache/nvim \
			$(ISAAC_IMAGE); \
	fi

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
			-e FASTRTPS_DEFAULT_PROFILES_FILE=/isaac-sim/.ros/fastdds.xml \
			-v "$(XDG_RUNTIME_DIR)":"$(XDG_RUNTIME_DIR)" \
			-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
			-v $(CURDIR)/$(ROS_DOT_DIR):/isaac-sim/.ros \
			-v $(CURDIR)/$(ROS_WS_DIR):/ros2_ws \
			-v $(BUILD_ROOT)/.config/lazyvim-starter:/root/.config/nvim \
			-v $(BUILD_ROOT)/.config/github-copilot:/root/.config/github-copilot \
			-v $(BUILD_ROOT)/.local/share/nvim:/root/.local/share/nvim \
			-v $(BUILD_ROOT)/.local/state/nvim:/root/.local/state/nvim \
			-v $(BUILD_ROOT)/.cache/nvim:/root/.cache/nvim \
			"$(ROS_IMAGE)"; \
	fi

########################################
# Stop / clean
########################################

.PHONY: stop-isaac
stop-isaac:
	@docker stop $(ISAAC_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop-ros
stop-ros:
	@docker stop $(ROS_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop-all
stop-all: stop-isaac stop-ros
	@echo "Stopped Isaac and ROS containers (if running)."

.PHONY: clean-isaac-container
clean-isaac-container:
	@docker rm -f $(ISAAC_CONTAINER_NAME) 2>/dev/null || true

.PHONY: clean-ros-container
clean-ros-container:
	@docker rm -f $(ROS_CONTAINER_NAME) 2>/dev/null || true

.PHONY: clean-container
clean-container: clean-isaac-container clean-ros-container
	@echo "Removed Isaac and ROS containers (if they existed)."

.PHONY: clean-isaac-image
clean-isaac-image:
	@docker rmi $(ISAAC_IMAGE) 2>/dev/null || true

.PHONY: clean-ros-image
clean-ros-image:
	@docker rmi $(ROS_IMAGE) 2>/dev/null || true

.PHONY: clean-images
clean-images: clean-isaac-image clean-ros-image
	@echo "Removed Isaac and ROS images (if they existed)."

.PHONY: clean-all
clean-all: stop-all clean-container clean-images
	@echo "All containers and images removed."

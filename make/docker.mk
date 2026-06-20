########################################
# Docker helpers and shared flags
########################################

ISAAC_DOCKER_RUNTIME = \
	--network=host \
	--user 0:0 \
	--gpus all \
	--device=/dev/dri

ISAAC_INTEGRATION_DOCKER_RUNTIME = \
	--user 0:0 \
	--gpus all \
	--device=/dev/dri

ISAAC_DOCKER_ENV = \
	-e ACCEPT_EULA=Y \
	-e OMNI_KIT_DISABLE_TELEMETRY=1 \
	-e PYTHONPATH=$(ISAAC_PROJECT_DIR)/src \
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

ISAAC_INTEGRATION_DOCKER_VOLUMES = \
	-v $(ISAAC_KIT_CACHE):/isaac-sim/kit/cache \
	-v $(ISAAC_OV_CACHE):/isaac-sim/.cache/ov \
	-v $(ISAAC_PIP_CACHE):/isaac-sim/.cache/pip \
	-v $(ISAAC_GLCACHE):/isaac-sim/.cache/nvidia/GLCache \
	-v $(ISAAC_COMPUTEC):/isaac-sim/.nv/ComputeCache \
	-v $${ISAAC_RUN_ROOT}/logs:/isaac-sim/.nvidia-omniverse/logs \
	-v $${ISAAC_RUN_ROOT}/data:/isaac-sim/.local/share/ov/data \
	-v $${ISAAC_RUN_ROOT}/documents:/isaac-sim/Documents \
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

ROS_DOCKER_RUNTIME = \
	--network=host \
	--gpus all \
	--device=/dev/dri

ROS_DOCKER_ENV = \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e DISPLAY="$(DISPLAY)" \
	-e XDG_RUNTIME_DIR="$(XDG_RUNTIME_DIR)" \
	-e QT_X11_NO_MITSHM=1 \
	-e QT_QPA_PLATFORM=xcb \
	-e ROS_DOMAIN_ID="$(ROS_DOMAIN_ID)" \
	-e RMW_IMPLEMENTATION="$(RMW_IMPLEMENTATION)" \
	-e FASTRTPS_DEFAULT_PROFILES_FILE=$(FASTRTPS_DEFAULT_PROFILES_FILE)

ROS_DOCKER_VOLUMES = \
	-v "$(XDG_RUNTIME_DIR)":"$(XDG_RUNTIME_DIR)" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $(CURDIR)/$(ROS_CONFIG_DIR):/etc/ros:ro \
	-v $(CURDIR)/$(ROS_WS_DIR):/ros2_ws \
	-v $(ROS_ZSH_CACHE):/cache/zsh \
	-v $(BUILD_ROOT)/.config/lazyvim-starter:/root/.config/nvim \
	-v $(BUILD_ROOT)/.config/github-copilot:/root/.config/github-copilot \
	-v $(BUILD_ROOT)/.local/share/nvim:/root/.local/share/nvim \
	-v $(BUILD_ROOT)/.local/state/nvim:/root/.local/state/nvim \
	-v $(BUILD_ROOT)/.cache/nvim:/root/.cache/nvim

PX4_DOCKER_RUNTIME = \
	--network=host

PX4_DOCKER_VOLUMES = \
	-v $(PX4_LOGS):$(PX4_ROOTFS)/log \
	-v $(PX4_EEPROM):$(PX4_ROOTFS)/eeprom

.PHONY: xhost
xhost:
	@xhost +local:root >/dev/null 2>&1 || true

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

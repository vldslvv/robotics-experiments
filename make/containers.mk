########################################
# Persistent container lifecycle
########################################

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

.PHONY: run
run:
	kitty --title ROS2 --hold $(MAKE) run-ros &
	kitty --title IsaacSim --hold $(MAKE) run-isaac &
	kitty --title PX4 --hold $(MAKE) run-px4 &

.PHONY: stop-isaac
stop-isaac:
	@docker stop $(ISAAC_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop-ros
stop-ros:
	@docker stop $(ROS_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop-px4
stop-px4:
	@docker stop $(PX4_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop
stop: stop-isaac stop-ros stop-px4
	@echo "Stopped Isaac, ROS, and PX4 containers (if running)."

.PHONY: remove-isaac-container
remove-isaac-container:
	@docker rm -f $(ISAAC_CONTAINER_NAME) 2>/dev/null || true

.PHONY: remove-ros-container
remove-ros-container:
	@docker rm -f $(ROS_CONTAINER_NAME) 2>/dev/null || true

.PHONY: remove-px4-container
remove-px4-container:
	@docker rm -f $(PX4_CONTAINER_NAME) 2>/dev/null || true

.PHONY: remove-containers
remove-containers: remove-isaac-container remove-ros-container remove-px4-container
	@echo "Removed Isaac, ROS, and PX4 containers (if they existed)."

.PHONY: remove-isaac-image
remove-isaac-image:
	@docker rmi $(ISAAC_IMAGE) 2>/dev/null || true

.PHONY: remove-ros-image
remove-ros-image:
	@docker rmi $(ROS_IMAGE) 2>/dev/null || true

.PHONY: remove-px4-image
remove-px4-image:
	@docker rmi $(PX4_IMAGE) 2>/dev/null || true

.PHONY: remove-images
remove-images: remove-isaac-image remove-ros-image remove-px4-image
	@echo "Removed Isaac, ROS, and PX4 images (if they existed)."

.PHONY: destroy
destroy: stop remove-containers remove-images
	@echo "All containers and images removed."

.PHONY: run-all
run-all: run

.PHONY: stop-all
stop-all: stop

.PHONY: clean-isaac-container
clean-isaac-container: remove-isaac-container

.PHONY: clean-ros-container
clean-ros-container: remove-ros-container

.PHONY: clean-px4-container
clean-px4-container: remove-px4-container

.PHONY: clean-container
clean-container: remove-containers

.PHONY: clean-isaac-image
clean-isaac-image: remove-isaac-image

.PHONY: clean-ros-image
clean-ros-image: remove-ros-image

.PHONY: clean-px4-image
clean-px4-image: remove-px4-image

.PHONY: clean-images
clean-images: remove-images

.PHONY: clean-all
clean-all: destroy

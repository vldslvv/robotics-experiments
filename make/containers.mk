########################################
# Persistent container lifecycle
########################################

isaac_DISPLAY_NAME = Isaac
isaac_CONTAINER_NAME = $(ISAAC_CONTAINER_NAME)
isaac_IMAGE = $(ISAAC_IMAGE)
isaac_RUN_DEPS = xhost sync-nvim-caches
isaac_RUNTIME_DIRS = $(ISAAC_RUNTIME_DIRS) "$(ISAAC_ZSH_CACHE)"
isaac_RECREATE_IF_ENV = ROBOTICS_EXPERIMENT_CONFIG_PATH=$(HYDRA_CONFIG_CONTAINER_PATH)
isaac_DOCKER_ARGS = \
	$(ISAAC_DOCKER_RUNTIME) \
	$(ISAAC_DOCKER_ENV) \
	$(ISAAC_INTERACTIVE_ENV) \
	$(ISAAC_DOCKER_VOLUMES) \
	$(ISAAC_INTERACTIVE_VOLUMES)

ros_DISPLAY_NAME = ROS2
ros_CONTAINER_NAME = $(ROS_CONTAINER_NAME)
ros_IMAGE = $(ROS_IMAGE)
ros_RUN_DEPS = xhost sync-nvim-caches
ros_DOCKER_ARGS = \
	$(ROS_DOCKER_RUNTIME) \
	$(ROS_DOCKER_ENV) \
	$(ROS_DOCKER_VOLUMES)

.PHONY: generate-drone-usd
generate-drone-usd:
	@uv run --extra simulation python isaac/runtime/standalone/build_copter_usd_standalone.py --all

.PHONY: run-isaac
run-isaac: generate-drone-usd $(isaac_RUN_DEPS)
	@mkdir -p $(isaac_RUNTIME_DIRS)
	@EXPECTED_CMD="$(ISAAC_CONTAINER_CMD)" \
	make/scripts/run_persistent_container.sh \
		"$(isaac_DISPLAY_NAME)" \
		"$(isaac_CONTAINER_NAME)" \
		"$(isaac_IMAGE)" \
		"$(isaac_RECREATE_IF_ENV)" \
		-- $(isaac_DOCKER_ARGS)

.PHONY: run-ros
run-ros: $(ros_RUN_DEPS)
	@make/scripts/run_persistent_container.sh \
		"$(ros_DISPLAY_NAME)" \
		"$(ros_CONTAINER_NAME)" \
		"$(ros_IMAGE)" \
		"$(ros_RECREATE_IF_ENV)" \
		-- $(ros_DOCKER_ARGS)

.PHONY: stop-isaac
stop-isaac:
	@docker stop $(isaac_CONTAINER_NAME) 2>/dev/null || true

.PHONY: stop-ros
stop-ros:
	@docker stop $(ros_CONTAINER_NAME) 2>/dev/null || true

.PHONY: remove-isaac-container
remove-isaac-container:
	@docker rm -f $(isaac_CONTAINER_NAME) 2>/dev/null || true

.PHONY: remove-ros-container
remove-ros-container:
	@docker rm -f $(ros_CONTAINER_NAME) 2>/dev/null || true

.PHONY: remove-isaac-image
remove-isaac-image:
	@docker rmi $(isaac_IMAGE) 2>/dev/null || true

.PHONY: remove-isaac-lab-image
remove-isaac-lab-image:
	@docker rmi $(ISAAC_LAB_IMAGE) 2>/dev/null || true

.PHONY: remove-ros-image
remove-ros-image:
	@docker rmi $(ros_IMAGE) 2>/dev/null || true

.PHONY: run-px4
run-px4:
	@HYDRA_CONFIG_HOST_PATH='$(HYDRA_CONFIG_HOST_PATH)' \
	PX4_IMAGE='$(PX4_IMAGE)' \
	MAKE_CMD='$(MAKE)' \
	make/scripts/run_px4_persistent_containers.sh

.PHONY: run-px4-instance
run-px4-instance:
	@PX4_INSTANCE='$(PX4_INSTANCE)' \
	PX4_SIM_MODEL='$(PX4_SIM_MODEL)' \
	PX4_SYS_AUTOSTART='$(PX4_SYS_AUTOSTART)' \
	PX4_TOTAL_INSTANCES='$(PX4_TOTAL_INSTANCES)' \
	PX4_CONTAINER_NAME='$(PX4_CONTAINER_NAME)' \
	PX4_IMAGE='$(PX4_IMAGE)' \
	PX4_LOGS='$(PX4_LOGS)' \
	PX4_EEPROM='$(PX4_EEPROM)' \
	PX4_ROOTFS='$(PX4_ROOTFS)' \
	PX4_DOCKER_RUNTIME='$(PX4_DOCKER_RUNTIME)' \
	px4/scripts/run_px4_persistent_instance.sh

.PHONY: stop-px4
stop-px4:
	@make/scripts/px4_containers.sh stop '$(PX4_CONTAINER_NAME)'

.PHONY: remove-px4-container
remove-px4-container:
	@make/scripts/px4_containers.sh remove '$(PX4_CONTAINER_NAME)'

.PHONY: remove-px4-image
remove-px4-image:
	@docker rmi $(PX4_IMAGE) 2>/dev/null || true

.PHONY: remove-cache-images
remove-cache-images:
	@docker rmi $(ISAAC_CACHE_IMAGE) 2>/dev/null || true
	@docker rmi $(ISAAC_LAB_CACHE_IMAGE) 2>/dev/null || true
	@docker rmi $(ROS_CACHE_IMAGE) 2>/dev/null || true
	@docker rmi $(PX4_CACHE_IMAGE) 2>/dev/null || true
	@echo "Removed Docker build cache anchor images (if they existed)."

.PHONY: run
run:
	kitty --title ROS2 --hold $(MAKE) run-ros &
	kitty --title IsaacSim --hold $(MAKE) run-isaac &
	$(MAKE) run-px4

.PHONY: stop
stop: stop-isaac stop-ros stop-px4
	@echo "Stopped Isaac, ROS, and PX4 containers (if running)."

.PHONY: remove-containers
remove-containers: remove-isaac-container remove-ros-container remove-px4-container
	@echo "Removed Isaac, ROS, and PX4 containers (if they existed)."

.PHONY: remove-images
remove-images: remove-isaac-image remove-isaac-lab-image remove-ros-image remove-px4-image
	@echo "Removed runnable Isaac, Isaac Lab, ROS, and PX4 image tags (build cache anchors preserved)."

.PHONY: destroy
destroy: stop remove-containers remove-images
	@echo "All containers and runnable image tags removed; Docker build cache anchors preserved."

.PHONY: run-all
run-all: run

# Like run-all, but auto-launches the custom Isaac runtime in the Isaac
# container (ISAAC_AUTORUN=1). The override is on the command line so it
# propagates to the kitty-spawned `$(MAKE) run-isaac` sub-make.
.PHONY: run-all-autorun
run-all-autorun:
	@$(MAKE) run-all ISAAC_AUTORUN=1

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

.PHONY: clean-cache-images
clean-cache-images: remove-cache-images

.PHONY: clean-all
clean-all: destroy

########################################
# Persistent container lifecycle
########################################

CONTAINER_STACKS = isaac ros

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

define RUN_CONTAINER_TARGET
.PHONY: run-$(1)
run-$(1): $$($(1)_RUN_DEPS)
	@$$(if $$($(1)_RUNTIME_DIRS),mkdir -p $$($(1)_RUNTIME_DIRS))
	@make/run_persistent_container.sh \
		"$$($(1)_DISPLAY_NAME)" \
		"$$($(1)_CONTAINER_NAME)" \
		"$$($(1)_IMAGE)" \
		"$$($(1)_RECREATE_IF_ENV)" \
		-- $$($(1)_DOCKER_ARGS)
endef

define STOP_CONTAINER_TARGET
.PHONY: stop-$(1)
stop-$(1):
	@docker stop $$($(1)_CONTAINER_NAME) 2>/dev/null || true
endef

define REMOVE_CONTAINER_TARGET
.PHONY: remove-$(1)-container
remove-$(1)-container:
	@docker rm -f $$($(1)_CONTAINER_NAME) 2>/dev/null || true
endef

define REMOVE_IMAGE_TARGET
.PHONY: remove-$(1)-image
remove-$(1)-image:
	@docker rmi $$($(1)_IMAGE) 2>/dev/null || true
endef

$(foreach stack,$(CONTAINER_STACKS),$(eval $(call RUN_CONTAINER_TARGET,$(stack))))
$(foreach stack,$(CONTAINER_STACKS),$(eval $(call STOP_CONTAINER_TARGET,$(stack))))
$(foreach stack,$(CONTAINER_STACKS),$(eval $(call REMOVE_CONTAINER_TARGET,$(stack))))
$(foreach stack,$(CONTAINER_STACKS),$(eval $(call REMOVE_IMAGE_TARGET,$(stack))))

.PHONY: run-px4
run-px4:
	@HYDRA_CONFIG_HOST_PATH='$(HYDRA_CONFIG_HOST_PATH)' \
	MAKE_CMD='$(MAKE)' \
	make/run_px4_persistent_containers.sh

.PHONY: run-px4-instance
run-px4-instance:
	@PX4_INSTANCE='$(PX4_INSTANCE)' \
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
	@make/px4_containers.sh stop '$(PX4_CONTAINER_NAME)'

.PHONY: remove-px4-container
remove-px4-container:
	@make/px4_containers.sh remove '$(PX4_CONTAINER_NAME)'

.PHONY: remove-px4-image
remove-px4-image:
	@docker rmi $(PX4_IMAGE) 2>/dev/null || true

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

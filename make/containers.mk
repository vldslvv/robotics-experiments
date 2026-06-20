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
	@if docker ps -a --format '{{.Names}}' | grep -q '^$$($(1)_CONTAINER_NAME)$$$$'; then \
		expected_env='$$($(1)_RECREATE_IF_ENV)'; \
		if [ -n "$$$$expected_env" ] && ! docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$$($(1)_CONTAINER_NAME)" | grep -qx "$$$$expected_env"; then \
			echo "Recreating $$($(1)_DISPLAY_NAME) container $$($(1)_CONTAINER_NAME) for updated runtime environment..."; \
			docker rm -f "$$($(1)_CONTAINER_NAME)" >/dev/null; \
		else \
		echo "Starting existing $$($(1)_DISPLAY_NAME) container $$($(1)_CONTAINER_NAME)..."; \
		docker start -ai $$($(1)_CONTAINER_NAME); \
			exit 0; \
		fi; \
	fi; \
	if docker ps -a --format '{{.Names}}' | grep -q '^$$($(1)_CONTAINER_NAME)$$$$'; then \
		echo "Starting existing $$($(1)_DISPLAY_NAME) container $$($(1)_CONTAINER_NAME)..."; \
		docker start -ai $$($(1)_CONTAINER_NAME); \
	else \
		echo "Creating and starting $$($(1)_DISPLAY_NAME) container $$($(1)_CONTAINER_NAME)..."; \
		docker run -it \
			--name "$$($(1)_CONTAINER_NAME)" \
			$$($(1)_DOCKER_ARGS) \
			"$$($(1)_IMAGE)"; \
	fi
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

PX4_ENABLED_INSTANCE_COUNT_CMD = ROBOTICS_EXPERIMENT_CONFIG_PATH='$(HYDRA_CONFIG_HOST_PATH)' uv run python -c 'from robotics_experiments.config import get_enabled_px4_vehicle_count; print(get_enabled_px4_vehicle_count())'

.PHONY: run-px4
run-px4:
	@count="$$( $(PX4_ENABLED_INSTANCE_COUNT_CMD) )"; \
	if [ "$$count" -eq 0 ]; then \
		echo "No PX4-enabled vehicles in Hydra config; skipping PX4 SITL."; \
		exit 0; \
	fi; \
	i=0; \
	while [ "$$i" -lt "$$count" ]; do \
		kitty --title "PX4[$$i]" --hold $(MAKE) run-px4-instance PX4_INSTANCE="$$i" PX4_TOTAL_INSTANCES="$$count" & \
		i=$$((i + 1)); \
	done

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
	@containers="$$(docker ps -a --format '{{.Names}}' | grep -E '^$(PX4_CONTAINER_NAME)(\.[0-9]+)?$$' || true)"; \
	if [ -n "$$containers" ]; then \
		echo "$$containers" | xargs docker stop >/dev/null 2>&1 || true; \
	fi

.PHONY: remove-px4-container
remove-px4-container:
	@containers="$$(docker ps -a --format '{{.Names}}' | grep -E '^$(PX4_CONTAINER_NAME)(\.[0-9]+)?$$' || true)"; \
	if [ -n "$$containers" ]; then \
		echo "$$containers" | xargs docker rm -f >/dev/null 2>&1 || true; \
	fi

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

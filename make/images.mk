########################################
# Image builds
########################################

docker_cache_from = $(if $(shell docker image inspect $(1) >/dev/null 2>&1 && printf yes),--cache-from $(1),)

.PHONY: require-host-nvim
require-host-nvim:
	@test -n "$(NVIM_VERSION)" || { \
		echo "Host Neovim is required to build the Isaac and ROS images." >&2; \
		exit 1; \
	}

.PHONY: build
build: build-isaac build-isaac-lab build-ros build-px4
	@echo "All images built/pulled."

.PHONY: build-isaac
build-isaac: require-host-nvim
	@echo "Building custom Isaac image: $(ISAAC_IMAGE)"
	docker buildx build \
		--progress=plain \
		$(call docker_cache_from,$(ISAAC_CACHE_IMAGE)) \
		--cache-to type=inline \
		--build-arg DOCKER_REPO="$(ISAAC_BASE_IMAGE)" \
		--build-arg DOCKER_TAG="$(ISAAC_BASE_TAG)" \
		--build-arg DOCKER_DIGEST="$(ISAAC_BASE_DIGEST)" \
		--build-arg NVIM_TAG="$(NVIM_VERSION)" \
		--build-arg SCRIPTS_DIR="$(ISAAC_DOCKER_DIR)" \
		--target isaac-sim \
		-t $(ISAAC_IMAGE) \
		-t $(ISAAC_CACHE_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

# Isaac Lab image: FROM isaac-sim, so buildx reuses the cached isaac-sim layers.
# Only the Phase F RL workflow (test-integration-press-hover-goto; future
# training) runs on it.
.PHONY: build-isaac-lab
build-isaac-lab: require-host-nvim
	@echo "Building Isaac Lab image: $(ISAAC_LAB_IMAGE) (ISAACLAB_REF=$(ISAACLAB_REF))"
	docker buildx build \
		--progress=plain \
		$(call docker_cache_from,$(ISAAC_LAB_CACHE_IMAGE)) \
		--cache-to type=inline \
		--build-arg DOCKER_REPO="$(ISAAC_BASE_IMAGE)" \
		--build-arg DOCKER_TAG="$(ISAAC_BASE_TAG)" \
		--build-arg DOCKER_DIGEST="$(ISAAC_BASE_DIGEST)" \
		--build-arg NVIM_TAG="$(NVIM_VERSION)" \
		--build-arg SCRIPTS_DIR="$(ISAAC_DOCKER_DIR)" \
		--build-arg ISAACLAB_REF="$(ISAACLAB_REF)" \
		--target isaac-lab \
		-t $(ISAAC_LAB_IMAGE) \
		-t $(ISAAC_LAB_CACHE_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

.PHONY: build-ros
build-ros: require-host-nvim
	@echo "Pulling ROS image: $(ROS_IMAGE)"
	docker buildx build \
		--progress=plain \
		$(call docker_cache_from,$(ROS_CACHE_IMAGE)) \
		--cache-to type=inline \
		--build-arg DOCKER_REPO="$(ROS_BASE_IMAGE)" \
		--build-arg DOCKER_TAG="$(ROS_BASE_TAG)" \
		--build-arg DOCKER_DIGEST="$(ROS_BASE_DIGEST)" \
		--build-arg NVIM_TAG="$(NVIM_VERSION)" \
		--build-arg TOPIC_BASED_ROS2_CONTROL_REF="$(TOPIC_BASED_ROS2_CONTROL_REF)" \
		--build-arg ISAAC_ROS_WORKSPACES_REF="$(ISAAC_ROS_WORKSPACES_REF)" \
		--target ros \
		-t $(ROS_IMAGE) \
		-t $(ROS_CACHE_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

.PHONY: print-airframes
print-airframes:
	@uv run python px4/scripts/generate_px4_airframes.py \
		--config-dir configs/airframes --print-manifest

.PHONY: build-px4
build-px4:
	@echo "Building PX4 SITL image: $(PX4_IMAGE)"
	docker buildx build \
		--progress=plain \
		$(call docker_cache_from,$(PX4_CACHE_IMAGE)) \
		--cache-to type=inline \
		--build-arg PX4_REF="$(PX4_REF)" \
		--target px4 \
		-t $(PX4_IMAGE) \
		-t $(PX4_CACHE_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

.PHONY: build-all
build-all: build

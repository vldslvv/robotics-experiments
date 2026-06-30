########################################
# Image builds
########################################

.PHONY: build
build: build-isaac build-ros build-px4
	@echo "All images built/pulled."

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

.PHONY: print-airframes
print-airframes:
	@uv run python px4/scripts/generate_px4_airframes.py \
		--config-dir configs/airframes --print-manifest

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
build-all: build

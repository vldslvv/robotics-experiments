########################################
# Images / containers
########################################

# Isaac custom image
ISAAC_BASE_IMAGE	= nvcr.io/nvidia/isaac-sim
ISAAC_BASE_TAG		= 5.1.0

ISAAC_IMAGE_NAME      = vv-isaac-sim-custom
ISAAC_IMAGE_TAG       = $(ISAAC_BASE_TAG)
ISAAC_IMAGE           = $(ISAAC_IMAGE_NAME):$(ISAAC_IMAGE_TAG)
ISAAC_CONTAINER_NAME  = vv-isaac-sim-custom

# Host paths for persistence
ISAAC_HOME       = $(CURDIR)/isaac-sim
ISAAC_KIT_CACHE  = $(ISAAC_HOME)/cache/kit
ISAAC_OV_CACHE   = $(ISAAC_HOME)/cache/ov
ISAAC_PIP_CACHE  = $(ISAAC_HOME)/cache/pip
ISAAC_GLCACHE    = $(ISAAC_HOME)/cache/glcache
ISAAC_COMPUTEC   = $(ISAAC_HOME)/cache/computecache
ISAAC_LOGS       = $(ISAAC_HOME)/logs
ISAAC_DATA       = $(ISAAC_HOME)/data
ISAAC_DOCS       = $(ISAAC_HOME)/documents
SRC_DIR          = $(CURDIR)/src
SCANS_DIR        = $(CURDIR)/scans

# ROS 2 image / container
ROS_DISTRO          = jazzy

ROS_BASE_IMAGE      = ros
ROS_BASE_TAG        = $(ROS_DISTRO)

ROS_IMAGE_NAME      = vv-ros-$(ROS_DISTRO)-custom
ROS_IMAGE_TAG       = $(ROS_DISTRO)
ROS_IMAGE           = $(ROS_IMAGE_NAME):$(ROS_IMAGE_TAG)
ROS_CONTAINER_NAME  = vv-ros-$(ROS_DISTRO)-custom

# ROS 2 middleware config
ROS_DOMAIN_ID										= 42
RMW_IMPLEMENTATION							= rmw_fastrtps_cpp
FASTRTPS_DEFAULT_PROFILES_FILE	= /etc/ros/fastdds.xml

########################################
# Phony targets
########################################

.PHONY: xhost \
        build-isaac build-ros build-all \
        run-isaac run-ros \
        stop-isaac stop-ros stop-all \
        clean-isaac-container clean-ros-container clean-container \
        clean-isaac-image clean-ros-image clean-image clean-all

########################################
# X11 helper
########################################

xhost:
	@xhost +local:root >/dev/null 2>&1 || true

########################################
# Build images
########################################

build-isaac:
	@echo "Building custom Isaac image: $(ISAAC_IMAGE)"
	docker build \
		--build-arg DOCKER_REPO=$(ISAAC_BASE_IMAGE) \
		--build-arg DOCKER_TAG=$(ISAAC_BASE_TAG) \
		-t $(ISAAC_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

build-ros:
	@echo "Pulling ROS image: $(ROS_IMAGE)"
	docker build \
		--build-arg DOCKER_REPO=$(ROS_BASE_IMAGE) \
		--build-arg DOCKER_TAG=$(ROS_BASE_TAG) \
		-t $(ROS_IMAGE) \
		-f Dockerfile \
		$(CURDIR)

build-all: build-isaac build-ros
	@echo "All images built/pulled."

########################################
# Interactive runs (persistent containers)
########################################

# Run Isaac container interactively with /bin/bash
run-isaac: xhost build-isaac
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
			--entrypoint "" \
			-e ACCEPT_EULA=Y \
			-e OMNI_KIT_DISABLE_TELEMETRY=1 \
			-e DISPLAY=$(DISPLAY) \
			-e XDG_RUNTIME_DIR=$(XDG_RUNTIME_DIR) \
			-e QT_X11_NO_MITSHM=1 \
			-e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) \
			-e RMW_IMPLEMENTATION=$(RMW_IMPLEMENTATION) \
			-e ROS_DISTRO=jazzy \
			-e isaac_sim_package_path=/isaac-sim \
			-e LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib:$$LD_LIBRARY_PATH \
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
			-v $(SRC_DIR)/robotics-experiments:/isaac-sim/robotics-experiments \
			-v $(SCANS_DIR):/isaac-sim/scans \
			-v $(CURDIR)/.ros:/isaac-sim/.ros \
			$(ISAAC_IMAGE) /bin/bash -c "tmux new-session \; split-window -h \; attach"; \
	fi

# Run ROS 2 container interactively with bash (ROS env pre-sourced)
run-ros: build-ros
	@if docker ps -a --format '{{.Names}}' | grep -q '^$(ROS_CONTAINER_NAME)$$'; then \
		echo "Starting existing ROS2 container $(ROS_CONTAINER_NAME)..."; \
		docker start -ai $(ROS_CONTAINER_NAME); \
	else \
		echo "Creating and starting ROS2 container $(ROS_CONTAINER_NAME)..."; \
		docker run -it \
			--name "$(ROS_CONTAINER_NAME)" \
			--network=host \
			--gpus all \
			--entrypoint "" \
			-e DISPLAY="$(DISPLAY)" \
			-e XDG_RUNTIME_DIR="$(XDG_RUNTIME_DIR)" \
			-e QT_X11_NO_MITSHM=1 \
			-e ROS_DOMAIN_ID="$(ROS_DOMAIN_ID)" \
			-e RMW_IMPLEMENTATION="$(RMW_IMPLEMENTATION)" \
			-e FASTRTPS_DEFAULT_PROFILES_FILE=/isaac-sim/.ros/fastdds.xml \
			-v "$(XDG_RUNTIME_DIR)":"$(XDG_RUNTIME_DIR)" \
			-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
			-v $(CURDIR)/.ros:/isaac-sim/.ros \
			-v "$(HOME)/ros2_ws":/ros2_ws \
			"$(ROS_IMAGE)" \
			/bin/bash -c "source /opt/ros/jazzy/setup.bash && cd /ros2_ws && bash"; \
	fi

########################################
# Stop / clean
########################################

stop-isaac:
	@docker stop $(ISAAC_CONTAINER_NAME) 2>/dev/null || true

stop-ros:
	@docker stop $(ROS_CONTAINER_NAME) 2>/dev/null || true

stop-all: stop-isaac stop-ros
	@echo "Stopped Isaac and ROS containers (if running)."

clean-isaac-container:
	@docker rm -f $(ISAAC_CONTAINER_NAME) 2>/dev/null || true

clean-ros-container:
	@docker rm -f $(ROS_CONTAINER_NAME) 2>/dev/null || true

clean-container: clean-isaac-container clean-ros-container
	@echo "Removed Isaac and ROS containers (if they existed)."

clean-isaac-image:
	@docker rmi $(ISAAC_IMAGE) 2>/dev/null || true

clean-ros-image:
	@docker rmi $(ROS_IMAGE) 2>/dev/null || true

clean-images: clean-isaac-image clean-ros-image
	@echo "Removed Isaac and ROS images (if they existed)."

clean-all: stop-all clean-container clean-image
	@echo "All containers and images removed."

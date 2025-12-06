########################################
# Images / containers
########################################

# Isaac custom image
ISAAC_IMAGE_NAME      = vv-isaac-sim-custom
ISAAC_IMAGE_TAG       = 5.1.0
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
ROS_IMAGE_NAME      = osrf/ros
ROS_IMAGE_TAG       = humble-desktop
ROS_IMAGE           = $(ROS_IMAGE_NAME):$(ROS_IMAGE_TAG)
ROS_CONTAINER_NAME  = vv-ros2

# ROS 2 middleware config
ROS_DOMAIN_ID       = 42
RMW_IMPLEMENTATION  = rmw_fastrtps_cpp

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
	docker build -t $(ISAAC_IMAGE) -f Dockerfile .

build-ros:
	@echo "Pulling ROS image: $(ROS_IMAGE)"
	docker pull $(ROS_IMAGE)

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
			-e DISPLAY=$(DISPLAY) \
			-e XDG_RUNTIME_DIR=$(XDG_RUNTIME_DIR) \
			-e QT_X11_NO_MITSHM=1 \
			-e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) \
			-e RMW_IMPLEMENTATION=$(RMW_IMPLEMENTATION) \
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
			-e DISPLAY="$(DISPLAY)" \
			-e XDG_RUNTIME_DIR="$(XDG_RUNTIME_DIR)" \
			-e QT_X11_NO_MITSHM=1 \
			-e ROS_DOMAIN_ID="$(ROS_DOMAIN_ID)" \
			-e RMW_IMPLEMENTATION="$(RMW_IMPLEMENTATION)" \
			-v "$(XDG_RUNTIME_DIR)":"$(XDG_RUNTIME_DIR)" \
			-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
			-v "$(HOME)/ros2_ws":/ros2_ws \
			"$(ROS_IMAGE)" \
			bash -c "source /opt/ros/humble/setup.bash && cd /ros2_ws && bash"; \
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

clean-image: clean-isaac-image clean-ros-image
	@echo "Removed Isaac and ROS images (if they existed)."

clean-all: stop-all clean-container clean-image
	@echo "All containers and images removed."


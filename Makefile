IMAGE_NAME = nvcr.io/nvidia/isaac-sim
IMAGE_TAG  = 5.1.0
CONTAINER_NAME = vv-isaac-sim-gui

# Host paths for persistence
ISAAC_HOME = $(HOME)/docker/isaac-sim
KIT_CACHE = $(ISAAC_HOME)/cache/kit
OV_CACHE  = $(ISAAC_HOME)/cache/ov
PIP_CACHE = $(ISAAC_HOME)/cache/pip
GLCACHE   = $(ISAAC_HOME)/cache/glcache
COMPUTEC  = $(ISAAC_HOME)/cache/computecache
LOGS      = $(ISAAC_HOME)/logs
DATA      = $(ISAAC_HOME)/data
DOCS      = $(ISAAC_HOME)/documents

.PHONY: build run start stop clean-container clean-image clean-all

# ------------------------------------------------------------------------
# Build image
# ------------------------------------------------------------------------
build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

# ------------------------------------------------------------------------
# Run GUI container (persistent)
# ------------------------------------------------------------------------
run:
	docker run \
	  --name $(CONTAINER_NAME) \
	  --network=host \
	  --user 0:0 \
	  --gpus all \
		-e ACCEPT_EULA=Y \
	  -e DISPLAY=$(DISPLAY) \
	  -e XDG_RUNTIME_DIR=$(XDG_RUNTIME_DIR) \
	  -e QT_X11_NO_MITSHM=1 \
	  --device=/dev/dri \
	  -v $(XDG_RUNTIME_DIR):$(XDG_RUNTIME_DIR) \
	  -v $(KIT_CACHE):/isaac-sim/kit/cache \
	  -v $(OV_CACHE):/isaac-sim/.cache/ov \
	  -v $(PIP_CACHE):/isaac-sim/.cache/pip \
	  -v $(GLCACHE):/isaac-sim/.cache/nvidia/GLCache \
	  -v $(COMPUTEC):/isaac-sim/.nv/ComputeCache \
	  -v $(LOGS):/isaac-sim/.nvidia-omniverse/logs \
	  -v $(DATA):/isaac-sim/.local/share/ov/data \
	  -v $(DOCS):/isaac-sim/Documents \
	  -it $(IMAGE_NAME):$(IMAGE_TAG)

# ------------------------------------------------------------------------
# Start an already-created container
# ------------------------------------------------------------------------
start:
	docker start -ai $(CONTAINER_NAME)

# ------------------------------------------------------------------------
# Stop the container if running
# ------------------------------------------------------------------------
stop:
	docker stop $(CONTAINER_NAME)

# ------------------------------------------------------------------------
# Remove the container completely
# ------------------------------------------------------------------------
clean-container:
	docker rm -f $(CONTAINER_NAME)

# ------------------------------------------------------------------------
# Remove image
# ------------------------------------------------------------------------
clean-image:
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG)

# ------------------------------------------------------------------------
# Remove container and image (full cleanup)
# ------------------------------------------------------------------------
clean-all: clean-container clean-image
	@echo "Container and image removed."

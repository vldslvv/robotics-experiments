# This dockerfile installs common linux utilities on top of ubuntu-based base image.

ARG DOCKER_REPO
ARG DOCKER_TAG
FROM ${DOCKER_REPO}:${DOCKER_TAG} AS base

ENV ORIGINAL_USER=${USER}

# # Verify ORIGINAL_USER is set; fail build if empty
# RUN if [ -z "$ORIGINAL_USER" ]; then echo "ERROR: USER is not set in the base image; cannot restore original user." >&2; exit 1; fi

# Run as root (you need this due to NVML/Vulkan permissions)
USER 0

# Versions are not pinned because this logic can run on both Ubuntu 22 and 24
# TODO: consider separate sets of versions for each version
# TODO: upgrade packages
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    tmux \
    ranger \
    htop \
    vim \
    git

# Restore the original user
USER ${ORIGINAL_USER}

FROM base AS isaac-sim

RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT []

FROM base AS ros
ARG ROS_DISTRO="humble"
# Use bash for RUN so that sourcing setup.bash works
SHELL ["/bin/bash", "-lc"]

# WARNING: On jazzy, ros-${ROS_DISTRO}-topic-based-ros2-control cannot be installed from registry
# Building from sources fails with deprecation warning. Best to use humble for now, but
# keep the ability to switch to jazzy easily
RUN apt-get install --no-install-recommends -y \
    ros-${ROS_DISTRO}-topic-based-ros2-control && \
    git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git && \
    cd IsaacSim-ros_workspaces/${ROS_DISTRO}_ws && \
    git submodule update --init --recursive

# This step has to be separate from previous
# Quote from # https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html#setting-up-workspaces
# cd humble_ws
# git submodule update --init --recursive # If using docker, perform this step outside the container and relaunch the container
# rosdep install -i --from-path src --rosdistro humble -y
RUN cd IsaacSim-ros_workspaces/${ROS_DISTRO}_ws && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep install -i --from-path src --rosdistro ${ROS_DISTRO} -y && \
    colcon build

# TODO:consider exporting this automatically
# cd humble_ws
# source IsaacSim-ros_workspaces/${ROS_DISTRO}_ws/install/local_setup.bash

# Commands above require apt indices populated
# RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT []

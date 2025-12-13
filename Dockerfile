# This dockerfile installs common linux utilities on top of ubuntu-based base image.

ARG DOCKER_REPO
ARG DOCKER_TAG
FROM ${DOCKER_REPO}:${DOCKER_TAG} AS base

ENV ORIGINAL_USER=${USER}

# Run as root (you need this due to NVML/Vulkan permissions)
USER 0

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    tmux=3.4-1ubuntu0.1 \
    ranger=1.9.3-5ubuntu0.24.04.1 \
    htop=3.3.0-4build1 \
    vim=2:9.1.0016-1ubuntu7.9 \
    git \
    && \
    rm -rf /var/lib/apt/lists/*

# Restore the original user
USER ${ORIGINAL_USER}

FROM base AS isaac-sim

ENTRYPOINT []

FROM base AS ros

# RUN apt-get update && \
#     apt-get install --no-install-recommends -y \
#     ros-jazzy-topic-based-ros2-control \
#     ros-jazzy-ros2-control ros-jazzy-ros2-controllers
#     && \
#     rm -rf /var/lib/apt/lists/*
#
# RUN git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git && \
#     cd IsaacSim-ros_workspaces/jazzy_ws && \
#     git submodule update --init --recursive && \
#     rosdep install -i --from-path src --rosdistro jazzy -y && \
#     colcon build

RUN git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git && \
    cd IsaacSim-ros_workspaces/jazzy_ws && \
    git submodule update --init --recursive

ENTRYPOINT []

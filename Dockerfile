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

FROM base AS ros-base
ARG ROS_DISTRO="humble"
# Use bash for RUN so that sourcing setup.bash works
SHELL ["/bin/bash", "-lc"]

# TODO: install python 3.11, this is for humble, maybe use pyenv instead
# Install Python 3.11 in ROS layer without altering system default python3
# Uses deadsnakes PPA which supports Ubuntu 22.04 and 24.04
# RUN apt-get update && \
#     apt-get install --no-install-recommends -y software-properties-common && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install --no-install-recommends -y \
#         python3.11 \
#         python3.11-venv \
#         python3.11-dev \
#         python3-pip && \
#     python3.11 -m pip install --upgrade pip

RUN apt-get install --no-install-recommends -y \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-ackermann-msgs \
    python3-rosdep build-essential \
    python3-colcon-common-extensions

# WARNING: On jazzy, ros-${ROS_DISTRO}-topic-based-ros2-control cannot be installed from registry
# Building from sources fails with deprecation warning. Best to use humble for now, but
# keep the ability to switch to jazzy easily
# RUN apt-get install --no-install-recommends -y \
#     ros-${ROS_DISTRO}-topic-based-ros2-control && \

WORKDIR /topic_based_ros2_control_ws
# We have to install topic_based_ros2_control from source, for jazzy it's mandatory as there's no
# apt package for it, ros-jazzy-topic-based-ros2-control
# Used as reference: https://github.com/PickNikRobotics/topic_based_ros2_control/blob/main/doc/installation.md
# The approach is to layer three workspaces: base -> where topic is built -> where isaac examples are
# To build this workspace, activate a base ros workspace first, then perform installation
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    export COLCON_WS=$(pwd) && \
    mkdir -p ${COLCON_WS}/src && \
    cd ${COLCON_WS}/src && \
    git clone https://github.com/PickNikRobotics/topic_based_ros2_control && \
    rosdep install --ignore-src --from-paths . -y && \
    cd ${COLCON_WS} && \
    colcon build --symlink-install --event-handlers log-

 WORKDIR /
 RUN git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git isaacsim_ros_workspaces && \
     cd isaacsim_ros_workspaces/${ROS_DISTRO}_ws && \
     git submodule update --init --recursive

 WORKDIR /isaacsim_ros_workspaces/${ROS_DISTRO}_ws
 # Install instructions tell to restart docker, therefore we run this in a separate RUN
 # First, source two workspaces, ros base and topic_based_ros2_control
 RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
     source /topic_based_ros2_control_ws/install/setup.bash && \
     rosdep install -i --from-path src --rosdistro jazzy -y && \
     colcon build

# This step has to be separate from previous
# Quote from # https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html#setting-up-workspaces
# cd humble_ws
# git submodule update --init --recursive # If using docker, perform this step outside the container and relaunch the container
# rosdep install -i --from-path src --rosdistro humble -y
# RUN cd IsaacSim-ros_workspaces/${ROS_DISTRO}_ws && \
#     source /opt/ros/${ROS_DISTRO}/setup.bash && \
#     rosdep install -i --from-path src --rosdistro ${ROS_DISTRO} -y && \
#     colcon build

# TODO:consider exporting this automatically
# cd ${ROS_DISTRO}_ws
# source IsaacSim-ros_workspaces/${ROS_DISTRO}_ws/install/local_setup.bash

# Commands above require apt indices populated
# RUN rm -rf /var/lib/apt/lists/*

FROM ros-base AS ros

RUN echo "we're here"

ENTRYPOINT []

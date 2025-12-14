# This dockerfile installs common linux utilities on top of ubuntu-based base image.

ARG DOCKER_REPO
ARG DOCKER_TAG
FROM ${DOCKER_REPO}:${DOCKER_TAG} AS base

ENV ORIGINAL_USER=${USER}

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
    ca-certificates \
    git

# Restore the original user
USER ${ORIGINAL_USER}

FROM base AS isaac-sim

# Optional, unless we care about image size
# RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["tmux"]
CMD ["new-session", ";", "split-window", "-h", ";", "attach"]

FROM base AS ros
ARG ROS_DISTRO="jazzy"
# Use bash for RUN so that sourcing setup.bash works
SHELL ["/bin/bash", "-lc"]

RUN apt-get install --no-install-recommends -y \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-ackermann-msgs \
    python3-rosdep build-essential \
    python3-colcon-common-extensions

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

# Commands above require apt indices populated
# Optional, unless we care about image size
# RUN rm -rf /var/lib/apt/lists/*

# TODO: consider using ./ros_entrypoint.sh
# Source all three workspaces we have in order
ENTRYPOINT ["/bin/bash", "-c", "\
  source /opt/ros/${ROS_DISTRO}/setup.bash && \
  source /topic_based_ros2_control_ws/install/setup.bash && \
  source /isaacsim_ros_workspaces/${ROS_DISTRO}_ws/install/setup.bash && \
  cd /ros2_ws && \
  exec \"$@\" \
", "--"]
CMD ["tmux", "new-session", "-A", "-s", "ros"]

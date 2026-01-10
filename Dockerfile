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

FROM base AS ros-base
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

FROM ros-base AS ros

RUN apt-get install ros-${ROS_DISTRO}-teleop-twist-keyboard

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

FROM base AS nvim
# Build Neovim from the latest tag
# Use bash for more complex RUN logic
SHELL ["/bin/bash", "-lc"]

# Elevate privileges to install build dependencies
USER 0

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  ninja-build \
  gettext \
  cmake \
  unzip \
  curl \
  git \
  build-essential \
  ca-certificates && \
  update-ca-certificates

# TODO: consider using same neovim version as host
# Clone and build Neovim
WORKDIR /opt
RUN git clone https://github.com/neovim/neovim.git && \
  cd neovim && \
  git fetch --tags && \
  LATEST_TAG=$(git describe --tags "$(git rev-list --tags --max-count=1)") && \
  echo "Checking out Neovim tag: ${LATEST_TAG}" && \
  git checkout "${LATEST_TAG}" && \
  make CMAKE_BUILD_TYPE=RelWithDebInfo && \
  make install && \
  echo "alias vim='nvim'" >> $HOME/.bashrc

# Optionally slim the layer by removing build tree
RUN rm -rf /opt/neovim

# Restore original user
ARG UID=1000 GID=1000
RUN mkdir -p /home/ubuntu/.config /home/ubuntu/.local/{share,state} /home/ubuntu/.cache && chown -R $UID:$GID /home/ubuntu
USER ubuntu

# Install nvm and Node.js v25 before cloning Neovim
# Reference: https://github.com/nvm-sh/nvm
# Note: We use bash SHELL; ensure nvm is sourced in the same RUN where needed
RUN export NVM_DIR="$HOME/.nvm" && \
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
  export NVM_DIR="$HOME/.nvm" && \
  source "$NVM_DIR/nvm.sh" && \
  nvm install 25 && \
  nvm alias default 25 && \
  node -v && npm -v

# RUN chown -R ubuntu:ubuntu /home/ubuntu

# TODO: mount directly from host, along with .local/share/nvim, .local/state/nvim and .cache/nvim
# Clone config repo to $HOME/.config/nvim
# RUN git clone -b patch/custom https://github.com/vldslvv/lazyvim-starter $HOME/.config/nvim

# ARG BUILD_ROOT_DIR=mybuild/docker
# COPY ${BUILD_ROOT_DIR}/.config/nvim/ /home/ubuntu/.config/nvim

ENTRYPOINT ["/bin/bash", "-c"]


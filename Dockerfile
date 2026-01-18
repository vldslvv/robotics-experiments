# This dockerfile installs common linux utilities on top of ubuntu-based base image.

ARG DOCKER_REPO
ARG DOCKER_TAG
FROM ${DOCKER_REPO}:${DOCKER_TAG} AS base

# Run as root (need this due to NVML/Vulkan permissions)
USER 0
ENV HOME=/root

# Versions are not pinned because this logic can run on both Ubuntu 22 and 24
# TODO: consider separate sets of versions for each version (ROS2 jazzy or humble)
# TODO: upgrade packages
RUN apt-get update && \
  apt-get install --no-install-recommends -y \
  tmux \
  ranger \
  htop \
  vim \
  ca-certificates \
  git \
  ripgrep \
  file \
  less \
  tree \
  fd-find


FROM base AS zsh

RUN apt-get update && apt-get install -y --no-install-recommends \
    zsh git curl ca-certificates \
    zsh-syntax-highlighting

# Cache location
ENV ZSH_CACHE_DIR=/cache/zsh

# Create cache dir
RUN mkdir -p /cache/zsh

# Installs into $HOME/.oh-my-zsh and creates $HOME/.zshrc (we'll overwrite)
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Enable autosuggestions
RUN git clone https://github.com/zsh-users/zsh-autosuggestions \
    ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Enable VI mode
RUN git clone https://github.com/jeffreytse/zsh-vi-mode \
    ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-vi-mode

# --- write a deterministic .zshrc ---
# Enables completion, autosuggestions, syntax highlighting, and uses persistent cache paths.
RUN cat > /root/.zshrc <<'EOF'
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"

# Where zsh writes caches (compdump, etc.)
export ZSH_CACHE_DIR="${ZSH_CACHE_DIR:-$HOME/.cache/zsh}"
mkdir -p "$ZSH_CACHE_DIR"

# Keep history in the mounted cache dir
export HISTFILE="$ZSH_CACHE_DIR/zsh_history"
export HISTSIZE=50000
export SAVEHIST=50000
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_FIND_NO_DUPS
setopt SHARE_HISTORY

plugins=(
  git
  zsh-autosuggestions
  zsh-vi-mode
)

# Ensure completion cache dir is secure
export ZSH_CACHE_DIR="${ZSH_CACHE_DIR:-/cache/zsh}"
mkdir -p "$ZSH_CACHE_DIR"

# Only attempt chown if we can (may fail on some bind mounts)
if command -v chown >/dev/null 2>&1; then
  chown -R "$(id -u)":"$(id -g)" "$ZSH_CACHE_DIR" 2>/dev/null || true
fi

chmod -R go-w "$ZSH_CACHE_DIR" 2>/dev/null || true

source "$ZSH/oh-my-zsh.sh"

# --- completions ---
autoload -Uz compinit
# -d forces compdump location (persisted)
compinit -d "$ZSH_CACHE_DIR/zcompdump"

# --- extras installed via apt ---
source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh

# Make completion faster / reduce disk churn
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path "$ZSH_CACHE_DIR"

# Useful defaults
setopt AUTO_CD
setopt INTERACTIVE_COMMENTS
EOF

ENV SHELL=/usr/bin/zsh
RUN chmod 755 /usr/local/share/zsh

# If neovim is not needed, this step can be skipped
FROM zsh AS nvim
SHELL ["/bin/bash", "-lc"]

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

WORKDIR /opt
ARG NVIM_TAG=stable
RUN git clone --depth 1 --branch ${NVIM_TAG} https://github.com/neovim/neovim.git && \
    cd neovim && \
    echo "Building Neovim tag: ${NVIM_TAG}" && \
    make CMAKE_BUILD_TYPE=RelWithDebInfo && \
    make install && \
    cd ..&& \
    rm -rf neovim && \
    rm -rf /opt/neovim

# Install nvm and Node.js v25 before cloning Neovim
# Reference: https://github.com/nvm-sh/nvm
# Note: We use bash SHELL; ensure nvm is sourced in the same RUN where needed
RUN export NVM_DIR="${HOME}/.nvm" && \
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
  source "$NVM_DIR/nvm.sh" && \
  nvm install 25 && \
  nvm alias default 25

# For escape delay, see if there are any issues
# https://superuser.com/questions/942677/consequences-of-escape-time-0-tmux-setting
RUN echo "alias vim='nvim'" >> ${HOME}/.bashrc && \
  echo 'set -g default-terminal "tmux-256color"' >> ${HOME}/.tmux.conf && \
  echo 'set -ga terminal-overrides ",*:Tc"' >> ${HOME}/.tmux.conf && \
  echo 'set -sg escape-time 0' >> ${HOME}/.tmux.conf


FROM nvim AS isaac-sim

# Fix for wrong characters being displayed in Neovim, and possible other issues
RUN apt-get update && apt-get install -y --no-install-recommends locales \
 && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen \
 && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /isaac-sim

# The problem being solved:
# https://forums.developer.nvidia.com/t/python-autocomplete-not-working-in-isaacsim-vscode-extension/329010
# Uses custom environment script to be able to use LSP properly
RUN echo "alias pyvim='source /isaac-sim/proj/scripts/set_python_env.sh && nvim'" >> ${HOME}/.bashrc
RUN echo "alias pyvim='source /isaac-sim/proj/scripts/set_python_env.sh && nvim'" >> ${HOME}/.zshrc

# This script will populate symlinks for editor's LSP to work
ARG SCRIPTS_DIR
COPY ${SCRIPTS_DIR}/make_symlinks.sh make_symlinks.sh
RUN chmod +x make_symlinks.sh

# If zsh setup is skipped, just use /bin/bash here
ENTRYPOINT ["/usr/bin/zsh", "-lc", "/isaac-sim/make_symlinks.sh || true; exec tmux \"$@\"", "tmux"]
CMD ["new-session", "/usr/bin/zsh", "-l", ";", "split-window", "-h", "/usr/bin/zsh", "-l", ";", "attach"]

FROM nvim AS ros-base

ARG ROS_DISTRO="jazzy"
# Use bash for RUN so that sourcing setup.bash works
SHELL ["/bin/bash", "-lc"]

RUN apt-get install --no-install-recommends -y \
  ros-${ROS_DISTRO}-vision-msgs \
  ros-${ROS_DISTRO}-ackermann-msgs \
  ros-${ROS_DISTRO}-teleop-twist-keyboard \
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


FROM ros-base AS ros

# TODO: consider using ./ros_entrypoint.sh
# Source all three workspaces we have in order
# ENTRYPOINT ["/bin/bash", "-c", "\
#   source /opt/ros/${ROS_DISTRO}/setup.bash && \
#   source /topic_based_ros2_control_ws/install/setup.bash && \
#   source /isaacsim_ros_workspaces/${ROS_DISTRO}_ws/install/setup.bash && \
#   cd /ros2_ws && \
#   exec \"$@\" \
#   ", "--"]
# CMD ["tmux", "new-session", "-A", "-s", "ros"]
ENTRYPOINT ["/usr/bin/zsh", "-lc", "\
  source /opt/ros/${ROS_DISTRO}/setup.zsh && \
  source /topic_based_ros2_control_ws/install/setup.zsh && \
  source /isaacsim_ros_workspaces/${ROS_DISTRO}_ws/install/setup.zsh && \
  cd /ros2_ws && \
  exec \"$@\" \
", "--"]
CMD ["tmux", "new-session", "-A", "-s", "ros"]


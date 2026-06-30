# This dockerfile installs common linux utilities on top of ubuntu-based base image.

# Defaults keep the `FROM ${DOCKER_REPO}:${DOCKER_TAG}` reference valid when this
# file is parsed for unrelated targets (e.g. `--target px4`); real builds of the
# isaac/ros chains override these via --build-arg.
ARG DOCKER_REPO=ubuntu
ARG DOCKER_TAG=22.04
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
  fd-find \
  python3-pip \
  python3-virtualenv


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
    "${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions"

# Enable VI mode
RUN git clone https://github.com/jeffreytse/zsh-vi-mode \
    "${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-vi-mode"

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
SHELL ["/bin/bash", "-o", "pipefail", "-lc"]

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
COPY docker/tmux.conf /root/.tmux.conf
RUN echo "alias vim='nvim'" >> "${HOME}/.bashrc"


FROM nvim AS ai-tools
SHELL ["/bin/bash", "-o", "pipefail", "-lc"]

# hadolint ignore=SC2016
RUN curl -fsSL https://claude.ai/install.sh | bash && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc

RUN export NVM_DIR="${HOME}/.nvm" && \
    source "$NVM_DIR/nvm.sh" && \
    npm install -g @openai/codex


FROM ai-tools AS isaac-sim

# Fix for wrong characters being displayed in Neovim, and possible other issues
RUN apt-get update && apt-get install -y --no-install-recommends locales \
 && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen \
 && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /isaac-sim

# Install hydra/omegaconf and their full runtime closure into the kit python's
# OWN site-packages. --ignore-installed is load-bearing: Isaac bundles several of
# these deps (PyYAML, packaging, antlr4) only on the kit-app augmented sys.path
# (omni.pip.compute pip_prebundle), so a plain install sees them already
# satisfied and skips them -- leaving the bare /isaac-sim/kit/python/bin/python3
# interpreter that runapp_custom.sh uses to resolve get_config().scene unable to
# import them. That failure silently empties SCENE_PATH and launches Isaac with
# no stage. Forcing the whole closure in keeps the standalone interpreter
# self-sufficient.
RUN PIP_ROOT_USER_ACTION=ignore ./python.sh -m pip install \
    --no-cache-dir \
    --disable-pip-version-check \
    --ignore-installed \
    "hydra-core>=1.3" \
    "omegaconf>=2.3"

# The problem being solved:
# https://forums.developer.nvidia.com/t/python-autocomplete-not-working-in-isaacsim-vscode-extension/329010
# Uses custom environment script to be able to use LSP properly
RUN echo "alias pyvim='source /isaac-sim/proj/isaac/runtime/standalone/set_python_env.sh && nvim'" >> "${HOME}/.bashrc"
RUN echo "alias pyvim='source /isaac-sim/proj/isaac/runtime/standalone/set_python_env.sh && nvim'" >> "${HOME}/.zshrc"

# This script will populate symlinks for editor's LSP to work
ARG SCRIPTS_DIR
COPY ${SCRIPTS_DIR}/make_symlinks.sh make_symlinks.sh
RUN chmod +x make_symlinks.sh

# If zsh setup is skipped, just use /bin/bash here
ENTRYPOINT ["/usr/bin/zsh", "-lc", "/isaac-sim/make_symlinks.sh || true; exec tmux \"$@\"", "tmux"]
CMD ["new-session", "/usr/bin/zsh", "-l", ";", "split-window", "-h", "/usr/bin/zsh", "-l", ";", "attach"]

FROM ai-tools AS ros-base

ARG ROS_DISTRO="jazzy"
# Use bash for RUN so that sourcing setup.bash works
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install --no-install-recommends -y \
  ros-${ROS_DISTRO}-vision-msgs \
  ros-${ROS_DISTRO}-ackermann-msgs \
  ros-${ROS_DISTRO}-teleop-twist-keyboard \
  ros-${ROS_DISTRO}-navigation2 \
  ros-${ROS_DISTRO}-nav2-bringup \
  python3-rosdep build-essential \
  python3-colcon-common-extensions

WORKDIR /topic_based_ros2_control_ws
# We have to install topic_based_ros2_control from source, for jazzy it's mandatory as there's no
# apt package for it, ros-jazzy-topic-based-ros2-control
# Used as reference: https://github.com/PickNikRobotics/topic_based_ros2_control/blob/main/doc/installation.md
# The approach is to layer three workspaces: base -> where topic is built -> where isaac examples are
# To build this workspace, activate a base ros workspace first, then perform installation
RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" && \
  COLCON_WS="$(pwd)" && export COLCON_WS && \
  mkdir -p "${COLCON_WS}/src" && \
  cd "${COLCON_WS}/src" && \
  git clone https://github.com/PickNikRobotics/topic_based_ros2_control && \
  rosdep install --ignore-src --from-paths . -y && \
  cd "${COLCON_WS}" && \
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
  PIP_BREAK_SYSTEM_PACKAGES=1 rosdep install -i --from-path src --rosdistro jazzy -y && \
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


# PX4 SITL stands in for the flight controller on a real drone; kept in its own
# container (not in isaac-sim or ros) to mirror the hardware split.
FROM ubuntu:22.04 AS px4

ARG PX4_TAG=v1.14.3
ENV PX4_DIR=/opt/PX4-Autopilot
ENV PX4_AIRFRAME_MANIFEST=/usr/local/share/robotics-experiments/px4-airframes.json
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
      git make cmake ninja-build build-essential ca-certificates \
      python3-pip python3-jinja2 python3-toml python3-numpy \
      python3-future python3-jsonschema python3-packaging \
      python3-setuptools python3-wheel python3-yaml \
      lsb-release sudo gosu \
      tmux ranger less vim \
 && rm -rf /var/lib/apt/lists/*

COPY docker/tmux.conf /root/.tmux.conf

RUN printf '%s\n' \
      "alias px4-iris='/usr/local/bin/run_px4_iris_fresh'" \
      >> /root/.bashrc

RUN pip3 install --no-cache-dir \
      kconfiglib empy==3.3.4 pyros-genmsg

RUN git clone https://github.com/PX4/PX4-Autopilot.git ${PX4_DIR} && \
    cd ${PX4_DIR} && git checkout ${PX4_TAG} && \
    git submodule update --init --recursive

# Generate the repo-managed posix airframes from configs/airframes before the
# SITL firmware build. The none_* model names still use PX4's external
# simulator_mavlink path; only the control-allocation scripts are generated.
COPY px4/scripts/generate_px4_airframes.py /usr/local/bin/generate_px4_airframes.py
COPY configs/airframes /tmp/robotics-experiments/airframes
RUN python3 /usr/local/bin/generate_px4_airframes.py \
      --config-dir /tmp/robotics-experiments/airframes \
      --output-dir ${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes \
      --cmake-list ${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt \
      --manifest ${PX4_AIRFRAME_MANIFEST}

# Headless SITL build; Pegasus provides dynamics over MAVLink, so no built-in sim backend.
# Build the firmware only — omit the sim target (e.g. `none`) since it would launch
# the nsh shell during `docker build` even with DONT_RUN=1 on PX4 v1.14.x.
RUN cd ${PX4_DIR} && make px4_sitl_default

COPY px4/docker/run_px4_iris_fresh.sh /usr/local/bin/run_px4_iris_fresh
COPY px4/docker/run_px4_iris_ci.sh /usr/local/bin/run_px4_iris_ci
COPY px4/docker/start_px4_tmux.sh /usr/local/bin/start_px4_tmux
COPY px4/docker/px4_iris_common.sh /usr/local/lib/px4/
COPY px4/docker/reset_px4_state.sh /usr/local/lib/px4/
RUN chmod +x \
      /usr/local/bin/run_px4_iris_fresh \
      /usr/local/bin/run_px4_iris_ci \
      /usr/local/bin/start_px4_tmux \
      /usr/local/lib/px4/px4_iris_common.sh \
      /usr/local/lib/px4/reset_px4_state.sh

WORKDIR ${PX4_DIR}
# Drop into an attached tmux with PX4 SITL running in one pane and a shell in another.
CMD ["/usr/local/bin/start_px4_tmux"]

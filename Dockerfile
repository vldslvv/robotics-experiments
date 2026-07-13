# This dockerfile installs common linux utilities on top of ubuntu-based base image.

# Defaults keep the `FROM ${DOCKER_REPO}:${DOCKER_TAG}` reference valid when this
# file is parsed for unrelated targets (e.g. `--target px4`); real builds of the
# isaac/ros chains override these via --build-arg.
ARG DOCKER_REPO=ubuntu
ARG DOCKER_TAG=22.04
ARG DOCKER_DIGEST=sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982
FROM ${DOCKER_REPO}:${DOCKER_TAG}@${DOCKER_DIGEST} AS base

# Run as root (need this due to NVML/Vulkan permissions)
USER 0
ENV HOME=/root

RUN apt-get update && \
  apt-get install --no-install-recommends -y \
    ca-certificates \
    curl \
    git \
    less \
    tmux \
 && rm -rf /var/lib/apt/lists/*

COPY docker/tmux.conf /root/.tmux.conf


# Terminal environment used by the interactive Isaac and ROS images. These
# shell customizations intentionally follow their upstream default branches so
# rebuilding an image refreshes the interactive experience.
FROM base AS terminal-tools

RUN apt-get update && \
  apt-get install --no-install-recommends -y \
    fd-find \
    file \
    htop \
    ranger \
    ripgrep \
    tree \
    zsh \
    zsh-syntax-highlighting \
 && rm -rf /var/lib/apt/lists/*

ENV SHELL=/usr/bin/zsh
ENV ZSH_CACHE_DIR=/cache/zsh
RUN mkdir -p /cache/zsh

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    "" --unattended
RUN git clone https://github.com/zsh-users/zsh-autosuggestions \
    "${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions" && \
  git clone https://github.com/jeffreytse/zsh-vi-mode \
    "${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-vi-mode"

RUN cat > /root/.zshrc <<'EOF'
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"

export ZSH_CACHE_DIR="${ZSH_CACHE_DIR:-/cache/zsh}"
mkdir -p "$ZSH_CACHE_DIR"
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

export ZSH_COMPDUMP="$ZSH_CACHE_DIR/zcompdump"
source "$ZSH/oh-my-zsh.sh"
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path "$ZSH_CACHE_DIR"
setopt AUTO_CD
setopt INTERACTIVE_COMMENTS

source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
EOF


# Interactive tools shared by the Isaac and ROS images. Neovim follows the host
# version supplied by Make so the mounted host configuration and caches run
# against the same editor release.
FROM terminal-tools AS development-tools
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
  xz-utils && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /opt
ARG NVIM_TAG
RUN test -n "${NVIM_TAG}" && \
    git clone --depth 1 --branch "${NVIM_TAG}" https://github.com/neovim/neovim.git && \
    cd neovim && \
    make CMAKE_BUILD_TYPE=RelWithDebInfo && \
    make install && \
    rm -rf /opt/neovim

ARG NODE_VERSION=22.17.0
ARG NODE_SHA256=325c0f1261e0c61bcae369a1274028e9cfb7ab7949c05512c5b1e630f7e80e12
RUN curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" && \
    echo "${NODE_SHA256}  node-v${NODE_VERSION}-linux-x64.tar.xz" | sha256sum -c - && \
    tar -xJf "node-v${NODE_VERSION}-linux-x64.tar.xz" --strip-components=1 -C /usr/local && \
    rm "node-v${NODE_VERSION}-linux-x64.tar.xz"

ARG CODEX_VERSION=0.142.5
ARG CLAUDE_CODE_VERSION=2.1.205
RUN npm install -g \
    "@openai/codex@${CODEX_VERSION}" \
    "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" && \
    echo "alias vim='nvim'" >> "${HOME}/.bashrc" && \
    echo "alias vim='nvim'" >> "${HOME}/.zshrc"


FROM development-tools AS isaac-sim

# Fix for wrong characters being displayed in Neovim, and possible other issues
RUN apt-get update && apt-get install -y --no-install-recommends locales \
 && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen \
 && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

WORKDIR /isaac-sim

# Install the pinned config and MAVLink runtime closure into the kit python's
# OWN site-packages. --ignore-installed is load-bearing: Isaac bundles several of
# these deps (PyYAML, packaging, antlr4) only on the kit-app augmented sys.path
# (omni.pip.compute pip_prebundle), so a plain install sees them already
# satisfied and skips them -- leaving the bare /isaac-sim/kit/python/bin/python3
# interpreter that runapp_custom.sh uses to resolve get_config().scene unable to
# import them. That failure silently empties SCENE_PATH and launches Isaac with
# no stage. Forcing the whole closure in keeps the standalone interpreter
# self-sufficient.
COPY isaac/docker/runtime-requirements.txt /tmp/isaac-runtime-requirements.txt
RUN PIP_ROOT_USER_ACTION=ignore ./python.sh -m pip install \
    --no-cache-dir \
    --disable-pip-version-check \
    --ignore-installed \
    -r /tmp/isaac-runtime-requirements.txt && \
    rm /tmp/isaac-runtime-requirements.txt && \
    ./python.sh -c "import hydra, omegaconf, pymavlink"

ARG SCRIPTS_DIR
COPY ${SCRIPTS_DIR}/make_symlinks.sh make_symlinks.sh
RUN chmod +x make_symlinks.sh && \
    echo "alias pyvim='source /isaac-sim/proj/isaac/runtime/standalone/set_python_env.sh && nvim'" \
      >> "${HOME}/.bashrc" && \
    echo "alias pyvim='source /isaac-sim/proj/isaac/runtime/standalone/set_python_env.sh && nvim'" \
      >> "${HOME}/.zshrc"
ENTRYPOINT ["/usr/bin/zsh", "-lc", "/isaac-sim/make_symlinks.sh || true; exec tmux \"$@\"", "tmux"]
CMD ["new-session", "/usr/bin/zsh", "-l", ";", "split-window", "-h", "/usr/bin/zsh", "-l", ";", "attach"]


FROM development-tools AS ros-base

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
ARG TOPIC_BASED_ROS2_CONTROL_REF=6bd8d55e1c4ad3188770fe5c8b93b942bcede4a2
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
  cd topic_based_ros2_control && git checkout "${TOPIC_BASED_ROS2_CONTROL_REF}" && cd .. && \
  rosdep install --ignore-src --from-paths . -y && \
  cd "${COLCON_WS}" && \
  colcon build --symlink-install --event-handlers log-

WORKDIR /
ARG ISAAC_ROS_WORKSPACES_REF=dd3eeede7912755996a18f4884285d9f50843f79
RUN git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git isaacsim_ros_workspaces && \
  cd isaacsim_ros_workspaces && git checkout "${ISAAC_ROS_WORKSPACES_REF}" && \
  cd / && \
  cd isaacsim_ros_workspaces/${ROS_DISTRO}_ws && \
  git submodule update --init --recursive

WORKDIR /isaacsim_ros_workspaces/${ROS_DISTRO}_ws
# Install instructions tell to restart docker, therefore we run this in a separate RUN
# First, source two workspaces, ros base and topic_based_ros2_control
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
  source /topic_based_ros2_control_ws/install/setup.bash && \
  rosdep install -i --from-path src --rosdistro jazzy -y \
    --skip-keys python3-pytorch-pip && \
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

# Isaac Lab layers on top of Isaac Sim: it shares the same kit python
# (/isaac-sim/python.sh), USD, and PhysX and cannot run standalone, so it is a
# superset of the isaac-sim stage rather than a separate image. Only the Phase F
# RL/DirectRLEnv workflow needs it (the test-integration-press-hover-goto gate and
# future F2/F3 training); the other integration targets stay on the leaner
# isaac-sim image. ISAACLAB_REF is pinned for reproducibility -- v2.3.2 is the
# Isaac Lab release built on Isaac Sim 5.1 (and it ships multirotor support).
# Override with --build-arg ISAACLAB_REF=... when moving Isaac Sim versions.
FROM isaac-sim AS isaac-lab

ARG ISAACLAB_REF=37ddf626871758333d6ed89cf64ad702aef127d0
# Build prerequisites: cmake/build-essential for the source packages / RL
# frameworks the full install resolves (some compile), and ncurses-term for the
# full terminfo DB. isaaclab.sh starts with `tabs 4`, which resolves the
# `ansi+tabs` terminfo building block; the base image's minimal terminfo lacks it,
# so without ncurses-term the install aborts with "'ansi+tabs': unknown terminal
# type".
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential ncurses-term \
 && rm -rf /var/lib/apt/lists/*
# Install only the project-used Isaac Lab base and rsl_rl adapter. The upstream
# `isaaclab.sh --install` default installs every RL framework and resolves broad
# dependency ranges at build time. The pinned requirements file plus --no-deps
# editable installs keep this image reproducible and omit unused frameworks and
# device-retargeting stacks.
COPY isaac/docker/lab-runtime-requirements.txt /tmp/isaac-lab-requirements.txt
RUN git clone https://github.com/isaac-sim/IsaacLab.git /isaac-lab \
 && cd /isaac-lab \
 && git -c advice.detachedHead=false checkout "${ISAACLAB_REF}" \
 && ln -s /isaac-sim _isaac_sim \
 && /isaac-sim/python.sh -m pip install --disable-pip-version-check --no-deps \
      --no-build-isolation \
      -r /tmp/isaac-lab-requirements.txt \
 && /isaac-sim/python.sh -m pip install --disable-pip-version-check --no-deps \
      --no-build-isolation -e source/isaaclab -e source/isaaclab_rl \
 && rm /tmp/isaac-lab-requirements.txt \
 && /isaac-sim/python.sh -c \
      "import flatdict, gymnasium, isaaclab, isaaclab_rl, rsl_rl; print('isaaclab OK:', isaaclab.__file__)"


# PX4 SITL stands in for the flight controller on a real drone; kept in its own
# container (not in isaac-sim or ros) to mirror the hardware split.
FROM ubuntu:22.04@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982 AS px4

ARG PX4_REF=1dacb4cdef2d7145754fc788fa8dc482eed74b40
ENV PX4_DIR=/opt/PX4-Autopilot
ENV PX4_AIRFRAME_MANIFEST=/usr/local/share/robotics-experiments/px4-airframes.json
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
      git make cmake ninja-build build-essential ca-certificates \
      python3-pip python3-jinja2 python3-toml python3-numpy \
      python3-future python3-jsonschema python3-packaging \
      python3-setuptools python3-wheel python3-yaml \
      lsb-release sudo gosu \
      tmux less \
 && rm -rf /var/lib/apt/lists/*

COPY docker/tmux.conf /root/.tmux.conf

RUN printf '%s\n' \
      "alias px4-sitl='/usr/local/bin/run_px4_sitl_fresh'" \
      >> /root/.bashrc

RUN pip3 install --no-cache-dir \
      kconfiglib==14.1.0 empy==3.3.4 pyros-genmsg==0.5.8

RUN git clone https://github.com/PX4/PX4-Autopilot.git ${PX4_DIR} && \
    cd ${PX4_DIR} && git checkout ${PX4_REF} && \
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

COPY px4/docker/run_px4_sitl_fresh.sh /usr/local/bin/run_px4_sitl_fresh
COPY px4/docker/run_px4_sitl_ci.sh /usr/local/bin/run_px4_sitl_ci
COPY px4/docker/start_px4_tmux.sh /usr/local/bin/start_px4_tmux
COPY px4/docker/px4_sitl_common.sh /usr/local/lib/px4/
COPY px4/docker/reset_px4_state.sh /usr/local/lib/px4/
RUN chmod +x \
      /usr/local/bin/run_px4_sitl_fresh \
      /usr/local/bin/run_px4_sitl_ci \
      /usr/local/bin/start_px4_tmux \
      /usr/local/lib/px4/px4_sitl_common.sh \
      /usr/local/lib/px4/reset_px4_state.sh

WORKDIR ${PX4_DIR}
# Drop into an attached tmux with PX4 SITL running in one pane and a shell in another.
CMD ["/usr/local/bin/start_px4_tmux"]

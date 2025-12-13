# This dockerfile installs common linux utilities on top of ubuntu-based base image.

ARG DOCKER_REPO
ARG DOCKER_TAG
FROM ${DOCKER_REPO}:${DOCKER_TAG} as base

ENV ORIGINAL_USER=${USER}

# Run as root (you need this due to NVML/Vulkan permissions)
USER 0

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    tmux=3.4-1ubuntu0.1 \
    ranger=1.9.3-5ubuntu0.24.04.1 \
    htop=3.3.0-4build1 \
    vim=2:9.1.0016-1ubuntu7.9 \
    && \
    rm -rf /var/lib/apt/lists/*

# Restore the original user
USER ${ORIGINAL_USER}

ENTRYPOINT ["/bin/bash"]

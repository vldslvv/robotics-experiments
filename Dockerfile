ARG ISAAC_REPO=nvcr.io/nvidia/isaac-sim
ARG ISAAC_TAG=5.1.0
FROM ${ISAAC_REPO}:${ISAAC_TAG}

# Run as root (you need this due to NVML/Vulkan permissions)
USER 0

RUN apt-get update && \
    apt-get install -y tmux ranger htop vim && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]

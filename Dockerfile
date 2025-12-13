FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Run as root (you need this due to NVML/Vulkan permissions)
USER 0

RUN apt-get update && \
    apt-get install -y tmux ranger htop vim && \
    rm -rf /var/lib/apt/lists/*

# Accept EULA + disable telemetry
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_DISABLE_TELEMETRY=1

# Set default ROS2 env
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV LD_LIBRARY_PATH=/isaac-sim/exts/omni.isaac.ros2_bridge/humble/lib:${LD_LIBRARY_PATH}

# Run two terminal sessions, one for isaac-sim gui, second for interactive session
COPY build_scripts/start.sh /start.sh
RUN chmod +x /start.sh

# ENTRYPOINT ["/bin/bash", "/start.sh"]
ENTRYPOINT ["/bin/bash"]

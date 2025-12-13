# Running

## Run Isaac-Sim.
On host:
`make run-isaac`

Inside docker:
`./runapp.sh` -- launches sim GUI

## Run ROS2
On host:
`make run-ros`

Both containers "should" be integrated to message each other

## ROS Bridge

Copy $PWD/.ros/fastdds.xml to host's $HOME/.ros directory
make

## Sometimes permissions are messed after creating something inside a container:

`sudo chown -R "$(whoami):" isaac-sim`

# Development

There's no good way to get autocomplete, neither locally nor while connected to running container.

The workflow is the following:
1. Create script using isaacsim
2. Mount its folder to a isaacsim container
3. Run script using `./python.sh ...` inside a container
4. For reference, use web documentation

VS Code can be used to connect to container, make sure to set interpreter path to the one used by `./python.sh` script.

To find out the interpreter, use:

```bash
./python.sh -c "import sys; print(sys.executable); print(sys.path)"
```

The output is likely to be the following:

```bash
/isaac-sim /isaac-sim/kit/python/bin/python3
['', '/isaac-sim', '/isaac-sim/kit/python/lib/python3.11', '/isaac-sim/kit/python/lib/python3.11/site-packages', '/isaac-sim/python_packages', '/isaac-sim/exts/isaacsim.simulation_app', '/isaac-sim/extsDeprecated/omni.isaac.kit', '/isaac-sim/kit/kernel/py', '/isaac-sim/kit/plugins/bindings-python', '/isaac-sim/exts/isaacsim.robot_motion.lula/pip_prebundle', '/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle', '/isaac-sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle', '/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle', '/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle', '/isaac-sim/exts/omni.pip.compute/pip_prebundle', '/isaac-sim/exts/omni.pip.cloud/pip_prebundle', '/isaac-sim/kit/python/lib/python311.zip', '/isaac-sim/kit/python/lib/python3.11/lib-dynload']
```

Tickets:
`https://forums.developer.nvidia.com/t/python-autocomplete-not-working-in-isaacsim-vscode-extension/329010`

## Creating a scene

Use Blender with BlenderKit to import some asset, or create your own. Export scene as USD, then open it with Isaac-Sim.

Make sure Blender is built with USD support, on Gentoo it has to be downloaded as binary from official site.


### Physics

Set rigid body for every object?
Uncheck "kinematic flag" for objects like walls or heavy objects that drone can't move
Set ground plane


## Remote Container Developing with VS Code

Connect to deployed container, and set python interpreter path to:

`.kit/python/bin/python3`

## Isaac Sim ↔ ROS 2 Integration (Docker, ROS 2 Jazzy)

Source: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html#enabling-the-ros-2-bridge

This section summarizes the **required steps** to integrate Isaac Sim with ROS 2 when both run in **separate Docker containers**.

### 1. Networking

Both containers must be started with:
```bash
--net=host
```

### 2. Fast DDS Middleware File

Put a file at known location for both Isaac Sim and ROS2 containers:

```xml
<?xml version="1.0" encoding="UTF-8" ?>

<license>Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.</license>


<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles" >
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>UdpTransport</transport_id>
            <type>UDPv4</type>
        </transport_descriptor>
    </transport_descriptors>

    <participant profile_name="udp_transport_profile" is_default_profile="true">
        <rtps>
            <userTransports>
                <transport_id>UdpTransport</transport_id>
            </userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
        </rtps>
    </participant>
</profiles>
```


### 3. Env vars

```bash
ROS_DISTRO=jazzy # Only set for Isaac Sim container
LD_LIBRARY_PATH+=<isaac_path>/exts/isaacsim.ros2.bridge/jazzy/lib # Only for Isaac Sim container
FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fastdds.xml # A path to file created on previous step
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ROS_DOMAIN_ID=<same as ROS 2 container>
```

### 4. ROS2 Bridge

ROS2 bridge has to be enabled for Isaac Sim. Usually by default, but worth
checking if things are not working as expected.

### 5. Add Test Graph

Isaac Sim publishes no ROS data by default. To publish, create an example Graph
in the scene by using `Tools -> Robotics -> ROS 2 OmniGraph -> Clock`

After pressing Play, Isaac Sim should start publishing events.

Topics and events can be checked by executing on ROS2 container:

```bash
ros2 topic list
ros2 topic echo /clock
```

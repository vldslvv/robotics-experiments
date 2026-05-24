# Robotics Experiments

This repository integrates Isaac Sim, ROS 2, PX4 SITL, and the shared Python
core package used by experiments.

## Repository Layout

- `src/robotics_experiments`: core Python package for reusable experiment logic.
- `isaac/isaac-sim`: Isaac Sim cache, config, logs, data, and documents
  populated by the container.
- `isaac/docker`: Isaac Sim Docker build and init inputs.
- `isaac/runtime/extensions`: Isaac Kit extension workflow.
- `isaac/runtime/standalone`: direct Isaac Sim scripts and launch helpers.
- `isaac/runtime/lab`: Isaac Lab configs, tasks, and future environment code.
- `ros/ros2_ws`: ROS 2 colcon workspace, persisted from the ROS 2 container.
- `ros/config`: ROS 2 middleware config mounted into Isaac Sim and ROS 2.
- `ros/docker` and `ros/scripts`: ROS 2 Docker and host-side helper area.
- `px4/docker` and `px4/scripts`: PX4 Docker and host-side helper area.
- `px4/logs` and `px4/eeprom`: PX4 SITL runtime persistence.
- `docs`: design and usage docs.

## Running

Build all Docker images:

```bash
make build
```

Run Isaac Sim:

```bash
make run-isaac
```

Inside the Isaac Sim container, `./runapp.sh` launches the GUI. For custom
extension development, use `isaac/runtime/standalone/runapp_custom.sh` so the
extensions under `isaac/runtime/extensions` are picked up.

Run ROS 2:

```bash
make run-ros
```

Run PX4 SITL:

```bash
make run-px4
```

The PX4 container opens a `px4` tmux session with two side-by-side panes. One
pane starts `make px4_sitl_default none_iris`; the other stays available as a
shell. Inside the container, use `px4-iris` to run the same SITL command on
demand.

Run host Python tests:

```bash
make test
```

Run container-backed integration tests:

```bash
make test-integration
```

Destroy containers and images:

```bash
make destroy
```

The runtime targets are interactive and persistent. If a container already
exists, the Makefile starts it again instead of creating a new one.
Legacy targets such as `build-all`, `run-all`, and `clean-all` remain as
compatibility aliases.

## Isaac Sim Workflows

- Use `isaac/runtime/extensions` for Isaac Kit extensions.
- Use `isaac/runtime/extensions/robotics_experiments_core` as the shared
  dependency that exposes `src/robotics_experiments` to other Isaac Sim
  extensions.
- Use `isaac/runtime/standalone` for scripts run directly by Isaac Sim through
  `/isaac-sim/python.sh`, `runapp.sh`, or `runapp_custom.sh`.
- Use `isaac/runtime/lab` for Isaac Lab-specific configs, task definitions, and
  future migration work.

Keep internal experiment logic in `src/robotics_experiments` where possible.
Isaac Sim extension code should handle Kit event wiring and delegate reusable
logic to the core package.

## ROS Bridge

Fast DDS config lives at `ros/config/fastdds.xml`. The Makefile mounts this
file into both Isaac Sim and ROS 2 containers at `/etc/ros/fastdds.xml` and sets
`FASTRTPS_DEFAULT_PROFILES_FILE` for both containers.

The Makefile also keeps the ROS 2 bridge defaults aligned:

```bash
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
FASTRTPS_DEFAULT_PROFILES_FILE=/etc/ros/fastdds.xml
```

Isaac Sim publishes no ROS data by default. To test the bridge, create a Clock
graph in Isaac Sim with `Tools -> Robotics -> ROS 2 OmniGraph -> Clock`, press
Play, then check from the ROS 2 container:

```bash
ros2 topic list
ros2 topic echo /clock
```

## Development Notes

VS Code can connect to a running container. For Isaac Sim, set the interpreter
to the Python used by `/isaac-sim/python.sh`.

To inspect the interpreter from inside the Isaac Sim container:

```bash
./python.sh -c "import sys; print(sys.executable); print(sys.path)"
```

Autocomplete for Isaac Sim APIs is limited. Use the container interpreter and
NVIDIA documentation as the main reference.

If container-created files have host permission issues, fix ownership from the
host. For example:

```bash
sudo chown -R "$(whoami):" isaac ros px4
```

## ROS Humble vs Jazzy

This repository defaults to ROS 2 Jazzy. Ubuntu 24.04, which Jazzy is based on,
does not provide every package used by older NVIDIA ROS 2 examples. In
particular, `topic_based_ros2_control` is built from source in the Docker image.

## Scene Notes

Use Blender with BlenderKit to import assets or create your own scene. Export
the scene as USD, then open it with Isaac Sim.

For simple physical scenes:

- Set rigid bodies for simulated objects.
- Leave static objects such as walls or heavy fixed objects non-kinematic.
- Add colliders and a ground plane.

Do not commit USD assets or other generated binary assets unless explicitly
requested.

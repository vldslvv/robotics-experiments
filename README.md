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
- `px4/logs`: PX4 SITL log persistence.
- `px4/eeprom`: PX4 SITL simulated EEPROM mount, cleared by the default
  `px4-iris` launch path so each run starts from fresh PX4 state.
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

`make run-px4` reads the active Hydra `vehicles` list and opens one terminal per
PX4-enabled vehicle. Each terminal starts a dedicated PX4 SITL container with an
isolated PX4 instance id, log directory, and EEPROM directory, then attaches to a
`px4` tmux session with the interactive PX4 console selected for commands such
as `commander arm` and `commander takeoff`. Automated integration tests use a
separate non-interactive PX4 launcher that suppresses prompt-only `pxh>` lines
while preserving PX4 logs under `px4/logs`. Set `INTEGRATION_VERBOSE=1` when
running a test target to print the full captured PX4 container log.

Select a different top-level Hydra experiment config with `HYDRA_CONFIG`, for
example:

```bash
make run-all HYDRA_CONFIG=configs/integration_test_multi_drone_takeoff.yaml
```

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

## PX4 Bridge Notes

The custom Isaac PX4 bridge is intentionally strict about simulation time and
frame conventions. The main failure modes fixed during flight debugging were:

- `HIL_SENSOR.time_usec` must be strictly increasing and derived from Isaac
  simulation time, not wall clock time.
- PX4 actuator lockstep should not block before PX4 has produced its first
  `HIL_ACTUATOR_CONTROLS` packet. After the first packet, the bridge drains
  queued actuator messages and applies the newest command only.
- The first actuator packet can arrive with startup timestamp lag while PX4 HIL
  output comes online. The bridge accepts that first lag as a bootstrap
  baseline, then disconnects and zeros commands if later actuator packets become
  materially staler than that baseline.
- Isaac articulation angular velocity is world-frame data. It must be rotated
  into body-FLU before the MAVLink adapter converts it to PX4 body-FRD gyro
  fields. Sending world-frame angular velocity as body gyro can look stable
  while level, then destabilize takeoff as soon as the vehicle rotates.
- Rotor thrust markers are configured in PX4 motor-index order for the
  generated Generic Quad X airframe (`none_x500_v2`): front-right CW, rear-left
  CW, front-left CCW, rear-right CCW. PX4 uses FRD geometry while Isaac marker
  positions are FLU, so the USD motor names look mirrored relative to PX4's
  Y-right convention.
- The bridge-side tilt cutoff zeros rotor forces after excessive tilt so an
  already-crashed or inverted vehicle does not keep applying thrust.

For flight debugging, start with the paired files under `px4/logs/<date>/`, for
example `HH_MM_SS.ulg` and `HH_MM_SS_bridge.log`. Use the bridge log to check
sensor timestamps, actuator lag, rotor mapping, applied forces, and tilt cutoff.
Use the ULog to confirm command acknowledgements, trajectory setpoints,
attitude setpoints, actuator outputs, estimator status, and failsafe flags.

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

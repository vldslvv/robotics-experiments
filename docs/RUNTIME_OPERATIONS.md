# Runtime operations

This is the reference for persistent Docker runtime commands, Isaac/PX4
container behavior, runtime logs, and flight-log debugging. Integration harness
details live in `docs/INTEGRATION_TEST_ARCHITECTURE.md`; experiment selection
details live in `docs/EXPERIMENT_CONFIG.md`.

Run ROS 2, Isaac Sim, and PX4 through the Docker targets below, not directly on
the host. They are meant to be used in the Dockerized environment.

## Docker targets

- `make build` builds the minimal Isaac Sim, Isaac Lab, ROS 2, and PX4
  runtime/integration images. These images contain execution dependencies but
  no editor or assistant tooling.
- `make build-isaac-dev` and `make build-ros-dev` build opt-in development
  variants with pinned Neovim, Node.js, Codex, and Claude Code versions. Run a
  persistent development variant by overriding the image, for example
  `make run-isaac ISAAC_IMAGE=vv-isaac-sim-custom:5.1.0-dev`.
- The Isaac Lab image is for RL training and RL-specific gates; see
  `docs/RL_WORKFLOW.md` before changing `build-isaac-lab`, `ISAAC_LAB_IMAGE`,
  `ISAACLAB_REF`, or `make train-rl`.
- `make destroy` stops/removes containers and removes built images (including the
  Isaac Lab image).
- `make run-isaac` starts the Isaac Sim container. By default
  (`ISAAC_AUTORUN=0`) it lands in the manual two-pane tmux shell. Pass
  `ISAAC_AUTORUN=1` (on `run-isaac` or `run-all`) to override the container's
  tmux CMD so the first pane auto-launches the custom runtime
  (`isaac/runtime/standalone/runapp_custom.sh`) and then drops to a login shell,
  leaving the second pane a plain shell.
- `make run-all-autorun` is the convenience target for `run-all` with
  `ISAAC_AUTORUN=1`, propagated to the kitty-spawned `run-isaac` sub-make. The
  selected startup command is passed to `run_persistent_container.sh` as
  `EXPECTED_CMD`, which recreates the persistent container when the desired
  command changes.
- `make run-ros` starts the ROS 2 container.
- `make run-px4` starts one PX4 SITL container per PX4-enabled vehicle.
- `make stop` stops persistent containers without removing them.
- Legacy targets such as `build-all`, `run-all`, and `clean-all` are
  compatibility aliases.

The Isaac, ROS, Ubuntu/PX4 base images are pinned by manifest digest. Source
checkouts used by the images are pinned to immutable commits, while the
human-readable release tags remain in comments or image tags. Update the digest
and corresponding tag together when intentionally upgrading a base image.
The ROS workspace deliberately skips the upstream humanoid example's
`python3-pytorch-pip` rosdep key: that example is outside this project's runtime
and allowing rosdep to install it would resolve a mutable, multi-gigabyte CUDA
closure during every ROS image build.

## Persistent Isaac logs

The persistent Isaac container (`make run-isaac` / `make run-all`) is named
`vv-isaac-sim-custom` (`ISAAC_CONTAINER_NAME`) and is retained after it exits, so
the quickest way to inspect the last manual run is
`docker logs vv-isaac-sim-custom`.

That console stream is the only place the shell-level scene-resolution step in
`runapp_custom.sh` (the `get_config().scene` subprocess that picks the stage)
reports its errors. Those errors never reach the Kit log file, so a blank stage
with no `open_stage` exec is diagnosed from `docker logs`, not from the
structured Kit log.

Kit's own log is written inside the container at
`/root/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/<ver>/kit_<timestamp>.log`
(container `HOME` is `/root`). It is not on the host, because the
`isaac/isaac-sim/logs` host mount maps to `/isaac-sim/.nvidia-omniverse/logs`, a
different path than `/root/...`. Copy it out with
`docker cp 'vv-isaac-sim-custom:/root/.nvidia-omniverse/logs/Kit/' <dest>` when
you need the structured Kit log rather than the console stream. The
`isaac/isaac-sim/logs/test_*.log` files there are integration-test output, which
redirects explicitly; manual runs do not land there.

## Persistent PX4

`make run-px4` derives the number of PX4 SITL containers from the active Hydra
`vehicles` list and opens one terminal per PX4-enabled vehicle. Each persistent
PX4 container attaches to a `px4` tmux session with the interactive PX4 console
selected, so manual commands such as `commander arm` and `commander takeoff` can
be entered directly.

Every persistent PX4 instance runs on host networking (`--network=host`, like
the Isaac container), so all instances share the host namespace and the PX4 `-i`
index keeps each one's simulator TCP `4560 + i` / GCS UDP `14550 + i` ports
distinct. The shared-private-namespace `--network container:<primary>` model is
specific to the integration harness, where instance 0 owns a private namespace
that the other instances and the ephemeral Isaac container join. Do not use that
model for the host-networked persistent path.

Persistent and integration Make targets accept `HYDRA_CONFIG=<path>` to select a
top-level Hydra experiment config for both host-side PX4 counting and Isaac
runtime config composition, for example:

```sh
make run-all HYDRA_CONFIG=configs/takeoff_quad_quad.yaml
```

## Flight log analysis and PX4 bridge debugging

- `flight_analysis/README.md` is the single source for the repeatable flight-log
  workflow. Read it when you start investigating an unstable flight, a diverging
  takeoff, or a sim-to-real comparison; do not reconstruct the procedure from
  `AGENTS.md`.
- That README documents the helper tools (`list_pairs.py`, `bridge_summary.py`,
  `ulog_timeline.py`, `compare_logs.py`), the bridge-log fields, the minimum
  ULog topic set and required real-drone SDLOG profile, the NED / aiding-source
  caveats, and the step-by-step order for deciding whether PX4 commanded the bad
  motion or the bridge applied it incorrectly.
- Make targets in `make/flight-analysis.mk` run those tools through a throwaway
  `uv run --no-project --with pyulog --with numpy` environment, so pyulog/numpy
  never enter the project `.venv`: `make compare-logs A=<log_a.ulg> B=<log_b.ulg>`
  prints a per-metric divergence table, and
  `make ulog-timeline ULOG=<log.ulg> [SAMPLES=...]` prints a single-log timeline.
- Debug from the persisted logs, not `docker logs` (PX4 terminal scrollback has
  repeated starts and ANSI redraw noise; use it only for high-level warnings).
  PX4 ULogs and bridge logs are persisted under `px4/logs/<yyyy-mm-dd>/` for
  manual runs and under `.build/docker/integration/<target>.<run-id>/` for
  integration tests; pair a ULog and bridge log by date directory and `HH_MM_SS`
  stem.

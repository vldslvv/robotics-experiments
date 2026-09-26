# Full-GUI flight benchmark

`make benchmark-flight` is the diagnostic Isaac-version comparison workload. It
runs the production GUI application, warehouse scene, live `experiment_runtime`
extension, PX4 lockstep bridge, and the two-drone routine from
`configs/flight_benchmark.yaml`. That config inherits
`configs/routines/fleet_circle_land_quad_quad.yaml` and only extends the cold-start bridge
deadline; measurement starts after connection, so the captured workload is
unchanged. The benchmark does not add or change a performance gate. A threshold
may be proposed only after both version reports have been reviewed.

Benchmark behavior is isolated in the opt-in `flight_benchmark` Kit extension.
The plugin depends on `experiment_runtime`, reads its immutable status snapshot,
and wraps selected public production callables with timers. Normal
`make run-isaac` / `make run-all` launches never load the plugin and execute no
benchmark imports, branches, global-recorder checks, or timing contexts.

## Method

The command stops persistent simulator and PX4 containers, builds the Isaac and
PX4 images serially, and runs one unreported warm-up,
three measured repetitions, and one separate Tracy repetition. Each repetition
uses a fresh pair of PX4 containers and a fresh GUI Isaac container. The
benchmark plugin presses Play after the production extension has configured the
warehouse stage and Kit reports no pending asset loads. It captures the phases
below, writes the repetition, and closes Kit without operator input.

- `connected_disarmed`: all simulator MAVLink bridges are connected and neither
  vehicle is armed; capture continues until the first vehicle arms.
- `armed_takeoff`: from first arm until every bridge-enabled vehicle completes
  its configured takeoff step. Each benchmark routine must contain exactly one.
- `airborne`: the configured `flight_benchmark.airborne_duration_s` after all
  takeoffs complete (five simulated seconds by default).

The default contract is 1280x720 Raytraced Lighting, the warehouse, two drones,
250 Hz physics, 60 Hz rendering, and TGS. A command-line `HYDRA_CONFIG` override
may select another workload, including the one-drone minimal-scene configuration
used by the earlier integration benchmark. Runs are comparable only when
hardware including CPU governor, the ordered bridge-enabled fleet, the fully
resolved experiment-config digest, PX4 revision, measurement level, and
methodology match. Isaac image/version and source commit are expected to differ.

The reproducible one-drone diagnostic uses the same measured workload as
`configs/routines/basic_takeoff_quad.yaml`, with only the cold-start bridge deadline extended:

```sh
make benchmark-flight HYDRA_CONFIG=configs/flight_benchmark_takeoff_quad.yaml
```

Use a fresh image build for the version run:

```sh
make benchmark-flight FLIGHT_BENCHMARK_BUILD_FRESH=1
```

`FLIGHT_BENCHMARK_BUILD_FRESH=1` adds `--pull --no-cache` to this target's Isaac
and PX4 image builds only.
Benchmark runs from the two checkouts must be serial; never leave the other
checkout's benchmark containers running.

## Metrics and artifacts

Artifacts are isolated under `.build/benchmarks/flight.<run-id>/`. The root
contains `fingerprint.json`, `report.json`, and `report.md`; `warmup/`,
`raw/repeat-{1,2,3}/`, and `profile/` contain the individual application, PX4,
Kit, resource, NVIDIA benchmark-service, and bridge logs. `profile/flight.tracy`
is attribution-only and is not mixed into the three-repeat distributions.

The report includes p50, p95, p99, and maximum values for app, render, GPU, and
PhysX timing; physics steps per frame; RTF; frame-budget violations; process and
container CPU/memory; per-drone bridge send/receive timing; MAVLink sensor and
actuator legs; articulation reads; thrust, rotor visuals, and dynamics writes;
and fleet routine-driver work. Raw component samples retain phase and app frame
IDs. The report uses those IDs for slow-frame correlations.

Project-component samples are benchmark-only wrapper timings around the real
`BridgeHost`, `Px4MavlinkPort`, `IsaacArticulationIO`, thrust-model, and driver
callables. The wrappers preserve results and exceptions and are restored when
the plugin shuts down. They add measurement overhead to benchmark runs, so
component durations are attribution signals rather than production-runtime
overhead. Reference and candidate runs must use the same methodology version.

To measure the observer effect in one Isaac version, repeat the same config and
hardware with `FLIGHT_BENCHMARK_MEASUREMENT_LEVEL=lifecycle`, `nvidia`, and
`full`. `lifecycle` keeps only autoplay, phase detection, and phase wall/sim
timing; it does not subscribe to pre-update or physics-step events. `nvidia`
adds app-frame and physics-step sampling, render intervals, and the NVIDIA
benchmark service. `full` is the default cross-version methodology and
additionally adds process-resource sampling and the project-component wrappers.
The reduced levels are overhead diagnostics, not cross-version comparison
workloads.

A failed lifecycle, incomplete phase set, short airborne capture, missing
NVIDIA metric group, missing container resource stream, inconsistent repetition,
or missing Tracy trace produces an `invalid` report rather than partial results.

## Cross-version comparison

After both runs are complete, invoke the command from either checkout:

```sh
make benchmark-flight-compare \
  REFERENCE=/path/to/isaac-5/flight.<run-id> \
  CANDIDATE=/path/to/isaac-6/flight.<run-id>
```

The command first enforces the comparable-run contract and exact numeric metric
tree, then writes timestamped JSON and Markdown under the sibling workspace
`.build/benchmarks/` directory. It reports every numeric delta, ranks airborne
project-component p95 deltas, and lists app, GPU, and PhysX p95 deltas without
claiming that any one timer caused the frame-time change. A zero reference value
uses an absolute delta and leaves percentage change unset.

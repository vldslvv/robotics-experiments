# Isaac Core Performance Benchmark

`make benchmark-isaac-core` is the Isaac Sim upgrade regression gate. It uses
the real minimal project scene and configured drone asset, but intentionally
stops at the Isaac boundary: PX4 SITL and the command routine are covered by
their integration tests, not this performance comparison.

Each unprofiled repetition runs six ordered phases:

| Phase | Isolated contribution | Limiter | Target |
| --- | --- | --- | --- |
| `kit_only_unlimited` | Headless Kit app updates with the timeline stopped | Off | 600 app updates |
| `render_only_unlimited` | Prior phase plus one 1280x720 RaytracedLighting render product | Off | 600 app updates |
| `scene_physics_unlimited` | Render workload plus the minimal scene's PhysX work | Off | 5,000 physics ticks |
| `drone_physics_unlimited` | Prior phase plus passive drone rigid-body/articulation physics | Off | 5,000 physics ticks |
| `drone_articulation_unlimited` | Prior phase plus per-tick state read, rotor visual update, and dynamics application | Off | 5,000 physics ticks |
| `drone_articulation_realtime` | Complete Isaac workload under production pacing | 60 Hz from config | 5,000 physics ticks |

The unlimited phases expose performance changes that would otherwise be hidden
under the 60 Hz sleep budget. The realtime phase checks production pacing and
real-time factor. Before each measured physics phase, the scenario warms 500
physics ticks; the Kit-only and render phases each warm 60 app updates.

The runner warms shared Isaac caches once, then runs three sequential,
unprofiled repetitions. It follows those with one separate Tracy run containing
named zones for all six phases. Profiling never contributes samples to the
gate. After a successful profile workload, the profile-only Python process exits
without Kit cleanup because the external Tracy capture owns profiler shutdown;
an unsuccessful workload still follows the normal error path. Do not run other
GPU- or CPU-heavy jobs on the host during a benchmark.

Artifacts live under `.build/benchmarks/isaac-core.<run-id>/`: normalized raw
samples, NVIDIA benchmark-service JSON, integration logs, the one-shot
`isaac_core.tracy` capture, `report.json`, and `report.md`. `history.jsonl`
provides a compact local history.

## Measurements and attribution

Every phase reports app-update p50/p95/p99. GPU p50/p95/p99 is required after
the render product exists and is also reported for the Kit-only phase if the
experience emits samples. A render-thread interval is reported when the selected
Kit experience emits that optional event; the 5.1 headless experience does not,
so renderer attribution comes from the adjacent phase delta, GPU samples, and
Tracy instead. Physics phases additionally report PhysX duration,
physics-step interval, real-time factor, simulated duration, and wall duration.
The two articulation phases directly time the project-owned Isaac calls for
state reads, visual rotor writes, and dynamics application.

The report includes signed deltas between adjacent unlimited phases to help
localize an upgrade change to rendering, scene physics, drone physics, or
articulation I/O. Those deltas are diagnostic, not additive: Isaac overlaps CPU
and GPU work, and the app-update duration is not expected to equal the sum of
PhysX, GPU, and project timers. Use the Tracy trace to inspect the remaining
Kit, USD, renderer, driver, synchronization, and idle zones when a delta is not
specific enough.

`physics_step_interval` is an inter-arrival latency, not a compute cost. It is
retained for scheduling diagnosis but is not gated because multiple 250 Hz
physics substeps are batched within a Kit update.

## Gate

The report takes the median of each per-repetition statistic. Only complete
workload outcomes fail the gate; isolated phase deltas and project sub-timers
remain diagnostic.

| Workload | Metric | Limit |
| --- | --- | --- |
| Unlimited full articulation | App p50, PhysX p50, GPU p50 | At most 10 percent above baseline |
| Unlimited full articulation | App p95, PhysX p95, GPU p95 | At most 15 percent above baseline |
| Realtime full articulation | App p50 | At most 10 percent above baseline |
| Realtime full articulation | App p95 | At most 15 percent above baseline |
| Realtime full articulation | Real-time factor | At least 95 percent of baseline and at least 0.95 |

The command exits zero only when every gated metric passes. A regression exits
1. Missing or malformed samples, a missing profiler trace, a missing baseline,
a changed workload contract, or different hardware exits 2. The hardware
fingerprint is the CPU model, logical CPU count, CPU governor, GPU count, and
ordered GPU model list. The Isaac image and version are recorded but may differ,
which is the intended upgrade comparison.

The workload contract records the scene, physics and rendering rates, solver,
vehicle identity and asset, renderer and resolution, limiter state, methodology
version, and sample target. A contract change requires a reviewed replacement
baseline.

## Upgrade workflow

Before changing the Isaac image, run:

```text
make benchmark-isaac-core
```

Review the aggregate metrics, phase deltas, and trace, then preserve the last
passing report. After changing or rebuilding the Isaac image, run the same
command and compare it against the accepted result.

Baseline replacement is explicit:

```text
make benchmark-isaac-core-accept \
  RESULT=.build/benchmarks/isaac-core.<run-id>/report.json
```

Acceptance permits a passing, failing, or first `no_baseline` result. It refuses
invalid or incomparable results, requires all three unprofiled repetitions and
the trace, and refuses hardware that differs from an existing baseline. Review
the report before accepting it. To change a tolerance intentionally, edit
`benchmarks/isaac_core/baseline.json` separately; acceptance preserves the
existing tolerance policy.

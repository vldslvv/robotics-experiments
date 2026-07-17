# Flight Performance Benchmark

`make benchmark-flight` is the upgrade regression gate for Isaac physics and
PX4 lockstep performance. It uses the real project workloads instead of a
synthetic bridge loop:

- `physics_only` reuses the configured drone articulation integration test. It
  warms 500 PhysX callbacks, then captures 5,000 callbacks while the timeline is
  playing.
- `px4_flight` reuses the strict single-drone PX4 takeoff routine. Capture starts
  when the arm step completes and ends after the takeoff and stable-hover steps
  complete.

The runner warms the shared Isaac caches once, then runs three sequential
repetitions of each workload. Do not run other GPU or CPU-heavy jobs on the host
during a benchmark. Raw NVIDIA benchmark-service samples, project physics-step
intervals, integration logs, `report.json`, and `report.md` are kept under
`.build/benchmarks/flight.<run-id>/`. `history.jsonl` provides a local compact
run history.

## Gate

The report takes the median of each per-repetition statistic. It records app,
PhysX, and physics-step interval p50/p95/p99 values, real-time factor, sample
counts, measured duration, and startup time when available. The committed
baseline gates these values:

| Workload | Metric | Limit |
| --- | --- | --- |
| Physics-only | PhysX p50 | At most 10 percent above baseline |
| Physics-only | PhysX p95 | At most 15 percent above baseline |
| Physics-only | Step interval p50 | At most 10 percent above baseline |
| Physics-only | Step interval p95 | At most 15 percent above baseline |
| PX4 flight | Step interval p50 | At most 10 percent above baseline |
| PX4 flight | Step interval p95 | At most 15 percent above baseline |
| PX4 flight | Real-time factor | At least 95 percent of baseline and at least 0.95 |

The command exits zero only when every gated metric passes. A regression exits
1. Missing or malformed samples, a missing baseline, a changed workload
contract, or different hardware exits 2. The hardware fingerprint is the CPU
model, logical CPU count, CPU governor, GPU count, and ordered GPU model list.
This deliberately refuses a comparison instead of weakening the threshold on a
different machine.

The PX4 source reference must also match the baseline so an Isaac upgrade is
not compared across a simultaneous flight-controller change. The Isaac image
and version are expected to differ during an upgrade and are recorded rather
than used to reject the comparison.

The workload contract records the scene, physics and rendering rates, solver,
vehicle count, vehicle assets and IDs, methodology version, and fixed tick
target. A contract change requires a newly reviewed baseline because it is no
longer the same benchmark.

## Upgrade workflow

Before changing the Isaac image, run:

```text
make benchmark-flight
```

Review both reports and keep the last passing result. After changing or
rebuilding the image, run the same command again. Use the per-workload metrics
to distinguish a PhysX regression from PX4 lockstep overhead. Startup and p99
values are diagnostic; they are reported but do not fail the gate because cache
and one-off scheduling effects make them noisier.

Baseline replacement is explicit:

```text
make benchmark-flight-accept RESULT=.build/benchmarks/flight.<run-id>/report.json
```

Acceptance permits a passing, failing, or first `no_baseline` result so an
intentional upgrade can establish the next reference. It refuses invalid or
incomparable results, requires all three repetitions, and refuses hardware that
differs from an existing baseline. Review the report before accepting it. To
relax a gate intentionally, edit the tolerance in
`benchmarks/flight/baseline.json` separately so the policy change is visible in
review; baseline acceptance preserves those tolerances.

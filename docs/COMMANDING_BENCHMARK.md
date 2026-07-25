# Isolated commanding performance benchmark

`make benchmark-commanding` is the explicit host-only performance gate for the
pure routine stepper and routine driver. It requires the normal host Python
environment only: it does not launch Docker, Isaac Sim, PX4, or pymavlink. It is
separate from `make check` so normal correctness gates remain stable and fast.

The accepted baseline is `benchmarks/commanding/baseline.json`. Transient runs
live under `.build/benchmarks/commanding.<run-id>/`, with one raw JSON document
per worker plus `report.json` and `report.md`. A compact local history is
appended to `.build/benchmarks/commanding-history.jsonl`.

## Workloads and sampling

The benchmark measures four fixed snapshots through the public
`robotics_experiments.commanding` contract:

| Workload | Snapshot |
| --- | --- |
| `goto_cached_transmit` | Accepted cached `goto` tick with setpoint transmission due |
| `policy_cached_transmit` | Policy tick with transmission due and inference not due |
| `policy_evaluate_transmit` | Policy inference and transmission both due, using immutable `()` state and a prebuilt deterministic setpoint |
| `driver_goto_drive` | Full public `tick_routine_driver` DRIVE tick over inert port/environment adapters and the accepted cached `goto` |

Fixtures validate state, decisions, cadence, and effects before measurement and
after every timed batch. A fresh worker also refuses to run if fixture creation
imports Isaac or pymavlink.

The parent launches nine fresh workers sequentially and pins each to one
available logical CPU. Workload order rotates between workers. For every worker
and workload, the runner calibrates to an approximately 50 ms batch, runs five
warm-up batches and twelve measured batches, and retains the median nanoseconds
per operation. Cyclic garbage collection is disabled only around each timed
batch and restored immediately afterward.

The raw records include CPU model, logical CPU count, governor, affinity,
platform, Python implementation/version, calibrated iterations, and every
warm-up and measured batch. Runs are comparable only when the recorded hardware
and methodology fingerprint matches the accepted baseline.

## Statistical gate

For each workload, the nine candidate worker medians are compared with the nine
accepted medians using an exact two-sided permutation test of the median
difference. Holm correction controls family-wise alpha at 0.05 across all four
workloads.

A workload fails only when both conditions hold:

- Holm-adjusted `p < 0.05`.
- Candidate median latency is at least 5% above baseline.

Statistically significant improvements, significant increases below 5%, and
large but noisy increases are reported diagnostically without failing. Reports
include median and percentage deltas, raw and adjusted p-values, baseline and
candidate worker ranges, and a deterministic bootstrap 95% interval for the
median percentage effect.

Exit status is 0 for a pass, 1 for a meaningful regression, and 2 for a missing
baseline, invalid samples, or an incomparable fingerprint.

## Running and acceptance

Use a quiet host with no competing CPU-heavy work and a stable CPU governor:

```sh
make benchmark-commanding
```

Review both the Markdown summary and raw worker distributions. Replace the
baseline explicitly:

```sh
make benchmark-commanding-accept \
  RESULT=.build/benchmarks/commanding.<run-id>/report.json
```

Acceptance permits a complete pass, fail, or first `no_baseline` result after
review. It refuses invalid or incomparable reports, requires all nine worker
medians for all four workloads, preserves the 5%/0.05 policy, and refuses a
fingerprint that differs from an existing baseline.

This benchmark attributes regressions within commanding quickly, but its inert
adapters do not represent socket, simulator, flight-controller, rendering, or
GPU costs. `make benchmark-flight` remains the end-to-end production-path
measurement for those system-level effects.

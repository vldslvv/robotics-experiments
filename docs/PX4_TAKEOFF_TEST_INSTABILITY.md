# `test-integration-px4-takeoff` instability -- investigation findings

> Status: **investigation complete; implementation deferred.** The implementation
> route is intentionally **left open** (see "Open implementation options"). No
> make/runner/scenario/config changes have been made; this document preserves the
> findings so the work can resume later.

## Context

Track P3 (code redundancy steps 3.1-3.4) was merged to `main` on 2026-06-16. Soon
after, the `test-integration-px4-takeoff` integration test was observed to be
unstable. This document records why it fails, the evidence, the failure taxonomy,
and the open options for either better understanding future failures or fixing the
flakiness.

## TL;DR

1. **Not caused by Track P3.** P3 is behavior-preserving; the live takeoff path
   takes the thrust constants from the Hydra `px4` config (unchanged), not the
   refactored library defaults, and the sensor-noise refactor only touches the
   `noise_enabled=false` path (takeoff runs with `sensor_noise_enabled=true`).
   Runs immediately after the merge passed; failures cluster in host-load-heavy
   wall-clock windows, not at the merge boundary.
2. **The test has a standing ~33% infra-flake rate** (historical PASS=42 /
   FAIL=21 / UNKNOWN=2 across recorded plain-takeoff run dirs), spiking to ~50%
   in busy windows (e.g. 16:20-16:55 UTC on 2026-06-16). It is environment /
   host-load dependent, not a code step-change.
3. **20 of 21 failures are `actuator polling timed out`** -- a PX4 SITL lockstep
   disconnect, in two forms (see taxonomy). Both are environmental: PX4 boot
   latency or a mid-flight process stall under host contention.
4. **1 failure is `ExpectedInstabilityDetected: PX4 entered failsafe state`**,
   which exposed a **taxonomy bug** (now **FIXED** -- see below): a rerun-worthy
   PX4 failsafe flake was raised as the *same* exception class as a genuine
   flight-envelope breach, and the `-instability` test caught
   `ExpectedInstabilityDetected`, so a failsafe flake there could make it pass for
   the wrong reason.

## Status of the fixes

- **Failsafe taxonomy bug: FIXED.** PX4 failsafe now raises a distinct
  `Px4FailsafeDetected` (an `AssertionError` that is **not** an
  `ExpectedInstabilityDetected`). All four failsafe detection sites in
  `isaac/runtime/standalone/tests/_px4_takeoff_scenario.py` route through it, so a
  failsafe never counts as an envelope-detected instability and the `-instability`
  test no longer passes on a failsafe flake. (Verified safe: all recent
  `-instability` runs detect via the tilt envelope, not failsafe.) Within the
  observation loop the per-tick stability-envelope gates are now evaluated
  *before* the failsafe check, so a fault-induced divergence that would trip both
  on the same tick is always attributed to the envelope; a failsafe is raised
  only when PX4 fails safe while the drone is still inside the envelope.
- **Deferred (route still open):** the per-run classification tool, bounded
  auto-retry of infra flakes, and the readiness-gate / concurrency hardening
  described under "Open implementation options" remain unimplemented.

## Evidence

### Failure-rate scan (all recorded plain-takeoff run dirs)

`.build/docker/integration/test-integration-px4-takeoff.*` (excluding
`-instability`): **PASS=42, FAIL=21, UNKNOWN=2**. Failures are not monotonic with
the P3 merge: a long PASS streak (00:22-04:19) precedes scattered failures, and
the recent window 16:20-16:55 shows ~50% failures. By signature, ~20/21 fails are
`PX4 bridge disconnected ... actuator polling timed out`; 1 is `PX4 entered
failsafe state`.

### The two latest failing runs, dissected

- **`...162919Z.50253` (mid-flight stall).** Bridge log shows healthy lockstep
  the whole time -- `actuator_lag_us=4000` (exactly one 250 Hz tick) repeatedly --
  with motors still `input_reference=[0,0,0,0]` (pre-arm/arming phase, ~12 s sim).
  Then PX4 goes silent and the bridge disconnects:
  `actuator polling timed out after 1.000 seconds (sensors=3044 actuators=3030
  last_sensor_us=12180000 last_actuator_us=12176000)`. Last actuator was only one
  tick stale -- this is the `lockstep_timeout_s=1.0` firing because PX4 stopped
  responding, **not** the sustained-lag tripwire.
- **`...165502Z.63196` (bootstrap).** `PX4 MAVLink port constructed ...
  bootstrap_prime_ticks=20 first_actuator_timeout_s=5.000`, then
  `actuator polling timed out after 5.000 seconds (sensors=21 actuators=0
  last_actuator_us=None)`. PX4 received 21 primed HIL_SENSOR packets but never
  emitted its first actuator within 5 s -- PX4 SITL not flight-ready in time.

### Why the bootstrap form happens

`make/run_px4_integration_test.sh:115-135` gates PX4 readiness on the docker-log
line `"Waiting for simulator to accept connection on TCP port 4560"`. That marks
the TCP port as *open*, not the PX4 commander/HIL stack as ready to produce
actuators. Isaac connects and primes immediately; under load PX4 can take longer
than `first_actuator_timeout_s=5.0` to answer.

### P3 non-causation (confirmed)

- `thrust_model.py` Step 3.4: `(_IRIS_ROTOR_CONSTANT,)*4 == (8.54858e-6,)*4` etc.
  -- value-identical. The live bridge constructs `PegasusQuadraticThrustModel`
  with `self._cfg.rotor_constant` (from `configs/experiment/px4/default.yaml`,
  unchanged), so the library defaults are not even on the runtime path.
- `_pegasus_sensors.py` Step 3.3: emitted dicts are semantically identical; the
  zeroing block only runs when `sensor_noise_enabled=false`, but takeoff runs with
  it `true`, so the block is never executed.
- Steps 3.1/3.2: config-resolution helper and USD-spawn dedup are not on the
  takeoff dynamics path.

## How the test works (mechanism reference)

- **Targets:** `make/integration-tests.mk:124-130` (`test-integration-px4-takeoff`
  and `-instability`); both go through the `RUN_PX4_INTEGRATION_TEST` macro
  (`:43-61`) -> `make/run_px4_integration_test.sh`.
- **Concurrency:** `make/config.mk:38` `INTEGRATION_TEST_JOBS ?= 2`;
  `test-integration-all` fans out with `xargs -P` (`integration-tests.mk:64-92`).
  CLAUDE.md documents lowering this to avoid GPU/host starvation. (Note: CLAUDE.md
  text still says default 3; config.mk says 2.)
- **Pass/fail:** the standalone test raises `AssertionError` -> printed as
  `[isaac-standalone-test] ERROR: ...` -> matched by
  `ISAAC_TEST_FAILURE_FILTER = ^\[isaac-standalone-test\] ERROR:`
  (`config.mk:32`) -> docker exits non-zero -> runner prints `TEST_TARGET: FAIL`
  (`run_px4_integration_test.sh:59-66`).
- **PX4 readiness gate:** `run_px4_integration_test.sh:115-135`,
  `PX4_TEST_READY_TIMEOUT_S ?= 60` (`config.mk:93`).
- **Scenario assertions:** `isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`
  - `_bridge_disconnect_reason` (`:119-122`) reads `port._last_disconnect_reason`.
  - Disconnect -> `AssertionError("PX4 bridge disconnected during takeoff
    scenario: {reason}")` (`:142-144`).
  - `ExpectedInstabilityDetected(AssertionError)` (`:833`); `_raise_instability`
    (`:133`); post-run bridge-log disconnect-marker scan (`:687-695`).
  - Envelope thresholds (hardcoded module constants `:23-51`): tilt 35 deg,
    lateral drift 2.0 m, vertical +8 / -4 m/s, angular 8 rad/s, hover altitude
    range 0.35 m, hover vertical speed 0.35 m/s, hover horizontal range 0.30 m /
    speed 0.30 m/s, min climb 1.5 m; takeoff target 2.5 m.
- **Bridge/lockstep config** (`configs/experiment/px4/default.yaml`):
  `lockstep_timeout_s=1.0`, `first_heartbeat_timeout_s=5.0`,
  `bootstrap_prime_ticks=20`, `first_actuator_timeout_s=5.0`,
  `max_sustained_actuator_lag_s=0.5`, `startup_settle_time_s=1.0`,
  `max_tilt_angle_deg=60.0`; `sim.physics_hz=250`.
- **`-instability` variant:** injects a post-takeoff rotor-0 thrust kill + 4.0 N*m
  roll bias and asserts the envelope raises `ExpectedInstabilityDetected`
  (`test_px4_takeoff_instability_detection.py`).

## Failure taxonomy (central artifact for any implementation)

- **INFRA_FLAKE -- rerun-worthy (environmental):**
  - `actuator polling timed out after 5.000 seconds` with `actuators=0 /
    last_actuator_us=None` (bootstrap: PX4 not flight-ready in time).
  - `actuator polling timed out after 1.000 seconds` mid-flight with small
    `actuator_lag_us` (PX4 process stall under load).
  - `PX4 entered failsafe state` (failsafe / system_status=5 -- per project
    memory, rerun-worthy).
  - first-heartbeat timeout; PX4 container exited before readiness;
    COMMAND_ACK *timeout* under load.
- **REGRESSION -- fail hard (never retry):**
  - `actuator lag exceeded sustained limit` (the stale-but-flowing-backlog
    tripwire; per CLAUDE.md/memory this signals a real regression).
  - `sensor cadence mismatch`; NaN/Inf vehicle-state raise.
- **STABILITY_BREACH -- fail hard (never retry):**
  - Envelope messages (tilt / lateral drift / vertical / angular / hover gates);
    `altitude did not increase enough`; COMMAND_ACK *rejected* (result != accepted).

Update: failsafe is now its own `Px4FailsafeDetected` exception (an
`AssertionError`, **not** an `ExpectedInstabilityDetected`), so it is already
separated from STABILITY_BREACH at the source. A future classifier should map
`Px4FailsafeDetected` to the INFRA_FLAKE bucket.

## Open implementation options (route deliberately left open)

Three orthogonal levers; a future PR may take any subset. Trade-offs noted so the
choice can be made later.

1. **Diagnosis / observability (foundation).** Emit a machine-readable per-run
   classification (PASS / INFRA_FLAKE:<subtype> / REGRESSION:<subtype> /
   STABILITY_BREACH:<metric>) from the run dir's isaac log + bridge log (+ ULog
   for failsafe/system_status). Sub-choice of home:
   - Host-side Python tool under `flight_analysis/` (tested; reusable for ad-hoc
     forensics; runner calls it on failure). Aligns with existing ULog tooling.
   - Inline shell grep in `run_px4_integration_test.sh` `cleanup()` (no new files,
     but fragile and not independently testable).
2. **Bounded auto-retry of INFRA_FLAKE only** (make-level wrapper around the PX4
   takeoff target(s)): up to N attempts, each with its own preserved run dir;
   stop immediately on REGRESSION / STABILITY_BREACH. Requires the failsafe
   re-bucketing (distinct exception/message) so a failsafe flake is retried, not
   treated as instability -- and so the `-instability` test is not satisfied by a
   failsafe.
   - Open question / trade-off: auto-retry hides a *rising* flake rate. Mitigate
     by logging attempts and/or a "classify-and-stop" alternative (no auto-retry;
     a human reruns), which is the conservative variant.
3. **Root-cause reduction (fewer triggers):**
   - Strengthen the PX4 readiness gate (`run_px4_integration_test.sh:115-135`) to
     wait for a *flight-ready* PX4 marker (or first heartbeat) rather than only
     TCP-port-open, shrinking the bootstrap first-actuator timeout.
   - Avoid co-scheduling the two heavy PX4 flights (`test-integration-px4-takeoff`
     and `-instability`) concurrently, or reconcile the `INTEGRATION_TEST_JOBS`
     guidance (CLAUDE.md says 3, config.mk says 2) to cut host contention.
   - Possibly relax `first_actuator_timeout_s` modestly (config) for slow boots --
     bounded by CLAUDE.md's rule that it must exceed PX4's boot need; the 1.0 s
     mid-flight `lockstep_timeout_s` should not be relaxed (weakens lockstep).

## Key files (for whoever implements)

- `make/run_px4_integration_test.sh` (readiness gate, cleanup/pass-fail print).
- `make/integration-tests.mk` (targets, macros, `test-integration-all` fan-out).
- `make/config.mk` (filters, `INTEGRATION_TEST_JOBS`, `PX4_TEST_READY_TIMEOUT_S`).
- `isaac/runtime/standalone/tests/_px4_takeoff_scenario.py` (assertions,
  `ExpectedInstabilityDetected`, envelope, failsafe raise site).
- `isaac/runtime/standalone/tests/test_px4_takeoff_instability_detection.py`
  (catches `ExpectedInstabilityDetected`).
- `configs/experiment/px4/default.yaml` (lockstep/boot timeouts).
- `flight_analysis/` (home candidate for a classifier; existing ULog tooling).

## Verification (when implementation resumes)

- Loop `make test-integration-px4-takeoff` N times; confirm the classifier labels
  each run correctly against the bridge/isaac logs.
- If retry is implemented: confirm INFRA_FLAKE retries (artifacts preserved per
  attempt) and REGRESSION / STABILITY_BREACH fail immediately with no retry.
- Confirm `test-integration-px4-takeoff-instability` still detects a genuine
  envelope breach and is **not** satisfied by a PX4 failsafe flake.
- CLAUDE.md must be updated for any behavior/workflow change (retry semantics,
  classification output, readiness-gate change).

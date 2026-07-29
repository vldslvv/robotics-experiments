# Controller alignment subsystem

This document is the contract and operational guide for aligning the
high-throughput Isaac Lab velocity controller to the PX4 SITL closed-loop
response. Alignment is upstream of RL: policy training consumes a published
alignment artifact, while trained-policy PX4 missions remain downstream
acceptance and never fit controller settings.

## Package and dependency boundary

`robotics_experiments.alignment` is host-safe and imports without Isaac Sim,
Kit, Isaac Lab, or a PX4 container. Its public API is exactly:

- `ControllerDefinition`
- `ControllerSettings`
- `AlignmentArtifact`
- `load_controller_definition(path)`
- `load_alignment_artifact(path)`
- `align(target, reference)`

The package direction is `config -> alignment -> rl -> workflows`.
Alignment may consume the public config boundary and RL may consume
`ControllerSettings`; alignment must not depend on RL. The V1 concrete
backends are `isaac-lab` as the target and `px4-sitl` as the reference.

Controller definitions live under `configs/alignment/controllers/`. A
definition fixes the controller identity, backend, experiment and vehicle,
policy-facing body-FLU velocity interface, 20 Hz command cadence, complete base
settings, fit and validation probes, declarative parameter candidates, named
selection metric, and image identity.

`ControllerSettings` owns all controller and actuator internals:

- command delay;
- per-axis velocity P and damping gains;
- attitude and body-rate gains;
- maximum commanded tilt;
- thruster rise and fall time constants;
- per-rotor maximum thrust rate.

`rl.action` owns only normalized action kind, dimension, and scales. Adding an
existing settings field to a fit is a controller-definition YAML edit. Adding a
new setting requires extending `ControllerSettings` and the Isaac Lab binding,
not the candidate expansion, ranking, Make, or publication loops.

## Probe and evaluator contract

Alignment uses policy-independent body-FLU velocity probes. Fit and validation
each cover x, y, and z independently. Fit profiles contain settling, positive
and negative steps, reversals, and return-to-zero settling. Validation uses
different amplitudes and ramp/reversal shapes.

The PX4 capture uses the normal fleet driver and an explicitly injected
`PolicyAdapter`; there is no new commanding routine kind. The adapter streams
the deterministic probes through the existing `{kind: policy}` OFFBOARD
lifecycle. Every trace records:

- the actual initial position, attitude, linear velocity, and angular velocity;
- exact command timestamps and body-FLU commands;
- measured body-FLU velocity;
- completion or flight-envelope failure.

The Isaac Lab sweep initializes each replay from the corresponding PX4 state
and advances the exact captured command intervals. It keeps one hover
environment alive for the sweep and reapplies the complete controller settings,
including a fresh delay buffer and actuator state, before each candidate. An
interval must be exactly representable at the configured physics rate.
Non-finite output, a backend exception, or an incomplete trace makes only that
candidate unusable. If all candidates are unusable, alignment fails and
publishes nothing.

## Metrics and ranking

Metric selection is by stable registry name. V1 uses
`worst-axis-validation-velocity-nrmse-v1` and ranks:

1. lowest worst-axis validation velocity NRMSE;
2. lowest aggregate validation velocity RMSE;
3. lowest fit velocity RMSE;
4. stable candidate order.

The artifact also reports per-axis error, aggregate error, overshoot, settling,
and stability diagnostics. Fit and validation samples are never pooled for
selection.

The advisory quality target is worst-axis validation NRMSE at most 15%, with
all selected traces finite and completed without a flight-envelope failure.
Poor quality is visible and produces a training warning, but a structurally
valid artifact remains loadable and consumable.

Changing the definition, probe suite, metric identity, resolved experiment,
image identity, or command interface changes the comparison fingerprint.
Results from different comparison fingerprints are separate generations and
are never ranked against one another.

## Artifact and publication contract

Completed immutable runs live at:

```text
.build/alignment/<alignment-id>/<run-id>/
```

`alignment.json` is the only training contract. Reference and target trace
JSON files in the run directory are diagnostics. The artifact includes:

- schema, alignment, and run identity;
- target/reference definition and comparison fingerprints;
- experiment, vehicle, airframe, source commit/dirty state, and images;
- probe and selection-metric identities;
- complete selected `ControllerSettings`;
- declared parameters and every evaluated candidate;
- selected ranking values, fit/validation metrics, and advisory quality.

Loading rejects malformed schemas, missing provenance, non-finite settings,
bad vector shapes, unsupported backend/interface contracts, and an invalid
selected candidate. Advisory quality failure is not a loader error.

Publication uses atomic relative directory symlinks:

- `latest` always advances to the newest completed run;
- `best` advances only for a better comparable ranking;
- the first completed run in a new comparison generation becomes `best`,
  regardless of advisory quality.

## Operational workflow

Defaults select the Isaac Lab and PX4 SITL x500 velocity definitions. Override
them with `ALIGNMENT_TARGET`, `ALIGNMENT_REFERENCE`, and an optional stable
`ALIGNMENT_RUN_ID`.

```console
make alignment-capture-reference
make alignment-sweep-target
make alignment-fit
```

The end-to-end wrapper preserves one run ID across all stages:

```console
make align-controllers
make test-integration-alignment-target
```

Separate Make invocations should author the same `ALIGNMENT_RUN_ID`. Capture
writes `reference.json`, sweep writes `target_candidates.json`, and fit writes
`alignment.json` before updating `latest` and `best`.

Training resolves `rl.training.alignment_artifact`, which defaults to the
alignment namespace's `best` link. `ALIGNMENT_ARTIFACT` or
`--alignment-artifact` selects an exact run for reproduction or ablation.
Training validates vehicle, airframe, cadence, interface, action dimension, and
action scales before constructing Isaac Lab. It stores the selected settings,
artifact digest, run identity, definition fingerprints, and comparison
fingerprint in `policy_meta.json` before PPO starts.

Checkpoint selection, export, and independent evaluation reconstruct the
controller from that embedded record. Deployment validates the alignment
record but continues to use only `PolicyContract` for OFFBOARD interpretation.
Policy artifacts without alignment provenance are unsupported.

After training and Lab acceptance, run the wall-free
`test-integration-press-policy-transfer` mission. It is a policy-transfer gate,
not an alignment input. The physical press mission remains separately gated.

## Deferred work

Parameter uncertainty distributions, domain randomization, real-flight
capture, residual response models, and PX4-in-loop training are deliberately
deferred. Adding a backend requires an evaluator adapter and a new stable
backend contract; it must not weaken V1 artifact validation or comparison
fingerprinting.

# Generated simulation assets

Prepare assets with one command:

```bash
make generate-assets HYDRA_CONFIG=configs/train/press_curriculum_quad.yaml
```

The entrypoint resolves an experiment or every experiment in a curriculum, then
calls the drone and switch builders in that order. It generates every named
profile and any distinct switch model authored by those experiments. Add future
generators as ordinary explicit calls; there is no discovery or workflow engine.
The individual builder scripts remain available for focused development.

One `.build/assets/generation.lock` serializes Make invocations. Drone files keep
their existing text-USD names and locations. Writes are atomic and identical files
retain their bytes and modification times. A failure returns nonzero, preserves
completed outputs, and prevents the dependent runtime from starting.

Isaac startup, training, evaluation, and the integration/alignment/flight targets
prepare assets before launching workers. Configuration loading and simulator
workers only read assets. All curriculum models are prepared before stage one.

Switch bundles live at `.build/assets/light-switches/<fingerprint>/`. Each contains
`switch.usda` and `manifest.json`, recording the resolved model, generator revision,
definition fingerprint and USD digest. Completed bundles are immutable. A changed
model or generator revision produces another directory; altered or stale contents
fail validation. Training pins the canonical path, fingerprint and digest before
worker startup, rechecks them in workers, and records them in schema-30 reports,
evaluation evidence and supplemental policy metadata. Switch profiles affect
comparison identity but not complete-learner compatibility.

## Switch definition

`configs/experiment/switches/model/nominal.yaml` owns the mechanics; `easy.yaml`
changes only the button dimensions. Both compose into `switches.model`.

| Property | Nominal | Easy |
| --- | --- | --- |
| Button width / height / thickness | 60 / 80 / 10 mm | 90 / 120 / 15 mm |
| Activation / maximum displacement | 10 / 20 mm | 10 / 20 mm |
| Activation force | 2.6 N | 2.6 N |
| Moving mass / damping ratio | 10 g / 0.7 | 10 g / 0.7 |
| Confirmation duration | 0.25 s | 0.25 s |

The selected force is the lower end of Legrand Mallia's documented 2.6-3.5 N
lighting push-button handling-force range:
[manufacturer datasheet](https://assets.legrand.com/pim/NP-FT-GT/F01402EN-00.pdf).
The displacement, travel, mass, damping, confirmation duration and linear spring
response are simulation choices, not manufacturer measurements.

Zero preload gives `stiffness = activation_force / activation_displacement`, or
260 N/m, and `damping = 2 * damping_ratio * sqrt(mass * stiffness)`, about
2.26 N s/m. The spring approaches 5.2 N immediately before the stop. Measured
contact force also includes damping while moving and the mechanical-stop reaction.

Local -Y remains outward, pressing travels along +Y, and the button rest center
remains `(0, -0.013399239124577633, 0)` metres. Four rails and a 3 mm back plate
retain the 100 by 150 mm housing face while leaving 2 mm clearance around the
complete swept button volume. Switch collision offsets are 0.5 mm, with zero rest
offset. Configuration rejects invalid mechanics, obstructed geometry, activation
outside usable travel, and insufficient clearance from the configured wall.
The binary switch, lamp and warehouse source assets are unchanged.

## Activation and contact

The host-safe `robotics_experiments.switches` transition is shared by Isaac and
Lab. Every physics sample contributes its simulation-time interval. A sustained
press toggles once; holding at the stop does not repeat. Releasing below activation
rearms it. Explicit resets and per-environment resets clear its memory. Policy
cadence has no effect on confirmation timing.

Lab observes the resulting lamp state and uses the same toggle events for reward
and episode counts. Pad/button contact is allowed throughout the entire stroke.
Pad/housing, pad/wall, other pad contacts and drone-body collisions remain failures;
hover and approach remain contact-free. Contact categories use sums of individual
normal-force magnitudes, so opposing contacts cannot cancel. Contact-task reports
retain each category and contacts simultaneous within a physics step. Retreat
holds, compression limits, rewards and acceptance remain task-owned.

The first four shipped press curriculum stages and matching runtime experiments
use easy; the final stage and generic switch scenes use nominal. There are no
post-load button geometry/stiffness scales or duplicate latch thresholds.

`make test-integration-switch-mechanics` runs controlled physical pushers against
both profiles, including full travel and deliberate housing, wall and simultaneous
contacts. Speeds and tolerances live in `switches.verification`. It retains raw
force/displacement/velocity/category samples under `.build/assets/switch-mechanics/`.
Feasibility `--output-dir` paths are relative to the repository, so reports remain
on the host after the container exits. Interrupting evaluation records failure
and makes the Make invocation fail. Both press diagnostics build pressure
gradually using their shared stage's Hydra feedback setting; approach speed and
acceptance limits are unchanged.
The aggregate Docker suite includes this check and the drone press/lamp/retreat
check. Feasibility checks use validation seeds and normal task termination;
procedural training smokes never use reserved test seeds or publish policies.

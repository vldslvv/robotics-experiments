# F3b -- Generic contact-policy deployment and scripted baseline

Status: PARTIALLY IMPLEMENTED; LEARNED-POLICY MISSION AWAITS AN ACCEPTED F3
ARTIFACT; SCRIPTED BASELINE REMAINS PLANNED.

Goal served: commanding must not know whether a policy hovers, presses a switch,
or performs another learned task. Every policy uses `{kind: policy}` and emits
the configured OFFBOARD setpoint type. Task-specific observation assembly,
progress, completion, and failure belong to a runtime adapter selected by the
artifact contract. There is intentionally no `{kind: press}`.

## Implemented boundary

The generic commanding contract is now:

```text
RoutineObservation + opaque frozen progress
                    |
                    v
              PolicyAdapter.step
                    |
                    v
PolicyStepResult(progress, setpoint, complete, failure_reason)
```

Commanding owns OFFBOARD priming, inference cadence, zero-order hold,
transmission cadence, mode acknowledgement, timeout, and flight-envelope
checks. It stores opaque adapter progress but never imports a press type or
branches on a task name. The runtime reads `policy_interface` from
`policy_meta.json` and currently composes either `hover-v1` or `press-v1`.
Exact legacy hover schema-v1 metadata is recognized; a press artifact without
an interface must be re-exported so it cannot be guessed from an ambiguous
shape.

`press-v1` keeps the trained contract deliberately concrete: exact 34-value
`PressObservation` in, normalized four-value action out, then the existing
configured action-to-OFFBOARD mapping. It is generic at the commanding seam,
not an attempt to invent a universal tensor format.

The private Isaac Sim 5.1 runtime adapter owns live external handles. Before
physics starts it applies the easy-task button geometry/stiffness and pusher
limits, requires enabled arm and swept-propeller collision, authors an invisible
static wall collider behind the selected warehouse switch, and creates the pad
contact sensor. At policy ticks it reads canonical switch pose, vehicle
kinematics, pusher compression/rate, contact force/flag, and lamp state, then
injects a frozen `PressObservation`. The pure `PressDeploymentState` transition
owns handoff, previous action, first-contact evidence, lamp-transition count,
and stable retreat. It fails closed on missing observation groups, an
initially-on lamp, invalid policy output, or more than one lamp transition.

The switch/lamp lifecycle also moved into the shared frozen `FleetState`.
`experiment_runtime` is the single owner; the light-switch extension is now a
thin ROS publisher over immutable ordered toggle events. This prevents two
extensions from spawning or advancing the same physical switch.

## Mission sanity gate

`configs/policy_press_quad.yaml` is the top-level experiment and
`configs/experiment/routine/policy_press.yaml` is the routine:

```text
arm -> takeoff -> hover -> goto handoff -> hover -> policy
     -> goto landing point -> land
```

`make test-integration-press-routine` runs the strict PX4/Isaac scenario. It
passes only when the routine records stable first contact, exactly one off-to-on
lamp transition, configured stable retreat, return waypoint, landing, and
disarm. It is artifact-dependent and therefore is not part of
`test-integration-all`.

The gate must be run only after `make eval-press` passes the F3 thresholds. The
current `.build/training/press/latest` predates `policy_interface` and measures
0% toggles, 100% stable first contact, and 0% retreat. Its exact legacy metadata
tuple is accepted with a warning for diagnostic deployment at the recorded 0.5
m/s scale, but it was trained against the obsolete floating pad-only geometry
and remains unaccepted. Do not use it to judge the corrected complete pusher
tool. The staged second tuning cycle has not been started.

## Remaining Step 1 -- produce an accepted learned artifact

First rerun `make test-integration-press-contact-chain` and confirm the corrected
mount/boom/plunger/pad geometry, full initial deployment, and arm/propeller/wall
collision contract. Then resume the exact staged F3 run documented in
`F3_TRAINING_DECISION_LOG.md`, evaluate all three non-training seeds, and stop
unless all configured thresholds pass. If an intermediate checkpoint is better
than the final checkpoint, export and evaluate it rather than rerunning PPO
blindly. The exported metadata will include `policy_interface: press-v1` and
Git provenance.

Done when: press success is at least 80%, stable first contact at least 85%, and
retreat among toggled episodes at least 95%.

## Remaining Step 2 -- execute the full learned-policy mission

Run `make test-integration-press-routine` with the accepted artifact. Diagnose
any gap between Lab evaluation and PX4 deployment as a contract/parity problem:
canonical pose, observation order, action scaling, pusher mechanics, contact
threshold, latch cadence, or flight-controller dynamics. Do not tune around a
schema mismatch.

Done when: the strict mission completes takeoff through disarm with one stable
press and retreat.

## Remaining Step 3 -- scripted baseline behind the same policy seam

Add a pure frozen scripted contact transition over `PressObservation`. It should
approach along the contact-frame normal, align laterally/vertically, ramp a
bounded post-contact press using compression as the force proxy, confirm one
lamp transition, and retreat. Detection loss, compression overshoot, or
sustained misalignment must retreat and fail after a bounded retry budget. All
thresholds belong in Hydra config.

The script must be exposed through `PolicyAdapter`; it must not add a new
routine kind. Because scripts have no TorchScript artifact, selection needs an
explicit policy-adapter descriptor or an artifact-like metadata directory. Pick
one representation once there are two working adapters, document it, and keep
commanding unchanged.

Done when: scripted and learned implementations can occupy the same `policy`
step and differ only in runtime adapter composition.

## Remaining Step 4 -- comparative metric floor

Run the scripted and learned adapters against the same press environment and
print the same table: confirmed toggle success, stable first contact, retreat
among toggles, median time-to-toggle, and peak compression overshoot. The
scripted result is the baseline floor; a learned policy is selected by config
only when its tradeoff is explicit.

## Exit gate

An accepted press artifact completes the strict full-flight mission through the
generic policy seam, and the scripted baseline uses that same seam with a
side-by-side metric table. Future contact tasks add observation/adapter logic,
not commanding branches.

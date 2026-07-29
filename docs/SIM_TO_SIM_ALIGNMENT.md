# Sim-to-sim control alignment

This guide defines how to interpret and improve the control-response alignment
between Isaac Lab training and PX4 SITL deployment. It complements
`docs/RL_WORKFLOW.md`; that document remains the source of truth for training,
artifact selection, and deployment gates.

## Current control boundary

The policy interface is shared, but the plants below it are not:

- In Isaac Lab, the normalized policy action enters the project velocity
  controller, rotor allocation, and Isaac Lab thruster dynamics. Lab applies
  the resulting wrench directly to the simulated articulation. PX4 is bypassed
  so hundreds of environments can collect transitions concurrently.
- In deployment, the same policy observation, action scaling, cadence, and
  frame convention produce a PX4 OFFBOARD velocity setpoint. PX4 SITL owns
  estimation, cascaded flight control, mixing, and the actuator commands
  returned through the HIL bridge before Isaac applies dynamics.

This makes Lab a controller surrogate, not a second execution of the PX4
control stack. Matching the policy-facing interface is necessary, but it does
not guarantee matching closed-loop velocity response, braking, saturation, or
settling.

PX4-guided `goto` speed and policy alignment solve different parts of the
mission. Guided repositioning controls the pre-policy staging and post-policy
return trajectories. Temporal alignment still matters during the policy step,
when the actor streams velocity commands through PX4 OFFBOARD.

## Why use a surrogate

Putting a complete external flight-controller instance behind every training
environment would constrain parallelism, add orchestration and synchronization
cost, and make resets more expensive. A compact simulator-native controller is
a practical way to obtain high-throughput RL. The cost is a model discrepancy
that must be measured and managed explicitly.

Public sim-to-real practice treats this as an empirical transfer problem:

- [Tan et al.](https://roboticsproceedings.org/rss14/p10.html) combine system
  identification, an actuator model, simulated latency, physical-parameter
  randomization, and real-robot evaluation.
- [Molchanov et al.](https://arxiv.org/abs/1903.04628) evaluate how noise and
  domain randomization affect low-level policies transferred to multiple
  quadrotors.
- [BayesSim](https://roboticsproceedings.org/rss15/p29.html) infers a posterior
  distribution over simulator parameters from observed trajectories instead of
  treating one fitted parameter vector as ground truth.
- [Swift](https://www.nature.com/articles/s41586-023-06419-4) trains its racing
  controller in simulation, fits empirical perception and dynamics residual
  models from physical data, performs controlled ablations, and validates the
  complete system on the physical vehicle.

The corresponding project practice is:

1. Preserve policy-interface parity across training and deployment.
2. Identify response parameters from policy-independent and policy-driven
   traces.
3. Select parameters on held-out trajectories, not only fitted traces.
4. Model actuator dynamics, response lag, command delay, and saturation as
   distinct effects where the data supports them.
5. Train over identified parameter distributions when a point estimate is not
   stable.
6. Use a residual model only when a simpler physical model leaves repeatable
   structure in the error.
7. Treat final PX4 SITL, HIL, or hardware evaluation as the acceptance
   authority.

## Current temporal-fit evidence

The wall-free temporal experiment used the production actor at nominal 0.460,
0.475, and 0.490 m entry ranges. It fitted the two outer traces and reserved
the middle trace for validation.

| Lab command delay | Outer-trace RMSE | Held-out RMSE |
| --- | ---: | ---: |
| 0.000 s | 0.03640 m/s | 0.02237 m/s |
| 0.200 s | 0.02455 m/s | 0.01302 m/s |
| 0.344 s | 0.01897 m/s | 0.01807 m/s |
| 0.392 s | 0.01814 m/s | 0.02076 m/s |
| 0.440 s | 0.01795 m/s | 0.02355 m/s |

The 0.440 s candidate cut outer-trace error by about half, so the experiment
was useful: it demonstrated that the original Lab response was materially too
fast and supplied a plant under which to retrain. It also performed worse on
the held-out trace than both zero delay and 0.200 s. The value therefore
overfit the two outer traces.

A fresh wall-free capture on 2026-07-29 used the retrained 0.200 s actor. The
measured PX4 entry ranges were 0.47310, 0.49502, and 0.49941 m; the nominal
labels remain identifiers, while the analyzer initializes Lab from the
measured state. Replaying the exact actions and 40/60 ms cadence produced:

| Lab command delay | Outer-trace RMSE | Held-out RMSE |
| --- | ---: | ---: |
| 0.000 s | 0.03197 m/s | 0.03037 m/s |
| 0.200 s | 0.02423 m/s | 0.02071 m/s |
| 0.440 s | 0.02579 m/s | 0.02540 m/s |

The fresh evidence selects 0.200 s on both fitted and held-out error. Its
closest-range deltas were -0.04117, -0.02319, and -0.01234 m: Lab still moved
farther than PX4, so the delay choice is validated but full plant alignment is
not complete. The run-scoped reports are under
`.build/benchmarks/press-temporal/20260729T031246Z/`.

`controller_command_delay_s` is an equivalent fitted proxy. It may absorb
estimator behavior, controller bandwidth, actuator response, state-dependent
braking, and unmodeled dynamics. It is not evidence that PX4 contains a literal
440 ms command delay.

The deployed diagnostic actor was trained with 0.440 s configured and remains
rejected by its original collision and minimum-range gates regardless of the
manual run evidence. New training uses the held-out-best 0.200 s candidate and
actor-only initializes from that diagnostic actor's latest checkpoint. This
plant change does not upgrade the old actor: the new run must repeat checkpoint
selection, independent Lab evaluation, and PX4 acceptance gates.

## Immediate alignment work

For the next identification cycle:

- Include zero and low-delay candidates in every sweep.
- Rank candidates by held-out evidence first, with fitted-trace error reported
  separately.
- Collect policy-independent step, ramp, and reversal commands across axes,
  amplitudes, and operating points. Keep policy traces as an end-to-end check,
  not the only excitation source.
- Fit pure command delay separately from response bandwidth, rise/fall
  behavior, slew limits, and saturation. A single delay should not stand in for
  every response effect when the traces distinguish them.
- Run controlled ablations that hold the actor fixed while changing the plant,
  and hold the plant fixed while changing the actor. Record exact artifact,
  config, and trace provenance.

## Longer-term alignment work

- Identify per-axis delay, bandwidth, saturation, and actuator rise/fall
  parameters rather than assuming one scalar describes the full vehicle.
- Estimate uncertainty or a credible parameter range, then randomize training
  within the identified range instead of optimizing against one fitted point.
- Reserve trajectories, entry states, and command shapes that are never used
  for fitting. Periodically rotate in new held-out data without leaking it into
  the fit.
- Re-identify after PX4, estimator, airframe, mass, actuator, or simulator
  changes.
- Where the remaining error is structured and repeatable, evaluate a small
  residual response model with a simpler model as an ablation.
- Use PX4-in-loop fine-tuning or residual adaptation only when its added runtime
  cost is justified, and retain independent PX4 SITL, HIL, or hardware
  acceptance afterward.

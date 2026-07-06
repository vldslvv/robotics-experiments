# F5 -- Synthetic pose estimator provider (introduce noise into ground truth)

Goal served: goal 3 (introduce noise into ground-truth pose). Depends on: F4 (a
policy robust to real geometry/force under DR). Ends Phase F.
No deferred decisions.

Why this exists: F2/F3/F4 feed the policy an exact ground-truth switch pose. Real
perception fails through latency, correlated bias, confidence collapse, bursty
dropouts, and gross outliers. This plan injects those failures BEHIND the F2
provider seam, so the same policy input hardens against imperfect pose -- with
"policy API change: none." It stops at synthetic noise: real RGB-D/LiDAR/fusion
providers are Phase G (decision recorded 2026-07-01), plugged into the same seam.

---

## Step 1 -- Pure synthetic-estimator math

- Action: create `src/robotics_experiments/rl/press/_synthetic_pose.py` (pure; numpy +
  stdlib, seedable). Implement the noise models from ROADMAP1's appendix as pure
  functions over a ground-truth `PoseSample`:
  - position noise (Gaussian), orientation noise (Gaussian on axis-angle),
  - latency (fixed + jitter) via a small delay buffer,
  - dropout (Bernoulli + burst process) -> `valid=False` / stale hold,
  - bias drift (random walk / OU),
  - outliers (mixture: rare large pos/yaw shifts),
  - confidence erosion (deterministic map from injected noise/age -> `confidence`),
  - `age_s` set from the latency/hold state.
- Keep it a deterministic function of `(ground_truth_sample, rng_state,
  estimator_state)` so it is reproducible and unit-testable.
- `__all__`: `SyntheticPoseEstimator` (the stateful wrapper) + config type.

## Step 2 -- Wire it as an `AbstractPoseProvider`

- Action: create `SyntheticPoseProvider` implementing F2's `AbstractPoseProvider`,
  composing a ground-truth provider (F2 Step 4) with the Step-1 estimator: it calls
  the ground-truth `sample()`, passes it through the estimator, and returns the
  degraded `PoseSample` (noisy pos/quat, lowered `confidence`, raised `age_s`,
  possibly `valid=False`).
- INVARIANT (assert in test): `PressObservation` shape and `to_vector` length are
  identical whether the env uses `GroundTruthPoseProvider` or
  `SyntheticPoseProvider`. Only the values change.
- Action: select the provider by config (`press.perception.provider:
  ground_truth | synthetic`) so training/eval can switch without code changes.

## Step 3 -- Config

- Action: add `PressPerceptionConfig` (provider selector + the synthetic-estimator
  parameters) to `config/types.py` and a `perception:` block to
  `configs/experiment/rl/default.yaml`, authored from ROADMAP1's appendix YAML:
  `publish_rate_hz`, `position_noise_std_m`, `orientation_noise_std_deg`,
  `latency` (base/jitter/max), `dropout` (bernoulli_p, burst enter p, burst
  length), `bias_random_walk`, `outliers` (p, shift ranges),
  `confidence_model` (min/nominal/age penalty). Add `perception: PressPerceptionConfig`
  to `RlConfig`.
- Also activate the `pose_noise` dimension of the F4 curriculum (which was 0 until
  now): the noisy-perception curriculum stages ramp these ranges.
- Rule: every stochastic parameter is config-driven and seedable.

## Step 4 -- Exercise handoff + a bounded retry stub

- Action: the degraded `confidence`/`age_s`/`valid` now make F2's `handoff_ready`
  meaningfully reject stale/low-confidence situations. Add a minimal, bounded
  local retry at the env/episode level: on a rejected handoff or lost detection,
  fall back to a hold/re-approach for a bounded number of attempts before aborting
  the episode. This is a LOCAL stub only -- full mission-level retry/abort is Phase
  H (mission manager). Keep the retry count in config.
- Done when: a stale/dropped pose sequence causes the handoff to reject and the
  bounded retry to trigger, rather than a bad-geometry contact.

## Step 5 -- Harden the policy

- Action: continue training / fine-tune the F4 policy with
  `press.perception.provider: synthetic` and the noisy-perception curriculum stages
  active. Log the stale-observation failure rate specifically.
- Metrics targets (ROADMAP1): press + retreat remain stable under the configured
  delay/dropout/noise; stale-observation failure rate near zero; false-handoff rate
  <= 1-2%.
- Done when: the same policy weights (or a light fine-tune -- record which) hold the
  press+retreat task under synthetic perception noise at target metrics.

## Step 6 -- Tests

- Action: `tests/test_press_synthetic_pose.py` (pure): seeded runs reproduce a
  sample sequence; dropout produces bursts (not i.i.d.); confidence decreases with
  injected noise/age; bias drift accumulates; outliers occur at the configured rate.
- Action: extend `tests/test_press_handoff.py`: fed synthetic samples, the gate
  rejects on `STALE` / `LOW_CONFIDENCE` at the configured thresholds.
- Action: a provider-equivalence test: for the same ground truth, ground-truth and
  synthetic providers yield identical-shape observations (values differ) -- the
  "policy API change: none" guard.

## Files touched by F5

- add `src/robotics_experiments/rl/press/_synthetic_pose.py`; edit `.../_provider.py`
  (or a new `SyntheticPoseProvider`), `rl/press/__init__.py`
- edit `config/types.py` (`PressPerceptionConfig`),
  `configs/experiment/rl/default.yaml`
- edit the env to select the provider by config + the bounded retry stub
- add `tests/test_press_synthetic_pose.py`; extend `tests/test_press_handoff.py`

## Exit gate (also the Phase F exit gate)

The policy holds the press + retreat task under synthetic pose noise
(latency/dropout/bias/outliers/confidence erosion) at the target metrics, with the
handoff gate correctly rejecting stale/low-confidence situations and the bounded
retry engaging instead of a bad contact -- all with the exact same policy input
tensor as ground truth. Phase F is complete. Phase G plugs real sensor/fusion
providers into the unchanged F2 seam.

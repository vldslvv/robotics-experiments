# Light-Switch Multirotor Roadmap

> Canonical merged plan for the light-switch multirotor project. Historical
> source documents are `docs/archive/QUADCOPTER_PLAN_REVISED.md` and
> `docs/archive/HOLYBRO_X8_STAGED_BOM_v2.md`; keep them for context, but update
> this document first for roadmap and platform decisions.

## Scope and Sources

This roadmap describes a PX4-driven Isaac Sim multirotor that can navigate to
wall light switches, press them with a compliant pusher, confirm the linked lamp
state, retreat, and eventually discover switches autonomously.

Source authority is split deliberately:

- `configs/airframes/holybro_x500_v2.yaml` is authoritative for the implemented
  Stage 1 Holybro X500 V2 X4 component dimensions, masses, placements, PX4
  airframe leaves, and bridge thrust leaves. It drives the generated USD body
  geometry/properties, PX4 control allocation, and bridge thrust curve.
- `docs/archive/HOLYBRO_X8_STAGED_BOM_v2.md` remains hardware/procurement context
  for staged platform decisions, Stage 2 planning, weight/cost targets, and
  rejected hardware alternatives.
- `docs/archive/QUADCOPTER_PLAN_REVISED.md` is historical source material for
  mission architecture, sensor strategy, RL/contact-task design, risk analysis,
  and the longer autonomous-discovery roadmap.

The current repository has already moved beyond the older PID-first simulator
sequence. Flight control is now PX4 SITL connected to Isaac through the project
bridge, with command routines layered on top. Any reference to an internal PID
controller in this roadmap is a future physical-platform tuning concern or a
historical/rejected-plan note, not the completed runtime architecture.

## Current State vs Roadmap

| Phase | State | Notes |
|-------|-------|-------|
| A. Runtime and PX4 SITL foundation | Done | Frozen `FleetState` / `Px4BridgeState` transitions, `BridgeHost`, `BridgeHostGroup`, `VehicleState -> ActuatorCommands -> RotorDynamics`, lockstep send/poll, multi-vehicle send-all/poll-all, per-vehicle logging, and rotor visual spin are implemented. |
| B. Generated X4/hexa assets and PX4 airframes | Mostly done | `configs/airframes/holybro_x500_v2.yaml` now owns the Stage 1 component leaves that generate `holybro_x500_v2.usda`, body properties, bridge thrust constants, and `4019_none_x500_v2`; `hexacopter_s550.usda` remains a placeholder bound to generated `6001_none_hexa_x`. Stage 2 X8 remains pending. |
| C. Command routines and navigation primitives | Done for demos | Pure command routines cover arm, takeoff, hover, forward, absolute world-ENU `goto`, and land; live and standalone shells drive the same per-drone `RoutineDriverState`. |
| D. Switch/lamp environment | Mostly done | Switch/lamp spawning and the N-step light-state latch exist. Closed-loop integration from navigation handoff into an RL or contact press policy remains pending. |
| E. Hardware-faithful X4/X8 asset/config | Partially done | Stage 1 Holybro X500 V2 X4 USD/config is implemented with computed mass properties and generated PX4 airframe binding. Stage 2 X8 geometry, rotor mapping, and tuning remain pending. |
| F. RL press policy | Not done | Isaac Lab has a no-op DirectRLEnv hover skeleton. The switch-press task, reward, observation/action surfaces, curriculum, and trained policy are future work. |
| G. Sensor simulation/fusion and ROS sensor stack | Not done | D435i/RPLiDAR/optical-flow simulation, ROS topics, LiDAR SLAM, and EKF/UKF fusion remain roadmap items. |
| H. Full mission state machine and autonomous discovery | Future | Retry/abort behavior, wall scanning, switch discovery, multi-switch task queues, and full autonomous missions are not implemented. |

## Canonical Hardware Platform

The hardware platform is a staged Holybro X500 V2 build. Stage 1 is a fully
flyable X4 quadcopter. Stage 2 upgrades the same frame to X8 coaxial only when
flight data shows a need for redundancy, vibration improvement, or thrust margin.

### Stage 1: Holybro X500 V2 X4

Detailed Stage 1 component dimensions, masses, placements, rotor ordering, PX4
airframe leaves, and bridge thrust leaves live in
`configs/airframes/holybro_x500_v2.yaml`. The bullets and metrics below are
roadmap summaries; update the YAML first when a component property changes.

- Frame: Holybro X500 V2, 500 mm wheelbase, carbon fiber tube arms, bolt-on arm
  attachment, tall landing gear retained for later lower-prop clearance.
- Motors: 4x T-Motor MN4006 EVO KV380.
- Props: T-Motor P134.4, 13x4.4 inch carbon fiber, matched CW/CCW pairs.
- Battery: 4S 5000 mAh LiPo; 4S is the selected baseline voltage.
- ESCs: HGLRC BS30A BLHeli_S, 30 A continuous, DShot600. Buy all 8 upfront even
  though Stage 1 uses 4.
- Protection: 4 partial-arc prop guards sized for 13-inch props, not full rings.
- Compute/sensors: Jetson Orin Nano Super, Intel RealSense D435i, RPLiDAR A2,
  and PMW3901-class optical flow/ToF.
- Manipulator: compliant centerline pusher with a spring-damper prismatic joint,
  roughly 10-15 mm travel, and a flat pad near 30 mm diameter.

Current Stage 1 summary metrics from the airframe config, except endurance:

| Metric | Value |
|--------|-------|
| AUW | ~2,339 g including the pusher pad |
| Max thrust | ~6,260 g |
| Thrust-to-weight | ~2.68:1 |
| Hover throttle | ~57% |
| Expected hover time | ~14-16 min BOM estimate |
| Motor redundancy | None; single motor failure is catastrophic |

Stage 1 is adequate but not generous. It is the right place to validate takeoff,
navigation, switch/lamp interaction, and early contact-policy training, but
payload growth or aggressive real contact may pull the X8 upgrade forward.

### Stage 2: X8 Coaxial Upgrade

Stage 2 adds four lower motors on the same four arms, with matched 13-inch lower
props and lower guards. No frame, flight controller, companion compute, battery,
or sensor replacement is planned for the baseline upgrade.

- Additional propulsion: 4x MN4006 EVO KV380 and 4x P134.4 13x4.4 inch props.
- Prop strategy: matched top/bottom 13-inch props. Mixed-size alternatives were
  rejected because the 3-4% thrust gain does not justify two prop SKUs, more
  spares, harder balancing, and per-motor scaling complexity.
- Redundancy: single motor loss remains controllable; two-motor loss can be
  survivable depending on geometry and should trigger return/land.
- Retuning: PX4 airframe/mixer, motor numbering, attitude/rate gains, and the
  learned press policy all need validation or fine-tuning after the upgrade.

Target Stage 2 metrics from the v2 BOM:

| Metric | Mid X8 | Max X8 with 3D LiDAR |
|--------|--------|----------------------|
| AUW | ~2,713 g | ~3,024 g |
| Max thrust after coaxial loss | ~11,268 g | ~11,268 g |
| Thrust-to-weight | 4.15:1 | 3.73:1 |
| Hover throttle | ~49% | ~52% |
| Expected hover time | ~12-15 min | ~10-13 min |
| Motor redundancy | 2-motor loss survivable in favorable cases | same |

Upgrade triggers:

- Normal press maneuvers show motor saturation, with any motor above roughly 85%
  throttle.
- Vibration limits IMU, LiDAR, camera, SLAM, or policy performance beyond what
  filtering and mounting can fix.
- Payload expansion, especially 3D LiDAR or heavier compute, pushes Stage 1 below
  the contact-task thrust margin.
- The operating environment makes a quad's single-motor-failure crash risk
  unacceptable.

Do not upgrade merely because X8 is available. The staged architecture exists so
the upgrade is evidence-driven.

### Hardware Notes That Must Carry Forward

- The old MN3110/12-inch plan is rejected. Historical context: MN3110 KV700 with
  12x4.5 inch props was found to have insufficient real 4S thrust margin for
  this payload; the catalog thrust depends on larger props and 6S operation.
- The MN4006 EVO saves mass, runs cooler at hover, uses upgraded EZO bearings,
  and has a lower vibration profile than the rejected MN3110 choice.
- The X500 V2's adjacent motor spacing leaves about 24 mm tip-to-tip clearance
  with 13-inch props. Do not grow beyond 13x4.4 on this frame.
- Guard CAD must be sized for 13-inch props: 165 mm radius plus about 15 mm
  outboard clearance. Older 12-inch guard designs need resizing.
- The BS30A ESC is correct for the 4S baseline but is rated 3-5S. A future 6S
  migration requires ESC replacement, such as a 3-6S 35 A BLHeli_32 class ESC.
- Pusher clearance must be verified statically. On Stage 2 the pusher threads
  between upper and lower prop planes; pitch excursions and mount deflection must
  not put the pad or boom into a prop disc.
- Track motor runtime and plan preventive bearing maintenance; research flights
  generate many repeated hover/contact cycles.

## Mission Architecture

The target mission remains:

1. Take off and establish a stable hover.
2. Navigate to an approximate switch location.
3. Localize the switch with RGB-D perception or known simulation coordinates.
4. Hand off to a contact policy once pose, velocity, angular rate, and detection
   confidence are inside gate limits.
5. Press with the compliant pusher.
6. Confirm the light-state change through a latch.
7. Retreat to a safe hover point.
8. Repeat for more switches or land.

The implemented command layer covers the early flight primitives through PX4:
arm, takeoff, hover verification, world-ENU `goto`, forward moves, and land. The
future press policy should sit above PX4 as a local contact behavior, not replace
the bridge or the command routine contract.

### Runtime Boundary

- The bridge owns physics-cadence sensor/actuator lockstep with PX4.
- The command layer owns per-drone high-level commands and routine verification.
- The future mission manager owns task sequencing, retry/abort policy, switch
  queues, and handoff into the press policy.
- The RL policy owns only the local approach/press/retreat contact behavior once
  the drone is already near the switch and inside the handoff gate. It is one
  implementation of the contact-behavior seam (`docs/phase_f/F3b_contact_behavior_seam.md`);
  a scripted model-based press is the other, and the baseline.

### Safety Harness

Real indoor flights run under a physical safety harness. The harness is a
physics change (tether disturbance, snag hazard near the wall) and a software
contract (hardware abort profile, geofence sized to the harness volume, contact
entry/exit rules), not only a catch rope. `docs/SAFETY_HARNESS_PLAN.md` is the
plan for the architecture accommodations and the harness geometry rules; its
software steps are preconditions for the first harnessed contact flight.

## Switch, Lamp, and Pusher Design

Switch assets should remain articulated buttons with a thresholded displacement.
The lamp state is binary and linked to the switch through an N-step confirmation
window. This latch prevents brief contact chatter from producing false success:
a button must stay at or beyond threshold for the configured number of samples,
then it toggles once until released.

The pusher is a compliant mechanical element, not a rigid spike. Its spring
compression is part of the policy observation and acts as a force proxy. The
simulated geometry should preserve:

- A flat pad near the drone centerline.
- A spring-damper prismatic joint along the press axis.
- Travel limits that absorb impact without allowing full collapse.
- Collision geometry simple enough for stable contact-rich simulation.
- Mechanical clearance for Stage 1 quad and Stage 2 X8 prop planes.

## RL Press Policy Roadmap

The press policy is the core learned component. It should learn the full local
approach, contact, press, toggle confirmation, and retreat behavior.

Observation candidates:

| Observation | Source |
|-------------|--------|
| Drone pose relative to switch | Ground truth during early training; fused state later |
| Drone orientation, velocity, angular velocity | Bridge state or fused estimator |
| Switch button displacement | Articulation query |
| Light state | USD query in sim; camera/lamp detector later |
| Pusher compression | Pusher joint displacement |
| Contact flag | Contact reporter |
| Previous action | Policy action buffer |

Recommended action surface:

- Start with high-level local velocity deltas plus yaw-rate command, with PX4 or
  the existing flight stack converting the request to low-level control.
- Add an exponential moving average and an action-delta penalty to suppress
  high-frequency jitter.
- Avoid training direct per-rotor commands until the high-level policy ceiling is
  known.

Reward structure:

- Approach reward before contact.
- Press displacement reward only during contact.
- One-time toggle reward gated by the light-state latch.
- Retreat reward after confirmed toggle.
- Always-on stability, alive, energy, and action-smoothness terms.

Curriculum:

- Start close to the switch with no randomization until contact/press is learned.
- Increase start distance to the normal handoff distance and require retreat.
- Add domain randomization for switch position, button stiffness, wall friction,
  drone mass, motor response lag, observation noise, and pusher stiffness.
- Benchmark 16-32 parallel Isaac Lab environments before scaling; contact-rich
  physics is heavier than simple hover tasks.

Handoff gates before policy activation:

- Position error to the target hover point below roughly 0.1 m.
- Linear velocity below roughly 0.2 m/s.
- Angular velocity below roughly 0.3 rad/s.
- Switch detection confidence above threshold or an explicit simulation fallback.
- Localization confidence healthy.

Failure behavior:

- If switch localization fails, return to hover and retry the approach a bounded
  number of times.
- If press fails or contact destabilizes the drone, retreat, then retry a bounded
  number of times.
- Critical faults transition to abort/land and log the reason.

## Sensor Strategy

Core hardware sensors:

| Sensor | Purpose |
|--------|---------|
| Forward RGB-D camera, D435i-class | Switch detection, lamp-state perception, close-range depth, visual odometry support |
| IMU | Attitude, angular velocity, estimator input |
| Optical flow + ToF | Low-altitude velocity/height hold in GPS-denied hover |
| 2D LiDAR, RPLiDAR A2-class | Indoor SLAM, obstacle awareness, drift-limited planar pose |

Future autonomous-discovery sensors:

| Sensor | Purpose |
|--------|---------|
| 3D LiDAR, Livox Mid-360-class | Volumetric mapping and wall-surface extraction |
| Downward camera, optional | Backup visual odometry over textured floor |

Fusion plan:

- Use LiDAR SLAM for low-drift planar pose.
- Use IMU for high-rate attitude and angular-rate propagation.
- Use optical flow and ToF for low-altitude velocity/height stabilization.
- Fuse those streams through an EKF/UKF for smooth state to navigation and the
  press handoff.
- During RL training, start from ground truth and add observation noise; exercise
  the sensor stack during integration and sim-to-real validation.

3D LiDAR is not for detecting small wall switches directly. It maps wall
surfaces and free space. The RGB-D camera detects the visual switch candidate.

## Autonomous Discovery Roadmap

Autonomous discovery is a later mission layer built on top of the existing press
pipeline:

1. Build or ingest a 3D occupancy map.
2. Extract planar wall segments and normals.
3. Generate wall-scanning trajectories at camera-friendly stand-off distances.
4. Run a camera-based switch detector on RGB-D frames.
5. Add detections to a task queue.
6. Use the normal press pipeline for each queued switch.
7. Record discovered, toggled, skipped, and failed switches with reasons.

The future mission state machine should include `SCAN_WALL`, `APPROACH`,
`RL_PRESS`, `RETREAT`, `HOVER`, `LAND`, and `ABORT` states. Retry limits should
be explicit and bounded.

## Implementation Roadmap

### Phase A: Runtime and PX4 SITL Foundation

Status: done. Preserve the bridge contract and command-routine layering.

Regression guardrails:

- `import robotics_experiments` must work without Isaac, ROS, PX4, or pymavlink.
- Bridge lockstep remains send-before-poll; multi-vehicle sends all vehicles
  before polling any vehicle.
- `VehicleState` remains pose, velocity, angular velocity, and sim time only.
- Safety gates zero dynamics only; they do not skip the per-tick exchange.

### Phase B: Generated X4/Hexa Assets and Airframes

Status: mostly done. The Stage 1 Holybro X500 V2 X4 airframe YAML is now the
source for the MN4006/P134.4/13-inch/pusher component geometry and physical
leaves. Generated consumers derive the USD body mass properties, PX4 airframe,
and bridge thrust constants from those leaves. The S550 hex remains a runtime
placeholder, and the Stage 2 X8 hardware config is still pending.

Pending:

- Create or regenerate a Stage 2 X8 coaxial USD/config with lower motor mounts,
  matched lower props, lower guards, and correct rotor directions.
- Keep generated USD assets, body properties, bridge settings, and PX4 SITL
  airframes in sync with `configs/airframes`.
- Treat `hexacopter_s550` as a placeholder until replaced by the Stage 2 X8
  config.

### Phase C: Command Routines and Navigation Primitives

Status: done for demos. Continue extending through the pure routine contract.

Possible next increments:

- Add mission-level retry/abort states outside the per-drone command driver.
- Add switch-target approach routines that end at the RL handoff gate.
- Add metrics around command failures, envelope breaches, and failsafe causes.

### Phase D: Switch/Lamp Environment

Status: mostly done.

Pending:

- Wire press-policy observations to switch button displacement, pusher
  compression, contact flag, and lamp latch state.
- Add closed-loop tests where a drone routine reaches a handoff pose and the
  policy or scripted press toggles the switch.
- Keep the USD-query lamp state as the training reward signal; later add camera
  lamp-state perception as an observation/evaluation path.

### Phase E: Hardware-Faithful X4/X8 Asset and Config

Status: partially done. Stage 1 X4 is represented by `holybro_x500_v2`; Stage 2
X8 remains pending.

Tasks:

- Implement X8 lower motor mount geometry and lower prop guards.
- Map Stage 2 motor order to generated PX4 airframes.
- Verify Stage 2 pusher/prop/boom clearance.
- Document retuning steps and expected parameter differences.

### Phase F: RL Press Policy

Status: not done. Detailed, implementation-ready step-by-step sub-plans live in
`docs/phase_f/` (F1 in-Lab flight actuation, F2 observation interface + handoff,
F3 switch-press env on ground truth + first policy, F3b contact-behavior seam +
scripted press baseline, F4 contact curriculum + domain randomization, F5
synthetic pose estimator). Phase F stops at synthetic pose noise; simulated/real
sensor providers plug into the F2 provider seam in Phase G. The in-Lab RL action
level (velocity / CTBR / attitude / rotor) is a deferred decision surveyed in F1.
F3b starts only after the F3 first-policy gate: it puts the press behavior
behind a `ContactBehavior` seam so a scripted model-based press and the learned
policy are interchangeable implementations, with the scripted baseline as the
published metric floor -- the project goal is the contact task family, not one
policy.

Tasks:

- Build the Isaac Lab switch-press environment from the current no-op hover
  scaffold.
- Implement observations, action surface, reward components, reset logic,
  contact reporting, and curriculum.
- Train and evaluate with domain randomization.
- Define handoff success/failure metrics before integrating into missions.

### Phase G: Sensor Simulation and Fusion

Status: not done.

Tasks:

- Simulate RGB-D camera, 2D LiDAR, optical flow/ToF, and IMU topics.
- Add ROS 2 topic publication/subscription for state, switch events, lamp state,
  camera images, and scans.
- Integrate LiDAR SLAM and estimator fusion.
- Stress localization with LiDAR occlusion, textureless optical-flow surfaces,
  IMU bias, and camera noise; abort instead of crashing when confidence degrades.

### Phase H: Full Mission and Discovery

Status: future.

Tasks:

- Implement mission-level state machine with bounded retry and abort handling.
- Add wall scanning and switch detector integration.
- Run multi-switch missions with known switch positions first, then autonomous
  discovery.
- Add mission reports: toggled count, skipped switches, failure reasons, total
  mission time, and safety events.

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Contact instability | High | Compliant pusher, smooth action surface, phase-aware rewards, strict handoff gates |
| Sim-to-real contact gap | High | Domain randomization over mass, friction, stiffness, actuator lag, sensor noise, and pusher parameters |
| Indoor localization failure | High | LiDAR SLAM + optical flow + IMU fusion; confidence gates and abort behavior |
| Stage 1 motor failure | High | Monitor throttle and motor health; upgrade to X8 if operational risk requires redundancy |
| Stage 1 thrust margin erosion | Medium | Treat payload additions as upgrade triggers; keep guards partial rather than full rings |
| X8 pusher/prop strike | Medium | Verify static clearance, limit pitch in policy, add sacrificial pusher mount if needed |
| Reward hacking | Medium | Phase-aware rewards, contact gating, approach constraints, retreat requirement |
| Training throughput | Medium | Benchmark env count before scaling and log reward components separately |
| Handoff instability | Medium | Gate on pose, velocity, angular rate, detection confidence, and localization confidence |
| Sensor vibration | Medium | MN4006 EVO bearings, isolation mounts, PX4 notch-filter tuning, Stage 2 lower hover throttle |
| Light timing false positives | Low | N-step light latch and persistent confirmation before success |
| Harness tether snag or taut-catch disturbance during contact | Medium | Slack-by-default harness geometry, static clearance check, taut-tether model in F4 randomization, abort profile + geofence (`docs/SAFETY_HARNESS_PLAN.md`) |

## Historical/Replaced Assumptions

These assumptions appear in the historical plan or calculators but are not the
current roadmap:

- MN3110 KV700 motors with 12x4.5 inch props are rejected hardware context.
- Historical thrust-to-weight figures such as 3.28:1 for Stage 1 and 4.66:1 for
  Stage 2 came from the rejected MN3110/12-inch analysis and should not be used
  for procurement or config work.
- A simulator-only PID-first hover sequence has been replaced by the PX4 SITL
  bridge and command-routine architecture. PID retuning remains relevant only for
  PX4/physical-airframe transitions such as X4 to X8.
- The removed `quadcopter_500mm` asset was a placeholder for runtime development,
  not a declaration that the final Stage 1 hardware was a generic square-X quad.
  The current `hexacopter_s550` asset remains a placeholder until the Stage 2 X8
  config replaces it.

# Multirotor Light Switch Interaction — Revised Architecture & Task Plan (v4)

> Historical source document. The canonical merged roadmap is
> `docs/LIGHT_SWITCH_MULTIROTOR_ROADMAP.md`; update that document first. This
> file preserves older architecture notes and contains stale hardware/control
> assumptions such as MN3110/12-inch propulsion and PID-first simulator phases.

> The platform decision is recorded as ADR 0001 (`docs/adr/`); this doc is the
> full architecture and task plan behind it.

## Overview

A simulated multirotor in Isaac Sim that can navigate to light switches on a warehouse wall, apply pressure to toggle them, and detect whether the associated lamp turned on or off. The system combines classical robotics for navigation and perception with RL for the fine-grained contact-based switch pressing task.

The platform is a **staged build on the Holybro X500 V2 frame**: Stage 1 is a conventional quadcopter (X4) for Phases 1–5 of the roadmap; Stage 2 upgrades to X8 coaxial octocopter by adding four motors to the undersides of the existing arms, providing motor redundancy and additional thrust margin for heavier sensor payloads. The X8 upgrade is triggered by empirical data from Stage 1, not built speculatively.

> **Revision notes (v4)**: This revision updates the platform from a generic 500mm-class hexacopter to the staged X500 V2 quad→X8 build based on detailed analysis of thrust-to-weight, prop geometry, vibration characteristics, and frontal clearance for the pusher mechanism. Also incorporates fixes from v3 for contact sensing, pusher compliance, handoff robustness, state machine error handling, reward conflicts, domain randomization coverage, action smoothing, light detection timing, retreat strategy, training resource sizing, sensor architecture with LiDAR/optical flow for indoor localization, and phase-evolving sensor strategy. Changed sections in v4 are marked with ✦; sections changed in v3 retain the ⚑ marker.

---

## 1. Drone Platform

### ✦ 1.1 Platform — Holybro X500 V2 Staged Build

The platform evolves across two hardware stages, sharing the same frame, flight controller, companion compute, sensors, and pusher mechanism. Only the propulsion layout changes.

**Stage 1 — Quadcopter (X4).** Four arms at ±45° from the forward axis, one motor per arm. Frame: Holybro X500 V2 (500mm wheelbase, 2mm carbon fiber twill plates, 16mm CF tube arms, bolt-on arm attachment). Motors: 4× T-Motor MN3110 KV700. Propellers: 12×4.5" polymer. Target AUW: ~1,830g. Thrust-to-weight ratio: 3.28:1.

**Stage 2 — X8 Coaxial Octocopter.** Same four arms, but with a second motor mounted on the underside of each arm spinning opposite direction, creating four upper/lower motor pairs. Added: 4× T-Motor MN3110 KV700 (identical to upper), 4× 12×4.5" polymer propellers (matched to upper props — mixed-size alternatives analyzed and rejected due to build complexity outweighing the 3–4% thrust gain), 4× lower motor brackets. Target AUW: ~2,320g (mid config) / ~2,631g (max config with 3D LiDAR). Thrust-to-weight ratio: 4.66:1 (mid) / 4.11:1 (max).

**Upgrade trigger.** Stage 2 is initiated when Stage 1 demonstrates specific need: motor saturation during contact (any motor >95% throttle), sensor vibration degrading SLAM/policy quality beyond software filtering, payload growth pushing quad T:W below 2.5:1, or operational risk assessment requiring motor-failure redundancy. Redundancy without demonstrated need is over-engineering; the quad is sufficient for most research needs.

**Why this platform over alternatives.** A 500mm hexacopter (S550/F550 class) was evaluated and rejected: the 60° arm spacing limits props to 9–10 inches, capping per-motor thrust and forcing higher RPM with correspondingly higher vibration. The X500 V2's 90° arm spacing allows 12–13" props and lower motor RPMs at hover, producing more thrust per motor with less vibration and better sensor data quality. A larger hexacopter (Tarot 680 with 13" props) is a viable alternative to the X8 upgrade but requires full rebuild rather than incremental upgrade; the staged X4→X8 path on one frame is the lower-risk option.

### 1.2 Low-Level Flight Controller (Classical)
- **PID or PX4-style rate/attitude controller** running at physics step rate.
- Inputs: desired thrust + torques (or attitude + thrust commands).
- Outputs: per-rotor RPM.
- This is deterministic controls math — no RL needed.
- Can use Isaac Sim's articulation or direct force API on each rotor joint.
- **Airframe mixing**: PX4 `Quad X` for Stage 1, `Octocopter Coaxial X` for Stage 2. Stage 2 transition requires PID retuning; budget 1–2 days.

### ✦ 1.3 Pusher Appendage Design

A rigid pusher striking a rigid wall creates brutal force spikes in simulation, causing instability and sim-to-real transfer problems. The pusher must include compliance.

**Mechanical design**: A small flat pad mounted on a spring-damper prismatic joint along the drone's forward axis.
- **Spring stiffness**: tuned so that the pad compresses ~5mm under the force needed to depress the switch button. This absorbs impact transients.
- **Damper coefficient**: set to ~0.7× critical damping to prevent oscillation.
- **Travel limit**: 10–15mm to prevent full collapse.
- **Collision geometry**: a flat cylinder or box, ~30mm diameter, centered on the drone's forward axis.
- **Total mass**: ~40–50g for the full assembly (pad + spring + bushing + carbon fiber boom + mount hardware).

The joint displacement of this compliant mount also provides a **contact force proxy** observable by the RL policy (see §4.2), removing the need for a simulated force-torque sensor.

#### Frontal clearance analysis

The pusher mounting location must account for propeller geometry. On any multirotor with forward-mounted props, the front prop discs can overlap the forward centerline, creating a no-go zone for centerline-mounted pushers at the prop plane.

On the X500 V2 with 12" props (152mm radius) and 354mm motor spacing, the two front motors sit at ±125mm lateral offset from centerline. Inner edges of the front prop discs reach to 125mm − 152mm = **−27mm**, meaning the prop discs overlap the forward centerline by 27mm. A pusher extending straight forward at the prop plane would be chopped.

**Solution: mid-plane forward mount (Stage 1 and Stage 2).**

The pusher mounts on the front edge of the center plate, extending forward as a carbon-fiber boom with the compliant pad at the tip. The boom passes at center-plate height, which is:
- **Stage 1 (quad)**: below the single prop plane — pusher sits beneath the props, no interference.
- **Stage 2 (X8)**: between the upper and lower prop planes — pusher threads through the ~80mm gap.

On the X8, the 80mm vertical gap between prop planes provides ~35mm clearance above and below a 10mm-thick pad. A 100mm-forward boom with the drone pitched 10° nose-down deflects the tip only 17mm, well within the 35mm budget. Normal operation with modest pitch variation stays in the safe envelope.

**Failure-mode consideration (X8 only)**: A violent pitch transient (>25° nose-down from a motor fault or disturbance) could deflect the pusher tip into the lower prop disc. The RL policy's axis-specific tilt termination (50° pitch max, see §4.5) provides the primary limit. For additional safety, the boom can be designed with a sacrificial shear point near the mount so a prop strike breaks off the pusher cleanly rather than damaging the motor.

**Alternative: under-frame mount (fallback for both stages).**

If mid-plane mounting proves problematic during build, mount the pusher beneath the center plate extending forward, passing under both prop planes. This trades a small nose-down approach angle (~5–10°) for larger vertical safety margin. The RL policy learns the approach angle naturally since forward flight requires forward pitch anyway.

**Why centerline-forward mounting works (contrary to offset alternatives)**: Earlier iterations of this plan considered offset pusher mounts (mounting between adjacent arms rather than on centerline). Analysis shows this doesn't solve the geometry problem — yawing the drone to aim an offset pusher places the drone in the same prop-overlap geometry relative to the target. The real constraint is vertical (prop disc height vs. pusher height), not horizontal (body orientation). Centerline mounting with correct vertical placement is the clean solution.

### ⚑ 1.4 Sensor Suite

The original plan specified only a forward RGB-D camera and IMU. This is insufficient for several reasons: GPS does not work indoors, a single forward camera leaves the drone blind on five sides, and VIO from one camera drifts significantly over time — especially during rotation and wall contact maneuvers. The revised sensor suite evolves by phase.

#### Core sensors (all phases)
| Sensor | Purpose | Mount Position | Notes |
|--------|---------|---------------|-------|
| Forward RGB-D camera | Switch detection, light state, visual odometry | Front-facing, angled slightly down | D435i-class; 87°×58° FOV, 0.1–10m depth range |
| IMU | Attitude estimation, angular velocity | Center of mass | Provided by flight controller |
| Optical flow sensor | Low-altitude velocity estimation, hover stability | Downward-facing | PMW3901-class; critical for GPS-denied hover |
| 2D LiDAR | Indoor SLAM, planar obstacle avoidance | Top-mounted, horizontal plane | 360° scan, 12m+ range; provides drift-free pose |

#### Phase 7 additions (autonomous discovery)
| Sensor | Purpose | Mount Position | Notes |
|--------|---------|---------------|-------|
| 3D LiDAR | Volumetric mapping, wall surface detection | Top-mounted, replaces 2D LiDAR | Livox Mid-360 class; 40m range, ~265g |
| Downward camera (optional) | Floor texture for visual odometry backup | Bottom-facing | Supplements optical flow at higher altitudes |

#### Why each sensor is needed

**Optical flow** solves the most immediate problem: stable hovering without GPS. The drone needs sub-centimeter position hold during the RL press maneuver, and neither VIO nor LiDAR SLAM alone provides the low-latency, high-frequency velocity estimate needed for tight hover control. Optical flow at 100+ Hz fills this gap. Weight is negligible (~5g for a PMW3901 module).

**2D LiDAR** provides the global localization that GPS cannot. A 360° scan at 10–15 Hz feeds a SLAM algorithm (e.g., Cartographer, Hector SLAM) that produces a drift-free 2D pose estimate. This is sufficient for Phases 1–6 where the drone operates at a known altitude in a largely planar environment. The LiDAR also provides all-around obstacle detection — the drone can see walls, shelving, and pillars in every direction, not just forward. Weight is ~170–190g.

**3D LiDAR** becomes necessary in Phase 7 for volumetric mapping. Discovering switches requires knowing where all wall surfaces are in 3D space (not just a 2D floor plan). The LiDAR builds a 3D occupancy map of the warehouse; the mission planner then generates wall-scanning trajectories that fly the camera close to each wall surface. The camera handles the actual switch detection — switches are too small and flat to resolve in a point cloud.

**The RGB-D camera** remains essential for switch localization (template matching or detection at close range), light state detection, and providing depth data for the RL policy's approach phase.

#### Sensor fusion architecture
```
                    ┌─────────────┐
                    │  2D/3D LiDAR │
                    │  (360° SLAM) │
                    └──────┬──────┘
                           │ pose @ 10-15 Hz
                           ▼
                    ┌─────────────┐       ┌──────────────┐
                    │  EKF / UKF   │◄──────│  IMU         │
                    │  State       │       │  (400 Hz)    │
                    │  Estimator   │       └──────────────┘
                    │              │       ┌──────────────┐
                    │              │◄──────│  Optical Flow │
                    │              │       │  (100+ Hz)   │
                    └──────┬──────┘       └──────────────┘
                           │
              fused pose + velocity estimate
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        Navigation    RL Policy    Obstacle
        Controller    (§4)        Avoidance
```

**In simulation (Isaac Sim)**: LiDAR and optical flow can be simulated using Isaac Sim's ray-casting and physics ground truth. During RL training, ground truth pose is used directly (no SLAM needed); the sensor fusion stack is exercised only during full-pipeline integration tests and sim-to-real validation.

**For real-world transfer**: The EKF fuses LiDAR SLAM pose (accurate but slow, 10–15 Hz), IMU (fast but drifts, 400 Hz), and optical flow velocity (fast, drift-free but only local, 100+ Hz). This produces a smooth, accurate, high-frequency state estimate suitable for both navigation and the RL policy.

---

## 2. Task Decomposition

| # | Task | Approach | Why |
|---|------|----------|-----|
| T1 | Takeoff & hover at fixed altitude | Classical PID | Well-understood, stable, no learning needed |
| T2 | Navigate to approximate switch coordinates | Classical path planning + LiDAR obstacle avoidance | Efficient, predictable, transferable |
| T3 | Detect & localize switch precisely | Classical CV / depth | Deterministic, reliable at close range |
| T4 | Approach & press the switch (contact maneuver) | **RL policy** | Contact dynamics are hard to model; RL handles disturbances, varying geometry, reaction forces |
| T5 | Detect light state change | Classical CV / USD query | Binary detection, no learning needed |
| T6 | Retreat from wall after press | **RL policy (same policy, retreat phase)** | Included in RL episode so policy learns end-to-end |
| T7 | Discover switches autonomously (future) | 3D LiDAR mapping + camera-based wall scanning | LiDAR finds wall surfaces; camera detects switches on them |

---

## 3. Classical Robotics Components

### ⚑ 3.1 Indoor Localization

**Problem**: GPS does not work indoors. The original plan hand-waved this as "ground truth or simulated GPS/VIO." For real-world operation, the drone needs a drift-free indoor pose estimate.

**Solution**: LiDAR SLAM + IMU + optical flow fusion (see §1.4 sensor fusion architecture).

- **Primary pose source**: 2D LiDAR SLAM (Cartographer or Hector SLAM) provides X, Y, yaw at 10–15 Hz with centimeter-level accuracy in structured indoor environments.
- **Altitude**: barometer + optical flow range sensor (integrated in PMW3901 modules with VL53L1X ToF). Accuracy ±2cm below 4m altitude.
- **Velocity**: optical flow provides direct ground-plane velocity at 100+ Hz, fused with IMU for full 6-DOF velocity estimate.
- **Fallback in simulation**: Isaac Sim ground truth pose. SLAM is not needed during RL training; it is only exercised during integration tests.

### 3.2 Navigation (T2)
- **Input**: current drone pose (from SLAM fusion or sim ground truth), target waypoint (approximate switch location).
- **Algorithm**: simple waypoint following with velocity profiling. **LiDAR point cloud provides real-time obstacle checking** — reject waypoints that pass through obstacles; simple reactive avoidance (stop or deflect) for dynamic obstacles.
- **Implementation**: position controller that outputs velocity commands → attitude controller → rotor RPMs.
- **Tolerance**: navigate to within ~0.5m of the switch, then hand off to the RL approach phase.

### 3.3 Switch Localization (T3)
- **Input**: RGB-D image from forward camera when within ~0.5m of expected switch location.
- **Algorithm**: template matching or simple object detection (the switch has a known appearance). Depth gives precise 3D position of the button surface.
- **Output**: precise 3D offset from drone to switch button center.
- **Fallback**: if visual detection fails, use the known spawn coordinates directly (for initial development).

### 3.4 Light Detection (T5)
- **Option A (simulation shortcut)**: query the SphereLight intensity attribute directly via USD API — instant, 100% reliable. Use this during RL training for reward signal.
- **Option B (realistic)**: use the drone's camera to detect brightness change in the lamp region. Thresholding on pixel intensity in the lamp's image region.
- **Recommendation**: use Option A for training reward, Option B as an observation the agent can learn to use for sim-to-real transfer later.

### ⚑ 3.5 Light State Latch (T5 timing)
The simulation may not change the light state instantaneously when the button reaches full depression. To handle this:
- After the switch button joint displacement exceeds the activation threshold, start a **confirmation window** of N steps (e.g. 5 physics steps).
- The light state is considered toggled only when the USD query confirms the new state has persisted for the full window.
- During RL training, the `r_toggle` reward fires only after this latch confirms the change, preventing premature episode termination.
- The episode does **not** end immediately on toggle — it continues for a short **retreat phase** (see §4.5) so the policy learns to back away cleanly.

### ⚑ 3.6 Autonomous Wall Scanning (T7, Phase 7)

Switch discovery requires two capabilities that no single sensor provides: mapping environment geometry (where are the walls?) and detecting small visual features on those walls (where are the switches?).

**Strategy — LiDAR maps, camera detects**:
1. **3D LiDAR** builds a volumetric occupancy map of the warehouse. Wall surfaces are extracted as planar segments with known position, orientation, and extent.
2. **Mission planner** generates wall-scanning trajectories: the drone flies parallel to each wall at ~0.5–1m distance, at the expected height range for switches, ensuring the forward camera covers the full wall surface with overlapping passes.
3. **Camera-based switch detector** (lightweight CNN or template matcher) runs on each RGB-D frame during the scan, flagging candidate switch locations in 3D.
4. Detected switch candidates are added to a **task queue** for the press state machine.

**Why LiDAR can't detect switches alone**: Switches are small (~7×12cm), flat, and wall-colored. They have <2mm of geometric relief — invisible in a LiDAR point cloud from 0.5m+ away. LiDAR tells you *where the walls are*; the camera tells you *what's on them*.

---

## 4. RL Policy — Switch Press Maneuver (T4)

This is the core learned behavior. The drone must approach the switch from ~0.5m away, make controlled contact with its pusher, apply enough force to depress the button, stabilize, and retreat.

### 4.1 Why RL?
- Contact dynamics between the drone pusher and switch button create discontinuous forces.
- The drone must simultaneously maintain attitude while applying forward force — a coupled control problem.
- Reaction force from the wall pushes the drone backward; the policy must compensate.
- Generalizing across slightly different switch positions, orientations, and heights is natural for RL.

### ⚑ 4.2 Observation Space
| Observation | Dims | Source | Notes |
|-------------|------|--------|-------|
| Drone position (relative to switch) | 3 | SLAM fusion / ground truth | |
| Drone orientation (quaternion) | 4 | IMU / ground truth | |
| Drone linear velocity | 3 | Optical flow + IMU fusion / ground truth | |
| Drone angular velocity | 3 | IMU / ground truth | |
| Switch button displacement (joint pos) | 1 | Articulation query | |
| Light state (on/off) | 1 | USD query | |
| Pusher compression (compliant joint pos) | 1 | Articulation query | Proxy for contact force |
| Contact flag (binary) | 1 | Contact reporter | Is pusher touching anything? |
| Previous action | 4 | Action buffer | |
| **Total** | **21** | | |

**Key addition**: The pusher compression and contact flag give the policy direct information about the contact state. Without these, the agent is blind during the critical pressing phase and must infer contact purely from velocity changes, which is noisy and slow.

**Note on LiDAR in obs space**: The RL policy does NOT receive raw LiDAR data. LiDAR feeds the SLAM system which provides the drone position observation. The RL policy operates in a small local frame relative to the switch — it doesn't need to know about the broader environment.

### 4.3 Action Space
- **Option A (high-level, recommended)**: 3D velocity command + yaw rate → classical attitude controller converts to rotor commands. Simpler to learn, more stable.
- **Option B (low-level)**: collective thrust + 3 body torques → direct motor mixing. Harder to learn but more agile.
- **Recommendation**: start with Option A. The RL policy outputs **desired velocity deltas** (Δv), not absolute velocities. This naturally limits jerk and encourages smooth trajectories.

### ⚑ 4.3.1 Action Smoothing
Raw RL policies produce jittery, high-frequency commands. Action smoothing is a first-class design choice, not a late-stage mitigation.

**Primary mechanism — delta action space**: The policy outputs Δv (change in velocity command) rather than absolute velocity. This limits how fast commands can change between steps.

**Secondary mechanism — EMA filter**: Apply an exponential moving average on the output:
```
v_cmd(t) = α * v_raw(t) + (1 - α) * v_cmd(t-1),   α ∈ [0.3, 0.5]
```

**Tertiary mechanism — action delta penalty** (in reward, see §4.4):
```
r_smooth = -k7 * ||action(t) - action(t-1)||^2
```

### ⚑ 4.4 Reward Design
```
# --- Phase-aware rewards ---
in_contact = (contact_flag == 1)

# Approach: only active BEFORE first contact
r_distance  = -k1 * ||drone_pusher_pos - switch_button_pos||  * (1 - in_contact)

# Press: only active DURING contact
r_press     = +k2 * button_joint_displacement                 * in_contact

# Toggle: large one-time bonus, gated by light state latch (§3.5)
r_toggle    = +k3 * (light_state_latched)

# Retreat: after toggle confirmed, reward distance FROM wall
r_retreat   = +k8 * ||drone_pos - wall_pos||                  * (toggle_confirmed)

# --- Always-on rewards ---
r_stability = -k4 * ||angular_velocity||
r_alive     = +k5
r_energy    = -k6 * ||action||^2
r_smooth    = -k7 * ||action(t) - action(t-1)||^2

r_total = r_distance + r_press + r_toggle + r_retreat
        + r_stability + r_alive + r_energy + r_smooth
```

**Key change from v1**: `r_distance` and `r_press` are **mutually exclusive** via the contact flag. This eliminates the conflict where pressing the switch (good) also increases distance from the switch (penalized). The reward now has three clean phases: approach → press → retreat.

### ⚑ 4.5 Episode Structure
1. Drone starts hovering at ~0.5m from a randomly positioned switch (randomize within a small volume).
2. Policy approaches, presses, and retreats.
3. Episode ends on:
   - **Success**: toggle confirmed via latch AND drone has retreated ≥0.3m from wall.
   - **Timeout**: 10s wall-clock (e.g. 500 steps at 50Hz).
   - **Crash**: ground contact, or tilt exceeding **axis-specific limits** (see below).
   - **Out-of-bounds**: drone leaves the training volume.

**⚑ Tilt termination — axis-specific limits**:
- **Pitch** (forward/back, press direction): terminate at **50°**. The drone needs some forward pitch to push, but 50° is already aggressive.
- **Roll** (left/right): terminate at **35°**. There's no reason to roll significantly during a press maneuver; large roll indicates loss of control.
- This prevents the policy from learning extreme maneuvers that won't transfer, while still allowing the forward lean needed for pressing.

**⚑ Retreat phase**: The episode does not end immediately on toggle. After `toggle_confirmed = True`, the policy has up to 2 additional seconds to back away. The `r_retreat` reward activates during this phase. This means a single RL policy handles the full approach-press-retreat sequence — no separate system needed for T6.

### ⚑ 4.6 Domain Randomization
| Parameter | Range | Notes |
|-----------|-------|-------|
| Switch Z position | ±0.3m from nominal | Original |
| **Switch X/Y position** | ±0.1m from nominal | Wall-relative offset |
| Drone start offset | ±0.15m from nominal | Original |
| Pusher friction | 0.3–0.8 | Original |
| Button stiffness | 0.5×–2× nominal | Original |
| **Wall surface friction** | 0.2–0.6 | Affects slide on contact |
| **Drone mass** | 0.9×–1.1× nominal | Affects thrust-to-weight |
| **Motor response lag** | 0–20ms | Simulates actuator delay |
| **Observation noise (pos)** | σ = 0.005m | Simulates SLAM localization error |
| **Observation noise (vel)** | σ = 0.02 m/s | Simulates optical flow / IMU noise |
| **Pusher spring stiffness** | 0.7×–1.3× nominal | Compliance variation |

### ⚑ 4.7 Training Infrastructure
- **Algorithm**: PPO (stable, well-supported in Isaac Lab / rl_games / skrl).
- **Parallel environments**: **Start with 16–32 envs** and benchmark GPU memory and step throughput before scaling. Contact-rich physics with articulated switch joints and collision detection is significantly heavier per-env than typical locomotion tasks. Scale to 64–128 only if memory and throughput allow.
- **Curriculum**:
  - Phase 1: start very close (0.1m), large approach reward, no domain randomization → learn to make contact and press.
  - Phase 2: increase start distance to 0.5m, include retreat phase → learn full approach + press + retreat.
  - Phase 3: add domain randomization → generalize.
- **Logging**: track per-episode success rate, mean episode length, mean reward per component, mean contact force, and training FPS.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mission Manager                           │
│  (state machine: takeoff → navigate → press → retreat → …)      │
└──────────┬───────────────────────┬───────────────────────────────┘
           │                       │
    ┌──────▼──────┐        ┌──────▼───────┐
    │  Navigation  │        │  RL Policy   │
    │  Controller  │        │ (approach +  │
    │  + LiDAR     │        │  press +     │
    │  Obstacle    │        │  retreat)    │
    │  Avoidance   │        │              │
    └──────┬──────┘        └──────┬───────┘
           │                       │
           └──────────┬────────────┘
                      │
              ┌───────▼────────┐
              │ Attitude / Rate │
              │  PID Controller │
              │  (classical)    │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Motor Mixing   │
              │  & Rotor Forces │
              └────────────────┘

    ┌─────────────────────────────────────┐
    │         Sensor Fusion (EKF)          │
    │                                      │
    │  LiDAR SLAM ──► pose (10–15 Hz)     │
    │  IMU ─────────► accel/gyro (400 Hz) │──► fused state estimate
    │  Optical Flow ► velocity (100+ Hz)  │    to all controllers
    │  Baro + ToF ──► altitude (50 Hz)    │
    └─────────────────────────────────────┘
```

### ⚑ State Machine Transitions (with error handling)

```
                          ┌────────────────────────┐
                          │         ABORT           │
                          │  (land immediately,     │
                          │   log failure reason)   │
                          └────────────────────────┘
                                    ▲
                           critical fault from any state
                                    │
IDLE ──► TAKEOFF ──► HOVER ──► NAVIGATE ──► APPROACH ──► RL_PRESS ──► RETREAT ──► HOVER
              │                    │            │            │            │
              │                    │            │            │            ▼
              ▼                    ▼            ▼            ▼        HOVER ──► NAVIGATE (next switch)
           (altitude           (timeout →    (switch not  (tilt/dist     │         or LAND (all done)
            not reached         ABORT)       detected →    exceeded →    │
            in 10s →                         HOVER, retry  RETREAT)      │
            ABORT)                           up to 3×,                   │
                                             then ABORT)                 │
                                                                        ▼
                                                                   (retry RL_PRESS
                                                                    up to 2×,
                                                                    then skip switch
                                                                    and log failure)
```

**Phase 7 additions to state machine**:
```
HOVER ──► SCAN_WALL ──► HOVER
              │
              ▼
         (fly parallel to wall, run switch detector on camera feed,
          add detected switches to task queue,
          proceed to next wall segment or return to HOVER when wall covered)
```

**Key additions over v1**:
1. **ABORT state**: emergency landing triggered by critical faults (excessive tilt recovery failure, navigation timeout, repeated localization failures, SLAM divergence). Logs the failure reason for debugging.
2. **Retry logic in APPROACH**: if the switch localizer fails, return to HOVER, re-approach, and retry up to 3 times before aborting.
3. **Retry logic in RL_PRESS**: if the press fails (timeout or excessive perturbation), retreat to safe distance and retry up to 2 times. After 2 failures, skip this switch and move on.
4. **RETREAT is now explicit**: always entered after RL_PRESS, whether success or failure. Uses the PID controller to return to a safe hover point ≥0.5m from the wall.
5. **SCAN_WALL (Phase 7)**: systematic wall-following trajectory using LiDAR map, with camera-based switch detection running continuously.

### ⚑ 5.1 Navigation-to-RL Handoff Gate
The transition from APPROACH to RL_PRESS requires all of the following conditions to be met simultaneously:
- Position error to target hover point < 0.1m (from SLAM fusion)
- Linear velocity magnitude < 0.2 m/s (from optical flow + IMU)
- Angular velocity magnitude < 0.3 rad/s
- Switch detected with confidence > 0.8 (or fallback coordinates available)
- SLAM pose confidence above threshold (no recent divergence)

If these conditions are not met within 5 seconds of entering APPROACH, the drone returns to HOVER and retries. This prevents handing off to the RL policy when the drone is still moving or poorly positioned.

---

## 6. Step-by-Step Implementation Roadmap

Implementation follows a strict simple-to-complex ordering. Each step produces a testable result before moving on. Steps within a phase are sequential — do not skip ahead.

### Phase 1 — Drone Foundation (Sim Only)

**Goal**: A quadcopter that hovers stably in Isaac Sim.

1. **Source or build a quadcopter USD asset** with collision meshes, 4 rotor joints, and physically plausible mass/inertia properties. Start with an existing Isaac Sim quadrotor asset if available (e.g., Crazyflie USD) and scale to 500mm class.
   - *Test*: asset loads in Isaac Sim, sits on ground under gravity, doesn't explode or clip through floor.

2. **Implement a PID attitude controller** (roll/pitch/yaw rate loops). Hard-code a constant thrust that equals gravity. Apply per-rotor forces via Isaac Sim's articulation API.
   - *Test*: drone lifts off and hovers at a fixed altitude with <5cm oscillation for 30 seconds.

3. **Add altitude hold PID** (outer loop on Z position → thrust command → attitude controller).
   - *Test*: drone takes off to 1.5m, holds altitude within ±3cm for 60 seconds.

4. **Design and attach the compliant pusher appendage**. Add a prismatic joint with spring-damper to the front of the drone USD. Add collision geometry (flat pad).
   - *Test*: manually teleport drone into a wall; pusher compresses, drone doesn't explode or tunnel through. Joint displacement is readable via API.

5. **Verify mass budget**. With the pusher attached, confirm the drone can still hover with at least 30% thrust margin. Adjust motor max force if needed.
   - *Test*: drone hovers stably with pusher attached, same quality as step 3.

### Phase 2 — Basic Navigation (Sim Only)

**Goal**: Drone can fly to arbitrary waypoints in the warehouse.

6. **Implement XY position controller** (PD on position → velocity command → attitude command → rotor forces). Cascade: position PD → velocity PD → attitude PID → rotor mixing.
   - *Test*: command drone to fly from hover point A to point B (3m away). Arrives within 10cm, settles within 5 seconds.

7. **Implement yaw controller** to keep the drone facing forward (or toward the target).
   - *Test*: drone navigates to waypoint while maintaining heading toward the wall.

8. **Add boundary constraints**: ceiling, floor, and warehouse walls. Simple clamp on position commands — if commanded position is out of bounds, clip to boundary.
   - *Test*: command drone to fly past a wall; it stops at the boundary instead of crashing.

9. **Implement multi-waypoint sequencing**: the drone flies through a list of waypoints in order, hovering briefly at each.
   - *Test*: drone visits 4 waypoints in sequence, returns to start, lands.

### Phase 3 — Switch & Light Environment (Sim Only)

**Goal**: The warehouse has interactive switches and lights.

10. **Build the switch USD asset**: a wall-mounted button with a prismatic joint (spring-loaded, ~5mm travel), collision geometry, and a visual toggle indicator. Define the activation threshold (e.g., 80% of travel).
    - *Test*: manually push the button via API; joint displaces and returns when released.

11. **Build the lamp USD asset**: a SphereLight linked to the switch's activation state. When button displacement exceeds threshold, light toggles.
    - *Test*: manually activate switch; light turns on. Activate again; light turns off. Query light state via USD API — returns correct boolean.

12. **Spawn multiple switch-lamp pairs** on the warehouse wall at known coordinates. Verify all work independently.
    - *Test*: manually toggle each switch; each lamp responds correctly.

13. **Implement the light state latch** (§3.5): N-step confirmation window before declaring toggle.
    - *Test*: rapid button tap that doesn't hold threshold for N steps does NOT count as toggle. Sustained press does.

### Phase 4 — RL Press Policy (Sim Only)

**Goal**: The RL policy can approach, press, and retreat from a switch.

14. **Set up the RL training environment** in Isaac Lab. Define the observation space (§4.2), action space (§4.3), and episode structure (§4.5). Start with ground truth observations (no sensor noise).
    - *Test*: environment resets correctly; observations are in expected ranges; random actions produce plausible (chaotic) behavior.

15. **Implement the reward function** (§4.4) with all components: r_distance, r_press, r_toggle, r_retreat, r_stability, r_alive, r_energy, r_smooth. Log each component separately.
    - *Test*: manually fly drone into switch via scripted trajectory; verify each reward component fires at the correct phase (approach → contact → press → toggle → retreat).

16. **Implement action smoothing** (§4.3.1): delta action space + EMA filter.
    - *Test*: random policy produces smoother motor commands than without EMA.

17. **Benchmark parallel environments**: start with 16 envs. Measure GPU memory usage and simulation steps per second. Scale up in powers of 2 until memory is >80% utilized or throughput plateaus.
    - *Test*: record steps/second and VRAM at 16, 32, 64 envs. Pick the best operating point.

18. **Train curriculum Phase 1**: drone starts 0.1m from switch. No domain randomization. Large approach reward weight.
    - *Success criterion*: >80% toggle success rate within 2000 training iterations.

19. **Train curriculum Phase 2**: increase start distance to 0.5m. Enable retreat phase.
    - *Success criterion*: >70% full approach-press-retreat success rate.

20. **Train curriculum Phase 3**: enable domain randomization (§4.6). Reduce learning rate.
    - *Success criterion*: >60% success rate across randomized conditions; policy doesn't overfit to nominal parameters.

21. **Evaluate trained policy**: run 500 evaluation episodes with randomized conditions. Record success rate, mean episode length, per-component reward breakdown, mean contact force, and failure modes.
    - *Test*: generate an evaluation report. Identify and log the top 3 failure modes for future improvement.

### Phase 5 — System Integration (Sim Only)

**Goal**: Full pipeline from takeoff to multiple switch presses, orchestrated by the state machine.

22. **Implement the state machine** (§5) with all states: IDLE, TAKEOFF, HOVER, NAVIGATE, APPROACH, RL_PRESS, RETREAT, LAND, ABORT.
    - *Test*: state transitions are logged correctly during a scripted mission.

23. **Implement the handoff gate** (§5.1): position/velocity/confidence criteria for APPROACH → RL_PRESS transition.
    - *Test*: drone in APPROACH with velocity > threshold stays in APPROACH; drone meeting all criteria transitions to RL_PRESS.

24. **Implement retry and abort logic**: retry counters for APPROACH (3×) and RL_PRESS (2×), ABORT on critical faults.
    - *Test*: deliberately cause a localization failure; verify retry and eventual abort behavior.

25. **Full pipeline test**: spawn 4 switch-lamp pairs at known coordinates. Drone takes off, navigates to each, presses, and confirms light toggle. Returns and lands.
    - *Test*: ≥3 of 4 switches successfully toggled. Full telemetry logged.

26. **ROS2 integration**: publish drone state, switch events, light state, and camera images as ROS2 topics. State machine subscribes to sensor topics instead of direct API calls.
    - *Test*: `ros2 topic echo` shows live data on all expected topics during mission execution.

### Phase 6 — Sensor-Based Localization (Sim → Real Bridge)

**Goal**: The drone localizes and navigates using only onboard sensors, no ground truth.

27. **Add simulated sensor stack**: simulate 2D LiDAR (ray-cast in Isaac Sim), optical flow (ground truth velocity + noise), and RGB-D camera (Isaac Sim render). Publish as ROS2 topics.
    - *Test*: simulated sensor data matches expected formats and rates.

28. **Implement LiDAR SLAM** (Cartographer or Hector SLAM) running on simulated 2D LiDAR scans.
    - *Test*: SLAM-produced map matches warehouse ground truth within 5cm. Pose estimate tracks ground truth within 3cm during navigation.

29. **Implement EKF sensor fusion**: fuse SLAM pose + IMU + optical flow velocity + barometric altitude.
    - *Test*: fused state estimate is smooth (no jumps) and accurate (<3cm position, <5cm/s velocity error).

30. **Switch RL policy to fused state estimate** instead of ground truth. Retrain or fine-tune if success rate drops significantly.
    - *Test*: RL press success rate with fused observations is within 10% of ground truth baseline.

31. **Stress test**: add simulated sensor degradation — LiDAR occlusion, optical flow failure over textureless floor, IMU bias drift. Verify graceful degradation (drone slows down or aborts, doesn't crash).
    - *Test*: drone detects degraded localization quality and enters ABORT rather than crashing.

### Phase 7 — Autonomous Discovery (Future)

**Goal**: Drone explores the warehouse, finds switches without prior coordinates, and presses them.

32. **Upgrade to 3D LiDAR** (simulated Livox Mid-360 ray-cast in Isaac Sim). Build 3D occupancy map.
    - *Test*: 3D map of warehouse matches ground truth geometry within 10cm.

33. **Implement wall surface extraction**: from 3D occupancy map, identify planar wall segments with position, normal, and extent.
    - *Test*: all warehouse walls detected; wall normals correct to within 5°.

34. **Implement wall-scanning trajectory planner**: generate parallel-to-wall flight paths that ensure camera coverage of the full wall surface at switch-detection range (~0.5–1m).
    - *Test*: scanning trajectory covers >95% of wall area. No collisions.

35. **Implement camera-based switch detector**: lightweight CNN or template matcher running on RGB-D frames. Output: 3D position of detected switches with confidence scores.
    - *Test*: detector finds >90% of switches in controlled conditions with <5% false positive rate.

36. **Integrate with state machine**: add SCAN_WALL state. Detected switches populate the task queue for the existing press pipeline.
    - *Test*: drone autonomously scans a wall, discovers 3 switches, navigates to each, presses them. ≥2 of 3 toggled successfully.

37. **Full autonomous end-to-end test**: drone explores entire warehouse, discovers all switches, presses all of them.
    - *Test*: >80% of all switches discovered and toggled. Total mission time logged.

---

## 7. Key Design Decisions

### ✦ Resolved decisions (v4)

1. **Frame**: Holybro X500 V2 (500mm wheelbase, CF arms, bolt-on attachment supporting both quad and X8 coaxial configurations).
2. **Motors**: T-Motor MN3110 KV700 across both stages. Buy 8 units upfront to ensure identical bearings and balance; 4 used in Stage 1, 4 reserved for Stage 2.
3. **Propellers**: T-Motor T1245 (12×4.5" polymer) for all positions — both upper and lower in Stage 2. Matched prop sizes chosen over mixed (12"/11") for operational simplicity; 3–4% thrust loss is negligible against 4.6:1+ T:W margin.
4. **ESCs**: HGLRC BS30A BLHeli_S (30A continuous, DShot600). Buy 8 units upfront.
5. **Flight controller**: Holybro Pixhawk 6C with Holybro PM07 power module (native 8-channel ESC passthrough, no PDB upgrade needed for Stage 2).
6. **Pusher mount**: Centerline-forward, mid-plane (between upper and lower prop planes on X8; below single prop plane on quad). Carbon fiber boom, 100mm forward extension, compliant spring-damper pad.
7. **Action space**: Velocity deltas (Δv) output from policy, not absolute velocity. Naturally limits jerk and encourages smooth trajectories.
8. **Compliance strategy**: Start in quad configuration through Phase 5. Upgrade to X8 only if empirical data demands it (motor saturation, vibration issues, payload growth, redundancy requirements).

### Open decisions

1. **Drone USD asset**: build from scratch or adapt an existing Isaac Sim quadcopter (e.g., Crazyflie, generic quadrotor)?
2. **RL framework**: Isaac Lab built-in, rl_games, skrl, or stable-baselines3?
3. **Compliant pusher parameters**: spring stiffness and damping ratio need empirical tuning in simulation — plan for a parameter sweep early in Phase 1.
4. **Multi-switch strategy**: sequential (one at a time) or learned ordering?
5. **LiDAR SLAM algorithm**: Google Cartographer (mature, well-documented) vs. Hector SLAM (no odometry needed) vs. KISS-ICP (modern, lightweight)?
6. **2D vs 3D LiDAR timing**: start with 2D for Phases 1–6, upgrade to 3D only in Phase 7? Or invest in 3D from the start for a single sensor to maintain?
7. **Stage 2 trigger threshold**: which specific metrics from Stage 1 flight data will justify the X8 upgrade? Define these before Phase 5 to avoid post-hoc rationalization.

---

## 8. Risk Areas

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Contact instability** — drone oscillates or bounces when pressing | High | Compliant pusher (§1.3), reward shaping for smooth approach (§4.4), action smoothing (§4.3.1) |
| **Sim-to-real gap** — contact dynamics differ significantly | High | Broad domain randomization (§4.6) covering friction, mass, stiffness, actuator delay, sensor noise |
| **Indoor localization failure** — SLAM diverges in featureless areas | High | Multi-sensor fusion (§1.4); LiDAR SLAM + optical flow + IMU provide redundant pose sources; SLAM divergence triggers ABORT |
| ✦ **Motor failure during contact (quad Stage 1)** — any motor loss = uncontrolled crash | High | Monitor throttle margin during testing; if any motor exceeds 85% during normal contact, upgrade to Stage 2 X8 for redundancy |
| ✦ **Prop strike on pusher (X8 Stage 2)** — aggressive pitch transient puts pusher into lower prop disc | Medium | Axis-specific pitch limit (50° max, §4.5); sacrificial shear point on pusher boom; domain randomization includes pitch excursions |
| **Reward hacking** — drone finds unintended ways to trigger the switch | Medium | Phase-aware rewards eliminate approach/press conflict; constrain approach direction via observation frame; penalize excessive velocity at contact |
| **Training time** — contact-rich RL slow to converge | Medium | Curriculum learning (§4.7), start with 16–32 envs and scale, shaped rewards with clear phases |
| **Handoff instability** — RL policy receives the drone in a bad state | Medium | Explicit handoff gate with position/velocity/confidence criteria (§5.1), retry logic |
| ✦ **Sensor vibration degrading SLAM/policy (Stage 2 X8)** — 8 motors in close proximity produce complex vibration signature | Medium | T-Motor MN3110 with EZO bearings chosen specifically for low vibration; silicone damper isolators on flight controller and LiDAR mounts; PX4 notch filters tuned post-build |
| ✦ **Stage 2 PID re-tune cost** — quad PIDs don't transfer to X8 | Medium | Budget 1–2 days for PID retuning in Stage 2; RL policy may need 500–1,000 additional training iterations for X8 dynamics |
| **Light detection timing** — policy releases button before light confirms | Low | Light state latch with N-step confirmation window (§3.5) |
| **GPU memory** — contact-rich parallel envs exceed VRAM | Medium | Start conservatively at 16–32 envs; benchmark before scaling (§4.7) |
| ✦ **Thrust margin on max config with 3D LiDAR (Stage 2)** — adding Livox Mid-360 pushes AUW to ~2,631g | Low | Stage 2 X8 with MN3110 motors provides 4.11:1 T:W at max config — comfortable margin; stage 2 triggered before adding 3D LiDAR |

---

## ✦ 9. Hardware Summary (v4)

This section summarizes hardware-platform decisions. Full bill of materials is maintained in the companion document `docs/archive/HOLYBRO_X8_STAGED_BOM_v2.md`.

### 9.1 Stage 1 (quadcopter) performance

| Metric | Value |
|--------|-------|
| Frame | Holybro X500 V2, 500mm wheelbase |
| Motors | 4× T-Motor MN3110 KV700 |
| Propellers | 4× T-Motor T1245 (12×4.5" polymer) |
| Battery | 4S 5000mAh LiPo (Tattu R-Line 75C) |
| AUW (mid config) | ~1,830g |
| Max thrust | ~6,000g |
| **Thrust-to-weight** | **3.28:1** |
| Hover throttle | ~30% |
| Flight time | ~12–14 min |
| Motor redundancy | **None** (single failure = crash) |

### 9.2 Stage 2 (X8 coaxial) performance

| Metric | Mid config | Max config |
|--------|-----------|-----------|
| Upper motors | 4× T-Motor MN3110 KV700 | same |
| Lower motors | 4× T-Motor MN3110 KV700 | same |
| Upper props | 4× T1245 (12×4.5") | same |
| Lower props | 4× T1245 (12×4.5") — matched to upper | same |
| AUW | ~2,320g | ~2,631g |
| Max thrust (after coaxial loss) | ~10,800g | ~10,800g |
| **Thrust-to-weight** | **4.66:1** | **4.11:1** |
| Hover throttle | ~19% | ~22% |
| Flight time | ~10–12 min | ~8–10 min |
| Motor redundancy | **2-motor loss survivable** | same |

### 9.3 Stage 2 upgrade trigger criteria

Stage 2 should be initiated when Stage 1 flight data demonstrates at least one of the following:

1. **Motor saturation during normal contact maneuvers** — any motor exceeds 85% throttle during successful press sequences. Above this threshold there is insufficient margin for disturbance rejection.
2. **Sensor data quality limited by vibration** — IMU FFT shows unfilterable peaks above PX4's notch filter bandwidth, LiDAR scan noise exceeds acceptable thresholds for SLAM, or camera-based switch detection confidence drops below 0.8 due to motion blur.
3. **Crash risk unacceptable for operational environment** — deployment context (proximity to humans, over critical equipment, in production areas) requires motor-failure survivability that quad cannot provide.
4. **Payload growth** — adding Phase 7 sensors (Livox Mid-360 3D LiDAR, Orin NX upgrade) pushes AUW above 2,200g. Quad T:W drops below 2.7:1 at this weight; X8 maintains 4.2:1.

Do not upgrade Stage 2 without at least one of these criteria demonstrated by data. Redundancy without demonstrated need is over-engineering.

### 9.4 Stage 2 software transition

Transitioning from Stage 1 (quad) to Stage 2 (X8 coaxial) requires software changes that cannot be skipped:

1. **PX4 airframe selection**: change from `Generic Quadrotor X` to `Generic Octocopter Coaxial`. Motor number mapping must be verified before first power-up.
2. **PID retune**: moment of inertia changes significantly; attitude controller gains need retuning. Budget 1–2 days of empirical tuning.
3. **RL policy fine-tuning**: policy trained on quad dynamics will not transfer zero-shot to X8. Plan for 500–1,000 additional training iterations in Isaac Sim with the X8 motor configuration before operational deployment.
4. **State machine update**: add motor health monitoring to detect individual motor failures and transition to degraded-mode operation. The X8's fault tolerance is only realized if the software actively uses it.

### 9.5 Alternative: larger hexacopter (Tarot 680)

If the X8 upgrade path proves unsuitable (e.g., vibration from 8 motors impacts sensor quality more than predicted), the alternative is a full rebuild on the Tarot 680 Pro hexacopter frame. This path:

- Keeps 6 of the 8 MN3110 motors already purchased (2 become spares)
- Uses larger props (13×4.4") for better thrust per motor and lower blade-pass frequency
- Provides single-motor-loss survivability (less than X8's 2-motor tolerance)
- Requires full frame rebuild, not incremental upgrade
- Larger indoor footprint (680mm vs 500mm)

This path is **not the default**, but is kept available as a contingency. The 8 MN3110 motors specified in the Stage 1 purchase support this pivot without motor rebuy.

---

## Appendix: Changelog

### v4 changes (✦)
1. **§1.1** — Platform: changed from generic quadcopter to Holybro X500 V2 staged build (quad → X8 coaxial). Explicit trigger criteria for Stage 2 upgrade. Rejected S550 hex due to 9" prop size limit.
2. **§1.2** — Added airframe mixing note (`Quad X` → `Octocopter Coaxial X` for Stage 2).
3. **§1.3** — Pusher design: added frontal clearance analysis. Solved prop-overlap problem via mid-plane forward mount (below quad prop plane, between X8 upper/lower prop planes). Documented rejection of offset pusher alternative (doesn't solve the vertical geometry problem).
4. **§7** — Resolved most open design decisions based on hardware analysis; remaining open items focused on software/training choices.
5. **§8** — Added risks specific to hardware: motor failure on quad, prop strike on X8, vibration from 8-motor layout, PID re-tune cost for Stage 2.
6. **§9 (NEW v4)** — Hardware summary section with Stage 1 and Stage 2 performance specs, upgrade trigger criteria, software transition requirements, and Tarot 680 hex contingency path.

### v3 changes (⚑)
1. **§1.3** — Pusher appendage design: specified compliant spring-damper mount with tuning guidelines.
2. **§1.4 (NEW v3)** — Sensor suite: added 2D LiDAR, optical flow sensor, sensor fusion architecture, phase-evolving sensor strategy.
3. **§3.1 (NEW v3)** — Indoor localization: LiDAR SLAM + IMU + optical flow fusion replaces GPS/VIO hand-wave.
4. **§3.5** — Light state latch: N-step confirmation window.
5. **§3.6 (NEW v3)** — Autonomous wall scanning strategy: LiDAR maps geometry, camera detects switches.
6. **§4.2** — Observation space: added pusher compression and contact flag; noted LiDAR feeds SLAM, not RL obs directly.
7. **§4.3** — Action space: changed to delta velocity formulation.
8. **§4.3.1** — Action smoothing: delta actions + EMA filter + reward penalty.
9. **§4.4** — Reward design: mutually exclusive r_distance/r_press via contact flag; added r_retreat and r_smooth.
10. **§4.5** — Episode structure: axis-specific tilt limits; retreat phase within RL episode.
11. **§4.6** — Domain randomization: expanded to cover wall friction, drone mass, motor lag, sensor noise, pusher stiffness.
12. **§4.7** — Training infrastructure: 16–32 starting envs with benchmark-first approach.
13. **§5** — System architecture: added sensor fusion block diagram; SCAN_WALL state for Phase 7; SLAM divergence as ABORT trigger.
14. **§5.1** — Handoff gate: added SLAM confidence criterion.
15. **§6 (REWRITTEN v3)** — Implementation roadmap: 37 sequential steps across 7 phases, each with explicit test criteria. Strict simple-to-complex ordering.

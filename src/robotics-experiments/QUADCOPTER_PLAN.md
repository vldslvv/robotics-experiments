# Quadcopter Light Switch Interaction — Architecture & Task Plan

## Overview

A simulated quadcopter in Isaac Sim that can navigate to light switches on a warehouse wall, apply pressure to toggle them, and detect whether the associated lamp turned on or off. The system combines classical robotics for navigation and perception with RL for the fine-grained contact-based switch pressing task.

---

## 1. Drone Platform

### 1.1 Quadcopter USD Asset
- 4-rotor quadcopter with a rigid forward-facing "pusher" appendage (a small flat plate or rod) mounted on the front, used to press switches.
- Collision geometry on the pusher for physics contact with the switch button.
- Sensors: forward-facing RGB-D camera, downward IMU (provided by Isaac Sim physics).
- Mass and inertia tuned so the drone can hover stably and absorb the reaction force of pressing a switch without tumbling.

### 1.2 Low-Level Flight Controller (Classical)
- **PID or PX4-style rate/attitude controller** running at physics step rate.
- Inputs: desired thrust + torques (or attitude + thrust commands).
- Outputs: per-rotor RPM.
- This is deterministic controls math — no RL needed.
- Can use Isaac Sim's articulation or direct force API on each rotor joint.

---

## 2. Task Decomposition

| # | Task | Approach | Why |
|---|------|----------|-----|
| T1 | Takeoff & hover at fixed altitude | Classical PID | Well-understood, stable, no learning needed |
| T2 | Navigate to approximate switch coordinates | Classical path planning | Efficient, predictable, transferable |
| T3 | Detect & localize switch precisely | Classical CV / depth | Deterministic, reliable at close range |
| T4 | Approach & press the switch (contact maneuver) | **RL policy** | Contact dynamics are hard to model; RL handles disturbances, varying geometry, reaction forces |
| T5 | Detect light state change | Classical CV / USD query | Binary detection, no learning needed |
| T6 | Recover from press & return to hover | RL or classical | RL if press causes significant perturbation; classical if gentle enough |
| T7 | Discover switches autonomously (future) | Classical CV + exploration | Frontier-based or coverage planning with object detection |

---

## 3. Classical Robotics Components

### 3.1 Navigation (T2)
- **Input**: current drone pose (from Isaac Sim ground truth or simulated GPS/VIO), target waypoint (approximate switch location).
- **Algorithm**: simple waypoint following with velocity profiling. No complex obstacle avoidance needed in the warehouse initially — the wall and switches are the only obstacles.
- **Implementation**: position controller that outputs velocity commands → attitude controller → rotor RPMs.
- **Tolerance**: navigate to within ~0.5m of the switch, then hand off to the RL approach phase.

### 3.2 Switch Localization (T3)
- **Input**: RGB-D image from forward camera when within ~0.5m of expected switch location.
- **Algorithm**: template matching or simple object detection (the switch has a known appearance). Depth gives precise 3D position of the button surface.
- **Output**: precise 3D offset from drone to switch button center.
- **Fallback**: if visual detection fails, use the known spawn coordinates directly (for initial development).

### 3.3 Light Detection (T5)
- **Option A (simulation shortcut)**: query the SphereLight intensity attribute directly via USD API — instant, 100% reliable. Use this during RL training for reward signal.
- **Option B (realistic)**: use the drone's camera to detect brightness change in the lamp region. Thresholding on pixel intensity in the lamp's image region.
- **Recommendation**: use Option A for training reward, Option B as an observation the agent can learn to use for sim-to-real transfer later.

---

## 4. RL Policy — Switch Press Maneuver (T4)

This is the core learned behavior. The drone must approach the switch from ~0.5m away, make controlled contact with its pusher, apply enough force to depress the button, and stabilize.

### 4.1 Why RL?
- Contact dynamics between the drone pusher and switch button create discontinuous forces.
- The drone must simultaneously maintain attitude while applying forward force — a coupled control problem.
- Reaction force from the wall pushes the drone backward; the policy must compensate.
- Generalizing across slightly different switch positions, orientations, and heights is natural for RL.

### 4.2 Observation Space
| Observation | Dimensions | Source |
|-------------|-----------|--------|
| Drone position (relative to switch) | 3 | Ground truth / localization |
| Drone orientation (quaternion) | 4 | IMU / ground truth |
| Drone linear velocity | 3 | IMU / ground truth |
| Drone angular velocity | 3 | IMU / ground truth |
| Switch button displacement (joint pos) | 1 | Articulation query |
| Light state (on/off) | 1 | USD query |
| Previous action | 4 | Action buffer |
| **Total** | **19** | |

### 4.3 Action Space
- **Option A (high-level)**: 3D velocity command + yaw rate → classical attitude controller converts to rotor commands. Simpler to learn, more stable.
- **Option B (low-level)**: collective thrust + 3 body torques → direct motor mixing. Harder to learn but more agile.
- **Recommendation**: start with Option A. The RL policy outputs desired velocity; the PID controller handles the rest.

### 4.4 Reward Design
```
r_distance  = -k1 * ||drone_pusher_pos - switch_button_pos||    # approach
r_press     = +k2 * button_joint_displacement                   # pressing
r_toggle    = +k3 * (light_state_changed)                       # sparse bonus on toggle (large, one-time)
r_stability = -k4 * ||angular_velocity||                        # don't tumble
r_alive     = +k5                                               # per-step survival bonus
r_energy    = -k6 * ||action||^2                                # efficiency

r_total = r_distance + r_press + r_toggle + r_stability + r_alive + r_energy
```

### 4.5 Episode Structure
1. Drone starts hovering at ~0.5m from a randomly positioned switch (randomize within a small volume).
2. Episode ends on: toggle success, timeout (e.g. 10s), crash (ground contact or excessive tilt > 60 deg), or out-of-bounds.
3. Domain randomization: switch Z position (within spawn bounds), drone start offset, pusher friction, button stiffness.

### 4.6 Training Infrastructure
- **Algorithm**: PPO (stable, well-supported in Isaac Lab / rl_games / skrl).
- **Parallel environments**: Isaac Sim's vectorized env support — run 64-256 envs in parallel for fast training.
- **Curriculum**:
  - Phase 1: start very close (0.1m), large approach reward → learn to make contact.
  - Phase 2: increase start distance to 0.5m → learn full approach + press.
  - Phase 3: add domain randomization → generalize.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Mission Manager                    │
│  (state machine: takeoff → navigate → press → done) │
└──────────┬──────────────────────┬────────────────────┘
           │                      │
    ┌──────▼──────┐       ┌──────▼───────┐
    │  Navigation  │       │  RL Policy   │
    │  Controller  │       │ (press task) │
    │  (classical) │       │              │
    └──────┬──────┘       └──────┬───────┘
           │                      │
           └──────────┬───────────┘
                      │
              ┌───────▼────────┐
              │ Attitude / Rate │
              │   PID Controller│
              │   (classical)   │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Motor Mixing   │
              │  & Rotor Forces │
              └────────────────┘
```

### State Machine Transitions
1. **IDLE** → TAKEOFF (on start)
2. **TAKEOFF** → HOVER (altitude reached)
3. **HOVER** → NAVIGATE (target switch assigned)
4. **NAVIGATE** → APPROACH (within 0.5m of switch)
5. **APPROACH** → RL_PRESS (switch localized, handoff to RL)
6. **RL_PRESS** → RETREAT (light state changed OR timeout)
7. **RETREAT** → HOVER (safe distance from wall)
8. **HOVER** → NAVIGATE (next switch) or LAND (all done)

---

## 6. Implementation Roadmap

### Phase 1 — Drone Foundation
- [ ] Source or build a quadcopter USD with collision, joints, and inertia
- [ ] Implement PID attitude controller
- [ ] Achieve stable hover in Isaac Sim
- [ ] Add a pusher appendage with collision geometry

### Phase 2 — Navigation (Classical)
- [ ] Implement waypoint position controller (P or PD on position → velocity → attitude)
- [ ] Test: drone takes off, flies to a fixed coordinate, hovers there
- [ ] Add simple boundary constraints (don't fly into the ground or ceiling)

### Phase 3 — RL Press Policy
- [ ] Set up Isaac Lab RL environment for the press task
- [ ] Define observation/action/reward as above
- [ ] Train Phase 1 curriculum (close start, contact learning)
- [ ] Train Phase 2 curriculum (full approach)
- [ ] Train Phase 3 curriculum (domain randomization)
- [ ] Evaluate success rate and average episode length

### Phase 4 — Integration
- [ ] Integrate RL policy with the classical navigation stack via the state machine
- [ ] Full pipeline: spawn switches + lamps → drone navigates → presses → light toggles
- [ ] ROS2 integration: publish drone state, switch events, light state
- [ ] Test with multiple switches in sequence

### Phase 5 — Autonomous Discovery (Future)
- [ ] Add camera-based switch detection (object detection or template matching)
- [ ] Implement exploration strategy (systematic wall scanning)
- [ ] Replace known switch coordinates with detected ones
- [ ] Test end-to-end: drone explores, finds switches, presses them

---

## 7. Key Design Decisions to Make

1. **Drone asset**: build from scratch or adapt an existing Isaac Sim quadcopter (e.g. Crazyflie, generic quadrotor)?
2. **RL framework**: Isaac Lab built-in, rl_games, skrl, or stable-baselines3?
3. **Action space**: velocity commands (easier) vs. thrust/torque (more capable)?
4. **Pusher design**: rigid rod, compliant pad, or the drone body itself?
5. **Multi-switch strategy**: sequential (one at a time) or learned ordering?

---

## 8. Risk Areas

- **Contact stability**: the drone may oscillate or bounce when pressing the switch. Mitigations: compliant pusher, reward shaping for smooth approach, action smoothing.
- **Sim-to-real gap**: contact dynamics differ significantly. Mitigation: domain randomization on friction, button stiffness, drone mass.
- **Reward hacking**: drone might find unintended ways to trigger the switch. Mitigation: constrain approach direction, penalize excessive velocity at contact.
- **Training time**: contact-rich RL can be slow to converge. Mitigation: curriculum learning, parallel envs, shaped rewards.

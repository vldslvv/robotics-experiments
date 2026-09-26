# Holybro X500 V2 staged hardware BOM

Historical procurement and X8 planning reference. Component estimates, prices,
and performance claims below describe the original BOM, not current vendor
quotes or verified flight capability. The implemented component values and
vendor corrections in `configs/airframes/holybro_x500_v2.yaml` take precedence.
Numbered sections remain stable because that config cites them as sources.

[ADR 0001](../adr/0001-staged-x500-platform.md) records the staged-platform
choice; the [roadmap](../LIGHT_SWITCH_MULTIROTOR_ROADMAP.md) owns project status.
Calculators are `docs/thrust_to_weight.py` and `docs/evaluate_motors_revised.py`.

## Stage 1 — Quadcopter Build

The planned Stage 1 quad retains its base components during the X8 upgrade.
Mission qualification is tracked separately in the roadmap.

### 1.1 Frame & Structure

| Component | Spec | Weight | Price | Source |
|-----------|------|--------|-------|--------|
| Holybro X500 V2 frame kit | 500mm wheelbase, 2mm CF twill plates, 16mm CF tube arms, bolt-on arm attachment | 410g (frame only) | $120 | Holybro, GetFPV, RobotShop |
| Landing gear (included) | Tall CF landing gear, gives lower-prop clearance for Stage 2 upgrade | 110g | $0 (included) | — |
| Vibration isolators | Silicone dampers between frame and FC | 5g | $5 | AliExpress / Amazon |
| Mounting hardware | M3 screws, standoffs, nylon spacers | 30g | $8 | Included in frame kit |

**Notes.** The X500 V2 includes landing gear tall enough that Stage 2 lower props (13″) clear the ground with ~25mm margin when the drone is on the ground. This is critical — the lower motor/prop pair sits directly beneath the upper motor on an X8 coaxial, which reduces ground clearance substantially compared to the quad configuration. Do not omit the landing gear. Note that the v1 BOM cited 12″ props with 30mm clearance; the 13″ upgrade reduces this to ~25mm, still sufficient but tighter — verify on first assembly.

### 1.2 Propulsion (4 motors)

| Component | Spec | Weight ×4 | Price ×4 | Source |
|-----------|------|-----------|----------|--------|
| T-Motor MN4006 EVO KV380 | Antigravity series, 18N24P stator, 40×6mm, 4mm shaft, EZO bearings, 4–6S, 22A continuous, 380W max | 272g (68g each) | $352 ($88 each, $176/pair from iNsightFPV-class distributors) | T-Motor official store, iNsightFPV, RobotShop |
| T-Motor P134.4 13×4.4″ CF prop (pair) | Carbon fiber, M6 hub, matched CW/CCW pairs | 28g (7g each prop, 2 pairs = 4 blades) | $40 (2 pairs @ $20) | T-Motor official store |
| Motor mounting hardware | M3 screws, lock washers | 10g | $5 | Included with motors |

**Per-motor thrust (4S 14.8V, static, 100% throttle).** Approximately 1,565g on 13×4.4 CF, drawing ~17.5A. This is below the motor's 22A continuous rating (80% headroom) and well below its 380W power ceiling (~252W on this prop). The motor is genuinely under-loaded on a 12–13″ prop — its design point is 14.4″–16″ props on 4–6S where it would deliver up to 1,600–2,000g per motor. The 13×4.4 sizing is dictated by X500 V2 frame geometry (354mm adjacent motor spacing limits prop diameter to 13.5″ max with safety clearance), not by motor capability.

Propeller direction: 2× CW props, 2× CCW props for Stage 1. **Buy upfront for Stage 2 too**: 5 pairs total (10 blades) covers Stage 1, Stage 2, and 2 spares. Single-batch purchasing improves dynamic balance across the airframe.

### 1.2a Prop Guards (Stage 1: 4 guards, Stage 2: 8 guards)

Prop guards protect against the dominant indoor failure mode — wall contact during navigation error or SLAM divergence. Without guards, a single clipped prop during development typically breaks the prop, may damage the motor, and can cause a full crash. With guards, the drone bumps the wall and recovers.

| Component | Spec | Weight ×4 | Price ×4 | Source |
|-----------|------|-----------|----------|--------|
| Partial arc bumper guards (3D-printed nylon PA12) | Covers outboard 90° of each prop arc, sized for 13″ props (~165mm radius) with 15mm clearance past prop tip, mounts to arm via bracket | 80g (20g each) | $44 (outsourced JLC3DP nylon PA12) or $10 filament cost for DIY PETG | JLC3DP / PCBWay, or DIY |
| Mounting hardware | M3 screws, nylon standoffs | 8g | $4 | — |

**Guard sizing for 13″ props.** The v1 BOM specified guards for 12″ props (15mm clearance past 152mm radius = 167mm from arm center). The v2 13″ props need guards sized to 165mm radius + 15mm clearance = 180mm from arm center. This adds ~13mm to the outboard reach of each guard but doesn't meaningfully change weight or cost — same nylon PA12, same partial-arc design. **If reusing existing 12″ guard CAD, the design must be scaled outward by ~13mm before printing.**

**Design rationale — partial arc rather than full ring.** Full circular ring guards surrounding each 13″ prop add 80–120g each (320–480g total), which would drop Stage 1 T:W from 2.78:1 to ~2.2:1 (below contact-task floor). Partial arc guards covering only the outboard side of each prop provide protection against the dominant threat (wall collision) at one-quarter the weight penalty. The inboard prop edge is protected by the frame body, so a ring guard there is wasted mass.

**Stage 2 addition.** The X8 upgrade adds 4 additional guards for the lower motors, same design as upper guards. Lower guards mount to the bottom of the same arm brackets as the lower motor mounts. Budget 4 additional guards at +$44 and +88g for Stage 2.

**Ground clearance check.** With partial arc guards on the outboard side of each prop, the guard's lowest point on the underside sits roughly at the prop plane level. On Stage 2 X8 with 13″ lower props, the guard must not extend below the lower prop tip — verify this during mechanical design.

**Alternative if weight margin is tight.** If max-config Stage 2 AUW (3,024g + extra payload) proves too heavy and pushes T:W below 3.0, the lower prop guards can be omitted (keeping upper guards only) to recover 88g and ~0.12 T:W points. Upper motors are the primary wall-strike risk since the drone's center of mass is above the prop plane.

### 1.3 ESCs (4 individual)

| Component | Spec | Weight ×4 | Price ×4 | Source |
|-----------|------|-----------|----------|--------|
| HGLRC BS30A BLHeli_S | 30A continuous, 40A burst, 3–5S, DShot600, 26×14.5mm, no BEC | 24g (6g each) | $40 (2× 4-packs at $20) | Amazon, HGLRC direct |

**Notes.** Buy two 4-packs (8 ESCs total) — this gives 4 ESCs for Stage 1 plus 4 additional ESCs ready for Stage 2. The Stage 2 ESCs can be kept on the shelf or mounted immediately during Stage 1 build as prep. The BS30A has comfortable headroom for the MN4006 EVO at 17.5A peak draw (58% of continuous rating, vs 73% for the v1 MN3110 spec). DShot600 protocol gives tight motor response needed for contact tasks.

**ESC sizing rationale.** Standard rule is ESC continuous rating ≥ 1.3× motor's max sustained draw. For the MN4006 EVO at 17.5A sustained, that's a minimum of 23A continuous. The BS30A's 30A continuous gives 71% headroom — comfortable for contact transients without ESCs running near thermal limit.

**6S upgrade caveat.** The BS30A is rated 3–5S only. If a future migration to 6S is contemplated (would meaningfully increase MN4006 EVO thrust to ~1,800–2,000g per motor since the motor is rated 4–6S), the BS30A becomes the binding constraint and would need replacement. The Holybro Tekko32 F4 35A ($22/each, $176/8) is the right alternative: 3–6S, DShot1200 BLHeli_32, 7g, Pixhawk-ecosystem-compatible. **Recommendation: stay with BS30A for now (4S is the platform's chosen voltage) and only upgrade ESCs if/when 6S migration is decided.**

### 1.4 Power System

| Component | Spec | Weight | Price | Source |
|-----------|------|--------|-------|--------|
| Holybro PM07 Power Module | XT60 input, dual BEC (5V 3A + 5V 3A), current/voltage sensing, 6x ESC signal passthrough | 36g | $28 | Holybro, GetFPV |
| Tattu 4S 5000mAh LiPo (R-Line or bashing series) | 14.8V, 75C continuous, 132×42×32mm, XT60 | 460g | $55 | GetFPV, Amazon |
| Battery strap | Hook-and-loop battery strap | 8g | $4 | — |

**Notes.** The PM07 is specifically designed for the Pixhawk ecosystem and comes pre-wired for the 6C. Its dual BEC outputs power the Pixhawk and companion computer independently (avoiding shared-rail issues). The six ESC signal passthroughs are a critical Stage 2 enablement — the PM07 natively supports up to 8 motors via the 6-channel passthrough plus two auxiliary outputs. No PDB upgrade needed for Stage 2.

**Battery C-rating analysis (revised for MN4006 EVO).** Stage 2 mid hover draw is approximately 17A total (2.1A per motor × 8) — only 3.4C from a 5000mAh pack. Peak draw at 100% throttle is 8 × 17.5A = 140A = 28C. The Tattu 75C rating provides ~2.7× margin over peak — comfortable. Stage 1 quad hover draw: ~15A (3.8A per motor × 4) = 3C. Both stages stay well within battery capability.

### 1.5 Flight Controller & Avionics

| Component | Spec | Weight | Price | Source |
|-----------|------|--------|-------|--------|
| Holybro Pixhawk 6C | STM32H743, triple IMU (BMI055, ICM-42688-P, BMP388), 32-bit MCU, plastic case | 35g | $141 | Holybro, Amazon |
| Holybro M9N GPS | u-blox M9N, compass, safety switch, buzzer | 32g | $55 | Holybro |
| SiK telemetry radio (optional, ground station link) | 433/915 MHz, 100mW, MAVLink | 14g | $60 (pair) | Holybro |
| RC receiver | FrSky R-XSR or ExpressLRS EP2 | 6g | $25 | — |
| Mounting foam | Pixhawk vibration isolation foam | 3g | $2 | Included |

**Notes.** GPS is not required for indoor operation but is included because (a) outdoor hover calibration testing is valuable during the build-up, and (b) the PX4 EKF benefits from having the GPS input available for fusion even indoors when dead-reckoning across localization dropouts.

### 1.6 Companion Compute

| Component | Spec | Weight | Price | Source |
|-----------|------|--------|-------|--------|
| NVIDIA Jetson Orin Nano Super Developer Kit | 67 TOPS, 8GB LPDDR5, integrated carrier board (remove plastic base for flight) | 165g (module + carrier + heatsink, no plastic base) | $249 | NVIDIA, Arrow, SparkFun |
| 256GB NVMe SSD | Samsung 980, WD Blue SN570, or similar M.2 2280 | 8g | $35 | Amazon |
| 5V/5A BEC | Matek UBEC DUO or Pololu 5V 6A step-down | 12g | $15 | — |
| Heatsink/fan (included with Orin) | Active cooling | (included) | — | — |
| USB-C to USB-A cable (to Pixhawk) | Short, shielded | 8g | $6 | — |
| Mounting plate (3D printed) | PLA/PETG standoff plate | 25g | $3 filament | DIY |

**Notes.** The Orin Nano Super runs Isaac ROS, ROS2 Jazzy, and the PPO policy inference simultaneously at 67 TOPS. The plastic base of the developer kit is flight-unsuitable (excessive weight, no vibration isolation, no standoff mounting) — remove it and mount the carrier board directly to the frame's top plate via 3D-printed standoffs with silicone vibration isolators. Power the Orin from a dedicated 5V/5A BEC, not the PM07's BEC, to avoid voltage sag affecting the flight controller during Orin compute spikes.

### 1.7 Sensor Suite

| Component | Spec | Weight | Price | Source |
|-----------|------|--------|-------|--------|
| Intel RealSense D435i | 87°×58° FOV, 0.1–10m depth range, IMU, USB 3.0 | 72g | $334 | Intel, Mouser, DigiKey |
| RPLiDAR A2M12 | 360° 2D LiDAR, 12m range, 10kHz sample rate, ROS2-supported | 190g | $300 | Slamtec, RobotShop |
| PMW3901 optical flow + VL53L1X ToF | Holybro OF module or MATEK optical flow module | 15g | $90 (Holybro optical flow) | Holybro |
| USB 3.0 cable (camera to Orin) | Short, shielded, 0.3m | 15g | $10 | — |
| Sensor mounting hardware | M3 screws, 3D-printed brackets | 20g | $8 | DIY |

**Notes.** The D435i mounts to the front of the frame angled ~15° downward for switch approach. The RPLiDAR mounts on the top plate for unobstructed 360° scan. The PMW3901 mounts underneath the center plate facing down — verify the ToF laser has clear line-of-sight to the floor through any frame openings.

### 1.8 Compliant Pusher Mechanism

| Component | Spec | Weight | Price | Source |
|-----------|------|--------|-------|--------|
| 3D-printed pusher assembly (PETG) | Flat pad ~30mm diameter, spring-damper prismatic joint, 10–15mm travel | 25g | $5 filament | DIY |
| Compression spring | Stainless steel, tuned for ~5mm compression under switch activation force | 3g | $3 | McMaster, local hardware |
| Linear bearing / bushing | Small brass or teflon bushing for smooth travel | 5g | $4 | McMaster |
| Mounting arm (carbon fiber tube, 6mm OD) | ~80mm length, forward-pointing under-frame mount | 12g | $6 | AliExpress |
| Mounting hardware | M3 screws, rubber dampers | 5g | $3 | — |

**Notes.** The pusher mounts to the underside of the center plate, extending forward past the propeller plane. This clears the front propeller disc overlap problem. The pusher is compliant by design: the spring absorbs impact transients, and the joint displacement is readable via an onboard linear encoder or estimated from visual cues.

### 1.9 Wiring, Connectors & Miscellaneous

| Component | Spec | Weight | Price | Source |
|-----------|------|--------|-------|--------|
| XT60 connectors | Gold-plated, battery + power module | 10g | $5 | Amazon |
| JST-GH cables | Pixhawk ecosystem standard, various lengths | 30g | $15 (assorted kit) | Holybro, Amazon |
| ESC signal wires | Dupont / servo style, 50cm | 15g | $5 | — |
| Zip ties, heat shrink, foam tape | Assorted | 40g | $10 | — |
| Spare props | 1 extra pair (CW + CCW) — included in propulsion total above | — | — | T-Motor |
| Prop balancer | Magnetic prop balancer (Du-Bro or similar) | — | $12 | Amazon |

### 1.10 Stage 1 Limitations Note

The MN4006 EVO + 13×4.4 quad delivers 2.78:1 T:W — adequate for indoor flight and most contact-task development, but tighter than the Stage 2 X8 configuration. Implications:

- **Hover throttle is ~60%**, leaving ~40% margin for disturbance rejection and contact transients. This is the lower bound of acceptable but not generous.
- **Single motor failure remains catastrophic** on the quad (no redundancy).
- **Payload growth past the current Stage 1 spec will degrade T:W quickly** — adding any of the Stage 2 max sensors (Livox Mid-360, D455 swap, Orin NX upgrade, totaling 311g) drops Stage 1 quad T:W to ~2.40:1, *below* the contact-task floor.

These are properties of running an X8-sized airframe (frame, sensors, compute, pusher) on 4 motors during Stage 1. The alternative — using lighter Stage 1-only sensors and compute, then upgrading sensors at Stage 2 transition — was rejected because it would force re-validation of the entire perception stack mid-project.

**Conclusion: Stage 1 is sufficient for Phases 1–5 of the implementation roadmap (hover, navigation, switch environment, RL press training in sim, system integration), but real-world contact testing in Phase 6 may require pulling Stage 2 forward earlier than planned.** This is a known accepted risk; the staged architecture allows the upgrade without rebuild.

### Stage 1 — Weight Budget

| Category | Weight |
|----------|--------|
| Frame + landing gear + mounting | 555g |
| 4× motors (MN4006 EVO 380) | 272g |
| 4× props (P134.4 13×4.4 CF) | 28g |
| 4× prop guards + mounting hardware | 88g |
| 4× ESCs (BS30A, of 8 purchased) | 24g |
| Power module + battery + strap | 504g |
| Flight controller + GPS + radio + RC | 90g |
| Companion compute (Orin + SSD + BEC + cables + mount) | 233g |
| Sensor suite (D435i + RPLiDAR A2 + optical flow + cables + mounts) | 312g |
| Pusher mechanism | 50g |
| Wiring, connectors, misc | 95g |
| **Total Stage 1 AUW** | **~2,251g** |

### Stage 1 — Thrust-to-Weight

- Total thrust: **6,260g** (4× 1,565g per MN4006 EVO on 13×4.4 at 4S, static)
- AUW: **2,251g**
- **T:W ratio: 2.78:1**
- Hover thrust needed per motor: ~563g
- Hover throttle: ~**60%**
- Estimated hover current per motor: ~3.8A × 4 = 15A
- Usable battery capacity: 4Ah (80% of 5000mAh)
- **Estimated flight time: ~14–16 minutes** (less under aggressive flight)

The prop guards add 88g and reduce T:W by ~0.11 points (without guards: 2.89). The drone has 1.78× the thrust needed to hover (hover at ~60% throttle), which is enough margin for nominal contact maneuvers and modest disturbance rejection. Aggressive maneuvers near payload limits are not advised on the quad — that's what Stage 2 is for.

### Stage 1 — Cost Summary

| Category | Cost |
|----------|------|
| Frame + landing gear | $130 |
| Propulsion (4 motors @ $88 + props/hardware, 4 motors of 8 purchased) | $397 |
| Prop guards (4 bumpers + hardware, of 8 purchased upfront) | $48 |
| ESCs (8 total, 4 used in Stage 1) | $40 |
| Power system (PM07 + battery + strap) | $87 |
| Flight controller + GPS + radio + RC | $283 |
| Companion compute | $302 |
| Sensors | $732 |
| Pusher mechanism | $21 |
| Wiring + misc | $63 |
| **Stage 1 total** | **~$2,103** |


---

## Stage 2 — X8 Coaxial Upgrade

Stage 2 adds four motors, four propellers, and integrates the four spare ESCs already purchased in Stage 1. No frame modification is required beyond adding the lower motor mounts. No flight controller, sensor, battery, or power system changes are needed.

### 2.1 Additional Components

| Component | Spec | Weight ×4 | Price ×4 | Source |
|-----------|------|-----------|----------|--------|
| T-Motor MN4006 EVO KV380 (×4 additional) | Same as Stage 1 | 272g | $352 | T-Motor |
| T-Motor P134.4 13×4.4″ CF prop (pair) | **13×4.4″ CF** — matched to upper props | 28g (7g each, 4 blades) | $40 (2 pairs) | T-Motor |
| Lower prop guards | Same partial-arc design as Stage 1 (see §1.2a), sized for 13″ props, mounts to lower motor brackets | 88g (22g each incl. mounting) | $48 (outsourced PA12) | JLC3DP / PCBWay or DIY |
| Lower motor mount brackets | 3D-printed (nylon PA12 recommended) or CNC aluminum, mounts to underside of existing arm | 32g (8g each) | $30 (4× outsourced JLC3DP nylon) | DIY + outsourced |
| M3 longer screws | For dual-motor arm mounting | 6g | $4 | — |
| Additional signal wires | Routing from ESC to lower motors | 12g | $4 | — |

**Prop size decision — matched 13″ top and bottom.** Earlier analyses considered mixed prop sizes (e.g. 13″ upper, 12″ lower) to reduce coaxial downwash interference on the lower row. This recovers only 3–4% of total system thrust at the cost of significant build complexity: two prop SKUs, four spare prop variants (CW/CCW × two sizes), harder matched-pair balancing, and per-motor thrust scaling in PX4. Given the X8 configuration provides 4.15:1+ T:W with matched props and guards, the 3–4% gain is not worth the operational complexity. Most commercial X8 platforms (DJI S1000, Freefly Alta X) use matched props for this reason.

**Ordering simplicity.** The same P134.4 13×4.4 prop pairs used in Stage 1 are used for Stage 2. **Strongly recommend buying all props upfront** (5 pairs total = 10 blades for Stage 1 + Stage 2 + spares) to guarantee identical prop batches across all positions — this improves dynamic balance across the airframe and avoids the risk of T-Motor revising the prop's mold mid-project.

### 2.2 Weight Budget — Stage 2 Additions

| Added component | Weight |
|-----------------|--------|
| 4× MN4006 EVO motors | 272g |
| 4× P134.4 13×4.4 CF props | 28g |
| 4× lower prop guards + mounting | 88g |
| 4× lower motor brackets | 32g |
| 4× ESCs (already purchased upfront, mounted in Stage 2) | 24g |
| Additional wiring + hardware | 18g |
| **Total added weight** | **~462g** |

**Stage 2 mid AUW: 2,251g + 462g = ~2,713g**

If running Max config (adds Livox Mid-360 3D LiDAR +265g, D455 instead of D435i +31g, Orin NX upgrade +15g): **Max Stage 2 AUW: ~3,024g**

### 2.3 Stage 2 Thrust-to-Weight

With matched 13×4.4″ CF props top and bottom, the upper motors produce full thrust and the lower motors experience coaxial downwash loss of ~20%.

- Upper motors: 4 × 1,565g = 6,260g
- Lower motors: 4 × 1,565g × 0.80 = 5,008g
- **Combined thrust: 11,268g**
- Stage 2 AUW (mid config): **2,713g → T:W 4.15:1**
- Stage 2 AUW (max config): **3,024g → T:W 3.73:1**
- Hover thrust per motor (averaged): ~377g (mid) / ~420g (max)
- Hover throttle: **~49% (mid) / ~52% (max)**

Both configurations comfortably exceed the 2.5:1 contact-task floor, and both exceed the 3.0:1 "ideal" threshold by a wide margin. The hover throttles in the 49–52% range sit in the engineering sweet spot (40–55%) — drone has plenty of margin for disturbance rejection while not being over-powered to the point of twitchy control.

### 2.4 Stage 2 Flight Time

Stage 2 hover current draw is roughly **17A total at hover (mid config)**, only 3.4C from a 5000mAh pack — much lower than the v1 BOM's estimate, because the MN4006 at 49% throttle is far more efficient than the MN3110 at 58% throttle. The MN4006's larger 18N24P stator provides higher torque at low RPM, which is exactly the regime hover requires.

- **Mid config: ~12–15 minutes** hover time
- **Max config: ~10–13 minutes** (additional ~310g of payload)

These figures assume 80% battery depth-of-discharge (4Ah usable from 5000mAh) and no aggressive maneuvering. Real flight time in mixed mission profiles with takeoff/landing, navigation transits, contact maneuvers, and climbs will be 70–80% of these estimates.

### 2.5 Redundancy Characteristics

X8 coaxial provides fault tolerance not available on either a quad or a hex:

- **Single motor failure:** remaining 7 motors maintain full attitude control with ~12% thrust loss. Hover throttle rises to ~55%. Mission can continue.
- **Two-motor failure (same arm, opposite directions — one upper + one lower):** remaining 6 motors maintain attitude with ~25% thrust loss. Hover throttle ~62%. Mission should return to home and land.
- **Two-motor failure (different arms):** attitude control degraded but maintained via mixer compensation. Mission aborts, emergency landing.
- **Three-motor failure:** attitude control may be lost depending on failure geometry. Emergency descent.

Compare to quad: **any single motor failure = immediate loss of attitude control, uncontrolled crash.** Compare to hex: **single motor failure survivable with degraded performance, two-motor failure typically fatal.**

For a contact-manipulation task where prop strikes, over-current events, or thermal stress from sustained pressing can damage motors, this redundancy is genuinely meaningful.

### 2.6 Stage 2 Cost Summary

| Category | Cost |
|----------|------|
| 4× additional motors (MN4006 EVO @ $88) | $352 |
| 4× additional props (2 pairs P134.4) | $40 |
| 4× lower prop guards + hardware | $48 |
| 4× lower motor brackets (outsourced nylon) | $30 |
| Additional wiring + hardware | $8 |
| **Stage 2 upgrade cost** | **~$478** |

### Total Project Cost

| | Cost |
|-|------|
| Stage 1 (complete quad build with guards) | $2,103 |
| Stage 2 upgrade (with lower guards) | $478 |
| **Combined total** | **$2,581** |


---

## Upgrade Path & Decision Criteria

The Stage 2 upgrade should be triggered by specific, measurable problems during Stage 1 operation. Do not upgrade speculatively.

### Upgrade if:

1. **Motor saturation during contact** — logs show any motor hitting >85% throttle during normal press maneuvers. The quad's 60% hover throttle leaves only 25% margin before saturation; saturation precedes loss of control.
2. **Single-motor-failure risk assessment** — if the operational environment (near humans, over expensive equipment, in critical infrastructure) makes an uncontrolled crash unacceptable, the X8's fault tolerance becomes required, not optional.
3. **Vibration-limited sensor performance** — if motor vibration (measured via IMU FFT, LiDAR scan noise, or camera image blur) is degrading SLAM or RL policy quality beyond what software filtering can address. Eight motors running at lower individual load produce less vibration per motor than four motors at higher load.
4. **Payload expansion requires more thrust margin** — adding any of the Phase 7 sensors (Livox Mid-360, D455, Orin NX) pushes Stage 1 quad T:W below 2.5:1. The X8 maintains 3.7:1+.

### Do not upgrade if:

1. **Stage 1 meets performance requirements** — if contact tasks succeed reliably, sensor data is clean, and flight time is adequate, the quad is sufficient.
2. **Only redundancy is desired** — redundancy without need is over-engineering. Most research drones in published contact-manipulation literature are quads.
3. **Cost pressure** — $478 is not trivial. If the project can demonstrate its goals on the quad, skip the upgrade.

### Software re-tuning required for Stage 2

- **PX4 airframe selection:** change from `Generic Quadrotor X` to `Generic Octocopter Coaxial`.
- **ESC motor mapping:** 8-channel output from PM07 — verify motor numbering matches PX4's octocopter X layout.
- **PID re-tune:** moment of inertia changes with dual motors, attitude controller gains need retuning. Budget 1–2 days for this.
- **RL policy fine-tuning:** the policy trained on quad dynamics will not transfer directly. Plan for 500–1,000 additional training iterations with X8 dynamics in Isaac Sim before operational deployment.
- **State machine update:** add motor health monitoring to detect individual motor failures and transition to degraded-mode operation.

---

## Appendix A — Rejected Alternatives

**Why MN4006 EVO over MN4006 base?** The EVO's upgraded EZO bearings (T-Motor cites 60–80h MTBF vs ~40–50h for the base variant), +8% peak thrust, and improved heat dissipation justify the $13/motor premium ($104 across 8 motors) for a research project where motors will see many hundreds of test cycles. If price is a hard constraint, the base MN4006 KV380 is acceptable — drops max-config T:W from 4.15 to 4.07 (negligible).

**Why not stay with MN3110 KV700 (v1 spec)?** Three issues: (1) heavier per-motor mass (99g vs 68g, total 248g penalty across 8 motors), (2) higher commutation frequency producing more vibration that degrades sensor data quality, (3) catalog max thrust of 1.2kg only achievable on 15″ props at 6S — on 12″ at 4S the real ceiling is ~1,150g, leaving Stage 2 max-config T:W at only 2.99:1.

**Why not MN3110 KV780 (cheaper alternative within MN3110 family)?** Same 100g motor mass penalty, same 12N14P stator vibration profile, only marginal thrust improvement (~1,200g on 12″). Costs the same as KV700 ($70/motor) but doesn't fix either of the MN3110's two main problems. The MN4006 EVO is a better $18/motor premium.

**Why not MN3508 KV380 or KV700?** Larger motor (87g, between MN4006 and MN3110), 1.8kg catalog max thrust, but 17A continuous current rating means it operates at >88% of continuous when run on 13×4.4 — no headroom for transients. Would need to stay at 12″ props to keep current safe, which under-uses the motor's thrust capability. The MN4006's 22A continuous rating provides more usable headroom.

**Why not stay with the S550 hex?** The S550's 60° arm spacing limits props to 9–10 inches. This caps per-motor thrust at ~1,400g even with the best motors. The X500 quad's 90° spacing allows 12–13″ props, producing meaningfully more thrust per motor.

**Why not start with X8 directly?** Unnecessary upfront cost ($478), PID tuning complexity during Phase 1 hover validation (X8 is harder to tune blind than a quad), and commits to a topology before data exists to justify it. Start simple, add complexity only when needed.

**Why not Tarot 680 (larger frame to fit MN4006's design 14–15″ props)?** The MN4006 EVO is genuinely under-utilized on 13″ props — its 1,800–2,000g per motor capability on 14.4–15″ props would yield Stage 2 X8 T:W of ~5.5:1. But: Tarot 680 is heavier (+300g base weight), bigger indoor footprint (may not fit through standard doorways), and requires full rebuild rather than incremental upgrade. The X500 V2 + 13×4.4 sweet spot trades 15% theoretical thrust for substantially better indoor practicality.

**Why not DJI Flame Wheel F550 clone?** No official dual-config support, integrated PCB center plate limits sensor mounting flexibility, plastic arms aren't rated for X8 loads, and the geometry is hex-only.

---

## Appendix B — Procurement Sequence

Order components in the following sequence to minimize blocking on long-lead items while maximizing build parallelism:

**Week 1 — Frame and flight basics:**
- Holybro X500 V2 frame kit
- Holybro Pixhawk 6C + M9N GPS + PM07 + SiK radios
- RC transmitter + receiver
- Battery (buy 2 for testing rotation)

**Week 2 — Propulsion (Stage 1 + Stage 2 motors and props upfront):**
- **8× T-Motor MN4006 EVO KV380** (buy full Stage 2 allocation now — avoids the risk of EVO stock being depleted mid-project and ensures all 8 motors come from the same production batch for matched bearings/balance). Distributors: iNsightFPV ($149.90/pair = $74.95/motor for base, EVO ~$88/motor), RobotShop, T-Motor official store. **Verify EVO variant specifically** — the base MN4006 KV380 is also good but is older spec.
- **5 pairs T-Motor P134.4 13×4.4″ CF props** (4 pairs serve all 8 motor positions on the X8, 1 pair spare; buying all from one batch improves dynamic balance)
- 2× 4-pack HGLRC BS30A ESCs (all 8 upfront)

**Week 3 — Compute and sensors:**
- NVIDIA Jetson Orin Nano Super (6–8 week lead time historically; order early)
- Intel RealSense D435i
- RPLiDAR A2M12
- PMW3901 optical flow module

**Week 4 — Mechanical details:**
- 3D-printed pusher parts (outsource to JLC3DP if quality matters; DIY if not)
- **Prop guards — 8 partial-arc bumpers sized for 13″ props** (4 for Stage 1 + 4 for Stage 2) — outsource to JLC3DP in nylon PA12 for best strength-to-weight. **CAD design must be sized for 13″ props (165mm radius + 15mm clearance = 180mm outboard reach)** — if reusing community designs for X500, verify they're for 12″ vs 13″ and scale outward by ~13mm if needed.
- Lower motor mount brackets (outsource for Stage 2 prep — ready to install when Stage 2 trigger hits)
- Hardware kits, connectors, wiring
- Prop balancer (Du-Bro magnetic or similar)

**Stage 2 trigger point (later):**
- Additional wiring if needed for routing to lower motors
- All motors, ESCs, props, and guards already on hand from Week 2 + Week 4 procurement

---

## Appendix C — Critical Build Notes

**Motor direction check.** On the X8 coaxial, each arm has two motors spinning in opposite directions. The upper motor's rotation direction determines the overall rotation class for that arm. Standard PX4 octocopter coaxial X assignments: arms 1 (front-right) and 3 (rear-left) have upper CW, lower CCW; arms 2 (rear-right) and 4 (front-left) have upper CCW, lower CW. Verify against PX4 `Octorotor Coaxial X` motor map before first power-up.

**Prop clearance verification.** Upper and lower props on the same arm must not contact each other or the arm hardware. On the X500 V2 with matched 13″ props top and bottom, axial spacing between prop planes should be at least 90mm to prevent downwash-induced flutter from causing prop-to-prop contact. Note: this is 10mm more than the v1 spec required for 12″ props — the larger prop's higher induced velocity demands more axial separation. Measure this before mounting props — 3D-printed motor mounts sometimes compress under load, reducing clearance.

**Adjacent-prop clearance.** The X500 V2's 354mm adjacent motor spacing leaves 354 − 2 × 165 = **24mm tip-to-tip clearance** between adjacent 13″ props. This is well within safe operating margin (recommended minimum is 10mm). Do not increase prop size beyond 13×4.4 on this frame.

**Pusher boom clearance.** With 13″ props, the pusher boom must thread between the front-mounted motor planes. On the X8, the 90mm vertical gap between upper and lower prop planes provides ~40mm clearance above and below a 10mm-thick pad. A 100mm-forward boom with the drone pitched 10° nose-down deflects the tip only 17mm, well within budget. **Verify the pusher boom mount on first assembly with a static test (drone powered off, manually pitch +/- 25°, check pusher-to-prop clearance).**

**Thermal management.** Eight motors running continuously generate significant heat in a compact frame. The Jetson Orin Nano also runs hot under AI workloads. The MN4006 EVO runs ~10–15°C cooler than the v1 MN3110 spec at hover (lower throttle, larger heat-dissipation surface), which improves the thermal margin for the Orin and surrounding electronics. Verify airflow paths during layout — do not mount the Orin directly above the battery (heat accumulation) or above any motor (prop wash turbulence).

**Battery strap orientation.** The battery sits directly above the pusher mechanism on this frame. In a crash, the battery can shear forward and impact sensors. Use two perpendicular hook-and-loop straps rather than one, and consider a battery tray with front lip.

**First flight order.** Do not attempt X8 first flight until Stage 1 has accumulated at least 30 minutes of stable hover time. The PID tuning knowledge from Stage 1 transfers partially but not completely — start Stage 2 in manual stabilize mode, not autonomous navigation, and gradually re-enable autonomy as each subsystem is verified.

**MN4006 EVO bearing replacement schedule.** T-Motor cites 60–80h flight time MTBF on EVO bearings. For a research project averaging 15-minute flights, that's roughly 240–320 flights before recommended bearing replacement. Track total powered-on time (PX4 logs this) and budget bearing replacement (T-Motor sells bearing kits for ~$15) at 50h cumulative motor runtime as preventive maintenance.

# S550 hexacopter for indoor contact tasks: a complete build analysis

**An S550 hexacopter can comfortably carry all specified sensor payloads while maintaining thrust-to-weight ratios above 2.7:1 — well beyond the 2.5:1 target needed for wall-pressing tasks.** The SunnySky X2212 980KV emerges as the optimal motor choice, delivering a verified **1,220g of thrust per motor** on 9×4.5 props at 4S, giving six motors a combined 7,320g of thrust against all-up weights of 1,700–2,600g depending on configuration. Even the heaviest configuration with both 2D and 3D LiDAR achieves roughly 8–9 minutes of hover time on a 4S 5000mAh battery — adequate for autonomous indoor missions. The hexacopter topology is specifically validated by IEEE-published research on aerial switch-pressing, where six-motor platforms provide the thrust margin, disturbance rejection, and partial motor-failure survivability that contact tasks demand.

## Three motor tiers match three mission profiles

The motor choice anchors the entire build. Four motors were evaluated against verified thrust test data at 4S (14.8V), and three emerge as viable candidates spanning budget to premium.

**SunnySky X2212 980KV ($15–23/unit)** is the best all-around choice. Official test data from SunnySky USA shows **1,220g max thrust with APC 9045 props at 18.2A** and 1,650g with 10×4.7 props at 29A. At typical hex hover loads of 300–440g per motor, each unit draws only **2.7–4.5A** — deep in the efficiency sweet spot at 7–9 g/W. The 57g weight, 22×12mm stator, and wide prop compatibility make it the workhorse of the 550mm hex community. Six units cost **$90–138**.

**T-Motor MN2212 920KV ($30–40/unit)** is the premium option. Though T-Motor only publishes 3S test data for the 920KV variant (617g max with 9.5×4.5 props), voltage scaling yields an estimated **900–1,100g per motor at 4S with 9×4.5 props**. The real value is build quality: EZO bearings, 1,000-hour MTBF, and exceptionally low vibration — critical when an IMU, LiDAR, and depth camera share the airframe. The conservative **15A continuous rating** (vs. SunnySky's 26A/30s) reflects honest engineering, not weakness. Six units run **$180–240**.

**SunnySky X2216 880KV ($20–30/unit)** is the thrust monster for the heaviest payloads. Its larger 22×16mm stator delivers **1,650g at full throttle with 10×4.7 props** (27.2A, 403W). At 67.5g per motor it adds 63g to the airframe versus the X2212, but provides the headroom needed when carrying both RPLiDAR A2 and Livox Mid-360 simultaneously. Six units cost **$120–180**.

The **Emax MT2213 935KV ($10–15/unit)** should be avoided for this application. Its 860g max thrust is marginal, the manufacturer recommends only 8-inch props at 4S, and build quality is inconsistent. The GARTT ML2212 920KV ($15–20) is a serviceable budget alternative with user-confirmed S550 compatibility.

| Motor | Weight | Max thrust (4S, 9045) | Hover efficiency | Price (×6) |
|-------|--------|-----------------------|------------------|------------|
| SunnySky X2212 980KV | 57g | **1,220g** (verified) | 7.5 g/W @ 300g | $90–138 |
| T-Motor MN2212 920KV | 54g | ~950g (estimated) | ~9 g/W @ 300g | $180–240 |
| SunnySky X2216 880KV | 67.5g | ~1,400g (w/ 1045) | 4.1 g/W @ max | $120–180 |
| Emax MT2213 935KV | 54g | ~860g (8045 only) | Basic | $60–90 |

## Detailed weight budgets reveal comfortable margins

Three configurations were modeled using verified component weights. Every sensor mass comes from manufacturer datasheets: **Pixhawk 6C plastic case at 34.6g**, Jetson Orin Nano Super dev kit at **155–176g** (module + carrier + heatsink, plastic base removed), Intel RealSense D435i at **72g**, RPLiDAR A2 at **190g**, and Livox Mid-360 at **265g**.

### Minimum configuration — 1,685g AUW

This stripped-down build carries only the depth camera and optical flow for basic autonomous navigation, plus the pusher mechanism. No landing gear — foam pads save 170–200g.

| Component | Weight |
|-----------|--------|
| S550 frame (no landing gear) | 480g |
| 6× T-Motor MN2212 920KV | 324g |
| 6× BLHeli_S 30A ESC | 42g |
| 6× 9045 nylon props | 48g |
| Pixhawk 6C Mini | 39g |
| Jetson Orin Nano Super | 155g |
| Intel RealSense D435i | 72g |
| PMW3901 optical flow | 5g |
| Spring-loaded pusher | 30g |
| 4S 5000mAh LiPo (Tattu) | 390g |
| Wiring, power module, mounting | 100g |
| **Total** | **~1,685g** |

### Mid configuration — 1,930g AUW

Adds RPLiDAR A2 for 2D SLAM-based navigation. This is the most practical indoor autonomy configuration — the 2D LiDAR enables robust localization while the D435i handles obstacle detection and target identification.

| Component | Weight |
|-----------|--------|
| S550 frame (no landing gear) | 480g |
| 6× SunnySky X2212 980KV | 342g |
| 6× BLHeli_S 30A ESC | 42g |
| 6× 9045 nylon props | 48g |
| Pixhawk 6C (plastic case) | 35g |
| Jetson Orin Nano Super | 165g |
| Intel RealSense D435i | 72g |
| RPLiDAR A2 | 190g |
| PMW3901 optical flow | 5g |
| Spring-loaded pusher | 40g |
| 4S 5000mAh LiPo (Tattu) | 390g |
| Wiring, power module, mounting | 120g |
| **Total** | **~1,929g** |

### Maximum configuration — 2,633g AUW

The full sensor suite with both 2D and 3D LiDAR, wider-baseline D455 camera, Orin NX compute, and landing gear for ground ops.

| Component | Weight |
|-----------|--------|
| S550 frame with landing gear | 680g |
| 6× SunnySky X2216 880KV | 405g |
| 6× BLHeli_S 30A ESC | 42g |
| 6× 1045 carbon nylon props | 68g |
| Pixhawk 6C (plastic case) | 35g |
| Jetson Orin NX (carrier + heatsink) | 180g |
| Intel RealSense D455 | 103g |
| RPLiDAR A2 | 190g |
| Livox Mid-360 3D LiDAR | 265g |
| PMW3901 optical flow | 5g |
| Spring-loaded pusher | 50g |
| 4S 5000mAh LiPo (CNHL) | 460g |
| Wiring, power module, mounting | 150g |
| **Total** | **~2,633g** |

## All three configs clear the 2.5:1 thrust target with room to spare

The thrust-to-weight analysis uses verified manufacturer test data for the SunnySky X2212 and scaled estimates for the T-Motor and X2216. Two figures matter: **maximum thrust** (full throttle, sustainable for 30 seconds) and **comfortable continuous thrust** (within long-duration current ratings).

| Config | AUW | Motor + Prop | Max thrust (×6) | T/W max | T/W continuous | Hover throttle |
|--------|-----|-------------|-----------------|---------|----------------|----------------|
| **Min** | 1,685g | MN2212 + 9045 | 5,700g | **3.38:1** | ~2.85:1 | ~30% |
| **Mid** | 1,929g | X2212 + 9045 | 7,320g | **3.79:1** | ~3.11:1 | ~26% |
| **Max** | 2,633g | X2216 + 1045 | 8,400g | **3.19:1** | ~2.74:1 | ~31% |

**The mid configuration is the sweet spot.** At 3.79:1 max T/W with the SunnySky X2212 on 9045 props, it hovers at just **26% throttle** — leaving enormous headroom for contact force generation, attitude corrections during wall pressing, and even single-motor-loss recovery. The per-motor hover current of only **~3A** (18A total) translates to roughly **13 minutes of flight** on a 5000mAh pack with 80% usable capacity.

For the min configuration, the T-Motor MN2212's lower thrust output still yields 3.38:1 — and the lower vibration signature meaningfully improves sensor data quality. Flight time stretches to **~14–15 minutes** thanks to the lighter payload.

The max configuration at 2.74:1 continuous T/W is the tightest, but still exceeds the 2.5:1 contact-task threshold. Hover time drops to **~9 minutes** as six X2216 motors at ~4.5A each consume ~27A. A 6000mAh battery (+100–150g, pushing AUW toward 2,800g) would recover 2–3 minutes but erode the T/W ratio below 3:1 max — a marginal tradeoff.

## The S550 frame, ESCs, props, and battery ecosystem

**The S550 frame** weighs **445–550g** depending on manufacturer (HobbyKing's version is lightest at ~445g). All versions use fiberglass-reinforced nylon arms with internal carbon fiber rod stiffening, FR4 PCB center plates with integrated power distribution (eliminating a separate PDB), and standard 16/19/25mm motor mount patterns that accept all 22xx-class motors. Landing gear adds **170–210g** — worth omitting for indoor use where takeoff/landing from a padded surface is acceptable. Retail price is **$25–40** on Amazon or AliExpress.

**Maximum prop size is 10 inches**, but **9-inch props are recommended** for safe inter-arm clearance. The 9×4.5 nylon prop (Gemfan 9045, ~8g each, $1.50–3/pair) is the default choice — cheap, durable, and safe for indoor use. The 10×4.5 carbon nylon (Gemfan 1045, ~11g, $3–5/pair) delivers 20–35% more thrust but fits tightly and draws significantly more current. **Avoid carbon fiber props indoors** — they shatter into dangerous fragments on impact, a serious risk near sensors and people. Keep a stock of cheap nylon 9045s for rapid replacement.

**For ESCs, six individual 30A BLHeli_S units are the clear winner.** The Hobbywing XRotor Micro 30A BLHeli_S ($8–12 each, **6g per unit**, DShot600) is the standard choice — proven, lightweight, and adequately rated for the 8–18A max current these motors draw with 9–10" props. Six units total $48–72 and 36–42g. BLHeli_32 ($12–18 each) adds bidirectional DShot for RPM filtering but is unnecessary for autonomous indoor flight where PID tuning precision matters less than reliability. A 6-in-1 board exists — the Pilotix BLS 65A ($100–130) on a 30.5mm mount — but limited community validation makes individual ESCs safer for a research platform.

**The 4S 5000mAh LiPo is the right battery.** The Tattu Bashing 5000mAh 4S 50C (**390g**, $45–55, 132×42×32mm) offers the best weight-to-capacity ratio and fits the S550 mounting area easily. The CNHL 5000mAh 4S (449–520g, $25–35) is a budget alternative at a 60–130g weight penalty. A 25C minimum C-rating is sufficient — hover draws only 5C from a 5000mAh pack. **6S is unnecessary and counterproductive** for this build: 2212-class motors at 800–1000KV are optimized for 4S, 6S batteries weigh 50–60% more at equivalent capacity, and the motor/ESC ecosystem would need to change entirely.

## Six motors beat four for pressing against walls

The hexacopter advantage for contact tasks is not theoretical — it is supported by published research. A 2018 IEEE IROS paper ("Contact Force Control of an Aerial Manipulator in Pressing an Emergency Switch Process") demonstrated a **hexa-rotor UAV with a 1-DOF manipulator** pressing emergency switches using impedance-based force control. More recently, a 2025 arXiv paper on safety-critical aerial physical interaction used a **DJI F550-based hexacopter** (the exact frame class under discussion) with KDE2314XF motors and 9-inch props for push-and-pull contact tasks at 3.5kg AUW.

The physics favor the hex in three specific ways. First, **disturbance rejection during contact** improves because six motors distribute corrective thrust across more actuators. When the drone presses a switch and experiences a reaction force, each motor needs smaller individual corrections — reducing the risk of any single motor saturating. A quad must make larger per-motor adjustments, and saturation of even one motor cascades into loss of attitude control. Second, **yaw authority is retained longer** under load. ArduPilot's mixer prioritizes roll and pitch over yaw, then yaw over thrust. With six motors providing three CW/CCW pairs instead of two, the yaw torque budget is 50% larger before yaw authority degrades. Third, **motor redundancy** allows degraded-but-survivable flight if one motor fails. ArduPilot's thrust-loss detection (merged as PR #9381) enables remaining motors to boost output when a failure is detected. The community consensus is that **T/W of 3:1 or higher** provides adequate margin for motor-loss recovery on a hex — all three configurations meet or approach this threshold.

The hex does carry disadvantages. The 550mm footprint is larger than a 450mm quad, limiting passage through narrow spaces. Six motors, ESCs, and props increase build complexity and cost by roughly **40–60%** over an equivalent quad. Prop wash from six rotors creates more turbulent downwash indoors, potentially affecting near-ground or near-ceiling operations. And flight time is inherently shorter — six motors consume more power than four at the same AUW, though the hex's lower per-motor loading partially offsets this.

For the specific application of pressing switches on walls, the hex's advantages are decisive. The contact force required to actuate a typical switch (2–10N) demands a **pitch angle of 10–20°** sustained against the wall. At 15° tilt, the drone diverts sin(15°) ≈ 26% of its thrust horizontally while retaining cos(15°) ≈ 97% vertically. The mid configuration's 3.79:1 T/W means the drone can simultaneously hover and push with **~5N of horizontal force** while keeping 40%+ throttle headroom for attitude stabilization.

## Conclusion: recommended build path and key decisions

The **mid configuration with SunnySky X2212 980KV motors and 9×4.5 nylon props** represents the optimal balance for this application. At ~1,930g AUW, it carries the Pixhawk 6C, Jetson Orin Nano Super, RealSense D435i, RPLiDAR A2, optical flow sensor, and pusher mechanism while achieving **3.79:1 max T/W and ~13 minutes of hover time** — both exceeding requirements. Total component cost for the propulsion system (motors, ESCs, props, battery) runs approximately **$190–260**.

Three decisions deserve careful attention. First, **start with 9×4.5 props** even though 10×4.5 fits; the thrust margin is already excellent and 9-inch props provide safer inter-arm clearance plus gentler failure modes indoors. Only move to 10-inch if the max configuration proves thrust-limited. Second, **invest in T-Motor MN2212s if budget allows** — the vibration reduction from EZO bearings measurably improves IMU and LiDAR data quality, which directly impacts autonomous navigation performance. The ~$100 premium for six motors pays for itself in reduced sensor noise and tuning time. Third, **omit landing gear and the Livox Mid-360** for initial development. The 2D RPLiDAR A2 provides excellent indoor SLAM, the D435i handles obstacle avoidance, and saving 430–470g keeps the drone in the lightweight regime where flight characteristics are most forgiving. Add the 3D LiDAR only after the contact task is proven with the mid configuration.
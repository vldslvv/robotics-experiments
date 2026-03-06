# Quadcopter Hardware Requirements — Bill of Materials

Drone hardware only. 500mm frame class, designed for indoor autonomous operation with switch-pressing capability. Three configurations: **Min** (functional prototype), **Mid** (robust R&D platform), **Max** (full autonomous discovery capable).

> Prices reflect retail USD as of early 2026. Component availability and pricing may shift — GPU/AI demand has inflated RAM, storage, and some electronics.

---

## 1. Frame & Structure

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **Frame** | S500 PCB frame kit — $25 | Holybro X500 V2 frame — $123 | Holybro X500 V2 frame — $123 |
| **Landing gear** | Included with S500 | Included with X500 V2 | Included with X500 V2 |
| **Notes** | PCB center plate, plastic arms. Adequate for indoor R&D but flexes under load. | 2mm carbon fiber twill plates, 16mm CF tube arms. Built-in companion computer mount points. | Same frame; payload capacity supports full sensor suite. |

---

## 2. Propulsion

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **Motors (×4)** | 2212 920KV generic (4-pack) — $35 | Holybro 2216 920KV (×4) — $120 | T-Motor AIR GEAR 450II kit (motors+ESCs+props) — $126 |
| **ESCs (×4)** | SimonK 30A (4-pack) — $36 | Hobbywing XRotor BLHeli_S 30A (×4) — $60 | Included in T-Motor kit, or Hobbywing 4-in-1 60A — $60 |
| **Propellers** | Plastic 1045 (8-pack) — $10 | Carbon fiber 1045 (4 pairs) — $20 | EMAX CF 1147 (4 pairs) — $26 |
| **Notes** | Budget motors have looser tolerances and shorter bearing life. Adequate for early testing. | Better bearings, tighter KV matching. Reliable for repeated testing. | T-Motor kit provides matched, balanced components. Larger 1147 props give more thrust headroom for heavy sensor payload. |

---

## 3. Flight Controller & GPS

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **Flight controller** | SpeedyBee F405 V4 — $37 | Holybro Pixhawk 6C Mini — $106 | Holybro Pixhawk 6C — $141 |
| **GPS module** | Beitian BN-880 (M8N + compass) — $18 | Holybro Micro M10 — $28 | Holybro M9N Standard (with compass, buzzer, safety switch) — $55 |
| **PDB** | Basic PDB — $5 | Matek PDB-XT60 (BEC 5V + 12V) — $14 | Matek PDB-XT60 — $14 |
| **Notes** | SpeedyBee runs Betaflight; ArduPilot support is limited. Fine for manual flight and basic waypoints. GPS is for outdoor testing only — indoor ops use LiDAR SLAM. | Pixhawk 6C Mini runs full PX4/ArduPilot. Triple IMU redundancy. Standard in robotics research. | Full-size Pixhawk 6C with more I/O ports for additional sensors. M9N offers faster fix and better multipath rejection. |

---

## 4. RC Control

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **Transmitter** | RadioMaster Pocket (ELRS) — $55 | RadioMaster Boxer (ELRS) — $130 | RadioMaster TX16S MKII (ELRS) — $225 |
| **Receiver** | Happymodel EP1 (ELRS) — $14 | RadioMaster RP3 V2 (ELRS) — $20 | RadioMaster RP4TD (ELRS, diversity) — $25 |
| **Notes** | Pocket is compact and fully capable. ELRS protocol gives excellent range and low latency. | Boxer has full-size gimbals — much better feel for development flying. | TX16S MKII is the flagship; useful only if operating multiple different aircraft. |

---

## 5. Battery & Power

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **Flight battery (×2)** | Zeee 4S 5200mAh 50C (×2) — $60 | Gens Ace 4S 5000mAh 45C G-Tech (×2) — $110 | Tattu R-Line 4S 5000mAh 95C (×2) — $140 |
| **Battery charger** | ISDT Q6 Pro (single port) — $40 | ISDT D2 Mark II (dual port) — $80 | SkyRC Q200 (4 port) — $120 |
| **Battery bag** | Fireproof LiPo bag — $10 | Fireproof LiPo bag — $10 | Fireproof LiPo bag — $15 |
| **Notes** | Two batteries minimum for continuous testing (one charging, one flying). Zeee packs are heavier (480g) but reliable. | Gens Ace G-Tech packs weigh only 468g — lightest mainstream option. Dual charger saves time. | Tattu R-Line for maximum burst current. 4-port charger for multi-battery rotation during intensive testing. |

---

## 6. Onboard Companion Computer

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **Computer** | Raspberry Pi 5 (4GB) — $75 | NVIDIA Jetson Orin Nano Super Dev Kit — $249 | NVIDIA Jetson Orin NX 16GB + carrier board — $750 |
| **Storage** | 64GB microSD — $10 | 256GB NVMe SSD — $35 | 512GB NVMe SSD — $55 |
| **Power regulation** | 5V BEC (from PDB) — $0 (included) | 5V/5A BEC module — $12 | Pololu 5V/6A step-down — $25 |
| **Mounting** | 3D-printed bracket — $3 | X500 V2 built-in mount — $0 | X500 V2 mount + vibration isolators — $10 |
| **AI performance** | No GPU; CPU-only inference. Sufficient for template matching and basic CV. | 67 TOPS; runs real-time object detection, SLAM, and policy inference simultaneously. | 157 TOPS; handles 3D LiDAR SLAM, CNN switch detection, and policy inference with headroom. |
| **Notes** | RPi 5 can run lightweight SLAM and simple CV but struggles with neural network inference at real-time rates. Good for Phases 1–4. | The Orin Nano Super is the sweet spot for robotics — enough compute for all Phase 1–6 tasks on one board. | Orin NX needed only for Phase 7 (simultaneous 3D SLAM + CNN detection + policy). Third-party carrier boards from Seeed or Waveshare ($100–$200 for the carrier alone). |

---

## 7. Sensors — Perception

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **Depth camera** | Luxonis OAK-D Lite — $149 | Intel RealSense D435i — $334 | Intel RealSense D455 — $419 |
| **Optical flow** | PMW3901 + VL53L1X module — $20 | PX4Flow (integrated flow + sonar) — $90 | PX4Flow — $90 |
| **2D LiDAR** | RPLiDAR A1 (12m, 360°) — $100 | RPLiDAR A2M12 (12m, 360°, 10kHz) — $300 | RPLiDAR A2M12 — $300 |
| **3D LiDAR** | — | — | Livox Mid-360 (40m, non-repetitive) — $550 |
| **Notes** | OAK-D Lite has on-device AI (4 TOPS) and stereo depth, but narrower FOV and shorter range than RealSense. | D435i is the robotics standard: global shutter, built-in IMU, excellent ROS2 support. RPLiDAR A2 gives higher scan rate for better SLAM. | D455 adds longer baseline for better depth at range. Livox Mid-360 is the lightest 3D LiDAR suitable for drone mounting (~265g). Only needed for Phase 7 autonomous discovery. |

---

## 8. Pusher Mechanism

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **3D-printed parts** | PLA pusher pad + mount (DIY) — $2 | PETG pad + spring guide (DIY) — $5 | Outsourced nylon PA12 (JLC3DP/PCBWay) — $15 |
| **Spring** | Small compression spring — $2 | Precision compression spring — $5 | Precision spring + damper — $8 |
| **Servo (if active)** | SG90 micro servo (9g) — $3 | MG90S metal gear (14g) — $6 | MG90S metal gear — $6 |
| **Hardware** | M3 screws, standoffs — $3 | M3 screws, standoffs, bearings — $8 | M3 hardware + linear rail — $15 |
| **Notes** | PLA is adequate for simulation validation and light indoor testing. Servo optional — passive spring-loaded design may suffice. | PETG is more durable and temperature-resistant. Metal gear servo survives repeated contact. | Outsourced nylon parts are the strongest and most dimensionally accurate. Linear rail gives the smoothest prismatic action for the compliant mount. |

---

## 9. Wiring, Connectors & Misc

| Component | Min | Mid | Max |
|-----------|-----|-----|-----|
| **Wire & connectors** | XT60, bullet connectors, servo wire — $15 | XT60, bullet connectors, JST-GH (for Pixhawk), wire loom — $25 | Same + shielded USB cables for camera/LiDAR — $35 |
| **Vibration damping** | Foam tape — $3 | Silicone damping balls (for FC) — $8 | Damping balls + isolator plate — $15 |
| **Zip ties, tape, heat shrink** | Assorted kit — $8 | Assorted kit — $8 | Assorted kit — $8 |
| **Spare props & hardware** | 1 extra prop set — $10 | 2 extra prop sets + spare arms — $25 | 3 extra prop sets + spare arms + spare motors (×2) — $50 |

---

## 10. Cost Summary

### Min Configuration — Functional Prototype
For Phases 1–4 in simulation with basic real-world hover testing. RPi companion, budget sensors.

| Category | Cost |
|----------|------|
| Frame & structure | $25 |
| Propulsion (motors, ESCs, props) | $81 |
| Flight controller & GPS | $60 |
| RC control | $69 |
| Battery & power | $110 |
| Companion computer | $88 |
| Sensors (camera, flow, 2D LiDAR) | $269 |
| Pusher mechanism | $10 |
| Wiring & misc | $36 |
| **Total** | **~$748** |

### Mid Configuration — Robust R&D Platform
For Phases 1–6. Pixhawk FC, Jetson Orin Nano, RealSense, reliable propulsion. The recommended build.

| Category | Cost |
|----------|------|
| Frame & structure | $123 |
| Propulsion (motors, ESCs, props) | $200 |
| Flight controller & GPS | $148 |
| RC control | $150 |
| Battery & power | $200 |
| Companion computer | $296 |
| Sensors (camera, flow, 2D LiDAR) | $724 |
| Pusher mechanism | $24 |
| Wiring & misc | $66 |
| **Total** | **~$1,931** |

### Max Configuration — Full Autonomous Discovery
For all phases including Phase 7. Adds 3D LiDAR and Jetson Orin NX for simultaneous 3D SLAM + detection.

| Category | Cost |
|----------|------|
| Frame & structure | $123 |
| Propulsion (motors, ESCs, props) | $212 |
| Flight controller & GPS | $210 |
| RC control | $250 |
| Battery & power | $275 |
| Companion computer | $840 |
| Sensors (camera, flow, 2D LiDAR, 3D LiDAR) | $1,359 |
| Pusher mechanism | $44 |
| Wiring & misc | $108 |
| **Total** | **~$3,421** |

---

## 11. Weight Budget Estimates

Payload weight matters — the 500mm frame with 2216 motors and 4S 5000mAh battery can typically produce ~2.5–3.0 kg of total thrust. A 2:1 thrust-to-weight ratio requires all-up weight ≤ ~1.3–1.5 kg.

| Component | Min (g) | Mid (g) | Max (g) |
|-----------|---------|---------|---------|
| Frame | 430 | 410 | 410 |
| Motors + ESCs + props | 280 | 320 | 340 |
| Flight controller + GPS | 50 | 65 | 80 |
| Battery | 480 | 468 | 468 |
| Companion computer | 45 | 85 | 130 |
| RGB-D camera | 58 | 72 | 100 |
| Optical flow module | 5 | 25 | 25 |
| 2D LiDAR | 170 | 190 | 190 |
| 3D LiDAR | — | — | 265 |
| Pusher mechanism | 20 | 35 | 50 |
| Wiring & misc | 40 | 60 | 80 |
| **Total** | **~1,578** | **~1,730** | **~2,138** |
| **Thrust-to-weight** | ~1.9:1 | ~1.7:1 | ~1.4:1 |

**Max config is tight on thrust margin.** The 3D LiDAR alone adds ~265g. Options to recover margin: upgrade to 2216 1000KV motors with 12" props (more thrust, requires larger frame), use a lighter 4S 4000mAh battery (saves ~80g, reduces flight time), or defer 3D LiDAR to a larger platform. The mid config has comfortable margins for sustained indoor flight.

---

## 12. Recommended Purchase Sequence

Aligned with the implementation roadmap phases:

**Buy first (Phases 1–3, mostly sim work):**
- Nothing — simulation phases require only a training workstation.

**Buy for Phase 4–5 integration testing:**
- Frame, propulsion, FC, GPS, RC control, batteries, charger — the core flyable drone.
- Companion computer (start with RPi 5 if budget-constrained; Jetson Orin Nano if serious).
- Cost: $500–$1,000.

**Buy for Phase 6 (sensor-based localization):**
- RGB-D camera, optical flow module, 2D LiDAR.
- Upgrade to Jetson Orin Nano if still on RPi.
- Cost: $400–$750 additional.

**Buy for Phase 7 (autonomous discovery):**
- 3D LiDAR (Livox Mid-360).
- Upgrade to Jetson Orin NX if compute-limited.
- Cost: $550–$1,100 additional.

This staged approach avoids buying expensive sensors before the software is ready to use them.

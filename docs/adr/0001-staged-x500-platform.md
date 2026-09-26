# ADR 0001: Staged Holybro X500 V2 platform (quad-first, X8 later)

- Status: Accepted
- Date: 2026-05-17
- Related: ADR 0010 (the quad asset's PX4 airframe)
- Full design: `docs/LIGHT_SWITCH_MULTIROTOR_ROADMAP.md`
- Hardware source: [staged BOM](../archive/HOLYBRO_X8_STAGED_BOM_v2.md)
- Historical mission plan: Git revision `a97389e0325dd54b3190752db3faf2f189509c33`,
  path `docs/archive/QUADCOPTER_PLAN_REVISED.md`
  (calculators: `docs/thrust_to_weight.py`, `docs/evaluate_motors_revised.py`)

## Context

The build needs a 500mm-class indoor platform that can hover, navigate, and apply
contact pressure to a wall switch. An earlier plan assumed a generic 500mm
hexacopter (S550/F550) with T-Motor MN3110 / 12" props. Detailed thrust analysis
showed the MN3110's catalog 1.2 kg max is only reached on 15" props at 6S; on a
12" prop at 4S the real per-motor thrust is far lower, pushing thrust-to-weight
below the ~2.5:1 contact-task floor.

## Decision

Adopt a **staged build on the Holybro X500 V2 frame**, quad-first:

- **Stage 1 -- Quadcopter (X4):** four arms at +/-45 deg, T-Motor MN4006 EVO
  KV380 + P13 (13x4.4") props. Fully flyable for the hover/navigate/RL roadmap
  phases. T/W ~2.78:1.
- **Stage 2 -- X8 coaxial octocopter:** add a second motor under each existing
  arm (four upper/lower pairs) for redundancy and thrust margin. Same frame, FC,
  compute, sensors, pusher.
- **Upgrade trigger is empirical, not speculative:** Stage 2 only when Stage 1
  shows motor saturation in contact, vibration degrading SLAM/policy, payload
  growth dropping T/W below 2.5:1, or a redundancy requirement.

X500 V2 chosen for its 90-deg arm spacing (allows 12-13" props at lower hover
RPM, better vibration / sensor data than a 60-deg hex) and official PX4/ArduPilot
dual-configuration support, enabling the incremental X4->X8 path on one frame.

## Consequences

- Defers ~$430 of Stage-2 hardware and avoids committing to an untested topology.
- Fixes the propulsion-margin gap of the rejected hexacopter/MN3110 plan.
- The simulated quad asset (`holybro_x500_v2.usda`, hardware-faithful per ADR
  0012) tracks this Stage-1 geometry; its PX4 SITL airframe binding is ADR 0010.

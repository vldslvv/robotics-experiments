# Light-switch multirotor roadmap

Build a PX4-controlled multirotor that approaches a known wall switch, hands off
safely, presses with a compliant pusher, confirms the lamp state, retreats, and
lands. Then replace ground-truth task state with perception and add discovery.
This is the single project roadmap; implementation contracts live in the area
documents linked from `AGENTS.md`.

## Implemented capabilities and qualification boundaries

| Area | Current boundary | Reference |
| --- | --- | --- |
| Flight runtime | PX4 lockstep, fleet lifecycle, navigation, recovery, and multi-vehicle isolation are implemented with integration coverage. | [Runtime operations](RUNTIME_OPERATIONS.md), [integration tests](INTEGRATION_TEST_ARCHITECTURE.md) |
| Airframes | Config generates the X500 body properties, USD, PX4 allocation, and bridge values. The S550 remains a placeholder; X8 is future work. | [Fleet config](FLEET_CONFIG.md) |
| Switch physics | Shared easy/nominal switch models, full-stroke mechanics, simulation-time activation, contact classification, and physical checks exist. Physical feasibility is distinct from learned-policy acceptance. | [Assets](ASSET_GENERATION.md) |
| Controller and flight evidence | Task-neutral recorded evidence and certified controller artifacts feed learning through strict contracts. Qualification belongs to an exact producer run. | [Alignment](SIM_TO_SIM_ALIGNMENT.md), [flight parameters](FLIGHT_PARAMETERS.md) |
| Learning | Hover, approach, contact, and press tasks and ordered complete-learner curricula are implemented. Historical hover acceptance does not qualify the complete nominal contact curriculum or current PX4 transfer. | [RL workflow](RL_WORKFLOW.md), [decisions](adr/README.md#learning-and-qualification) |
| Deployment | Generic policy adapters support hover/approach through PX4. PX4 contact/press deployment remains unsupported. | [Commanding](COMMAND_ROUTINE_CONTRACT.md) |
| Perception and mission | Synthetic pose degradation, sensor fusion, bounded mission retries, and autonomous discovery remain future work. | [RL follow-ups](backlog/rl_followups.md), [ROS validation](backlog/ros2_sensor_tf_validation.md) |

Implementation presence is not a new validation claim. Qualification requires the
producer's exact inputs, actor/controller digests, workload, and terminal report;
a deployment selector or dated roadmap status is not evidence.

## Next outcomes

1. Establish nominal contact feasibility using the current generated switch
   models and task termination rules, then obtain fresh acceptance for every
   stage of the complete contact curriculum. Tighten the provisional contact
   requirements after trainability is established.
2. Qualify learned hover/approach transfer and a complete known-switch mission.
   Extend PX4 contact support only with explicit physics, observation, failure,
   and acceptance evidence. Compare learned behavior with a scripted baseline.
3. Add seeded physical randomization, then synthetic latency, drift, dropouts,
   outliers, and confidence loss behind the existing pose-provider contract.
   Demonstrate rejection of stale observations and bounded recovery.
4. Add tracked ROS sensor/fusion inputs and cross-container validation. RGB-D
   supplies switch/lamp observations; LiDAR supplies mapping and wall geometry;
   IMU and optical flow/ToF support state estimation and GPS-denied hover.
5. Add mission-level approach, handoff, press, confirmation, retreat, retry, and
   abort/land sequencing. Later discovery maps free space, extracts walls,
   generates scanning routes, detects switches, and maintains a task queue with
   explicit success/skip/failure reasons.

Detailed unfinished learning and transfer work is in
[RL follow-ups](backlog/rl_followups.md). Sensor publication and frame checks are
in [ROS validation](backlog/ros2_sensor_tf_validation.md). These tasks extend the
existing runtime, command, and provider boundaries rather than creating a second
flight executor.

## Hardware track

[ADR 0001](adr/0001-staged-x500-platform.md) selects the staged Holybro X500 V2:
quad first, X8 only when measured saturation, vibration, payload growth, or a
redundancy requirement justifies it. Hardware work is a separate track, not a
prerequisite for every simulator milestone.

`configs/airframes/holybro_x500_v2.yaml` owns implemented component properties.
The [hardware BOM](archive/HOLYBRO_X8_STAGED_BOM_v2.md) preserves procurement
sources and X8 planning estimates. Battery SKU, pusher hardware, prop guards,
and measured dynamics still need physical verification; simulation traceability
alone does not establish hardware fidelity.

Before real contact flight, qualify landing recovery, independent PX4 protection,
the harness operating volume, and tether/prop/pusher clearance. See
[hardware follow-ups](backlog/drone_platform_followups.md) and
[the harness plan](backlog/safety_harness.md). Any X8 implementation needs matching
geometry, motor order/allocation, controller qualification, and policy tests.

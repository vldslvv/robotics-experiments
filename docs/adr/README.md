# Architecture Decision Records

Keep each ADR focused on one durable decision: Context / Decision / Consequences
and brief evidence references. Number records in recording order; preserve known
decision dates. Consolidations use a `Recorded` date and cite the source Git
revision/path rather than keeping execution transcripts. Source files removed
from the working tree remain readable with `git show <revision>:<path>`.

Current behavior belongs in the owning contract document, execution details in
producer run artifacts, and unfinished work in [the backlog](../backlog/rl_followups.md)
or [roadmap](../LIGHT_SWITCH_MULTIROTOR_ROADMAP.md). Mark superseded decisions
explicitly; historical acceptance is not current qualification.

## Platform, runtime, and development

| ADR | Decision |
| --- | --- |
| [0001](0001-staged-x500-platform.md) | X500 quad first; X8 only on demonstrated need. Hardware sources remain in the [BOM](../archive/HOLYBRO_X8_STAGED_BOM_v2.md). |
| [0002](0002-three-layer-feature-design.md) | Core behavior, Kit wiring, and standalone integration have distinct owners. |
| [0003](0003-px4-takeoff-instability-mitigation.md) | Separate takeoff dynamics failures from infrastructure and failsafe failures. |
| [0004](0004-enforce-package-layering.md) | Enforce dependency direction and imports with Isaac absent. |
| [0005](0005-pegasus-vendoring-policy.md) | Preserve vendored fixes and maintain their provenance. |
| [0006](0006-command-routine-contract-layer.md) | Pure routine decisions over a control-only MAVLink port. |
| [0007](0007-tests-assert-public-behavior.md) | Test public behavior and enforce private-member boundaries. |

## PX4 airframes

| ADR | Decision |
| --- | --- |
| [0008](0008-match-px4-sitl-airframe-to-drone-config.md) | Match each vehicle's SITL airframe to config. |
| [0009](0009-hexacopter-airframe-rotor-geometry.md) | Match hexa allocation to simulated rotor geometry. |
| [0010](0010-quad-asset-uses-x500-airframe.md) | Use the X500 model for the square-X quad. |
| [0011](0011-config-driven-px4-airframe-generation.md) | Generate airframes from config; supersedes hand-written mechanisms in 0008-0010. |
| [0012](0012-hardware-faithful-x500-and-computed-airframe-properties.md) | Cite component leaves and compute X500 body/rotor properties. |
| [0015](0015-px4-1.17-and-custom-airframe-ids.md) | PX4 1.17 and reserved custom model IDs; supersedes earlier numeric bindings. |

## Learning and qualification

| ADR | Decision |
| --- | --- |
| [0013](0013-in-lab-rl-action-level.md) | Velocity and CTBR action levels; controller ownership/frame details superseded by 0016/0018. |
| [0014](0014-press-observation-provider-contract.md) | Versioned observations, pose provider, and handoff; actor-layout details superseded by 0018. |
| [0016](0016-measured-flight-evidence-and-controller-qualification.md) | Independent measured evidence and certified controllers before learning. |
| [0017](0017-policy-selection-acceptance-and-publication.md) | Validation selection, fresh acceptance, and explicit deployment semantics. |
| [0018](0018-training-deployment-observation-and-actuation-parity.md) | Body-relative observations and matching command timing across training/deployment. |
| [0019](0019-complete-learners-and-isolated-attempts.md) | Complete learner restoration and isolated attempt/curriculum lifecycles. |

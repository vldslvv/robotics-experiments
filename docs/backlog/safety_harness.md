# Harnessed hardware flight prerequisites

Planned hardware work; it does not block simulator-only development. A tether
changes both flight dynamics and the available operating volume. Complete these
checks before real contact flights, using the existing command and policy
contracts rather than introducing another executor.

- **Landing recovery:** qualify the shared driver on the hardware shell. An
  armed failure enters landing recovery; failed Land acknowledgement retains
  PX4's configured offboard-loss fallback. `on_failure` remains notification-only.
  Demonstrate an injected envelope failure and terminal recovery in SITL first.
  RC kill and PX4 failsafes remain independent of companion software.
- **Operating volume:** add an explicit configured world-ENU geofence invariant
  and match it with independent PX4 hardware geofencing. Test every boundary and
  companion-link loss; a software check alone is insufficient.
- **Contact entry and exit:** use the existing handoff predicate and generic
  `{kind: policy}` interface. Exercise stale-pose rejection, bounded re-approach,
  release/retreat when recoverable, and exhausted-retry abort-to-land. PX4 contact
  deployment is currently unsupported and must be qualified before hardware use.
- **Tether dynamics:** model zero force while slack and a spring-damper pull
  toward a configured anchor when taut. Author anchor, slack, stiffness, damping,
  and randomization in complete stage experiments; curriculum labels must not
  implicitly select behavior. Test recovery after taut events as well as nominal
  free flight.
- **Mechanical clearance:** verify the whole operating volume, worst-case pitch,
  taut-tether direction, and pusher/prop paths with power off. Keep the attachment
  above the center of mass and clear of props and the GPS mast; prevent a catch
  from swinging the drone or tether into the wall. Record the measured clearance
  before flight.

[RL follow-ups](rl_followups.md) owns the related contact/randomization work.
[The hardware BOM](../archive/HOLYBRO_X8_STAGED_BOM_v2.md) retains build sources;
its estimates do not replace the physical clearance and recovery checks.

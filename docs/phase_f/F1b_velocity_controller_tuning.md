# F1b -- In-Lab velocity-controller tuning

Goal served: the F1 Step-7 `test-integration-press-hover-goto` gate is a
trustworthy precondition for F3 training. It drives the Isaac Lab
`DroneHoverEnv` with a scripted outer position loop, verifies hover and a 2 m
lateral goto, and requires the final 1.0 s window to stay inside tolerance.

## Current Gate Contract

- Default seed: `7`, so the make target is reproducible.
- Default fleet scale: the configured `press.num_envs` (`16` in the default
  config).
- Hover phase: 4.0 s, final-window max position error must stay below `0.1 m`.
- Goto phase: 6.0 s to a 2 m +X target, final-window max position error must
  stay below `0.2 m`.
- Gate episode length is owned by the script and set to hover + goto + 1.0 s, so
  Isaac Lab timeouts cannot reset an env during validation.
- Any crash / out-of-bounds termination or episode timeout fails the gate.
- Diagnostic mode:
  `test_press_hover_goto.py --diag --seed <N> --num-envs 1 --log-every <N>`
  logs sim time, position error, world/body velocity, yaw, tilt, body angular
  velocity, commanded velocity, rotor omega min/max, and final-window metrics.

## Final Gains

`configs/experiment/press/default.yaml`:

```yaml
action:
  controller_kp_vel: [2.0, 2.0, 3.0]
  controller_kd_vel: [1.2, 1.2, 1.5]
  controller_kp_att: [6.0, 6.0, 2.0]
  controller_kp_rate: [1.2, 1.2, 0.4]
```

The fix is config-only. No integral term, explicit yaw hold, or controller
structure change was needed, so the numpy reference in
`press/actuation.py` and the torch implementation in
`workflows/isaac_lab/tasks/_flight.py` remain structurally unchanged.

## Measured Results

Seeded 16-env make gate, run three times back-to-back:

```text
hover final=0.060 m, last1.0s_max=0.092 m
goto  final=0.005 m, last1.0s_max=0.039 m
```

One-env diagnostics across seeds:

```text
seed 7:  hover last1.0s_max=0.090 m, goto last1.0s_max=0.037 m
seed 11: hover last1.0s_max=0.089 m, goto last1.0s_max=0.037 m
seed 23: hover last1.0s_max=0.021 m, goto last1.0s_max=0.039 m
```

Rotor omega stayed away from sustained saturation during hover and goto. The
goto starts with the expected transient demand (`omega` roughly
`[186, 414] rad/s` on the X500) and settles near hover (`[319, 323] rad/s`).

CTBR sanity mode (`--ctbr-sanity --seed 31 --num-envs 1`) remains stable with the
same shared rate-loop gains:

```text
hover_fraction=0.374
max_tilt=0.106 rad
max_yaw_rate=0.355 rad/s
z_drift=0.027 m
omega=[319.2,322.7] rad/s
```

## Verification Commands

Host-side:

```bash
uv run pytest tests/test_pressactuation.py tests/test_config.py tests/test_core_imports.py
```

Isaac Lab:

```bash
make test-integration-press-hover-goto
make test-integration-press-hover-goto
make test-integration-press-hover-goto
```

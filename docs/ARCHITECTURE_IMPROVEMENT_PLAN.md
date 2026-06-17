# Architecture Improvement Plan

Whole-project architecture review for the Isaac Sim + ROS 2 + PX4 stack, captured
as step-by-step improvements. Each step is sized to one PR and follows a
consistent format (Problem with `file:line` anchors, Files, Outline, Tests,
Acceptance, Status) so it can be handed straight to a coding agent.

## Why this exists

The repository recently went through a focused PX4-bridge correctness
investigation, now resolved -- the resulting fixes are in git history and are
summarized in Track P1 below. This document zooms out from bridge correctness to
the **whole project**: where code is redundant or over-concentrated, and how to
bring the structure, build system, test infrastructure, and a few larger design
choices in line with how current Isaac Sim + ROS 2 + PX4 stacks are built. The
macro architecture is already strong; the open work is structural.

## Conventions every step must honor

- Host Python runs through the project venv: `uv run pytest` / `make test`
  (CLAUDE.md "Python environment"). Never call `pip`/system Python.
- Tunable thresholds, timeouts, and rates live in Hydra `configs/` plus a bare
  annotation in `src/robotics_experiments/config/types.py`; never hardcode them
  (CLAUDE.md "Hard rules").
- Keep lazy imports that defer Isaac / Pegasus / PX4 dependencies.
- Do not rename, remove, or broaden public `__all__` exports as part of cleanup
  without explicit sign-off.
- Any behavior or convention change is mirrored into `CLAUDE.md`.
- No new tooling unless explicitly justified by the step; vendored Pegasus stays
  excluded from ruff/mypy.

## Current-state assessment

Verified during review (line counts are current):

- `src/robotics_experiments/` is ~3,200 LOC across 28 files. The one outlier is
  `px4_bridge/mavlink.py` at **847 LOC** -- the single god-module, with two
  god-methods: `step()` (`mavlink.py:471`) and `_poll_for_actuator()`
  (`mavlink.py:773`).
- `tests/` is ~4,000 LOC across 15 files with **no `conftest.py`**;
  `tests/test_px4_bridge.py` alone is **1,246 LOC / 41 tests** with fakes defined
  inline (`_FakeArticulationIO`, `_FakeFlightStack`, `_CapturingThrustModel`).
- Build system: `Dockerfile` is 322 LOC (zsh setup ~50 LOC and nvim/tmux ~45 LOC
  duplicated across the isaac and ros stages; the px4 stage re-does tmux
  separately). `make/containers.mk` is 162 LOC: three near-identical `run-*`
  blocks plus ~18 trivial one-line delegating targets.
- `px4/docker/run_px4_iris_fresh.sh` and `run_px4_iris_ci.sh` share an
  **identical** state-reset prologue (lines 1-22); the CI variant differs only in
  its `sed` log-filter tail.
- Vendored Pegasus is ~2,100 LOC under
  `isaac/runtime/extensions/px4_sitl_brigde/pegasus/`, excluded from lint; the
  bridge reuses its sensors and MAVLink backend.
- **No CI** (`.github/workflows` does not exist); quality is gated only by local
  `make test` and pre-commit (ruff / mypy).

Maturity snapshot: Layering A, Testability A, Frame conventions A-, Complexity
management B (god-methods), Redundancy control B-, Pegasus integration B+,
CI/CD F (absent).

---

## Track P1 - Bridge correctness and lockstep integrity (DONE - context only)

The flight-critical correctness work from the recent PX4-bridge investigation is
already implemented. This track is retained so the architecture roadmap is
complete; no action is required here beyond awareness. The `#` column enumerates
the original investigation items (several map to the `-- Step N` commit
messages). Verified against the tree and git history:

| # | Subject | Status | Evidence |
|---|---|---|---|
| 1 | Tilt-cutoff / NaN guard keeps the lockstep exchange | DONE | `bridge.py:137-161` calls `port.step` every tick, gates only forces; NaN -> raise by design (`bridge.py:110-135`), documented in CLAUDE.md |
| 2 | Sensor rate equals physics cadence, validated | DONE | `mavlink.py:316-328` pins `physics_tick_us`; "sensor cadence mismatch" disconnect; commit `Sync frequencies for sim and px4` |
| 3 | Barometer accumulated drift | DONE | `barometer.py:131-133` with intentional-divergence comment; commit `Accumulated barometer drift` |
| 4 | `frames.py` / Pegasus equivalence | DONE | `tests/test_frames_pegasus_equivalence.py`; commit `Frame helpers and pegasus alignment` (see Step 5.1 for the remaining consolidation) |
| 5 | Remove dead acceleration finite-diff | DONE | no `linear_acceleration_enu` remains in `px4_bridge/`; commit `Cleanup redundant calculations -- step 5` |
| 6 | GPS ground-truth units | DONE | commit `Proper GPS handling -- Step 6` |
| 7 | Drop px4 autolaunch | DONE | commit `Get rid of px4 autolaunch option -- Step 7` |
| 8 | HIL_SENSOR temperature guard | DONE | commit `Small comment about divergence with Pegasus -- Step 8` |
| 9 | Remove dead vendored Pegasus modules | DONE | commit `Cleanup unused pegasus modules -- Step 9` |
| 10 | Dead `Px4SitlBridge_python` shim files | OBSOLETE | `articulation_io.py` removed; `global_variables.py` is now used by `extension.py:14`, so the premise no longer holds |
| 11 | Remove dead `max_rotor_force_n` config | DONE | no references remain; commit `Cleanup unused config key` |

Action: none. If a future "sync with upstream Pegasus" is attempted, re-check the
intentional divergences (barometer drift, HIL temperature field, sim-time
lockstep) -- this is folded into P5.2.

---

## Track P2 - Decompose the `mavlink.py` god-module (DONE)

Goal: turn one 847-LOC module with two long god-methods into single-responsibility
units, without changing the public seam. `Px4MavlinkPort` stays the only export
(`mavlink.py:27`).

All three steps below are landed: the Pegasus sensor layer is extracted to
`_pegasus_sensors.py`, `step()` is a linear three-phase sequence, and
`_poll_for_actuator` is split into named helpers. `mavlink.py` is now ~739 LOC,
`Px4MavlinkPort` is still the only export, and the full unit suite is green.

### Step 2.1 - Extract the Pegasus sensor layer

**Problem.** `mavlink.py` mixes the MAVLink port with Pegasus sensor wrapping. The
`_PegasusSensorSuite` class (`mavlink.py:86`) and the sensor-config factories
`_pegasus_imu_config` ... `_pegasus_magnetometer_config` (`mavlink.py:176-238`)
are ~150 LOC that have nothing to do with the lockstep transport.

**Files.** New `src/robotics_experiments/px4_bridge/_pegasus_sensors.py` (private);
`mavlink.py` (imports).

**Outline.**
1. Move `_PegasusSensorSuite` and the four `_pegasus_*_config` helpers into
   `_pegasus_sensors.py`, keeping the lazy Pegasus import inside them.
2. Import them back into `mavlink.py`. Pure move; no behavior change. Do not add
   a public export.

**Tests.** Existing sensor-path cases in `tests/test_px4_bridge.py` pass unchanged.

**Acceptance.** `mavlink.py` drops ~150 LOC; public API unchanged; suite is
importable and unit-testable on its own.

**Status.** DONE. `_PegasusSensorSuite` and the four `_pegasus_*_config`
factories live in `src/robotics_experiments/px4_bridge/_pegasus_sensors.py`
(with `_DEFAULT_SENSOR_RATE_HZ`, re-imported into `mavlink.py` to avoid a cycle);
lazy Pegasus imports preserved, no new public export. Verified `uv run pytest
tests/test_px4_bridge.py` green; mypy/ruff at baseline.

### Step 2.2 - Decompose `Px4MavlinkPort.step()`

**Problem.** `step()` (`mavlink.py:471`, ~100 LOC) sequentially does connection
check, timestamp-monotonicity guard, sensor fill, heartbeat send, first-heartbeat
wait, IMU-refresh validation, sensor+GPS send, and actuator poll, with several
early-return paths. The happy path is hard to read without jumping ~100 lines.

**Files.** `src/robotics_experiments/px4_bridge/mavlink.py`.

**Outline.**
1. Extract three private phase methods on the class: `_heartbeat_phase` (send +
   wait first heartbeat), `_sensor_phase` (fill + cadence/refresh validation +
   send sensors/GPS), `_actuator_phase` (delegates to `_poll_for_actuator`).
2. `step()` becomes a linear sequence calling the three phases, with early returns
   reserved for the connection/timeout exits only.
3. No semantic change to ordering, timeouts, or the cadence-mismatch disconnect.

**Tests.** All existing lockstep / heartbeat / sensor-send / actuator-lag tests in
`tests/test_px4_bridge.py` pass unchanged -- they are the safety net for this
refactor.

**Acceptance.** No method over ~40 LOC; `step()` reads top-to-bottom as
heartbeat -> sensor -> actuator.

**Status.** DONE. `step()` is a linear connection-check -> monotonicity-guard ->
`_heartbeat_phase` -> `_sensor_phase` -> `_actuator_phase` sequence; each phase
returns a bool and owns its own disconnect. No ordering/timeout/disconnect-reason
change. Done standalone against the existing inline test net (P4.1 not required);
all sensor/heartbeat/cadence tests pass unchanged.

### Step 2.3 - Split the `_poll_for_actuator` lockstep policy

**Problem.** `_poll_for_actuator()` (`mavlink.py:773`, ~75 LOC) tangles three
concerns: the bootstrap prime-then-block decision, the blocking-recv timeout
loop, the post-wait backlog drain, and the `max_sustained_actuator_lag` freshness
tripwire. These are exactly the invariants CLAUDE.md calls out, but they cannot be
tested in isolation today.

**Files.** `src/robotics_experiments/px4_bridge/mavlink.py`.

**Outline.**
1. Extract `_needs_to_block_for_actuator()` (the prime-then-block decision) and
   `_drain_actuator_backlog()` (apply newest queued packet) as named helpers.
2. Keep the timeout/deadline arithmetic in the poll loop; keep the freshness
   tripwire (the "actuator lag exceeded sustained limit" disconnect) intact.

**Tests.** Add a focused unit test that feeds a synthetic backlog and asserts the
newest packet is applied; keep existing actuator-lag and prime-then-block tests.

**Acceptance.** Each CLAUDE.md actuator invariant (prime-then-block, drain newest,
sustained-lag tripwire) is covered by a named test.

**Status.** DONE. `_poll_for_actuator` now delegates the prime-then-block decision
to `_needs_to_block_for_actuator()` and the post-wait drain to
`_drain_actuator_backlog()`, keeping the timeout/deadline loop and the
"actuator lag exceeded sustained limit" tripwire intact. Added
`test_drain_actuator_backlog_applies_newest_queued_packet`; existing actuator-lag
and prime-then-block tests unchanged and green.

---

## Track P3 - Redundancy elimination

Independent, low-risk steps. Each removes a duplicated pattern.

### Step 3.1 - Config-resolution helper

**Problem.** The fallback pattern
`if config is None: from robotics_experiments.config import get_config; config = get_config().<section>`
repeats in four modules: `drone/spawn.py`, `environment/stage.py:185-188`,
`environment/step.py`, `simulation/_isaac_settings.py`.

**Files.** New `src/robotics_experiments/config/_resolve.py`; the four call sites.

**Outline.**
1. Add `resolve_section(name: str, provided)` that returns `provided` if not
   `None`, else `getattr(get_config(), name)` with the `get_config` import kept
   inside the function (preserves laziness).
2. Replace each inline fallback with a call to it.

**Tests.** `tests/test_config.py` plus the existing per-module tests.

**Acceptance.** The fallback-resolve pattern is defined once.

**Status.** DONE. `resolve_section(name, provided)` lives in
`src/robotics_experiments/config/_resolve.py` (generic over the section type,
`get_config` import kept inside the function). Applied to all five call sites:
`drone/spawn.py` (spawn + despawn), `environment/stage.py`,
`environment/step.py`, `simulation/_isaac_settings.py`. Added
`test_resolve_section_*` in `tests/test_config.py`.

### Step 3.2 - Deduplicate USD asset spawning in `stage.py`

**Problem.** `_spawn_switches` (`environment/stage.py:82-115`) and `_spawn_lamps`
(`stage.py:118-170`) share the same body: `DefinePrim(path, "Xform")` ->
`GetReferences().AddReference(usd_path)` -> `UsdGeom.Xformable` ->
`ClearXformOpOrder()` -> `AddTranslateOp().Set(pos)`.

**Files.** `src/robotics_experiments/environment/stage.py`.

**Outline.**
1. Extract `_reference_asset_at(stage, prim_path, usd_path, pos) -> Usd.Prim`
   holding the shared reference+transform body.
2. `_spawn_switches` becomes a thin loop over it; `_spawn_lamps` calls it and
   keeps only its light-intensity / shader-emission post-step (`stage.py:154-165`).

**Tests.** `tests/test_environment_switches.py`.

**Acceptance.** The reference+transform body exists in exactly one place.

**Status.** DONE. `_reference_asset_at(stage, prim_path, usd_path, pos)` holds
the shared DefinePrim/AddReference/Xformable/ClearXformOpOrder/AddTranslateOp
body; `_spawn_switches` loops over it and `_spawn_lamps` keeps only its
SphereLight/OmniPBR post-step. `tests/test_environment_switches.py` green
unchanged.

### Step 3.3 - Sensor-noise config factory

**Problem.** The four `_pegasus_*_config` factories each repeat the same
"if noise disabled, zero the noise_density / random_walk / bias keys" logic
(`mavlink.py:176-238`, relocating to `_pegasus_sensors.py` after Step 2.1).

**Files.** `src/robotics_experiments/px4_bridge/_pegasus_sensors.py` (post-2.1).

**Outline.** Add `_zero_noise(cfg: dict, keys: Iterable[str]) -> None` and call it
per sensor with that sensor's noise keys.

**Tests.** Existing noise-disabled assertions in `tests/test_px4_bridge.py`.

**Acceptance.** The zeroing logic is defined once; behavior identical.

**Status.** DONE. `_zero_noise(cfg, keys)` zeroes the flat noise terms for the
GPS, barometer, and magnetometer factories (the magnetometer's
`bias_correlation_time` is set separately as a structural constant). IMU keeps
its nested gyro/accel sub-dicts; its `config` dict was annotated
`dict[str, object]` so the nested assignments type-check. Sensor-config tests
in `tests/test_px4_bridge.py` green unchanged.

### Step 3.4 - Symbolic thrust constants

**Problem.** `PegasusQuadraticThrustModel.__init__`
(`thrust_model.py:118-127`) repeats the iris motor constants per axis:
`rotor_constant` `8.54858e-6` x4, `rolling_moment_coefficient` `1e-6` x4,
`max_rotor_velocity` `1100.0` x4, `min_rotor_velocity` `0.0` x4.

**Files.** `src/robotics_experiments/px4_bridge/thrust_model.py`.

**Outline.** Define module-level constants
(`_IRIS_ROTOR_CONSTANT`, `_IRIS_ROLLING_MOMENT`, `_IRIS_MAX_ROTOR_VELOCITY`) and
build the defaults as `(_IRIS_ROTOR_CONSTANT,) * 4`, etc. These remain
library-default fallbacks; the Hydra config stays the runtime source of truth.

**Tests.** `tests/test_thrust_model.py`.

**Acceptance.** Each constant has one literal definition; defaults unchanged.

**Status.** DONE. Module-level `_IRIS_ROTOR_CONSTANT`, `_IRIS_ROLLING_MOMENT`,
`_IRIS_MIN_ROTOR_VELOCITY`, `_IRIS_MAX_ROTOR_VELOCITY` defined; defaults built as
`(_IRIS_*,) * 4`. `tests/test_thrust_model.py` green; defaults unchanged.

### Step 3.5 - Collapse `make/containers.mk` boilerplate

**Problem.** Three near-identical `run-*` targets (`containers.mk:5-71`) repeat the
`docker ps -a | grep -> docker start else docker run` pattern, and ~18 trivial
`stop-* / remove-* / clean-*` targets (`containers.mk:79-162`) are one-line
delegations.

**Files.** `make/containers.mk`.

**Outline.**
1. Define a GNU-make `define run-container ... endef` template parameterized by
   container name, image, and a per-container args variable; instantiate it for
   isaac/ros/px4. Keep the per-container `docker run` arg lists in `make/docker.mk`
   variables (already mostly there for isaac).
2. Generate the `remove-*-container`, `remove-*-image`, and `stop-*` targets with
   `$(foreach ...)`. Keep the `clean-*` aliases as the documented legacy
   compatibility shims (CLAUDE.md lists them).

**Tests.** Manual: `make run-isaac`, `make run-ros`, `make run-px4`, `make stop`,
`make destroy` behave identically (dry-run with `make -n` first).

**Acceptance.** Adding a fourth container is a few variables, not a copied block.

**Status.** DONE. `make/containers.mk` now uses a single
`run-container` template for Isaac, ROS 2, and PX4, with per-container runtime
arguments centralized in `make/docker.mk`. Stop/remove targets are generated from
`CONTAINER_STACKS`, while documented compatibility aliases remain unchanged.
Verified with dry-run lifecycle targets.

### Step 3.6 - Deduplicate Docker and PX4 shell boilerplate

**Problem.** (a) The zsh setup (`Dockerfile:59-109`) and nvim/tmux setup
(`Dockerfile:115-159`) are duplicated across the isaac and ros stages, and the px4
stage re-does tmux separately. (b) `px4/docker/run_px4_iris_fresh.sh` and
`run_px4_iris_ci.sh` share an identical state-reset prologue (lines 1-22); the CI
script differs only in its trailing `sed` log filter (lines 25-28).

**Files.** `Dockerfile`; new shared shell fragment(s) under an appropriate docker
build-input dir; `px4/docker/run_px4_iris_fresh.sh`, `run_px4_iris_ci.sh`.

**Outline.**
1. Move the zsh and nvim/tmux setup into COPY-ed scripts sourced by the isaac and
   ros stages; normalize the px4 tmux block to the same source.
2. Extract the px4 state-reset prologue into a sourced fragment (or a small helper
   script) used by both run scripts; the CI script keeps only its `sed` tail.

**Tests.** `make build` still builds all three images headless; a px4 smoke launch
(`make run-px4`) starts cleanly.

**Acceptance.** Each setup/reset block is defined once.

**Status.** DONE. The current Dockerfile had already collapsed the zsh/nvim setup
through shared `zsh -> nvim -> ai-tools` stages; the remaining duplicate tmux
configuration is now sourced from `docker/tmux.conf` for both nvim-based images
and PX4. PX4 SITL state reset lives in `px4/docker/reset_px4_state.sh` and is
used by both fresh and CI launch scripts; the CI script keeps only its log-filter
tail.

---

## Track P4 - Test infrastructure and CI

### Step 4.1 - Shared test fixtures via `conftest.py`

**Problem.** There is no `tests/conftest.py`. The fakes `_FakeArticulationIO`,
`_FakeFlightStack`, `_CapturingThrustModel`, and a fake MAVLink port are defined
inline, and `tests/test_px4_bridge.py` is a 1,246-LOC / 41-test monolith.

**Files.** New `tests/conftest.py` (or `tests/_fakes.py`); split
`tests/test_px4_bridge.py`.

**Outline.**
1. Promote the shared fakes into `conftest.py` fixtures (or a `_fakes.py` helper
   imported by tests).
2. Split `test_px4_bridge.py` along seams into `test_lockstep.py`,
   `test_actuator_lag.py`, `test_sensor_send.py`, `test_bridge_gating.py`.
   Mechanical; no behavior change.

**Tests.** `make test` stays green; no fake class defined more than once.

**Acceptance.** Each fake has a single definition; no test file over ~400 LOC.

**Status.** DONE. Shared fakes/helpers live in `tests/_fakes.py` (imported as
`from _fakes import ...`); the 42-test monolith is split by concern into
`tests/test_px4_bridge_sensors.py` (15), `tests/test_px4_bridge_lockstep.py`
(8), `tests/test_px4_bridge_actuator.py` (7), and
`tests/test_px4_bridge_gating.py` (12). `tests/test_px4_bridge.py` is removed.
Each fake is defined once; no split file exceeds ~400 LOC; suite count
unchanged (128 total).

### Step 4.2 - Local pre-commit gate (no CI for now)

**Problem.** Quality is gated only by ad hoc local runs. A red unit test, type
error, or lint error can land. The original plan proposed a GitHub Actions CI
gate; the project is local-only for now, so the gate lives in pre-commit
instead.

**Files.** `.pre-commit-config.yaml`.

**Outline.** Keep ruff/mypy as mirror hooks (fast, self-bootstrapping,
per-changed-file, ruff auto-fixes on commit; `rev:` kept in sync with the `dev`
extra) and add a `repo: local` `pytest` hook (`pass_filenames: false` /
`always_run: true`) that runs the whole suite through the project venv on every
commit. The utility (`pre-commit-hooks`) and `Lucas-C` crlf/tabs/chmod hooks are
unchanged. Container-backed `make test-integration-*` targets stay manual.

This is the hybrid choice: commits stay fast and self-fixing, but the mirror
hooks only see staged files, so they are not a whole-repo gate -- the
pre-existing-error class (stale lint/type errors in untouched files) can recur.
Run `uv run ruff check` / `uv run ruff format --check` / `uv run mypy src tests`
for a full sweep. If CI is added later, those three plus `pytest` port directly
to a workflow as the whole-repo gate.

**Tests.** `uv run pre-commit run --all-files` passes every hook; confirm
`pytest` fails the commit on an intentionally broken test.

**Acceptance.** A failing unit test blocks the commit; ruff/mypy block on
changed files.

**Status.** DONE. `.pre-commit-config.yaml` keeps the mirror ruff/mypy hooks and
adds the local `pytest` hook alongside the utility/crlf/tabs hooks; all pass on
`--all-files`. Documented in CLAUDE.md (Isaac Sim test workflow).

---

## Track P5 - Larger structural shifts (SOTA alignment)

Deliberate, separately reviewed initiatives. Each may spawn sub-PRs; treat the
step as the design + first slice.

### Step 5.1 - Consolidate frame math into one canonical library

**Problem.** `px4_bridge/frames.py` performs quaternion algebra directly, while
vendored `pegasus/.../rotations.py` keeps a parallel static-quaternion /
scipy-Rotation implementation. An equivalence test already exists
(`tests/test_frames_pegasus_equivalence.py`, from the Track P1 work), but the two
implementations still coexist with no single owner.

**Files.** `src/robotics_experiments/px4_bridge/frames.py`; the equivalence test.

**Outline.** Confirm `frames.py` is the canonical source for all bridge-side
conversions; ensure every bridge call path uses it (not ad hoc math). Keep the
equivalence test as the drift guard against the vendored `rotations.py`, and note
in `frames.py` that it is canonical. Do not change public exports without sign-off.

**Tests.** Extend `tests/test_frames_pegasus_equivalence.py` to cover any
conversion not yet asserted.

**Acceptance.** A test fails on any divergence beyond tolerance (noise disabled);
`frames.py` is documented as the single source.

**Status.** DONE. `frames.py` is documented as the canonical source for
bridge-owned frame math, `_PegasusSensorSuite` uses its quaternion-order helper
when adapting `VehicleState` into Pegasus `State`, and the Pegasus equivalence
test now covers quaternion order, ENU/NED position and velocity, FLU/FRD gyro,
attitude rotation convention, body velocity, GPS, magnetometer, and barometer.

### Step 5.2 - Pegasus de-vendoring strategy

**Problem.** ~2,100 LOC of Pegasus is vendored and lint-excluded. The bridge
relies on it but the version is unpinned and the project's intentional divergences
(HIL temperature fix, accumulated barometer drift, sim-time lockstep) are spread
across the tree.

**Files.** `CLAUDE.md` (new "Pegasus vendoring policy" section); the vendored tree
(slimming); `pyproject.toml` if a pin is recorded.

**Outline.**
1. Record the upstream Pegasus commit/version the vendor tree is based on.
2. Keep only the modules the bridge transitively imports (sensors, state,
   rotations, geo_mag_utils, backend) -- building on the already-done dead-module
   removal (commit `Cleanup unused pegasus modules`) -- and flag every intentional
   divergence inline so a future sync is mechanical.
3. Document the policy: when and how to re-sync, and that divergences are
   load-bearing fixes, not drift.

**Tests.** Smoke tests under
`isaac/runtime/extensions/px4_sitl_brigde/tests/` plus `tests/` unit suite still
pass after slimming.

**Acceptance.** A short vendoring-policy section in CLAUDE.md; the vendor tree is
reduced to the reachable set with divergences marked.

**Status.** DONE (documentation/policy; no runtime change). New
`isaac/runtime/extensions/px4_sitl_brigde/pegasus/PROVENANCE.md` records the
honest unpinned-snapshot provenance (upstream repo/BSD-3 license; commit
`UNKNOWN`, not guessed), the bridge-reachable module set, and a divergence index
(GPS degrees, accumulated barometer drift, `HIL_SENSOR` temperature field,
bridge-side sim-time lockstep) with each divergence's `file:line` and guarding
test. CLAUDE.md gains a "Pegasus vendoring policy" section; the three in-tree
inline divergence comments now point at PROVENANCE.md. The reachable-set slimming
was already complete (dead modules removed in `Cleanup unused pegasus modules --
Step 9`); `px4_launch_tool.py` + the autolaunch branches are dead at runtime
(`px4_autolaunch=False`) and were deliberately kept and documented rather than
removed, to minimize divergence.

### Step 5.3 - Isaac Lab task/env alignment

**Problem.** `src/robotics_experiments/workflows/isaac_lab/` and
`isaac/runtime/lab/` (only `drone_cfg.py` + `README.md`) are stubs. Future
RL/perception work has no defined home and would otherwise accrete under
`standalone/`.

**Files.** `isaac/runtime/lab/`, `src/robotics_experiments/workflows/isaac_lab/`;
a short design note (can live in `isaac/runtime/lab/README.md`).

**Outline.** Define the target structure following current Isaac Lab conventions
(an env class in the `DirectRLEnv` / manager-based style plus a task/env config),
and land one minimal compiling skeleton task. Keep it lazy-import clean so the
core package still imports without Isaac.

**Tests.** A skeleton import/instantiation test guarded to skip when Isaac Lab is
absent.

**Acceptance.** A documented env/task layout and one minimal skeleton that
compiles/imports.

**Status.** DONE. Importable Isaac Lab task code now lives under
`src/robotics_experiments/workflows/isaac_lab/` with package initializers kept
lazy-import clean. Added guarded Gymnasium registration for
`Isaac-Drone-Hover-Direct-v0` and a minimal DirectRLEnv drone-hover scaffold
(`DroneHoverEnvCfg` / `DroneHoverEnv`) that defines the action/observation
surfaces, timeout-only termination, and no-op action application without
duplicating PX4/thrust policy. The scaffold's tunable task defaults live in the
Hydra `isaac_lab` group. `isaac/runtime/lab/README.md` documents the task/env
layout, and AGENTS.md records the convention. Host tests cover import behavior
without Isaac Lab and registration metadata; the task import/config test is
skipped when Isaac Lab/Gymnasium are absent.

### Step 5.4 - Enforce package layering

**Problem.** The intended dependency direction is
`config -> domain (drone / environment / simulation / px4_bridge) -> Isaac
extensions`, with Isaac/carb/pxr imports lazy at function scope (as `mavlink.py`
does). But `drone/spawn.py`, `environment/stage.py` (`stage.py:5-6`), and
`environment/step.py` import `carb` / `pxr` / `isaacsim` at module top level,
coupling import-time to a simulator and breaking `import robotics_experiments`
outside Isaac.

**Files.** `src/robotics_experiments/drone/spawn.py`,
`environment/stage.py`, `environment/step.py`; a new import-boundary test.

**Outline.**
1. Move the simulator imports to function scope (or document the exception where a
   module is legitimately Isaac-only and is never imported by the core path).
2. Add a lightweight test asserting `import robotics_experiments` (and the core
   subpackages) succeeds with Isaac absent, guarding the layering.

**Tests.** New boundary test; existing per-module tests still pass.

**Acceptance.** `python -c "import robotics_experiments"` works outside Isaac; a
test guards the boundary.

**Status.** OPEN.

---

## Suggested execution order

1. **P4.1** first -- shared fixtures make every later refactor cheaper to verify.
2. **P2** (2.1 pure-move, then 2.2 / 2.3) behind the now-complete P1 lockstep
   tests.
3. **P3** in any order; 3.1-3.4 are code, 3.5-3.6 are build-only and can run in
   parallel. 3.3 depends on 2.1.
4. **P4.2** any time once the unit suite is the gate of record.
5. **P5** as deliberate, separately reviewed initiatives: 5.1 extends the existing
   equivalence guard, 5.2 builds on the committed Pegasus dead-module cleanup, 5.4
   pairs naturally with P2, 5.3 is independent.

Every step is independently testable with `uv run pytest tests/...`. Build steps
(P3.5 / P3.6) additionally gate on a clean `make build` and a headless smoke run.
Track P1 needs no action; it is recorded for completeness and as the regression
context for P5.2.

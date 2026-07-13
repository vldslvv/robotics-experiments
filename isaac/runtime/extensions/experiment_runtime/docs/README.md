# ExperimentRuntime

Live Kit shell for `make run-all` / `make run-isaac`. It owns no flight logic: it
creates frozen `robotics_experiments.runtime.FleetState` and wires Kit events to
its stateless lifecycle functions, explicitly rebinding every returned state.

- `on_startup`: create fleet state from the active experiment config and
  subscribe to Kit events. An app-update callback opens the configured `scene`
  once `IApp.is_app_ready()` is true (the runtime owns scene-open, so the
  live boot does not use Kit's `--exec open_stage.py` or a separate
  `sim_settings` extension). The stage is adopted and sim settings applied on the
  `OPENED` event.
- Timeline `PLAY`: spawn the fleet, start the PX4 lockstep bridge group, and -- if
  any drone's resolved `routine.steps` is non-empty -- build one frozen
  `RoutineDriverState` per flying drone and advance them with `tick_drivers`
  from a separate app-update subscription.
- `STOP` / `PAUSE` / `on_shutdown`: stop the control tick and tear down.

Always forgiving: a routine failure, PX4 failsafe, or phase timeout is logged and
that drone idles (manual control remains). The strict counterpart is the
standalone scenario runner under `--strict`.

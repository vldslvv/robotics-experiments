# ExperimentRuntime

Live Kit shell for `make run-all` / `make run-isaac`. It owns no flight logic: it
builds the host-agnostic
`robotics_experiments.runtime.FleetOrchestrator` and wires Kit events to its
phases.

- `on_startup`: build the orchestrator from the active experiment config and open
  the configured `scene` (the orchestrator owns scene-open, so the live boot no
  longer uses Kit's `--exec open_stage.py` or a separate `sim_settings`
  extension). The stage is adopted and sim settings applied on the `OPENED` event.
- Timeline `PLAY`: spawn the fleet, start the PX4 lockstep bridge group, and -- if
  any drone's resolved `routine.steps` is non-empty -- build one `RoutineDriver`
  per flying drone and tick them from a separate app-update subscription.
- `STOP` / `PAUSE` / `on_shutdown`: stop the control tick and tear down.

Always forgiving: a routine failure, PX4 failsafe, or phase timeout is logged and
that drone idles (manual control remains). The strict counterpart is the
standalone scenario runner under `--strict`.

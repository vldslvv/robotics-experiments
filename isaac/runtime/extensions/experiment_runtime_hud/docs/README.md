# Experiment Runtime HUD

`experiment_runtime_hud` is a pure-observer Isaac Sim extension for manual GUI
runs. It opens a dockable `Experiment Runtime HUD` window containing one card
per flying vehicle. The card identifies the current command owner and displays
navigation, handoff, policy, transmitted-setpoint, actual-motion, completion,
and recovery telemetry.

The extension only calls `experiment_runtime.get_instance().read_status()` from
Kit's post-update stream. It does not read Isaac sensors, evaluate actors, send
MAVLink commands, subscribe to physics, create USD prims, or register callbacks
with the routine driver. Hiding or disabling it therefore cannot change flight
behavior. The runtime produces the same immutable status boundary whether or
not the HUD is loaded.

The window is enabled by the manual `runapp_custom.sh` launcher, opens by
default, and can be reopened from
`Window > Robotics Experiments > Runtime HUD`. It keeps the current and previous
semantic control states by default. Change the presentation-only Kit setting
`/exts/experiment_runtime_hud/transition_history_capacity` to retain a different
number of transitions.

For `press-approach`, the current artifact is explicitly labelled as an
approach-only policy. Contact or switch activation is not expected during that
stage.

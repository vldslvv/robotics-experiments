# PX4 Docker Inputs

Put PX4-specific Docker setup inputs here if the root `Dockerfile` needs separate support files later.

## Multi-instance launch (`PX4_INSTANCE`)

Both PX4 launchers read `PX4_INSTANCE` (default `0`). **Every** instance
(including `0`) is launched uniformly by `launch_px4_iris`: the prebuilt PX4
binary run directly with `-i ${PX4_INSTANCE}` and an interactive `pxh>` console
(no `-d`/daemon flag, so manual `commander arm` / `commander takeoff` work in
every per-vehicle terminal), from its own per-instance rootfs subdir
(`build/px4_sitl_default/rootfs/${PX4_INSTANCE}`). The airframe is selected by
`PX4_SIM_MODEL` / `PX4_SYS_AUTOSTART` (defaulting to the `none_iris` quad when
unset, e.g. the fresh/manual launch path): the host-side launch scripts derive
the per-vehicle pair from the drone USD asset via
`config.get_enabled_px4_launch_configs()` (`none_iris`/`10016` for the quad,
`none_hexa_x`/`6001` for the generic hexarotor-x geometry) and thread it into
each container, so a hexacopter asset no longer flies the quad airframe. The
`none_hexa_x` airframe is a custom SITL airframe shipped in this repo
(`airframes/6001_none_hexa_x`) and baked into the image by the Dockerfile's px4
stage: upstream PX4 v1.14.3 ships `6001_hexa_x` only in the hardware/NuttX
airframe set, so the SITL build has no stock none-family hexarotor. Like every
non-gz/sih/jmavsim airframe it falls through to `simulator_mavlink` in
`px4-rc.simulator`, so the bridge connects identically to the quad. `-i` shifts
every port by the index, so instance
`0` keeps simulator TCP `4560` / GCS UDP `14550` (the single-vehicle baseline)
and instance `i` listens on simulator TCP `4560 + i` / GCS UDP `14550 + i`. There
is intentionally no `instance == 0` special case in the launch, reset, or mount
logic -- a divergent instance-0 launch (`make px4_sitl_default none_iris` with a
console vs. additional instances started in daemon mode) is exactly how the extra
instances silently lost their `pxh>` console. PX4 v1.14.3 shifts the GCS local
UDP port for `-i` but leaves its remote GCS port at `14550`, so the launchers
patch `px4-rc.mavlink` at container startup to set the GCS remote port to
`14550 + PX4_INSTANCE`. `reset_px4_state.sh` clears `dataman` /
`parameters*.bson` / `eeprom` under the per-instance rootfs so instances never
share non-volatile state.

The two workflows differ only in Docker networking:

- Persistent `make run-px4` / `run-all`: one interactive tmux-backed PX4
  container per PX4-enabled vehicle (via `start_px4_tmux.sh` ->
  `run_px4_iris_fresh.sh`). **Every** instance runs on host networking
  (`--network=host`, the same as the Isaac container), so they all share the host
  namespace and `-i` keeps the ports distinct. This is the standard
  multiple-SITL-on-one-host pattern.
- Integration harness: the same instance count via the non-interactive
  `run_px4_iris_ci.sh` launcher. There instance `0` owns a private bridge
  namespace and every other instance plus the ephemeral Isaac container joins it
  with `--network container:<primary>` (see
  `make/scripts/run_px4_integration_test.sh`). That shared-namespace model is
  specific to the integration harness; do not apply it to the host-networked
  persistent path.

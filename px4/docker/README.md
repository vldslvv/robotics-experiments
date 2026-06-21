# PX4 Docker Inputs

Put PX4-specific Docker setup inputs here if the root `Dockerfile` needs separate support files later.

## Multi-instance launch (`PX4_INSTANCE`)

Both PX4 launchers read `PX4_INSTANCE` (default `0`). Instance `0` is the
single-vehicle baseline (`make px4_sitl_default none_iris`, simulator TCP port
`4560`, GCS UDP `14550`). Each additional instance `i` launches the prebuilt PX4
binary directly with `-i ${PX4_INSTANCE}`, which shifts every port by the index
(simulator TCP `4560 + i`, GCS UDP `14550 + i`) and runs from a per-instance
rootfs subdir. PX4 v1.14.3 shifts the GCS local UDP port for `-i` but leaves its
remote GCS port at `14550`, so the launchers patch `px4-rc.mavlink` at container
startup to set the GCS remote port to `14550 + PX4_INSTANCE`.
`reset_px4_state.sh` clears `dataman` / `parameters*.bson` / `eeprom` under the
per-instance rootfs so instances never share non-volatile state.

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

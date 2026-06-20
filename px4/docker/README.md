# PX4 Docker Inputs

Put PX4-specific Docker setup inputs here if the root `Dockerfile` needs separate support files later.

## Multi-instance launch (`PX4_INSTANCE`)

`run_px4_iris_ci.sh` reads `PX4_INSTANCE` (default `0`). Instance `0` is the
single-vehicle baseline (`make px4_sitl_default none_iris`, simulator TCP port
`4560`, GCS UDP `14550`). Each additional instance `i` shares instance 0's
network namespace, so it launches the prebuilt PX4 binary directly with
`-i ${PX4_INSTANCE}`, which shifts every port by the index (simulator TCP
`4560 + i`, GCS UDP `14550 + i`) and runs from a per-instance rootfs subdir.
PX4 v1.14.3 shifts the GCS local UDP port for `-i` but leaves its remote GCS
port at `14550`, so `run_px4_iris_ci.sh` patches `px4-rc.mavlink` at container
startup to set the GCS remote port to `14550 + PX4_INSTANCE`.
`reset_px4_state.sh` clears `dataman` / `parameters*.bson` / `eeprom` under the
per-instance rootfs so instances never share non-volatile state. The integration
harness (`make/run_px4_integration_test.sh`) starts `PX4_INSTANCE_COUNT`
containers in one shared namespace; the dual-PX4 test target sets it to `2`.

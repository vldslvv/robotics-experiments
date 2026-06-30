#!/usr/bin/env bash
# shellcheck disable=SC2016

px4_dir="${PX4_DIR:-/opt/PX4-Autopilot}"

px4_iris_configure_instance() {
  instance="${PX4_INSTANCE:-0}"
  export PX4_INSTANCE="${instance}"

  if ! [[ "${instance}" =~ ^[0-9]+$ ]]; then
    echo "PX4_INSTANCE must be a non-negative integer, got ${instance}" >&2
    exit 2
  fi
}

configure_gcs_remote_port() {
  local mavlink_rc
  local mavlink_rc_paths=(
    "${px4_dir}/ROMFS/px4fmu_common/init.d-posix/px4-rc.mavlink"
    "${px4_dir}/build/px4_sitl_default/etc/init.d-posix/px4-rc.mavlink"
  )

  for mavlink_rc in "${mavlink_rc_paths[@]}"; do
    if [ ! -f "${mavlink_rc}" ]; then
      continue
    fi

    if grep -Fq 'udp_gcs_port_remote=' "${mavlink_rc}"; then
      continue
    fi

    if ! grep -Fq 'udp_gcs_port_local=$((18570+px4_instance))' "${mavlink_rc}"; then
      echo "ERROR: PX4 mavlink startup script has unexpected GCS port layout: ${mavlink_rc}" >&2
      exit 1
    fi
    if ! grep -Fq 'mavlink start -x -u $udp_gcs_port_local -r 4000000 -f' "${mavlink_rc}"; then
      echo "ERROR: PX4 mavlink startup script has unexpected GCS start command: ${mavlink_rc}" >&2
      exit 1
    fi

    sed -i \
      '/udp_gcs_port_local=/a udp_gcs_port_remote=$((14550+px4_instance))' \
      "${mavlink_rc}"
    sed -i \
      's|mavlink start -x -u \$udp_gcs_port_local -r 4000000 -f|mavlink start -x -u $udp_gcs_port_local -r 4000000 -f -o $udp_gcs_port_remote|' \
      "${mavlink_rc}"
  done
}

launch_px4_iris() {
  # Every instance (including 0) launches the prebuilt binary the same way: an
  # interactive pxh console (no -d, so manual `commander arm` / `commander
  # takeoff` work), -i ${instance} for port-shifting (instance 0 keeps simulator
  # TCP 4560 / GCS UDP 14550), from its own per-instance rootfs subdir so the
  # dataman / parameters / eeprom / log state never collides between instances
  # sharing the host network namespace. The airframe is selected by the caller
  # via PX4_SIM_MODEL / PX4_SYS_AUTOSTART (the host-side launch scripts derive
  # the per-vehicle pair from the drone asset: none_quad_x/4001 quad,
  # none_hexa_x/6001 hex), defaulting to the repo's Generic Quad X when unset by
  # a manual/fresh launch path. There is intentionally no
  # `instance == 0` special case here: a divergent instance-0 launch is exactly
  # how additional instances silently lost their pxh console.
  build_path="${px4_dir}/build/px4_sitl_default"
  base_rootfs="${PX4_ROOTFS:-${build_path}/rootfs}"
  instance_dir="${base_rootfs}/${instance}"
  mkdir -p "${instance_dir}"
  cd "${instance_dir}" || exit
  export PX4_SIM_MODEL="${PX4_SIM_MODEL:-none_quad_x}"
  export PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4001}"
  "${build_path}/bin/px4" -i "${instance}" \
    "${build_path}/etc" \
    -s "${px4_dir}/ROMFS/px4fmu_common/init.d-posix/rcS" \
    -t "${px4_dir}/test_data"
}

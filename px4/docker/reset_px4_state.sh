#!/usr/bin/env bash

reset_px4_state() {
  local px4_dir="${PX4_DIR:-/opt/PX4-Autopilot}"
  local base_rootfs="${PX4_ROOTFS:-${px4_dir}/build/px4_sitl_default/rootfs}"
  local instance="${PX4_INSTANCE:-0}"
  local fresh_state="${PX4_FRESH_STATE:-1}"

  if [[ "${fresh_state}" == "0" ]]; then
    return
  fi

  # Every instance (including 0) runs from its own per-instance rootfs subdir so
  # its non-volatile state (dataman / parameters / eeprom) never collides with
  # the other instances sharing the network namespace. This matches the uniform
  # per-instance launch in px4_iris_common.sh::launch_px4_iris.
  local rootfs="${base_rootfs}/${instance}"

  mkdir -p "${rootfs}"

  # Keep rootfs/log mounted and persistent for post-run debugging, but remove
  # simulated EEPROM, dataman, and parameter snapshots that PX4 reuses as
  # non-volatile state across SITL launches.
  rm -f \
    "${rootfs}/dataman" \
    "${rootfs}/parameters.bson" \
    "${rootfs}/parameters_backup.bson"

  if [[ -d "${rootfs}/eeprom" ]]; then
    find "${rootfs}/eeprom" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  fi
}

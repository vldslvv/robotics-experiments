#!/usr/bin/env bash

reset_px4_state() {
  local px4_dir="${PX4_DIR:-/opt/PX4-Autopilot}"
  local rootfs="${PX4_ROOTFS:-${px4_dir}/build/px4_sitl_default/rootfs}"
  local fresh_state="${PX4_FRESH_STATE:-1}"

  if [[ "${fresh_state}" == "0" ]]; then
    return
  fi

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

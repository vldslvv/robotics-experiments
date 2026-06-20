#!/usr/bin/env bash
# shellcheck disable=SC2016
set -euo pipefail

px4_dir="${PX4_DIR:-/opt/PX4-Autopilot}"
reset_helper="${PX4_RESET_HELPER:-/usr/local/lib/px4/reset_px4_state.sh}"
# PX4 instance index. 0 is the single-vehicle baseline. Each additional instance
# (i>0) shares instance 0's network namespace, so PX4 adds the index to every
# port: simulator TCP 4560 + i, GCS UDP 14550 + i, offboard 14540 + i.
instance="${PX4_INSTANCE:-0}"
export PX4_INSTANCE="${instance}"

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

# shellcheck source-path=SCRIPTDIR
# shellcheck source=reset_px4_state.sh
source "${reset_helper}"
reset_px4_state

cd "${px4_dir}"
set -o pipefail
configure_gcs_remote_port

px4_filter() {
  sed -u -E 's/\r/\n/g' | sed -u -E 's/^pxh>[[:space:]]*//; /^$/d'
}

if [[ "${instance}" -eq 0 ]]; then
  # Instance 0: unchanged baseline. Builds (cached) and launches none_iris on the
  # default simulator TCP port 4560 / GCS UDP 14550.
  make px4_sitl_default none_iris 2>&1 | px4_filter
else
  # Additional instances share instance 0's network namespace, so each must run
  # with a distinct PX4 instance index. The firmware is prebuilt in the image
  # (Dockerfile: make px4_sitl_default), so launch the binary directly with -i,
  # which shifts the simulator/GCS ports by the index and uses a per-instance
  # rootfs working dir.
  #
  # NOTE (Step 0 probe): confirm against the live image that `-i ${instance}`
  # shifts the simulator TCP port to 4560 + instance, emits the
  # "Waiting for simulator to accept connection on TCP port 456${instance}"
  # readiness line, and that these rootfs/etc/rcS paths match this PX4 tag before
  # relying on this branch in CI.
  build_path="${px4_dir}/build/px4_sitl_default"
  base_rootfs="${PX4_ROOTFS:-${build_path}/rootfs}"
  instance_dir="${base_rootfs}/${instance}"
  mkdir -p "${instance_dir}"
  cd "${instance_dir}"
  export PX4_SIM_MODEL="none_iris"
  "${build_path}/bin/px4" -i "${instance}" \
    -d "${build_path}/etc" \
    -s "${px4_dir}/ROMFS/px4fmu_common/init.d-posix/rcS" \
    -t "${px4_dir}/test_data" \
    2>&1 | px4_filter
fi

#!/usr/bin/env bash
set -euo pipefail

session="${PX4_TMUX_SESSION:-px4}"
px4_dir="${PX4_DIR:-/opt/PX4-Autopilot}"

if tmux has-session -t "${session}" 2>/dev/null; then
  exec tmux attach-session -t "${session}"
fi

tmux new-session -d -s "${session}" -c "${px4_dir}" \
  "/bin/bash -lc 'cd \"${PX4_DIR:-/opt/PX4-Autopilot}\" && exec make px4_sitl_default none_iris'"
tmux split-window -h -t "${session}:0" -c "${px4_dir}" "/bin/bash"
tmux select-pane -t "${session}:0.0"
exec tmux attach-session -t "${session}"

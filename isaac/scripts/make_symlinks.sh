#!/usr/bin/env bash

# Basic idea is to populate symlinks so that editor's LSP
# can discover, as isaac exts core have issues with discoverability
# Example: populate isaacsim.core.api
# module_name="isaacsim.core.api"
# mkdir -p $SCRIPT_DIR/symlinks/exts/isaacsim/core
# Populate __init__.py at every level, if not already created
# [ ! -e "$SYMLINKS/isaacsim/__init__.py" ] && touch "$SYMLINKS/isaacsim/__init__.py"
# [ ! -e "$SYMLINKS/isaacsim/core/__init__.py" ] && touch "$SYMLINKS/isaacsim/core/__init__.py"
# Now, create a symlink for api, if doens't exist already

# TODO: create symlinks for /isaac-sim/kit/kernel/py/omni/ext/
# and /isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy/
# Consider also /isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy/rclpy/

set -euo pipefail

# -------- CONFIG --------
EXTS_DIR="/isaac-sim/exts"
SYMLINKS="/isaac-sim/symlinks/exts"

# ------------------------

mkdir -p "$SYMLINKS"

log() {
  echo "[linker] $*"
}

ensure_init_py() {
  local dir="$1"
  if [[ ! -e "$dir/__init__.py" ]]; then
    touch "$dir/__init__.py"
  fi
}

link_python_module() {
  local module_name="$1"

  # Convert dotted name to path
  local module_path="${module_name//./\/}"

  local src="$EXTS_DIR/$module_name/$module_path"
  local dst="$SYMLINKS/$module_path"

  if [[ ! -d "$src" ]]; then
    log "Skipping (source not found): $src"
    return
  fi

  log "Linking $module_name"

  # Create parent dirs for destination
  mkdir -p "$(dirname "$dst")"

  # Populate __init__.py at every intermediate level
  local current="$SYMLINKS"
  IFS='/' read -ra parts <<< "$module_path"
  for part in "${parts[@]:0:${#parts[@]}-1}"; do
    current="$current/$part"
    mkdir -p "$current"
    ensure_init_py "$current"
  done

  # Create / update symlink
  ln -sfn "$src" "$dst"
}

# -------- MAIN --------

log "Scanning extensions in $EXTS_DIR"

for ext in "$EXTS_DIR"/*; do
  [[ -d "$ext" ]] || continue
  module_name="$(basename "$ext")"

  # Only handle isaacsim.* and omni.*
  if [[ "$module_name" == isaacsim.* || "$module_name" == omni.* ]]; then
    link_python_module "$module_name"
  fi
done

log "Done."

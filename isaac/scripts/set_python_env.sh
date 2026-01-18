#!/bin/bash
# This script should be sourced prior to launching the editor
# To be able to develop with LSP reading symbols from Isaac Sim's Python environment
# export PYTHONPATH="/isaac-sim/exts/isaacsim.simulation_app/isaacsim::/isaac-sim:/isaac-sim/kit/python/lib/python3.11:/isaac-sim/kit/python/lib/python3.11/site-packages:/isaac-sim/python_packages:/isaac-sim/exts/isaacsim.simulation_app:/isaac-sim/extsDeprecated/omni.isaac.kit:/isaac-sim/kit/kernel/py:/isaac-sim/kit/plugins/bindings-python:/isaac-sim/exts/isaacsim.robot_motion.lula/pip_prebundle:/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle:/isaac-sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle:/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle:/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle:/isaac-sim/exts/omni.pip.compute/pip_prebundle:/isaac-sim/exts/omni.pip.cloud/pip_prebundle:/isaac-sim/kit/python/lib/python311.zip:/isaac-sim/kit/python/lib/python3.11/lib-dynload"
# export PYTHONPATH="/isaac-sim/exts/isaacsim.simulation_app/isaacsim:/isaac-sim/exts/isaacsim.core.api/isaacsim/core/api::/isaac-sim:/isaac-sim/kit/python/lib/python3.11:/isaac-sim/kit/python/lib/python3.11/site-packages:/isaac-sim/python_packages:/isaac-sim/exts/isaacsim.simulation_app:/isaac-sim/extsDeprecated/omni.isaac.kit:/isaac-sim/kit/kernel/py:/isaac-sim/kit/plugins/bindings-python:/isaac-sim/exts/isaacsim.robot_motion.lula/pip_prebundle:/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle:/isaac-sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle:/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle:/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle:/isaac-sim/exts/omni.pip.compute/pip_prebundle:/isaac-sim/exts/omni.pip.cloud/pip_prebundle:/isaac-sim/kit/python/lib/python311.zip:/isaac-sim/kit/python/lib/python3.11/lib-dynload"
# export PATH="/isaac-sim/kit/python/bin:$PATH"
# alias python3="/isaac-sim/kit/python/bin/python3"
# alias python="/isaac-sim/kit/python/bin/python3"

SCRIPT_DIR="/isaac-sim"
# symlinks dir is here as well
# TODO: append to $PYTHONPATH:
SYMLINKS="$SCRIPT_DIR/symlinks/exts"
export PYTHONPATH=$SCRIPT_DIR/kit/python/lib/python3.11:$SCRIPT_DIR/kit/python/lib/python3.11/site-packages:$SCRIPT_DIR/python_packages:$SCRIPT_DIR/exts/isaacsim.simulation_app:$SCRIPT_DIR/extsDeprecated/omni.isaac.kit:$SCRIPT_DIR/kit/kernel/py:$SCRIPT_DIR/kit/plugins/bindings-python:$SCRIPT_DIR/exts/isaacsim.robot_motion.lula/pip_prebundle:$SCRIPT_DIR/exts/isaacsim.asset.exporter.urdf/pip_prebundle:$SCRIPT_DIR/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle:$SCRIPT_DIR/exts/omni.isaac.core_archive/pip_prebundle:$SCRIPT_DIR/exts/omni.isaac.ml_archive/pip_prebundle:$SCRIPT_DIR/exts/omni.pip.compute/pip_prebundle:$SCRIPT_DIR/exts/omni.pip.cloud/pip_prebundle:$SYMLINKS
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SCRIPT_DIR/.:$SCRIPT_DIR/exts/isaacsim.robot.schema/plugins/lib:$SCRIPT_DIR/exts/isaacsim.robot_motion.lula/pip_prebundle:$SCRIPT_DIR/exts/isaacsim.asset.exporter.urdf/pip_prebundle:$SCRIPT_DIR/kit:$SCRIPT_DIR/kit/kernel/plugins:$SCRIPT_DIR/kit/libs/iray:$SCRIPT_DIR/kit/plugins:$SCRIPT_DIR/kit/plugins/bindings-python:$SCRIPT_DIR/kit/plugins/carb_gfx:$SCRIPT_DIR/kit/plugins/rtx:$SCRIPT_DIR/kit/plugins/gpu.foundation
export PATH="/isaac-sim/kit/python/bin:$PATH"
alias python3="/isaac-sim/kit/python/bin/python3"
alias python="/isaac-sim/kit/python/bin/python3"

# Example: populate isaacsim.core.api
# module_name="isaacsim.core.api"
# mkdir -p $SCRIPT_DIR/symlinks/exts/isaacsim/core
# # Populate __init__.py at every level, if not already created
# [ ! -e "$SYMLINKS/isaacsim/__init__.py" ] && touch "$SYMLINKS/isaacsim/__init__.py"
# [ ! -e "$SYMLINKS/isaacsim/core/__init__.py" ] && touch "$SYMLINKS/isaacsim/core/__init__.py"
# # Now, create a symlink for api, if doens't exist already

# ln -sfn $SCRIPT_DIR/exts/isaacsim.core.api/isaacsim/core/api $SYMLINKS/isaacsim/core

# module_name_api="isaacsim.core.api"
# module_name_prims="isaacsim.core.prims"
#
# link_python_module() {
#   local module_name="$1"
#   local module_path="${module_name//./\/}"
#   mkdir -p "$SCRIPT_DIR/symlinks/exts/isaacsim/core"
#   mkdir -p "$SYMLINKS/$(dirname "$module_path")"
#   # Populate __init__.py at every level, if not already created
#   [ ! -e "$SYMLINKS/isaacsim/__init__.py" ] && touch "$SYMLINKS/isaacsim/__init__.py"
#   [ ! -e "$SYMLINKS/isaacsim/core/__init__.py" ] && touch "$SYMLINKS/isaacsim/core/__init__.py"
#   # Create symlink for the module
#   ln -sfn "$SCRIPT_DIR/exts/${module_name}/${module_path}" "$SYMLINKS/${module_path}"
# }
#
# # Link both api and prims modules
# link_python_module "$module_name_api"
# link_python_module "$module_name_prims"

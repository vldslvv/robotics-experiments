#!/bin/bash
# This script should be sourced prior to launching the editor
# To be able to develop with LSP reading symbols from Isaac Sim's Python environment
# Before that, $SYMLINKS directory has to be populated using make_symlinks.sh script

SCRIPT_DIR="/isaac-sim"
SYMLINKS="$SCRIPT_DIR/symlinks/exts"
PROJECT_PYTHONPATH="${PYTHONPATH:-}"
unset PYTHONPATH
# shellcheck source=/dev/null
source "$SCRIPT_DIR/setup_python_env.sh"
if [[ -n "$PROJECT_PYTHONPATH" ]]; then
  export PYTHONPATH="$PROJECT_PYTHONPATH:$PYTHONPATH:$SYMLINKS"
else
  export PYTHONPATH="$PYTHONPATH:$SYMLINKS"
fi
unset PROJECT_PYTHONPATH

alias python3="/isaac-sim/kit/python/bin/python3"
alias python="/isaac-sim/kit/python/bin/python3"

#!/bin/bash
tmux new-session -d -s isaac "cd /isaac-sim && ./runapp.sh"

# second pane for your terminal
tmux split-window -h "bash"

tmux attach-session -t isaac

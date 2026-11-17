#!/usr/bin/env bash
# Run a command inside the RL virtualenv (~/rl), from the rl/ folder:
#   bash vorpal-mujoco/rl/wslrun.sh python train.py ...
source ~/rl/bin/activate
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
exec "$@"

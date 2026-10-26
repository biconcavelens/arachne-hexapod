#!/usr/bin/env bash
# Run a command inside the WSL RL virtualenv, from the rl/ folder:
#   wsl -d Ubuntu -- bash /mnt/c/projects/arachne/vorpal-mujoco/rl/wslrun.sh python train.py ...
source ~/rl/bin/activate
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
exec "$@"

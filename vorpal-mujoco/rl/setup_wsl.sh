#!/usr/bin/env bash
# One-time setup of the RL environment inside WSL2 Ubuntu (GPU JAX + MJX + Playground).
# Run from Windows:  wsl -d Ubuntu -- bash /mnt/c/projects/arachne/vorpal-mujoco/rl/setup_wsl.sh
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~
[ -d rl ] || uv venv rl --python 3.12
source ~/rl/bin/activate
uv pip install "jax[cuda12]" mujoco mujoco-mjx playground matplotlib pillow
python - << 'EOF'
import jax, mujoco
print("jax", jax.__version__, jax.devices())
print("mujoco", mujoco.__version__)
import mujoco_playground
print("playground ok")
EOF

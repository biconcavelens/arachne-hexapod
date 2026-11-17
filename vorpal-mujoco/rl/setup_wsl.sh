#!/usr/bin/env bash
# One-time setup of the RL environment (GPU JAX + MJX + Playground) in ~/rl.
# Runs on Linux with an NVIDIA GPU (originally WSL2 Ubuntu; now native Ubuntu). Needs uv.
#   bash vorpal-mujoco/rl/setup_wsl.sh
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

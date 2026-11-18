#!/usr/bin/env bash
# One-time setup of the RL environment (GPU JAX + MJX + Playground) in ~/rl.
# Runs on Linux with an NVIDIA GPU (originally WSL2 Ubuntu; now native Ubuntu). Needs uv.
#   bash vorpal-mujoco/rl/setup_wsl.sh
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~
[ -d rl ] || uv venv rl --python 3.12
source ~/rl/bin/activate
# Pinned 2026-11-17: Playground 1.0 / MuJoCo 4 / JAX 0.12 changed the APIs this code uses.
uv pip install "jax[cuda12]==0.11.*" "mujoco==3.16.*" "mujoco-mjx==3.16.*" "playground<1.0" \
  ml_collections matplotlib pillow
python - << 'EOF'
import jax, mujoco
print("jax", jax.__version__, jax.devices())
print("mujoco", mujoco.__version__)
import mujoco_playground
print("playground ok")
EOF

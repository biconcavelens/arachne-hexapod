#!/usr/bin/env bash
# One-time setup on the lab machine (Ubuntu + NVIDIA GPU). Creates the Python environment in ~/rl.
#   bash vorpal-mujoco/rl/lab/setup.sh
set -e
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cd ~
[ -d rl ] || uv venv rl --python 3.12
source ~/rl/bin/activate
# Same versions as the laptop (pinned 2026-11-17; newer Playground/MuJoCo/JAX changed the APIs used here).
uv pip install "jax[cuda12]==0.11.*" "mujoco==3.16.*" "mujoco-mjx==3.16.*" "playground<1.0" \
  ml_collections matplotlib pillow imageio-ffmpeg
python - << 'EOF'
import jax, mujoco, mujoco_playground
print("jax", jax.__version__, jax.devices())
print("mujoco", mujoco.__version__)
assert jax.devices()[0].platform == "gpu", "JAX does not see the GPU"
print("setup ok")
EOF

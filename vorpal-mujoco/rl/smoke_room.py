"""Smoke test for room_env: obs sizes, a camera image, a top-down layout, faults, throughput.

    bash wslrun.sh python smoke_room.py            # writes media/room_smoke_*.png
"""
import os
import time

import jax
import jax.numpy as jp
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import room_env as E
import room_scene as RS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
os.makedirs(OUT, exist_ok=True)

for s in ("imu", "cam", "both"):
    env = E.FaultRoom(config_overrides={"sensors": s})
    print(s, env.observation_size)

env = E.FaultRoom()
reset, step = jax.jit(env.reset), jax.jit(env.step)

# camera images and layouts for 4 episodes
fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
for e in range(4):
    st = reset(jax.random.PRNGKey(e))
    for _ in range(10):
        st = step(st, jp.zeros(12))
    img = np.asarray(st.info["cam"][0]).reshape(RS.CAM_H, RS.CAM_W)
    axes[0, e].imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, e].set_title(f"episode {e}: body camera 24x16", fontsize=8)
    axes[0, e].axis("off")
    ax = axes[1, e]
    pts, g = np.asarray(st.info["path"]), int(st.info["goal_idx"])
    ax.plot(pts[:g + 1, 0], pts[:g + 1, 1], "b-", lw=1.5)
    ax.plot(*pts[g], "go")
    xy, pres = np.asarray(st.info["obj_xy"]), np.asarray(st.info["obj_present"])
    for (k, _, name), p, on, f in zip(RS.OBJECTS, xy, pres, np.asarray(env._obj_foot)):
        if on:
            ax.add_patch(plt.Circle(p, max(f, 0.05), color="tab:orange" if f else "tab:purple", alpha=0.5))
            ax.text(p[0], p[1], k, fontsize=5, ha="center")
    q = np.asarray(st.data.qpos)
    ax.plot(q[0], q[1], "k^")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "room_smoke_layout.png"), dpi=120)
print("kinds", np.asarray(st.info["fault_kind"]), "onset", np.asarray(st.info["fault_onset"]))

# throughput, batched
N = 4096
keys = jax.random.split(jax.random.PRNGKey(0), N)
vr, vs = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
st = vr(keys)
a = jp.zeros((N, 12))
st = vs(st, a); jax.block_until_ready(st.obs["state"])
t0 = time.time()
for _ in range(20):
    st = vs(st, a)
jax.block_until_ready(st.obs["state"])
print(f"{N * 20 / (time.time() - t0):,.0f} env-steps/s (sensors=both, {N} envs); done frac",
      float(st.done.mean()), "bumped", float(st.metrics["bumped"].mean()))

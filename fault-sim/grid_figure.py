"""
Figure of what Stage A hands the diagnoser: the body camera while walking, its optical flow
(same DIS settings and 160x120 resolution as Stage A), and the 4x4 grid of mean flow that becomes
32 of the 38 camera features. Light: a few seconds of simulation and about 30 renders.

    python grid_figure.py [--seed 7]        # -> runs/report/flow_grid.png
"""
import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

import build_scene
import config as C
import simulate
from egomotion import FlowEstimator, PlaneEgomotion, camera_axes_in_body, grid_summary, intrinsics
from gaits_lib import Gait

DISP = 4                                  # display frames are 4x the model's 160x120
OUT = os.path.join(C.HERE, "runs/report/flow_grid.png")


def record(seed, t0=2.0, t1=3.5):
    frames = []                           # (t, gray 160x120 as Stage A sees it, rgb for display)
    ren = {}

    def on_step(m, d, t):
        if not t0 <= t <= t1:
            return
        if not ren:
            ren["s"] = mujoco.Renderer(m, C.IMG_H, C.IMG_W)
            ren["d"] = mujoco.Renderer(m, C.IMG_H * DISP, C.IMG_W * DISP)
        ren["s"].update_scene(d, camera="ego")
        ren["d"].update_scene(d, camera="ego")
        frames.append((t, ren["s"].render().mean(axis=2).astype(np.uint8), ren["d"].render()))

    rec = simulate.run_episode(seed, [], Gait(), seconds=t1 + C.CTRL_DT, egomotion="gt", on_step=on_step)
    for r in ren.values():
        r.close()
    return frames, rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not os.path.exists(C.SCENE_XML):
        build_scene.build()
    frames, rec = record(a.seed)

    flow_of = FlowEstimator("dis")
    flows = [flow_of(frames[i][1], frames[i + 1][1]) for i in range(len(frames) - 1)]
    k = int(np.argmax([np.linalg.norm(f, axis=2).mean() for f in flows]))     # the step with the clearest motion
    flow, (t, _, rgb) = flows[k], frames[k + 1]
    grid = grid_summary(flow).reshape(C.GRID, C.GRID, 2) * C.IMG_W          # back to pixels per frame
    pose = C.camera_pose()
    pe = PlaneEgomotion(intrinsics(), camera_axes_in_body(pose), pose["pos_m"], simulate.BODY_HEIGHT + pose["pos_m"][2])
    est = pe.solve(flow, C.CTRL_DT)
    true = rec["twist_true"][int(round(t / C.CTRL_DT))]

    H, W = rgb.shape[:2]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    s1 = 6.0                                                              # arrow length = flow x s1
    ys, xs = np.mgrid[4:C.IMG_H:8, 4:C.IMG_W:8]
    u, v = flow[ys, xs, 0], flow[ys, xs, 1]
    axs[0].imshow(rgb)
    axs[0].quiver(xs * DISP, ys * DISP, u * DISP * s1, v * DISP * s1, color="#ffd23f", angles="xy", scale_units="xy",
                  scale=1, width=0.0022, headwidth=4)
    axs[0].set_title(f"(a) Body camera at t = {t:.2f} s and its optical flow (arrows ×{s1:g})", fontsize=10)

    s2 = 20.0
    axs[1].imshow((rgb * 0.55).astype(np.uint8))
    ch, cw = H / C.GRID, W / C.GRID
    for i in range(1, C.GRID):
        axs[1].axhline(i * ch, color="white", lw=0.8, alpha=0.8)
        axs[1].axvline(i * cw, color="white", lw=0.8, alpha=0.8)
    for i in range(C.GRID):
        for j in range(C.GRID):
            gu, gv = grid[i, j]
            cx, cy = (j + 0.5) * cw, (i + 0.5) * ch
            axs[1].annotate("", xy=(cx + gu * DISP * s2, cy + gv * DISP * s2), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="-|>", color="#ffd23f", lw=1.8))
            axs[1].text(j * cw + 6, (i + 1) * ch - 8, f"u {gu:+.2f}\nv {gv:+.2f}", color="white", fontsize=7, va="bottom")
    axs[1].set_title(f"(b) 4×4 grid: mean flow per cell = 32 model inputs (arrows ×{s2:g}, px/frame)", fontsize=10)
    for ax in axs:
        ax.set_xticks([]), ax.set_yticks([])
    fig.text(0.5, 0.015,
             f"Healthy walking, seed {a.seed}. The other 6 camera features are body motion fitted to the flow on the floor plane: "
             f"forward speed {est[0] * 1000:+.0f} mm/s (true {true[0] * 1000:+.0f}), "
             f"turn rate {np.degrees(est[5]):+.1f} deg/s (true {np.degrees(true[5]):+.1f}).",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("written", OUT, f"(step at t={t:.2f}s, mean |flow| {np.linalg.norm(flow, axis=2).mean():.2f} px)")

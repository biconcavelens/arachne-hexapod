"""
5-second video of Stage A while walking: the body camera with the 4x4 grid, what DIS optical flow sees
(colour = direction, brightness = speed) with each cell's mean flow, and the 6 body-motion numbers
fitted from the flow, against the true values. Light: 5 s of simulation, 100 frames.

    python flow_video.py [--seed 7]      # -> runs/report/flow_grid.mp4
"""
import argparse
import os
import shutil
import subprocess

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_scene
import config as C
import simulate
from egomotion import FlowEstimator, PlaneEgomotion, camera_axes_in_body, grid_summary, intrinsics
from grid_figure import DISP, record

OUT = os.path.join(C.HERE, "runs/report/flow_grid.mp4")
NAMES = [("Forward speed", "mm/s", 1000), ("Sideways speed", "mm/s", 1000), ("Vertical speed", "mm/s", 1000),
         ("Roll rate", "deg/s", np.degrees(1)), ("Pitch rate", "deg/s", np.degrees(1)), ("Turn rate", "deg/s", np.degrees(1))]
FMAX = 6.0                                   # px/frame shown at full brightness
AMBER, GREY = "#ffb000", "#8a8984"


def flow_colour(flow):
    mag, ang = cv2.cartToPolar(np.ascontiguousarray(flow[..., 0]), np.ascontiguousarray(flow[..., 1]))
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / FMAX * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def wheel(n=61):
    u, v = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    img = flow_colour(np.dstack([u, v]).astype(np.float32) * FMAX)
    img[u ** 2 + v ** 2 > 1] = 255
    return img


def grid_lines(ax, H, W):
    for i in range(1, C.GRID):
        ax.axhline(i * H / C.GRID, color="white", lw=0.7, alpha=0.8)
        ax.axvline(i * W / C.GRID, color="white", lw=0.7, alpha=0.8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if not os.path.exists(C.SCENE_XML):
        build_scene.build()
    frames, rec = record(a.seed, t0=2.0 - C.CTRL_DT, t1=7.0)
    dis = FlowEstimator("dis")
    pose = C.camera_pose()
    pe = PlaneEgomotion(intrinsics(), camera_axes_in_body(pose), pose["pos_m"], simulate.BODY_HEIGHT + pose["pos_m"][2])
    steps = []                                                    # (t, rgb, flow, estimated twist, true twist)
    for (_, g0, _), (t, g1, rgb) in zip(frames, frames[1:]):
        f = dis(g0, g1)
        steps.append((t, rgb, f, pe.solve(f, C.CTRL_DT), rec["twist_true"][int(round(t / C.CTRL_DT))]))
    ts = np.array([s[0] for s in steps])
    est = np.array([s[3] for s in steps]) * [n[2] for n in NAMES]
    true = np.array([s[4] for s in steps]) * [n[2] for n in NAMES]
    lims = [(min(est[:, i].min(), true[:, i].min()), max(est[:, i].max(), true[:, i].max())) for i in range(6)]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 6, height_ratios=[2.3, 1], hspace=0.42, wspace=0.42, left=0.03, right=0.985, top=0.9, bottom=0.07)
    ax_cam, ax_dis = fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])
    ax_ts = [fig.add_subplot(gs[1, i]) for i in range(6)]
    wh = wheel()
    tmp = OUT + ".mp4v.mp4"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    vw = None
    for k, (t, rgb, flow, _, _) in enumerate(steps):
        H, W = rgb.shape[:2]
        cells = grid_summary(flow).reshape(C.GRID, C.GRID, 2) * C.IMG_W          # px/frame per cell
        for ax in (ax_cam, ax_dis, *ax_ts):
            ax.cla()

        ax_cam.imshow(rgb)
        grid_lines(ax_cam, H, W)
        for i in range(C.GRID):
            for j in range(C.GRID):
                cx, cy = (j + 0.5) * W / C.GRID, (i + 0.5) * H / C.GRID
                u, v = cells[i, j] * DISP * 10
                ax_cam.annotate("", xy=(cx + u, cy + v), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))
        ax_cam.set_title("Body camera (what the robot sees), mean flow per cell ×10", fontsize=10)

        ax_dis.imshow(cv2.resize(flow_colour(flow), (W, H), interpolation=cv2.INTER_NEAREST))
        grid_lines(ax_dis, H, W)
        for i in range(C.GRID):
            for j in range(C.GRID):
                u, v = cells[i, j]
                ax_dis.text(j * W / C.GRID + 5, (i + 1) * H / C.GRID - 6, f"u {u:+.2f}\nv {v:+.2f}", color="white",
                            fontsize=7.5, va="bottom", bbox=dict(facecolor="black", alpha=0.45, pad=1.2, lw=0))
        ins = ax_dis.inset_axes([0.885, 0.02, 0.11, 0.147])
        ins.imshow(wh), ins.set_xticks([]), ins.set_yticks([]), ins.set_title("direction", fontsize=6.5, color="white", pad=1)
        for sp in ins.spines.values():
            sp.set_visible(False)
        ax_dis.set_title(f"How DIS sees it: colour = direction, brightness = speed (full at {FMAX:g} px/frame)", fontsize=10)
        for ax in (ax_cam, ax_dis):
            ax.set_xticks([]), ax.set_yticks([])

        for i, ax in enumerate(ax_ts):
            name, unit, _ = NAMES[i]
            ax.plot(ts, true[:, i], color=GREY, lw=1, alpha=0.5)
            ax.plot(ts[:k + 1], true[:k + 1, i], color=GREY, lw=1.6, label="true")
            ax.plot(ts[:k + 1], est[:k + 1, i], color=AMBER, lw=1.6, label="from camera")
            ax.axvline(t, color="black", lw=0.6)
            pad = 0.1 * (lims[i][1] - lims[i][0] + 1e-9)
            ax.set_xlim(ts[0], ts[-1]), ax.set_ylim(lims[i][0] - pad, lims[i][1] + pad)
            ax.set_title(f"{name}\ncam {est[k, i]:+.0f}  true {true[k, i]:+.0f} {unit}", fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("time (s)", fontsize=7)
        ax_ts[0].legend(fontsize=7, loc="lower left", frameon=False)
        fig.suptitle(f"Stage A while walking (healthy, seed {a.seed})   t = {t:4.2f} s\n"
                     "camera  →  DIS optical flow  →  4×4 grid (32 numbers)  +  body motion fitted to the floor (6 numbers)",
                     fontsize=11)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if vw is None:
            vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / C.CTRL_DT), (img.shape[1], img.shape[0]))
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if k == len(steps) // 2:
            cv2.imwrite(OUT.replace(".mp4", "_still.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()
    ok = shutil.which("ffmpeg") and subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT]).returncode == 0
    os.remove(tmp) if ok else os.replace(tmp, OUT)
    print("written", OUT, f"({len(steps)} frames)")

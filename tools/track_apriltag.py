"""
Phone video -> AprilTag -> x, y, heading vs time.

    python track_apriltag.py run.mp4 [--tag-size 0.10] [--t0 S] [--t1 S] [--fps F] [--id N]
    python track_apriltag.py --selftest

Setup: tag36h11, printed so the BLACK square is --tag-size metres (measure it),
flat on top of the robot with its top edge (the tag's +x) pointing to the front
(L0). Phone fixed overhead, looking straight down, tag fully in view.

Output next to the video: <name>_track.csv (+ .png). Position is zeroed at the
first frame of the window and given twice: x/y in the camera frame (x right,
y up) and fwd/left in the robot's starting frame. Heading is CCW from above,
zeroed at the start, unwrapped.

Limits (ponytail):
- one global metres-per-pixel from the tag's size, so it assumes the phone looks
  straight down and the tag height doesn't change; lens distortion is ignored.
  Tilting the phone 5 deg costs ~0.4% at the image centre, more at the edges.
- time = frame index / fps. Phones record slightly variable frame rate; fine for
  10-30 s runs, pass --fps if the container reports it wrong (slow-mo files).
"""
import argparse
import csv
import math
import os
import sys
import tempfile

import cv2
import numpy as np


def detect(video, tag_id=None, fps=None):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, n = [], 0                          # rows: frame, t, cx, cy, side_px, heading_rad
    while True:
        ok, img = cap.read()
        if not ok:
            break
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                if tag_id is None or i == tag_id:
                    c = c.reshape(4, 2)
                    side = np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])
                    h = math.atan2(-(c[1][1] - c[0][1]), c[1][0] - c[0][0])   # corner0 -> corner1, y up
                    rows.append((n, n / fps, *c.mean(axis=0), side, h))
                    break
        n += 1
    return np.array(rows), n, fps


def analyse(rows, n_frames, fps, tag_size, t0=None, t1=None):
    """-> dict of arrays (t, x, y, fwd, left, heading_deg) and a summary dict."""
    t = rows[:, 1]
    keep = np.ones(len(rows), bool)
    if t0 is not None:
        keep &= t >= t0
    if t1 is not None:
        keep &= t <= t1
    r = rows[keep]
    if len(r) < 2:
        sys.exit("fewer than 2 frames with the tag in the chosen window")
    scale = tag_size / np.median(r[:, 4])              # metres per pixel at the tag
    x = (r[:, 2] - r[0, 2]) * scale
    y = -(r[:, 3] - r[0, 3]) * scale
    th = np.unwrap(r[:, 5])
    h0 = th[0]
    fwd = x * math.cos(h0) + y * math.sin(h0)
    left = -x * math.sin(h0) + y * math.cos(h0)
    heading = np.degrees(th - h0)
    tt = r[:, 1] - r[0, 1]
    in_window = max(1, round((r[-1, 1] - r[0, 1]) * fps) + 1)
    dur = tt[-1]
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())
    summary = {
        "duration_s": dur, "frames_with_tag": len(r), "detection_rate": len(r) / in_window,
        "mm_per_px": scale * 1000, "path_mm": path * 1000,
        "forward_mm": fwd[-1] * 1000, "left_mm": left[-1] * 1000,
        "forward_speed_mm_s": fwd[-1] * 1000 / dur, "heading_change_deg": heading[-1],
        "turn_rate_deg_s": heading[-1] / dur,
    }
    return dict(t=tt, x=x, y=y, fwd=fwd, left=left, heading=heading), summary


def write_outputs(base, d, s):
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x_m", "y_m", "fwd_m", "left_m", "heading_deg"])
        w.writerows(zip(*(np.round(d[k], 6) for k in ("t", "x", "y", "fwd", "left", "heading"))))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(d["left"] * 1000, d["fwd"] * 1000)
        ax[0].set(xlabel="left (mm)", ylabel="forward (mm)", title="path in start frame", aspect="equal")
        ax[1].plot(d["t"], d["heading"])
        ax[1].set(xlabel="t (s)", ylabel="heading (deg, CCW)", title="heading")
        fig.tight_layout()
        fig.savefig(base + ".png", dpi=110)
    except ImportError:
        pass


def show(s):
    print(f"  window {s['duration_s']:.1f} s, tag seen in {s['detection_rate']*100:.0f}% of frames "
          f"({s['mm_per_px']:.2f} mm/px)")
    print(f"  forward {s['forward_mm']:+.0f} mm ({s['forward_speed_mm_s']:+.1f} mm/s), "
          f"left {s['left_mm']:+.0f} mm, path {s['path_mm']:.0f} mm")
    print(f"  heading change {s['heading_change_deg']:+.1f} deg ({s['turn_rate_deg_s']:+.2f} deg/s, + = CCW from above)")


def selftest():
    """Synthetic tag driving a known arc; the tracker must recover it."""
    ppm, tag_m, fps, secs = 1500.0, 0.10, 30, 6.0     # 150 px tag
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dic, 7, 150)
    patch = np.full((300, 300), 255, np.uint8)
    patch[75:225, 75:225] = tag                         # white quiet zone around it
    v, w, h_start = 0.05, math.radians(6), math.radians(25)   # 50 mm/s, 6 deg/s turn, starts at 25 deg
    path = os.path.join(tempfile.mkdtemp(), "synth.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (1280, 720))
    px, py, truth = 320.0, 560.0, []                    # pixels, y down
    for i in range(int(secs * fps)):
        t = i / fps
        h = h_start + w * t
        M = cv2.getRotationMatrix2D((150, 150), math.degrees(h), 1.0)
        M[:, 2] += (px - 150, py - 150)
        img = cv2.warpAffine(patch, M, (1280, 720), borderValue=200)
        out.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        truth.append((t, px, py, h))
        px += v * ppm * math.cos(h) / fps               # moves along its own heading
        py -= v * ppm * math.sin(h) / fps
    out.release()
    rows, n, fps_read = detect(path)
    d, s = analyse(rows, n, fps_read, tag_m)
    show(s)
    tr = np.array(truth)
    dx, dy = (tr[:, 1] - tr[0, 1]) / ppm, -(tr[:, 2] - tr[0, 2]) / ppm       # metres, y up
    true_fwd = dx * math.cos(h_start) + dy * math.sin(h_start)
    true_left = -dx * math.sin(h_start) + dy * math.cos(h_start)
    assert s["detection_rate"] > 0.98, s
    assert abs(d["heading"][-1] - math.degrees(w * (len(tr) - 1) / fps)) < 0.5, d["heading"][-1]
    assert abs(d["fwd"][-1] - true_fwd[-1]) < 0.003 and abs(d["left"][-1] - true_left[-1]) < 0.003, \
        (d["fwd"][-1], true_fwd[-1], d["left"][-1], true_left[-1])
    os.remove(path)
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--tag-size", type=float, default=0.10, help="black-square side in metres")
    ap.add_argument("--id", type=int, default=None, help="tag id to follow (default: first seen)")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--t0", type=float, default=None, help="start of the walk in the video (s)")
    ap.add_argument("--t1", type=float, default=None, help="end of the walk (s)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit()
    if not a.video:
        ap.error("video required")
    rows, n, fps = detect(a.video, a.id, a.fps)
    if len(rows) == 0:
        sys.exit("no tag36h11 found: check print size/contrast and that it is in frame")
    d, s = analyse(rows, n, fps, a.tag_size, a.t0, a.t1)
    base = os.path.splitext(a.video)[0] + "_track"
    write_outputs(base, d, s)
    print(base + ".csv")
    show(s)

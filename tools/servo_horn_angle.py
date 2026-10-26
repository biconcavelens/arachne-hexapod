"""
Slow-motion video of a servo horn -> horn angle vs time.

    python servo_horn_angle.py bench.mp4 [--fps 240] [--hue 60] [--center X,Y] [--mask]
    python servo_horn_angle.py --selftest

Setup: stick ONE small, bright, saturated dot (green/pink/blue sticker or paint)
near the tip of the horn, nothing else of that colour in view. Camera fixed,
looking along the servo shaft (straight at the horn). The shaft centre is found
by fitting a circle to the dot's path, so the horn must sweep at least ~20 deg
in the clip, or pass --center X,Y (pixels) yourself.

Colour: --hue is OpenCV hue 0-179 (red 0/179, orange 10, yellow 25, green 60,
blue 110, pink 165). Run once with --mask to write <name>_mask.png and check
that only the dot is white; tune --hue-tol / --smin / --vmin if not.

Output: <name>_angle.csv with time, absolute angle (CCW from the image +x axis)
and angle relative to the rest pose (median of the first --rest-s seconds), plus
a summary with the range and peak angular rate.

Time is frame index / --fps. Slow-motion phone files often store 30 fps
playback, so pass the real capture rate (240) explicitly.

This only measures the horn. Delay and deadband also need the time the command
was sent; log that on the sender side and align it with the first motion.
"""
import argparse
import csv
import math
import os
import sys
import tempfile

import cv2
import numpy as np


def dot_mask(img, hue, hue_tol, smin, vmin):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dh = np.abs(hsv[..., 0].astype(int) - hue)
    dh = np.minimum(dh, 180 - dh)                     # hue is circular
    return ((dh <= hue_tol) & (hsv[..., 1] >= smin) & (hsv[..., 2] >= vmin)).astype(np.uint8)


def find_dot(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    m = cv2.moments(max(cnts, key=cv2.contourArea))
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def track(video, fps, hue, hue_tol, smin, vmin, min_area, save_mask=None):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 240.0
    pts, n = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        mask = dot_mask(img, hue, hue_tol, smin, vmin)
        if save_mask and n == 0:
            cv2.imwrite(save_mask, mask * 255)
        p = find_dot(mask, min_area)
        pts.append((n / fps, *(p if p else (np.nan, np.nan))))
        n += 1
    return np.array(pts), fps


def fit_circle(x, y):
    """Algebraic (Kasa) circle fit -> cx, cy, r, rms residual in px."""
    A = np.c_[x, y, np.ones_like(x)]
    sol = np.linalg.lstsq(A, -(x ** 2 + y ** 2), rcond=None)[0]
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r = math.sqrt(max(cx ** 2 + cy ** 2 - sol[2], 0))
    return cx, cy, r, float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))


def analyse(pts, centre=None, rest_s=0.3):
    t, x, y = pts.T
    ok = ~np.isnan(x)
    if ok.sum() < 10:
        sys.exit("dot found in fewer than 10 frames: run with --mask and tune the colour")
    info = {"detection_rate": ok.mean()}
    if centre is None:
        cx, cy, r, rms = fit_circle(x[ok], y[ok])
        info.update(fit_rms_px=rms, radius_px=r)
        span = np.ptp(np.unwrap(np.arctan2(-(y[ok] - cy), x[ok] - cx)))
        if math.degrees(span) < 20:
            sys.exit(f"horn only swept {math.degrees(span):.0f} deg: too little to find the centre, pass --center X,Y")
    else:
        cx, cy = centre
    ang = np.full(len(t), np.nan)
    ang[ok] = np.degrees(np.unwrap(np.arctan2(-(y[ok] - cy), x[ok] - cx)))
    rest = np.nanmedian(ang[t < t[0] + rest_s])
    rel = ang - rest
    # rate: 3-frame moving average of the angle, then central differences
    sm = np.convolve(np.nan_to_num(ang, nan=np.nanmean(ang)), np.ones(3) / 3, mode="same")
    rate = np.gradient(sm, t)
    info.update(centre=(cx, cy), range_deg=float(np.nanmax(rel) - np.nanmin(rel)),
                peak_rate_deg_s=float(np.abs(rate[2:-2]).max()))
    return dict(t=t, abs=ang, rel=rel, rate=rate), info


def show(i):
    fit = f", circle fit rms {i['fit_rms_px']:.2f} px (radius {i['radius_px']:.0f} px)" if "fit_rms_px" in i else ""
    print(f"  dot seen in {i['detection_rate']*100:.0f}% of frames, centre ({i['centre'][0]:.1f}, {i['centre'][1]:.1f}) px{fit}")
    print(f"  horn range {i['range_deg']:.1f} deg, peak rate {i['peak_rate_deg_s']:.0f} deg/s")


def selftest():
    """Synthetic green dot on a rate-limited 90 -> 150 -> 90 deg sweep at 240 fps."""
    fps, rate, c = 240, 300.0, (320.0, 240.0)           # deg/s
    path = os.path.join(tempfile.mkdtemp(), "horn.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    truth = []
    for i in range(int(1.6 * fps)):
        t = i / fps
        a = 90 + np.clip(rate * (t - 0.2), 0, 60) - np.clip(rate * (t - 1.0), 0, 60)
        truth.append(a)
        img = np.full((480, 640, 3), 90, np.uint8)
        px, py = c[0] + 100 * math.cos(math.radians(a)), c[1] - 100 * math.sin(math.radians(a))
        cv2.circle(img, (round(px * 16), round(py * 16)), 7 * 16, (0, 255, 0), -1, cv2.LINE_AA, shift=4)   # sub-pixel
        out.write(img)
    out.release()
    pts, fps_read = track(path, fps, hue=60, hue_tol=12, smin=120, vmin=80, min_area=20)
    d, info = analyse(pts)
    show(info)
    truth = np.array(truth)
    err = np.sqrt(np.nanmean((d["abs"] - truth) ** 2))
    assert abs(info["centre"][0] - c[0]) < 1.5 and abs(info["centre"][1] - c[1]) < 1.5, info["centre"]
    assert err < 0.7, f"angle rms error {err:.2f} deg"
    assert abs(info["peak_rate_deg_s"] - rate) < 0.1 * rate, info["peak_rate_deg_s"]
    assert abs(info["range_deg"] - 60) < 1.0, info["range_deg"]
    os.remove(path)
    print(f"  angle rms error {err:.2f} deg\nselftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?")
    ap.add_argument("--fps", type=float, default=None, help="real capture rate (default: from the file)")
    ap.add_argument("--hue", type=int, default=60)
    ap.add_argument("--hue-tol", type=int, default=12)
    ap.add_argument("--smin", type=int, default=120)
    ap.add_argument("--vmin", type=int, default=80)
    ap.add_argument("--min-area", type=float, default=20)
    ap.add_argument("--center", type=lambda s: tuple(map(float, s.split(","))), default=None)
    ap.add_argument("--rest-s", type=float, default=0.3, help="seconds at the start used as the rest pose")
    ap.add_argument("--mask", action="store_true", help="save the first frame's colour mask for tuning")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit()
    if not a.video:
        ap.error("video required")
    base = os.path.splitext(a.video)[0]
    pts, fps = track(a.video, a.fps, a.hue, a.hue_tol, a.smin, a.vmin, a.min_area, base + "_mask.png" if a.mask else None)
    d, info = analyse(pts, a.center, a.rest_s)
    with open(base + "_angle.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "angle_deg_abs", "angle_deg_rel_rest", "rate_deg_s"])
        w.writerows(zip(*(np.round(d[k], 4) for k in ("t", "abs", "rel", "rate"))))
    print(base + "_angle.csv")
    show(info)

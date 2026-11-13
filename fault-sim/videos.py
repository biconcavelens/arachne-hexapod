"""
Videos of the trained diagnoser running closed loop: the body camera (ego) beside a
third-person follow camera (exo), with the true fault, the model's top suspect, alarms
and recovery actions written underneath.

    MUJOCO_GL=egl python videos.py --ckpt runs/gt_transformer/model.pt --out runs/videos

One video per case: a single loss, sag and dead-band fault from the closed-loop test seeds,
and a healthy run (any false alarms show up). The diagnosis gets the same inputs as in
training (--egomotion gt: true body motion plus noise, unless --egomotion flow); the ego
frames are rendered for viewing.
"""
import argparse
import os
import shutil
import subprocess

import cv2
import mujoco
import numpy as np

import config as C
import simulate
from faults import sample_faults
from gaits_lib import Gait
from recovery import ModelDiagnosis, RecoveryController

EGO, EXO, BAR = (320, 240), (480, 360), 92      # (w, h) of each view; text bar height


def find_cases(first_seed):
    """The first closed-loop test seed with a single fault of each kind (sampled as closed_loop.py does)."""
    cases, seed = {}, first_seed
    while len(cases) < 3:
        rng = np.random.default_rng([seed, 1])
        params = Gait.random(rng).params
        faults = sample_faults(rng, simulate.movable_servos(Gait(**params)))
        if len(faults) == 1 and faults[0]["kind"] not in cases:
            cases[faults[0]["kind"]] = (seed, params, faults)
        seed += 1
    hs = first_seed + 50000                                     # closed_loop.py's healthy seeds
    cases["healthy"] = (hs, Gait.random(np.random.default_rng([hs, 1])).params, [])
    return cases


def overlay(ego, exo, t, ctrl):
    ego = cv2.resize(ego, (EXO[1] * EGO[0] // EGO[1], EXO[1]))
    img = np.zeros((EXO[1] + BAR, ego.shape[1] + EXO[0], 3), np.uint8)
    img[:EXO[1], :ego.shape[1]], img[:EXO[1], ego.shape[1]:] = ego, exo
    types, sizes = ctrl.bank.labels(t)
    unit = {1: "", 2: " deg", 3: " deg"}
    truth = ", ".join(f"{C.JOINT_NAMES[j]} {C.FAULT_TYPES[types[j]]} {sizes[j]:.2f}{unit[types[j]]}"
                      for j in np.flatnonzero(types)) or "healthy"
    if ctrl.last_p is None:
        suspect = "-"
    else:
        pf = 1 - ctrl.last_p[:, 0]
        j = int(pf.argmax())
        suspect = f"{C.JOINT_NAMES[j]} p={pf[j]:.2f} ({C.FAULT_TYPES[1 + int(ctrl.last_p[j, 1:].argmax())]})"
    alarms = [e for e in ctrl.events if "alarm" in e]
    acts = [e for e in ctrl.events if "action" in e]
    lines = [f"t {t:5.2f} s   state: {ctrl.state}   truth: {truth}",
             f"model's top suspect: {suspect}",
             "last alarm: " + (f"{C.JOINT_NAMES[alarms[-1]['servo']]} {alarms[-1]['alarm']} at {alarms[-1]['t']:.1f} s" if alarms else "-"),
             "last action: " + (f"{acts[-1]['action']} on {C.JOINT_NAMES[acts[-1]['servo']]} at {acts[-1]['t']:.1f} s"
                                + (f" (estimate {acts[-1]['estimate_deg']} deg)" if "estimate_deg" in acts[-1] else "")
                                if acts else "-")]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, EXO[1] + 20 + 21 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    for x, s in ((8, "EGO: body camera"), (ego.shape[1] + 8, "EXO: follow camera (viewing only)")):
        cv2.putText(img, s, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def record(path, seed, params, faults, diag, seconds, egomotion):
    ctrl, frames, ren = RecoveryController("model", diag), [], {}
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.8, 60.0, -25.0

    def on_step(m, d, t):
        if not ren:
            ren["ego"], ren["exo"] = mujoco.Renderer(m, EGO[1], EGO[0]), mujoco.Renderer(m, EXO[1], EXO[0])
        cam.lookat[:] = d.xpos[m.body("base").id]
        ren["ego"].update_scene(d, camera="ego")
        ego = ren["ego"].render()
        ren["exo"].update_scene(d, camera=cam)
        frames.append(overlay(ego, ren["exo"].render(), t, ctrl))

    simulate.run_episode(seed, faults, Gait(**params), seconds=seconds, egomotion=egomotion, controller=ctrl, on_step=on_step)
    for r in ren.values():
        r.close()
    tmp = path + ".mp4v.mp4"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), round(1 / C.CTRL_DT), (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    ok = shutil.which("ffmpeg") and subprocess.run(                # H.264 plays in browsers; mp4v often does not
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path]
    ).returncode == 0
    if ok:
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    return ctrl.events


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/videos")
    ap.add_argument("--first-seed", type=int, default=100000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--egomotion", choices=["gt", "flow"], default="gt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    diag = ModelDiagnosis(a.ckpt)
    for name, (seed, params, faults) in find_cases(a.first_seed).items():
        path = os.path.join(a.out, f"{name}_seed{seed}.mp4")
        events = record(path, seed, params, faults, diag, a.seconds, a.egomotion)
        print(f"{path}: faults {[(C.JOINT_NAMES[f['servo']], f['kind'], round(f['size'], 2), round(f['onset'], 1)) for f in faults]}; "
              f"{sum('alarm' in e for e in events)} alarms, actions {[e['action'] for e in events if 'action' in e]}", flush=True)

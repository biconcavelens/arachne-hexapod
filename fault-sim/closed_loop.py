"""
Closed-loop test: walk with faults, diagnose online, recover, and measure walking.

    python closed_loop.py --mode model  --ckpt runs/gt_transformer/model.pt --episodes 60
    python closed_loop.py --mode oracle --episodes 60    # true labels: the best these recoveries can do
    python closed_loop.py --mode none   --episodes 60    # faults, no recovery

Every seed runs three times with the same scene, gait, faults and noise: healthy (the
reference), faulty without recovery, and faulty with the chosen mode. In model mode, extra
healthy episodes count false alarms. Walking is measured over the last LAST_S seconds.
"""
import argparse
import json
import math
import os

import numpy as np

import config as C
import simulate
from faults import sample_faults
from gaits_lib import Gait
from recovery import ModelDiagnosis, RecoveryController

LAST_S = 4.0


def walking(rec):
    """Forward speed (mm/s) along the heading at the start of the last LAST_S, and turn rate (deg/s)."""
    n = int(LAST_S / C.CTRL_DT)
    p = rec["pose"][-n:]
    if len(p) < n or "fell_at" in rec:
        return {"speed_mm_s": 0.0, "turn_deg_s": float("nan"), "fell": True}
    d = p[-1, :2] - p[0, :2]
    fwd = d[0] * math.cos(p[0, 2]) + d[1] * math.sin(p[0, 2])
    dpsi = (p[-1, 2] - p[0, 2] + math.pi) % (2 * math.pi) - math.pi
    return {"speed_mm_s": fwd / LAST_S * 1000, "turn_deg_s": math.degrees(dpsi) / LAST_S, "fell": False}


def episode(seed, mode, diag, seconds, egomotion):
    rng = np.random.default_rng([seed, 1])
    params = Gait.random(rng).params
    faults = sample_faults(rng, simulate.movable_servos(Gait(**params)))
    run = lambda f, ctrl=None: simulate.run_episode(seed, f, Gait(**params), seconds=seconds, egomotion=egomotion, controller=ctrl)
    ref, none = walking(run([])), walking(run(faults))
    out = {"seed": seed, "faults": faults, "gait": params, "healthy": ref, "no_recovery": none}
    if mode != "none":
        ctrl = RecoveryController(mode, diag)
        out["recovery"] = walking(run(faults, ctrl))
        out["events"] = ctrl.events
        alarms = [e for e in ctrl.events if "alarm" in e]
        onset = min(f["onset"] for f in faults)
        truth = {(f["servo"], f["kind"]) for f in faults}
        if alarms:
            first = alarms[0]
            out["first_alarm"] = {"delay_s": first["t"] - onset, "correct_servo": first["servo"] in {s for s, _ in truth},
                                  "correct_servo_and_type": (first["servo"], first["alarm"]) in truth}
    return out


def false_alarms(seed, diag, seconds, egomotion):
    rng = np.random.default_rng([seed, 1])
    ctrl = RecoveryController("model", diag)
    simulate.run_episode(seed, [], Gait.random(rng), seconds=seconds, egomotion=egomotion, controller=ctrl)
    return sum("alarm" in e for e in ctrl.events)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["model", "oracle", "none"], default="model")
    ap.add_argument("--ckpt")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--healthy-episodes", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=100000)      # away from the training seeds
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--egomotion", choices=["gt", "flow"], default="gt")
    ap.add_argument("--out", default="runs/closed_loop.json")
    ap.add_argument("--drop-imu", action="store_true", help="IMU missing at test (inputs held at their training mean)")
    a = ap.parse_args()
    diag = ModelDiagnosis(a.ckpt) if a.mode == "model" else None
    if a.drop_imu:
        diag.norm["no_imu"] = True
    eps = []
    for i in range(a.episodes):
        eps.append(episode(a.first_seed + i, a.mode, diag, a.seconds, a.egomotion))
        e = eps[-1]
        line = f"seed {e['seed']}: {[(f['servo'], f['kind']) for f in e['faults']]}  healthy {e['healthy']['speed_mm_s']:.0f}  " \
               f"no recovery {e['no_recovery']['speed_mm_s']:.0f}"
        if "recovery" in e:
            line += f"  recovery {e['recovery']['speed_mm_s']:.0f} mm/s"
        print(line, flush=True)

    ratio = lambda k: [e[k]["speed_mm_s"] / e["healthy"]["speed_mm_s"] for e in eps if e["healthy"]["speed_mm_s"] > 5]
    summary = {"mode": a.mode, "episodes": len(eps), "egomotion": a.egomotion,
               "speed_vs_healthy_no_recovery": float(np.median(ratio("no_recovery")))}
    if a.mode != "none":
        summary["speed_vs_healthy_recovery"] = float(np.median(ratio("recovery")))
        fa = [e["first_alarm"] for e in eps if "first_alarm" in e]
        summary.update(detected=len(fa) / len(eps),
                       first_alarm_correct_servo=float(np.mean([f["correct_servo"] for f in fa])) if fa else None,
                       first_alarm_correct_type=float(np.mean([f["correct_servo_and_type"] for f in fa])) if fa else None,
                       median_delay_s=float(np.median([f["delay_s"] for f in fa])) if fa else None)
        for kind in ("loss", "sag", "deadband"):                    # by the kind of the first fault
            r = [e["recovery"]["speed_mm_s"] / e["healthy"]["speed_mm_s"] for e in eps
                 if e["faults"][0]["kind"] == kind and e["healthy"]["speed_mm_s"] > 5]
            if r:
                summary[f"speed_vs_healthy_recovery_{kind}"] = float(np.median(r))
    if a.mode == "model":
        n = sum(false_alarms(a.first_seed + 50000 + i, diag, a.seconds, a.egomotion) for i in range(a.healthy_episodes))
        summary["false_alarms_per_min_healthy"] = n / (a.healthy_episodes * a.seconds / 60)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary, "episodes": eps}, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1))

"""
Gait generator + clearance check for the Vorpal hexapod.

Gaits are defined in SIM joint angles (degrees; 0 = neutral = servos at 90)
and verified in MuJoCo (hexapod_neutral.xml) before they're written out for
the web page. The page converts sim angles to servo angles with the channel
mapping saved on the board, so changing the mapping never needs a regenerate.

Clearance: every frame (plus the ramps from/to neutral) must keep at least
CLEARANCE_MM between any geometry of two different legs. Distances use
MuJoCo's convex hulls of the meshes, which can only under-estimate the real
gap, so this errs on the safe side.

Run:  python gaits.py   -> writes servo-web-control/assets/gaits/*.json
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

# The sim is ~5mm optimistic: at the real legs-touch point (servo ch0=112,
# ch1=46 -> L0 hip -22, L5 hip +44) it still reports a 4.8mm gap. 10mm covers
# that error plus ~5mm of real air.
CLEARANCE_MM = 10.0
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "..", "servo-web-control", "assets", "gaits")

# Standing pose: knees pushed down (sim knee - = foot down). At 0 (servos at
# 90) the knee hinges clear the ground by only ~13mm and scrape when walking;
# at -30 (knee servos 60) they clear by ~40mm, body ~96mm up, reach -24mm.
STANCE_KNEE = -30.0
HIP_DEG = [3.3 + 60 * k for k in range(6)]   # sim leg k sits here around the base
TRIPOD_A = (0, 2, 4)                          # swings in the first half of the cycle
NUM_LEGS = 6


def forward_walk(hip_amp=18.0, knee_lift=25.0, cycle_s=1.2, fps=20, stance_knee=STANCE_KNEE):
    """Tripod gait toward +x (L0 = front).

    Hip-yaw legs move their feet on arcs around the hip axis: dx ~ -sin(theta) * dhip,
    so only the side legs can push along x. L0/L3 (theta ~ 0/180) keep their hip
    centered and just lift/plant with their tripod for support.
    Swing: foot moves forward (+x) while lifted. Stance: foot moves back on the
    ground, which moves the body forward.
    """
    n = int(round(cycle_s * fps))
    frames = []
    for i in range(n):
        phi = i / n
        hips, knees = [0.0] * NUM_LEGS, [stance_knee] * NUM_LEGS
        for k in range(NUM_LEGS):
            s = math.sin(math.radians(HIP_DEG[k]))
            # sign so that decreasing p (stance) moves the foot to -x
            fwd = 0.0 if abs(s) < 0.2 else -math.copysign(1.0, s)
            ph = phi if k in TRIPOD_A else (phi + 0.5) % 1.0
            if ph < 0.5:                      # swing: p -1 -> +1, foot lifted
                u = ph / 0.5
                p = -math.cos(math.pi * u)
                knees[k] = stance_knee + knee_lift * math.sin(math.pi * u)
            else:                             # stance: p +1 -> -1, foot down
                u = (ph - 0.5) / 0.5
                p = math.cos(math.pi * u)
            hips[k] = fwd * hip_amp * p
        frames.append([round(v, 2) for v in hips + knees])
    return {"name": "Walk forward", "fps": fps, "cycle_s": cycle_s,
            "params": {"hip_amp_deg": hip_amp, "knee_lift_deg": knee_lift, "stance_knee_deg": stance_knee},
            "rest": [0.0] * NUM_LEGS + [stance_knee] * NUM_LEGS,
            # +1 left side legs (L1, L2), -1 right (L4, L5), 0 front/back: see steered()
            "hip_side": [0 if abs(math.sin(math.radians(h))) < 0.2 else int(math.copysign(1, math.sin(math.radians(h))))
                         for h in HIP_DEG],
            "joints": [f"hip{k}" for k in range(6)] + [f"knee{k}" for k in range(6)],
            "frames": frames}


def steered(pose, side, steer):
    """Steer = left/right stride difference: hip swing x (1 + steer * side).
    + makes the left legs stride longer -> turns right. Also cancels the
    chiral (pinwheel knee offset) drift: in physics the untrimmed walk turns
    right ~4 deg/s, trim ~-0.15 holds heading (open loop, still wanders)."""
    return [v * (1 + steer * side[k]) if k < NUM_LEGS else v for k, v in enumerate(pose)]


# ---------------------------------------------------------------- verification
model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "hexapod_neutral.xml"))
data = mujoco.MjData(model)
JOINT_ADR = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j]
             for j in range(model.njnt)}
LEG_GEOMS = {k: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{p}{k}")
                 for p in ("hingeA", "hingeB", "leg")] for k in range(NUM_LEGS)}
PAIRS = [(g1, g2, a, b) for a in range(NUM_LEGS) for b in range(a + 1, NUM_LEGS)
         for g1 in LEG_GEOMS[a] for g2 in LEG_GEOMS[b]]


def min_clearance_mm(pose):
    """pose: [hip0..5, knee0..5] in sim degrees -> (min gap mm, legs a, b)."""
    for name, v in zip([f"hip{k}" for k in range(6)] + [f"knee{k}" for k in range(6)], pose):
        data.qpos[JOINT_ADR[name]] = math.radians(v)
    mujoco.mj_forward(model, data)
    best = (1e9, None, None)
    for g1, g2, a, b in PAIRS:
        d = mujoco.mj_geomDistance(model, data, g1, g2, 0.05, None) * 1000
        if d < best[0]:
            best = (d, a, b)
    return best


def verify(gait, steer=0.0, ramp_steps=10):
    neutral = [0.0] * 12
    frames = [steered(f, gait["hip_side"], steer) for f in gait["frames"]]
    rest, first = gait["rest"], frames[0]
    lerp = lambda a, b: [[x + (y - x) * t / ramp_steps for x, y in zip(a, b)] for t in range(ramp_steps + 1)]
    ramp = lerp(neutral, rest) + lerp(rest, first)
    worst = (1e9, None, None, None)
    for label, poses in (("ramp in/out", ramp), ("cycle", frames)):
        for i, pose in enumerate(poses):
            d, a, b = min_clearance_mm(pose)
            if d < worst[0]:
                worst = (d, a, b, f"{label} frame {i}")
    return worst


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    neutral_gap = min_clearance_mm([0.0] * 12)[0]
    print(f"neutral pose: min gap between legs {neutral_gap:.1f} mm")
    gait = forward_walk()
    d, a, b, where = verify(gait)
    print(f"{gait['name']}: min gap {d:.1f} mm (L{a}-L{b}, {where}), required {CLEARANCE_MM} mm")
    if d < CLEARANCE_MM:
        print("FAILED clearance - not written")
        sys.exit(1)
    # largest steer (0.05 steps) that keeps clearance both ways
    steer_max = 0.0
    while steer_max < 0.6 and all(verify(gait, sg * (steer_max + 0.05))[0] >= CLEARANCE_MM for sg in (1, -1)):
        steer_max = round(steer_max + 0.05, 2)
    print(f"steer: +-{steer_max} keeps clearance")
    gait["steer_max"] = steer_max
    gait["verified"] = {"min_gap_mm": round(d, 1), "required_mm": CLEARANCE_MM,
                        "model": "hexapod_neutral.xml", "worst": f"L{a}-L{b} {where}",
                        "steer_max_checked": steer_max}
    out = os.path.join(OUT_DIR, "forward.json")
    json.dump(gait, open(out, "w"), indent=1)
    print(f"written: {os.path.abspath(out)}")

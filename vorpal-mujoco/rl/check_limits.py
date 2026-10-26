"""
Largest symmetric hip range [-H, +H] (and a knee range) for which NO combination
of two neighbouring legs gets closer than CLEARANCE_MM, using the same
convex-hull distance check as gaits.py. A learned policy can pick any
combination, so its action limits must hold for the worst one.

Usage: python check_limits.py [knee_lo=-55] [knee_hi=10]
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gaits as G                      # noqa: E402  (loads hexapod_neutral.xml, no side effects)

args = dict(a.split("=") for a in sys.argv[1:])
KLO, KHI = float(args.get("knee_lo", -55)), float(args.get("knee_hi", 10))
KNEES = np.linspace(KLO, KHI, 4)


def worst_gap(H, steps=5):
    hips = np.linspace(-H, H, steps)
    worst = (1e9, None)
    for k in range(6):
        a, b = k, (k + 1) % 6
        for ha, hb, ka, kb in itertools.product(hips, hips, KNEES, KNEES):
            pose = [0.0] * 6 + [G.STANCE_KNEE] * 6
            pose[a], pose[b], pose[6 + a], pose[6 + b] = ha, hb, ka, kb
            d, la, lb = G.min_clearance_mm(pose)
            if d < worst[0]:
                worst = (d, f"L{la}-L{lb} hips {ha:+.0f}/{hb:+.0f} knees {ka:+.0f}/{kb:+.0f}")
    return worst


print(f"required gap {G.CLEARANCE_MM} mm, knees {KLO:+.0f}..{KHI:+.0f}")
best = None
for H in (15, 20, 25, 30, 35):
    d, where = worst_gap(H)
    ok = d >= G.CLEARANCE_MM
    print(f"  hip +-{H:2d} deg: worst gap {d:6.1f} mm  ({where})  {'OK' if ok else 'too close'}")
    if ok:
        best = H
print(f"=> hip limit +-{best} deg" if best else "=> no tested hip range keeps the clearance")

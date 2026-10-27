"""
Generate training episodes: random gait variant, random faults (or none), random scene.

    python generate.py --out data/gt   --episodes 2000 --egomotion gt   --workers 4
    python generate.py --out data/flow --episodes 400  --egomotion flow --workers 2   # renders; needs OpenGL

One file per episode (ep_<seed>.npz). Existing files are skipped, so an interrupted
run (for example a Kaggle session limit) continues where it stopped.
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np


def one(job):
    seed, out, egomotion, flow = job
    path = os.path.join(out, f"ep_{seed:06d}.npz")
    if os.path.exists(path):
        return 0
    import config as C
    import simulate
    from faults import sample_faults
    from gaits_lib import Gait
    rng = np.random.default_rng([seed, 1])
    gait = Gait.random(rng)
    faults = [] if rng.random() < C.HEALTHY_FRACTION else sample_faults(rng, simulate.movable_servos(gait))
    rec = simulate.run_episode(seed, faults, gait, egomotion=egomotion, flow_method=flow)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **rec, faults=json.dumps(faults), gait=json.dumps(gait.params), egomotion=egomotion)
    os.replace(tmp, path)
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--egomotion", choices=["gt", "flow"], default="gt")
    ap.add_argument("--flow", choices=["dis", "raft"], default="dis")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import config as C
    json.dump({k: getattr(C, k) for k in dir(C) if k.isupper() and isinstance(getattr(C, k), (int, float, str, tuple, list))},
              open(os.path.join(a.out, "config.json"), "w"), indent=1)
    jobs = [(s, a.out, a.egomotion, a.flow) for s in range(a.first_seed, a.first_seed + a.episodes)]
    t0, done = time.time(), 0
    with Pool(a.workers) as pool:
        for i, new in enumerate(pool.imap_unordered(one, jobs), 1):
            done += new
            if i % 50 == 0 or i == len(jobs):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"{i}/{len(jobs)} episodes ({done} new, {rate:.2f}/s)", flush=True)

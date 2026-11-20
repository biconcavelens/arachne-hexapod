"""
Evaluate room policies under fixed fault scenarios (paired: every scenario and policy uses the same
rooms, paths and start poses).

    bash wslrun.sh python eval_room.py policies=room_both,room_cam,room_imu,room_both_healthy n=256 out=eval_room
    bash wslrun.sh python eval_room.py policies=dyn_both,dyn_imu,room_both env=room_dyn out=eval_dyn

env= overrides the scene for every policy (e.g. to test a static-room policy among movers).
Writes runs/<out>/results.json, table.md and traj_<policy>_<scenario>.npz (16 episodes each).
"""
import json
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

import fault_rollout as R
import scenes
import room_env as E

args = dict(a.split("=", 1) for a in sys.argv[1:])
POLICIES = args.get("policies", "room_both").split(",")
N = int(args.get("n", 256))
OUT = os.path.join(R.ROOT, "runs", args.get("out", "eval_room"))
os.makedirs(OUT, exist_ok=True)
MID = 40                                               # mid-walk onset: 2 s; further faults 2 s apart
SIZE = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.7, 4: float(np.radians(15))}
NAMES = {1: "dead", 2: "stuck", 3: "signal loss", 4: "worn gears"}

rng = np.random.default_rng(0)
keys = jax.random.split(jax.random.PRNGKey(4321), N)


def servos(k):                                          # k servos on k different legs
    legs = np.stack([rng.permutation(6)[:k] for _ in range(N)])
    return 2 * legs + rng.integers(0, 2, (N, k))


S1 = servos(1)
S2, S3 = servos(2), servos(3)
mixed = rng.integers(1, 5, (N, 3))
scen = [("healthy", np.zeros((N, 1), int), -np.ones((N, 1), int), np.zeros((N, 1), int))]
for k in (1, 2, 3, 4):
    for when, on in (("from start", 0), ("mid-walk", MID)):
        scen.append((f"1 {NAMES[k]}, {when}", np.full((N, 1), k), S1, np.full((N, 1), on)))
scen.append(("2 dead, mid-walk", np.full((N, 2), 1), S2, np.tile(MID + 40 * np.arange(2), (N, 1))))
scen.append(("3 dead, mid-walk", np.full((N, 3), 1), S3, np.tile(MID + 40 * np.arange(3), (N, 1))))
scen.append(("3 mixed, from start", mixed, S3, np.zeros((N, 3), int)))

results = {}
for pname in POLICIES:
    cfg = json.load(open(os.path.join(R.ROOT, "runs", pname, "config.json")))
    scene = args.get("env", cfg.get("env", "room"))
    cls = scenes.env_class(scene)
    env = cls(config_overrides={"sensors": cfg["config"]["sensors"]})
    run = R.make_runner(env, R.load_policy(pname, env))
    res = results[pname] = {"scene": scene, "sensors": cfg["config"]["sensors"], "faults_in_training": cfg["faults"]}
    t_healthy = None
    for name, kinds, sv, onsets in scen:
        sizes = np.vectorize(lambda k: SIZE[int(k)])(kinds).astype(np.float32)
        tr = jax.tree_util.tree_map(np.asarray, run(keys, jp.asarray(kinds), jp.asarray(sv), jp.asarray(onsets),
                                                    jp.asarray(sizes)))
        m = R.summarise(tr, env.dt)
        if name == "healthy":
            t_healthy = m["t_goal"]
        both = np.isfinite(m["t_goal"]) & np.isfinite(t_healthy)
        alive = ~tr["done"]
        xt = np.where(alive, tr["cross_track"], np.nan) * 1000
        res[name] = {
            "success_pct": float(m["goal"].mean() * 100),
            "ended_pct": float(m["fell"].mean() * 100),           # fell, left the path (>15 cm) or bumped
            "path_pct": float(m["path_frac"].mean() * 100),
            "xtrack_mean_mm": float(np.nanmean(xt)),
            "xtrack_p95_mm": float(np.nanpercentile(xt, 95)),
            "t_goal_s": float(np.nanmedian(m["t_goal"])) if m["goal"].any() else None,
            "speed_vs_healthy_pct": float(np.median(t_healthy[both] / m["t_goal"][both]) * 100) if both.any() else None,
        }
        if "struck" in tr:                                      # hit from the side or behind by a mover
            res[name]["struck_pct"] = float((tr["struck"] & alive).any(-1).mean() * 100)
        if "heading_err" in tr:                                 # camera vs path direction while walking
            walk = alive & (tr["remaining"] > 0.03)
            he = np.degrees(np.where(walk, tr["heading_err"], np.nan))
            res[name]["heading_mean_deg"] = float(np.nanmean(he))
            res[name]["heading_within30_pct"] = float(np.nanmean(np.where(walk, he < 30, np.nan)) * 100)
        r = res[name]
        print(f"{pname:18s} {name:26s} success {r['success_pct']:5.1f}%  ended {r['ended_pct']:5.1f}%"
              f"  xtrack {r['xtrack_mean_mm']:5.1f} / p95 {r['xtrack_p95_mm']:5.1f} mm  speed {r['speed_vs_healthy_pct']}"
              f"  struck {r.get('struck_pct', '-')}  heading {r.get('heading_mean_deg', '-')}",
              flush=True)
        tag = name.replace(" ", "").replace(",", "_")
        np.savez_compressed(os.path.join(OUT, f"traj_{pname}_{tag}.npz"), xy=tr["xy"][:16], path=tr["path"][:16],
                            goal_idx=tr["goal_idx"][:16], done=tr["done"][:16], remaining=tr["remaining"][:16],
                            cross_track=tr["cross_track"][:16], servos=sv[:16], kinds=kinds[:16])

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
rows = [s[0] for s in scen]
with open(os.path.join(OUT, "table.md"), "w") as f:
    f.write("| scenario | " + " | ".join(POLICIES) + " |\n|---|" + "---|" * len(POLICIES) + "\n")
    for r in rows:
        f.write(f"| {r} | " + " | ".join(
            f"{results[p][r]['success_pct']:.0f}% · {results[p][r]['xtrack_mean_mm']:.0f} mm" for p in POLICIES) + " |\n")
print(open(os.path.join(OUT, "table.md")).read())

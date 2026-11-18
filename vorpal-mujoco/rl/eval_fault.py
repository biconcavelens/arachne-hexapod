"""
Evaluate path-following policies under fixed fault scenarios (paired: every scenario
uses the same paths and start poses).

    bash wslrun.sh python eval_fault.py policies=fault_v1,healthy_v1 n=128 out=eval_v1

Scenarios: healthy; each fault kind on a hip and on a knee (random leg, onset at 3 s,
mid-range size); two dead servos on different legs. Writes runs/<out>/results.json,
runs/<out>/table.md and runs/<out>/paths_<policy>.png.
"""
import json
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

import fault_path_env as E
import fault_rollout as R

args = dict(a.split("=", 1) for a in sys.argv[1:])
POLICIES = args.get("policies", "fault_v1").split(",")
N = int(args.get("n", 128))
OUT = os.path.join(R.ROOT, "runs", args.get("out", "eval"))
os.makedirs(OUT, exist_ok=True)
ONSET = int(args.get("onset", 60))
GOAL = int(args["goal"]) if "goal" in args else None      # path index, e.g. 190 = 1.9 m

env = E.FaultPath()
rng = np.random.default_rng(0)
keys = jax.random.split(jax.random.PRNGKey(1234), N)
legs = rng.integers(0, 6, N)
legs2 = (legs + rng.integers(1, 6, N)) % 6
dead2 = rng.choice([4, 5], (N, 2))

scen = [("healthy", "-", np.zeros((N, 2), int), -np.ones((N, 2), int))]
for k in range(1, 6):
    for j, jn in ((0, "hip"), (1, "knee")):
        scen.append((E.KINDS[k], jn, np.stack([np.full(N, k), np.zeros(N, int)], 1),
                     np.stack([2 * legs + j, -np.ones(N, int)], 1)))
scen.append(("2 dead", "any", dead2, np.stack([2 * legs + rng.integers(0, 2, N), 2 * legs2 + rng.integers(0, 2, N)], 1)))
legs3 = np.stack([rng.permutation(6)[:3] for _ in range(N)])               # three different legs
scen.append(("3 dead", "any", rng.choice([4, 5], (N, 3)), 2 * legs3 + rng.integers(0, 2, (N, 3))))

results = {}
for pname in POLICIES:
    run = R.make_runner(env, R.load_policy(pname, env), goal_idx=GOAL)
    results[pname] = {}
    for name, joint, kinds, servos in scen:
        sizes = np.vectorize(lambda k: R.EVAL_SIZE[int(k)])(kinds).astype(np.float32)
        onsets = np.tile(ONSET + 40 * np.arange(kinds.shape[1]), (N, 1))      # extra faults 2 s apart
        traj = jax.tree_util.tree_map(np.asarray, run(keys, jp.asarray(kinds), jp.asarray(servos),
                                                      jp.asarray(onsets), jp.asarray(sizes)))
        m = R.summarise(traj, env.dt)
        key = f"{name} ({joint})"
        results[pname][key] = {
            "goal_pct": float(m["goal"].mean() * 100), "fell_pct": float(m["fell"].mean() * 100),
            "path_pct": float(m["path_frac"].mean() * 100),
            "t_goal_s": float(np.nanmedian(m["t_goal"])) if m["goal"].any() else None,
            "cross_track_mm": float(np.nanmedian(m["cross_track_mm"])),
        }
        print(f"{pname:12s} {key:18s} goal {results[pname][key]['goal_pct']:5.1f}%  path {results[pname][key]['path_pct']:5.1f}%"
              f"  fell {results[pname][key]['fell_pct']:4.1f}%  xtrack {results[pname][key]['cross_track_mm']:5.1f} mm"
              f"  t_goal {results[pname][key]['t_goal_s']}", flush=True)
        if (name in ("healthy", "limp", "frozen") and joint in ("-", "knee")) or name in ("2 dead", "3 dead"):
            np.savez_compressed(os.path.join(OUT, f"traj_{pname}_{name.replace(' ', '')}_{joint}.npz"),
                                xy=traj["xy"][:16], path=traj["path"][:16], goal_idx=traj["goal_idx"][:16],
                                done=traj["done"][:16], remaining=traj["remaining"][:16],
                                servos=servos[:16], kinds=kinds[:16])

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)

# markdown table: goal reached % (median cross-track mm) per policy
rows = list(results[POLICIES[0]].keys())
with open(os.path.join(OUT, "table.md"), "w") as f:
    f.write("| scenario | " + " | ".join(POLICIES) + " |\n|---|" + "---|" * len(POLICIES) + "\n")
    for r in rows:
        f.write(f"| {r} | " + " | ".join(
            f"{results[p][r]['goal_pct']:.0f}% goal, {results[p][r]['path_pct']:.0f}% path, {results[p][r]['cross_track_mm']:.0f} mm"
            for p in POLICIES) + " |\n")
print(open(os.path.join(OUT, "table.md")).read())

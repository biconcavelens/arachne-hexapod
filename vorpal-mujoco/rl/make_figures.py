"""
Figures for the results report, from eval_final/ and a diagnosis run.

    python make_figures.py eval=eval_final diag=diag_v2_ctx3 out=media/report

Writes goal_by_scenario.png, time_by_scenario.png, paths_3dead.png, diag_by_kind.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
args = dict(a.split("=", 1) for a in sys.argv[1:])
EV = os.path.join(ROOT, "runs", args.get("eval", "eval_final"))
DG = os.path.join(ROOT, "runs", args.get("diag", "diag_v2_ctx3"))
OUT = os.path.join(ROOT, args.get("out", "media/report"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"                   # fault-aware, healthy-only
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POL = [("fault_v2", "trained with faults", BLUE), ("healthy_v2", "trained healthy only", ORANGE)]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6})

res = json.load(open(os.path.join(EV, "results.json")))
singles = [k for k in res["fault_v2"] if not k.startswith(("healthy", "2 dead", "3 dead"))]
groups = [("no fault", ["healthy (-)"]), ("1 faulty servo\n(10 scenarios)", singles),
          ("2 dead servos", ["2 dead (any)"]), ("3 dead servos", ["3 dead (any)"])]


def agg(p, keys, field):
    v = [res[p][k][field] for k in keys if res[p][k][field] is not None]
    return float(np.mean(v)) if v else np.nan


def bars(field, ylabel, fname, fmt):
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    x = np.arange(len(groups))
    w = 0.36
    for j, (p, lab, c) in enumerate(POL):
        v = [agg(p, ks, field) for _, ks in groups]
        b = ax.bar(x + (j - 0.5) * (w + 0.02), v, w, color=c, label=lab, zorder=2)
        for xi, vi in zip(x + (j - 0.5) * (w + 0.02), v):
            ax.text(xi, vi, fmt(vi), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [g for g, _ in groups])
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=170)
    plt.close(fig)


bars("goal_pct", "goal reached within 20 s (%)", "goal_by_scenario.png", lambda v: f"{v:.0f}")
bars("t_goal_s", "median time to goal (s)", "time_by_scenario.png", lambda v: f"{v:.1f}")

# distance still to go over time, 3 dead servos (onsets 1, 3, 5 s); 16 paired test episodes
fig, ax = plt.subplots(figsize=(6.2, 2.9))
for p, lab, c in POL:
    z = np.load(os.path.join(EV, f"traj_{p}_3dead_any.npz"))
    rem = np.where(z["done"], np.nan, z["remaining"]) * 100
    t = np.arange(rem.shape[1]) * 0.05
    for r in rem:
        ax.plot(t, r, color=c, lw=0.6, alpha=0.25)
    ax.plot(t, np.nanmedian(rem, 0), color=c, lw=2.2, label=f"{lab} (median of 16)")
for on in (1, 3, 5):
    ax.axvline(on, color=INK2, lw=0.8, ls=(0, (2, 2)))
ax.text(5.2, 148, "dashed: 1st, 2nd, 3rd servo dies", fontsize=7, color=INK2, va="top")
ax.set_xlabel("time (s)"); ax.set_ylabel("distance to goal (cm)")
ax.set_ylim(0, 152); ax.set_xlim(0, 20)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "progress_3dead.png"), dpi=170)
plt.close(fig)

# diagnosis: right servo per fault kind (while walking)
d = json.load(open(os.path.join(DG, "results.json")))
kinds = list(d["walking_right_servo_by_kind"].keys())
fig, ax = plt.subplots(figsize=(5.2, 2.6))
v = [d["walking_right_servo_by_kind"][k] for k in kinds]
ax.bar(kinds, v, 0.55, color=BLUE, zorder=2)
for i, vi in enumerate(v):
    ax.text(i, vi, f"{vi:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
ax.axhline(100 / 12, color=INK2, lw=1, ls=(0, (3, 2)))
ax.text(len(kinds) - 0.5, 100 / 12 + 2, "chance (1 in 12)", ha="right", fontsize=7, color=INK2)
ax.set_ylabel("faulty servo named correctly (%)")
ax.set_ylim(0, 100)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "diag_by_kind.png"), dpi=170)
plt.close(fig)
print("wrote", sorted(os.listdir(OUT)))

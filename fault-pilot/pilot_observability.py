"""
Observability pilot (SIMULATION ONLY): can a single hobby-servo fault be detected
and localised from the body's own motion?

The robot walks the open-loop tripod gait (servo-web-control/assets/gaits/forward.json)
in the lean MuJoCo twin. At a random moment one servo develops a fault:
  stuck   : it ignores all further commands (dropped commands / stall / dead servo)
  offset  : its angle is shifted by +-8 deg (slipped horn / spline step / backlash)
We record what a body-mounted sensor could see at 20 Hz and try to recover the fault:
  CAM  : body-frame linear + angular velocity, i.e. ego-motion from visual odometry
         (linear part has an unknown per-episode scale error, a monocular-VO property)
  IMU  : specific force + gyro + gravity direction (no translation)
  BOTH : both of the above
Noise is synthetic (level 1 ~ what we measured on the real IMU at rest while handled,
level 3 = 3x). This is an UPPER-BOUND style test of the information in the signal: no
real camera, no sim-to-real gap, one gait, classifier trained and tested on the same
simulator. It says whether the idea is worth building, not that it works.

Detection: unsupervised, trained on healthy windows only (PCA-whitened distance of the
window's residual from the healthy template).  Localisation / type: logistic regression.
One window = one gait cycle (24 steps = 1.2 s), aligned to the commanded gait clock.

    python pilot_observability.py          # ~1-2 min on a many-core machine
"""
import json
import math
import os
import time

import numpy as np
import mujoco
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = r"C:\projects\arachne"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "vorpal-mujoco", "hexapod_mjx.xml")
G = json.load(open(os.path.join(ROOT, "servo-web-control", "assets", "gaits", "forward.json")))
FRAMES, REST, FPS = np.array(G["frames"]), np.array(G["rest"]), G["fps"]
N = len(FRAMES)
ORDER = [k + 6 * j for k in range(6) for j in (0, 1)]     # actuator i (hip0, knee0, hip1, ...) -> index in gait vector
CTRL_DT, SUB = 0.05, 25
T_SETTLE, T_RAMP = 1.0, 0.6
T0 = T_SETTLE + T_RAMP                                     # gait starts here
CYC = int(round(N / FPS / CTRL_DT))                        # 24 control steps per cycle
CYC_S = CYC * CTRL_DT
OFFSET_DEG = 8.0
JOINT = [f"{n}{k}" for k in range(6) for n in ("hip", "knee")]


def pose_deg(t):
    if t < T_SETTLE:
        return REST
    t -= T_SETTLE
    if t < T_RAMP:
        return REST + (FRAMES[0] - REST) * (0.5 - 0.5 * math.cos(math.pi * t / T_RAMP))
    f = ((t - T_RAMP) * FPS) % N
    i = int(f)
    return FRAMES[i] + (FRAMES[(i + 1) % N] - FRAMES[i]) * (f - i)


def episode(seed, fault, T=14.0):
    """fault = None or (actuator index, 'stuck' | 'offset', degrees)."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    m.geom_friction[[m.geom(f"foot{k}").id for k in range(6)], 0] = rng.uniform(0.6, 1.2)
    base = m.body("base").id
    m.body_mass[base] *= rng.uniform(0.95, 1.05)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    t_on = T0 + 3 * CYC_S + rng.uniform(0, CYC_S) if fault else None
    frozen, n = None, int(T / CTRL_DT)
    V, W, F, Gv = (np.zeros((n, 3)) for _ in range(4))
    v_prev = d.qvel[0:3].copy()
    up = 1.0
    for k in range(n):
        for s in range(SUB):
            t = (k * SUB + s) * m.opt.timestep
            u = np.radians(pose_deg(t))[ORDER]
            if fault and t >= t_on:
                a, kind, mag = fault
                if kind == "stuck":
                    frozen = u[a] if frozen is None else frozen
                    u[a] = frozen
                else:
                    u[a] += math.radians(mag)
            d.ctrl[:] = u
            mujoco.mj_step(m, d)
        R = d.xmat[base].reshape(3, 3)
        v = d.qvel[0:3].copy()
        V[k], W[k] = R.T @ v, d.qvel[3:6]
        F[k] = R.T @ ((v - v_prev) / CTRL_DT + np.array([0, 0, 9.81]))
        Gv[k] = R.T @ np.array([0, 0, -1.0])
        v_prev = v
        up = R[2, 2]
        if up < 0.3:
            return None                                    # fell over: drop the episode
    return dict(V=V, W=W, F=F, G=Gv, t_on=t_on, fault=fault, seed=seed)


def windows(ep):
    """-> list of (window arrays dict, label) where label is 'healthy' or 'faulty'."""
    out, n = [], len(ep["V"])
    for c in range(1, 100):
        start = (int(round(T0 / CTRL_DT)) + c * CYC)
        if start + CYC > n:
            break
        t0, t1 = start * CTRL_DT, (start + CYC) * CTRL_DT
        w = {k: ep[k][start:start + CYC] for k in "VWFG"}
        if ep["t_on"] is None or t1 <= ep["t_on"]:
            out.append((w, "healthy"))
        elif t0 >= ep["t_on"] + CYC_S:                     # skip the transition cycle
            out.append((w, "faulty"))
    return out


def feat(w, sensor, level, rng, vscale):
    cam = np.c_[w["V"] * vscale + rng.normal(0, 0.005 * level, (CYC, 3)), w["W"] + rng.normal(0, 0.05 * level, (CYC, 3))]
    imu = np.c_[w["F"] / 9.81 + rng.normal(0, 0.05 * level, (CYC, 3)), w["W"] + rng.normal(0, 0.05 * level, (CYC, 3)),
                w["G"] + rng.normal(0, 0.03 * level, (CYC, 3))]
    x = {"cam": cam, "imu": imu, "both": np.c_[cam, imu]}[sensor]
    return x.ravel()


def build(eps, sensor, level, seed):
    rng = np.random.default_rng(seed)
    X, y, meta = [], [], []
    for ep in eps:
        vscale = rng.lognormal(0, 0.1 * level)             # monocular scale error, fixed per episode
        for w, lab in windows(ep):
            X.append(feat(w, sensor, level, rng, vscale))
            y.append(lab)
            meta.append(ep["fault"])
    return np.array(X), np.array(y), meta


def main():
    t_start = time.time()
    configs = []                                           # (actuator, kind, magnitude)
    for a in range(12):
        for kind, mag in (("offset", +OFFSET_DEG), ("offset", -OFFSET_DEG), ("stuck", 0.0)):
            if kind == "stuck" and a in (0, 6):            # hip0 / hip3 never move in this gait: stuck = no fault
                continue
            configs.append((a, kind, mag))
    REPS = 10
    jobs = [(s, None) for s in range(120)]
    jobs += [(1000 + 100 * i + r, c) for i, c in enumerate(configs) for r in range(REPS)]
    sev = [(50000 + 1000 * j + 10 * a + r, (a, "offset", mg * sg)) for j, mg in enumerate((2, 4, 8)) for a in range(12)
           for sg in (1, -1) for r in range(2)]
    jobs += sev
    res = Parallel(n_jobs=-1)(delayed(episode)(s, f) for s, f in jobs)
    eps = [e for e in res if e is not None]
    print(f"{len(eps)}/{len(jobs)} episodes ok ({len(jobs) - len(eps)} fell), simulated in {time.time() - t_start:.0f} s")

    def is_test(e):
        return (e["seed"] % 10) >= 7
    main_eps = [e for e in eps if e["seed"] < 50000]
    sev_eps = [e for e in eps if e["seed"] >= 50000]
    tr, te = [e for e in main_eps if not is_test(e)], [e for e in main_eps if is_test(e)]

    results = {}
    cm_store = {}
    for level in (1, 3):
        for sensor in ("cam", "imu", "both"):
            Xtr, ytr, mtr = build(tr, sensor, level, 1)
            Xte, yte, mte = build(te, sensor, level, 2)
            htr = Xtr[ytr == "healthy"]
            tmpl = htr.mean(0)
            sc = StandardScaler().fit(htr - tmpl)
            pca = PCA(n_components=20, whiten=True).fit(sc.transform(htr - tmpl))
            score = lambda X: (pca.transform(sc.transform(X - tmpl)) ** 2).sum(1)
            s_te = score(Xte)
            auc = roc_auc_score(yte == "faulty", s_te)
            auc_by = {}
            for kind in ("stuck", "offset"):
                idx = [i for i, f in enumerate(mte) if f is None or f[1] == kind]
                auc_by[kind] = roc_auc_score(yte[idx] == "faulty", s_te[idx])
            # supervised localisation / type on faulty windows
            ftr, fte = ytr == "faulty", yte == "faulty"
            Rtr, Rte = sc.transform(Xtr[ftr] - tmpl), sc.transform(Xte[fte] - tmpl)
            jtr = np.array([f[0] for f, k in zip(mtr, ftr) if k])
            jte = np.array([f[0] for f, k in zip(mte, fte) if k])
            ttr = np.array([f[1] + ("+" if f[2] > 0 else "-" if f[2] < 0 else "") for f, k in zip(mtr, ftr) if k])
            tte = np.array([f[1] + ("+" if f[2] > 0 else "-" if f[2] < 0 else "") for f, k in zip(mte, fte) if k])
            clf = LogisticRegression(max_iter=3000, C=0.05).fit(Rtr, jtr)
            pj = clf.predict(Rte)
            clt = LogisticRegression(max_iter=3000, C=0.05).fit(Rtr, ttr)
            results[f"{sensor}/noise x{level}"] = dict(
                detect_auroc=float(auc), detect_auroc_stuck=float(auc_by["stuck"]), detect_auroc_offset=float(auc_by["offset"]),
                joint_acc=float((pj == jte).mean()), leg_acc=float((pj // 2 == jte // 2).mean()),
                type_acc=float((clt.predict(Rte) == tte).mean()), n_test_windows=int(fte.sum()))
            if sensor == "both" and level == 1:
                cm_store["true"], cm_store["pred"] = jte, pj
            if sensor in ("cam", "both"):                    # severity sweep, detection only
                Xs, ys, ms = build(sev_eps, sensor, level, 3)
                Xh, yh, _ = build([e for e in te if e["fault"] is None], sensor, level, 4)
                for mg in (2, 4, 8):
                    sel = [i for i, f in enumerate(ms) if abs(f[2]) == mg and ys[i] == "faulty"]
                    sc_s = np.r_[score(Xs[sel]), score(Xh)]
                    lab = np.r_[np.ones(len(sel)), np.zeros(len(Xh))]
                    results.setdefault(f"severity/{sensor}/x{level}", {})[f"{mg}deg"] = float(roc_auc_score(lab, sc_s))
            print(f"  {sensor:4s} x{level}: detect AUROC {auc:.3f} | which joint {results[f'{sensor}/noise x{level}']['joint_acc']:.2f}"
                  f" (leg {results[f'{sensor}/noise x{level}']['leg_acc']:.2f}) | type {results[f'{sensor}/noise x{level}']['type_acc']:.2f}")

    # per-joint localisation accuracy (both, x1)
    t, p = cm_store["true"], cm_store["pred"]
    results["per_joint_acc_both_x1"] = {JOINT[j]: float((p[t == j] == j).mean()) for j in range(12) if (t == j).any()}
    json.dump(results, open(os.path.join(HERE, "pilot_results.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    names = ["cam", "imu", "both"]
    for lv, off in ((1, -0.2), (3, 0.2)):
        ax[0].bar(np.arange(3) + off, [results[f"{s}/noise x{lv}"]["joint_acc"] for s in names], 0.4, label=f"noise x{lv}")
    ax[0].axhline(1 / 12, ls=":", c="gray")
    ax[0].set(xticks=range(3), xticklabels=["camera ego-motion", "IMU", "both"], ylim=(0, 1), ylabel="accuracy",
              title="Which of the 12 servos is faulty? (chance = 0.08)")
    ax[0].legend()
    cm = np.zeros((12, 12))
    for a, b in zip(t, p):
        cm[a, b] += 1
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    ax[1].imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax[1].set(xticks=range(12), yticks=range(12), xlabel="predicted servo", ylabel="true servo", title="Confusion (cam + IMU, noise x1)")
    ax[1].set_xticklabels(JOINT, rotation=90, fontsize=7)
    ax[1].set_yticklabels(JOINT, fontsize=7)
    for s, c in (("cam", "C0"), ("both", "C2")):
        for lv, ls in ((1, "-"), (3, "--")):
            r = results[f"severity/{s}/x{lv}"]
            ax[2].plot([2, 4, 8], [r["2deg"], r["4deg"], r["8deg"]], ls, marker="o", c=c, label=f"{s} noise x{lv}")
    ax[2].axhline(0.5, ls=":", c="gray")
    ax[2].set(xlabel="servo angle offset (deg)", ylabel="detection AUROC", ylim=(0.4, 1.02), title="Detecting small offsets")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pilot_results.png"), dpi=130)
    print(f"done in {time.time() - t_start:.0f} s -> pilot_results.json / .png")


if __name__ == "__main__":
    main()

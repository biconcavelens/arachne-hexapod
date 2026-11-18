"""
Diagnosis head: from the policy's own input (1 s of IMU, its past commands and the path
features; no joint sensing) say, for each of the 12 servos, which fault it has
(none, loss, deadband, sag, frozen, limp). Trained supervised on episodes of a trained
policy with random faults, evaluated on held-out episodes.

    bash wslrun.sh python diagnose.py policy=fault_v1 episodes=3072 out=diag_v1

Writes runs/<out>/results.json, diag_params.pkl, confusion.png.
"""
import json
import os
import pickle
import sys
import time

import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np
import optax

import fault_path_env as E
import fault_rollout as R

args = dict(a.split("=", 1) for a in sys.argv[1:])
POLICY = args.get("policy", "fault_v1")
EPIS = int(args.get("episodes", 3072))
OUT = os.path.join(R.ROOT, "runs", args.get("out", "diag"))
os.makedirs(OUT, exist_ok=True)
CHUNK, SUB = 512, 4                     # envs per rollout batch; keep every 4th step for training
CTX = int(args.get("ctx", 1))           # seconds of history: the policy's 1 s input at t, t-1 s, ...
LAGS = np.arange(CTX) * E.HIST
FULL = E.HIST * CTX                     # only windows whose whole history is past the reset

env = E.FaultPath()
GOAL = int(args["goal"]) if "goal" in args else None    # e.g. 190: long walk so faults start mid-walk
run = R.make_runner(env, R.load_policy(POLICY, env), record_obs=True, random_faults=True, goal_idx=GOAL)

t0 = time.time()
TT = env._config.episode_length
X = np.empty((EPIS, TT, E.HIST * E.OBS_STEP), np.float16)                    # filled in place: RAM is tight
Y = np.empty((EPIS, TT, 12), np.int8)
DONE = np.empty((EPIS, TT), bool)
REM = np.empty((EPIS, TT), np.float32)
ONS = np.empty((EPIS, 12), np.int32)
KND = np.empty((EPIS, 12), np.int32)
for c in range(EPIS // CHUNK):
    keys = jax.random.split(jax.random.PRNGKey(10_000 + c), CHUNK)
    z = jp.zeros((CHUNK, 1))
    tr = run(keys, z.astype(jp.int32), z.astype(jp.int32) - 1, z.astype(jp.int32), z)
    sl = slice(c * CHUNK, (c + 1) * CHUNK)
    X[sl] = np.asarray(tr["obs"].astype(jp.float16)); Y[sl] = np.asarray(tr["label"]); DONE[sl] = np.asarray(tr["done"])
    ONS[sl] = np.asarray(tr["fault_onset"]); KND[sl] = np.asarray(tr["fault_kind"]); REM[sl] = np.asarray(tr["remaining"])
    del tr
    print(f"rollouts {(c+1)*CHUNK}/{EPIS}  {time.time()-t0:.0f}s", flush=True)
WALK = REM > 0.03                                                              # still walking to the goal
N, TT = Y.shape[:2]
n_tr = int(N * 0.8)
valid = ~DONE                                                                  # drop steps after a fall
valid[:, :FULL] = False



def stack(Xe, t):
    """Xe (T, D) one episode, t (M,) step indices -> (M, D*CTX): the 1 s windows at t, t-1 s, ..."""
    return np.concatenate([Xe[np.maximum(t - l, 0)] for l in LAGS], -1)


def windows(sl, sub):
    """(episode, step) index pairs of usable training windows; stacked lazily per batch to save RAM."""
    e, t = np.nonzero(valid[sl, ::sub])
    return e + sl.start, t * sub


def gather(e, t):
    return np.concatenate([X[e, np.maximum(t - l, 0)] for l in LAGS], -1), Y[e, t]


Etr, Ttr = windows(slice(0, n_tr), SUB)
sub_i = np.random.default_rng(0).choice(len(Etr), min(60000, len(Etr)), replace=False)
Xs, Ytr_s = gather(Etr[sub_i], Ttr[sub_i])
mean = Xs.astype(np.float32).mean(0)
std = Xs.astype(np.float32).std(0) + 1e-3
D = Xs.shape[-1]
del Xs


class Diag(nn.Module):
    @nn.compact
    def __call__(self, x, train=False):
        x = (x - mean) / std
        for h in (512, 512, 256):
            x = nn.gelu(nn.Dense(h)(x))
            x = nn.Dropout(0.1, deterministic=not train)(x)
        return nn.Dense(12 * E.NK)(x).reshape(x.shape[:-1] + (12, E.NK))


Ytr = Y[Etr, Ttr]
print("train windows", len(Etr), "faulty fraction", float((Ytr > 0).any(1).mean()), flush=True)
# class weights: faults are rare per servo
cnt = np.bincount(Ytr.reshape(-1), minlength=E.NK).astype(np.float64)
w = jp.asarray(np.clip((cnt.sum() / (E.NK * np.maximum(cnt, 1))) ** 0.5, 0.2, 10.0), jp.float32)

model = Diag()
params = model.init(jax.random.PRNGKey(0), jp.zeros((1, D)))
opt = optax.adamw(1e-3, weight_decay=1e-4)
ost = opt.init(params)


@jax.jit
def train_step(params, ost, x, y, key):
    def loss_fn(p):
        lg = model.apply(p, x, train=True, rngs={"dropout": key})
        ce = optax.softmax_cross_entropy_with_integer_labels(lg, y)
        return (ce * w[y]).mean()
    l, g = jax.value_and_grad(loss_fn)(params)
    up, ost = opt.update(g, ost, params)
    return optax.apply_updates(params, up), ost, l


predict = jax.jit(lambda p, x: jax.nn.softmax(model.apply(p, x), -1))
BS, EPOCHS = 2048, int(args.get("epochs", 12))
key = jax.random.PRNGKey(1)
for ep in range(EPOCHS):
    perm = np.random.default_rng(ep).permutation(len(Etr))
    ls = []
    for i in range(0, len(perm) - BS + 1, BS):
        b = perm[i:i + BS]
        key, k = jax.random.split(key)
        xb, yb = gather(Etr[b], Ttr[b])
        params, ost, l = train_step(params, ost, jp.asarray(xb, jp.float32), jp.asarray(yb, jp.int32), k)
        ls.append(float(l))
    print(f"epoch {ep}: loss {np.mean(ls):.4f}  {time.time()-t0:.0f}s", flush=True)
pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "mean": mean, "std": std, "ctx": CTX},
            open(os.path.join(OUT, "diag_params.pkl"), "wb"))

# ------------------------------------------------------------------ held-out evaluation, every step
P = np.zeros((N - n_tr, TT, 12, E.NK), np.float32)
for i in range(n_tr, N):
    P[i - n_tr] = np.asarray(predict(params, jp.asarray(stack(X[i], np.arange(TT)), jp.float32)))
Yv, Vv = Y[n_tr:], valid[n_tr:]
score = 1.0 - P[..., 0].min(-1)                        # P(some servo is faulty), roughly
faulty = (Yv > 0).any(-1)


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


res = {"policy": POLICY, "episodes": int(N), "heldout_episodes": int(N - n_tr), "ctx_s": CTX, "goal_idx": GOAL}
res["detect_auroc"] = auroc(score[Vv & faulty], score[Vv & ~faulty])

# windows with exactly one faulty servo, at least 1 s after its onset (history fully faulty)
t_idx = np.arange(TT)[None, :]
ons_v = np.where(KND[n_tr:] > 0, ONS[n_tr:], 10_000).min(1)[:, None]
one = Vv & ((Yv > 0).sum(-1) == 1) & (t_idx >= ons_v + E.HIST)
true_servo = (Yv > 0).argmax(-1)
pred_servo = (1 - P[..., 0]).argmax(-1)
true_kind = Yv.max(-1)
pk = P[np.arange(P.shape[0])[:, None], np.arange(TT)[None, :], true_servo][..., 1:].argmax(-1) + 1
res["right_servo_pct"] = float((pred_servo == true_servo)[one].mean() * 100)
res["right_kind_on_servo_pct"] = float((pk == true_kind)[one].mean() * 100)
corr = np.asarray(E.CORRECTABLE)
res["correctable_vs_dead_pct"] = float((corr[pk] == corr[true_kind])[one].mean() * 100)
wk = one & WALK[n_tr:]
res["walking_right_servo_pct"] = float((pred_servo == true_servo)[wk].mean() * 100)
res["walking_right_kind_on_servo_pct"] = float((pk == true_kind)[wk].mean() * 100)
res["walking_correctable_vs_dead_pct"] = float((corr[pk] == corr[true_kind])[wk].mean() * 100)
res["walking_detect_auroc"] = auroc(score[Vv & faulty & WALK[n_tr:]], score[Vv & ~faulty & WALK[n_tr:]])
res["walking_right_servo_by_kind"] = {E.KINDS[k]: float((pred_servo == true_servo)[wk & (true_kind == k)].mean() * 100)
                                      for k in range(1, E.NK)}
res["right_servo_by_kind"] = {E.KINDS[k]: float((pred_servo == true_servo)[one & (true_kind == k)].mean() * 100)
                              for k in range(1, E.NK)}
res["right_servo_by_joint"] = {j: float((pred_servo == true_servo)[one & (true_servo % 2 == jj)].mean() * 100)
                               for jj, j in ((0, "hip"), (1, "knee"))}
conf = np.zeros((E.NK - 1, E.NK - 1), int)
for a_, b_ in zip(true_kind[one], pk[one]):
    conf[a_ - 1, b_ - 1] += 1
res["kind_confusion"] = conf.tolist()

# alarm: score above a threshold for 3 consecutive steps; threshold = 99.5th percentile on healthy windows
healthy_scores = score[Vv & ~faulty]
thr = float(np.quantile(healthy_scores, 0.995))
above = (score > thr) & Vv
run3 = above & np.roll(above, 1, 1) & np.roll(above, 2, 1)
run3[:, :2] = False
pre = Vv & ~faulty
minutes = pre.sum() * env.dt / 60
starts = run3 & ~np.roll(run3, 1, 1)
res["false_alarms_per_min"] = float((starts & pre).sum() / minutes)
delays = []
for i in range(N - n_tr):
    o = int(ons_v[i, 0])
    if o >= TT:
        continue
    hit = np.nonzero(run3[i, o:])[0]
    delays.append(hit[0] * env.dt if len(hit) else np.nan)
delays = np.array(delays)
res["detected_pct"] = float(np.isfinite(delays).mean() * 100)
res["median_delay_s"] = float(np.nanmedian(delays))
res["threshold"] = thr
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4.2, 3.6))
cn = conf / np.maximum(conf.sum(1, keepdims=True), 1)
ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
for a_ in range(E.NK - 1):
    for b_ in range(E.NK - 1):
        ax.text(b_, a_, f"{cn[a_, b_]*100:.0f}", ha="center", va="center", color="white" if cn[a_, b_] > 0.5 else "black")
ax.set_xticks(range(E.NK - 1), E.KINDS[1:], rotation=30); ax.set_yticks(range(E.NK - 1), E.KINDS[1:])
ax.set_xlabel("predicted kind"); ax.set_ylabel("true kind"); ax.set_title("Fault kind on the faulty servo (%)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "confusion.png"), dpi=150)
print("done", time.time() - t0)

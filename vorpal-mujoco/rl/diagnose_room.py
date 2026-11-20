"""
Which servos are faulty? A per-servo detector trained on rollouts of a room policy.

The input is what the robot itself knows whatever sensors the policy uses: the IMU (gravity
direction, gyro) and the 12 commands it last sent, every 0.2 s over the last 2 s. (Until 2026-11-20 it
was the policy's own observation at 0, 1 and 2 s; with camera pixels in it the detector did no better
than chance on terrain.) The output is, for each of the 12 servos,
the probability that it is faulty now. The kind of fault is not asked for: the policy deals with the
consequence whatever the kind, and the diagnosis only has to say where.

    bash wslrun.sh python diagnose_room.py policy=room_both [episodes=1536] [out=diag_room_both]

Metrics (held-out episodes, steps after a fall excluded):
  detect_auroc        any servo faulty vs none, from max_i p_i
  alarm               max_i p_i above the 99.5th percentile of healthy steps for 3 steps in a row:
                      detected %, median delay after onset, false alarms per minute of healthy walking
  right_servo_pct     exactly one faulty servo, >= 1 s after onset: argmax_i p_i is that servo
  all_found_pct       k faulty servos (k = 1-3), >= 1 s after the last onset: the top-k are exactly them
  by_kind             right servo % per fault kind (dead, stuck, signal loss, worn gears)
Writes runs/<out>/results.json and diag_params.pkl.
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

import fault_rollout as R
import scenes
import room_env as E

args = dict(a.split("=", 1) for a in sys.argv[1:])
POLICY = args.get("policy", "room_both")
EPIS = int(args.get("episodes", 1536))
OUT = os.path.join(R.ROOT, "runs", args.get("out", f"diag_{POLICY}"))
os.makedirs(OUT, exist_ok=True)
CHUNK = int(args.get("chunk", 128))
KEEP = 2                                    # store every 2nd control step (0.1 s)
LAGS = np.arange(0, 21, 2)                  # stored steps: now, 0.2 s, ... 2 s ago
FULL = 20 // KEEP

cfg = json.load(open(os.path.join(R.ROOT, "runs", POLICY, "config.json")))
scene = args.get("env", cfg.get("env", "room"))
cls = scenes.env_class(scene)
env = cls(config_overrides={"sensors": cfg["config"]["sensors"]})
run = R.make_runner(env, R.load_policy(POLICY, env), record_obs=True, random_faults=True)
T = env._config.episode_length
TS = T // KEEP
D = 18                                      # gravity 3, gyro 3, last commands 12
print(f"{POLICY}: scene {scene}, sensors {cfg['config']['sensors']}, obs {D}, {EPIS} episodes", flush=True)

t0 = time.time()
X = np.empty((EPIS, TS, D), np.float16)
Y = np.empty((EPIS, TS, 12), bool)
DONE = np.empty((EPIS, TS), bool)
KND = np.empty((EPIS, 12), np.int32)
ONS = np.empty((EPIS, 12), np.int32)
for c in range(EPIS // CHUNK):
    k = jax.random.split(jax.random.PRNGKey(20_000 + c), CHUNK)
    z = jp.zeros((CHUNK, 1))
    tr = run(k, z.astype(jp.int32), z.astype(jp.int32) - 1, z.astype(jp.int32), z)
    sl = slice(c * CHUNK, (c + 1) * CHUNK)
    X[sl] = np.asarray(tr["diag_x"][:, ::KEEP].astype(jp.float16))
    Y[sl] = np.asarray(tr["label"][:, ::KEEP] > 0)
    DONE[sl] = np.asarray(tr["done"][:, ::KEEP])
    KND[sl], ONS[sl] = np.asarray(tr["fault_kind"]), np.asarray(tr["fault_onset"])
    del tr
    print(f"rollouts {(c + 1) * CHUNK}/{EPIS}  {time.time() - t0:.0f}s", flush=True)

n_tr = int(EPIS * 0.8)
valid = ~DONE
valid[:, :FULL] = False


def gather(e, t):
    return np.concatenate([X[e, np.maximum(t - l, 0)] for l in LAGS], -1)


Etr, Ttr = np.nonzero(valid[:n_tr])
sub = np.random.default_rng(0).choice(len(Etr), min(50_000, len(Etr)), replace=False)
Xs = gather(Etr[sub], Ttr[sub]).astype(np.float32)
mean, std = Xs.mean(0), Xs.std(0) + 1e-3
del Xs
pos = Y[:n_tr][valid[:n_tr]].mean(0)
pw = jp.asarray(np.clip((1 - pos) / np.maximum(pos, 1e-3), 1, 20) ** 0.5, jp.float32)   # rarer -> heavier


class Diag(nn.Module):
    @nn.compact
    def __call__(self, x, train=False):
        x = (x - mean) / std
        for h in (512, 512, 256):
            x = nn.gelu(nn.Dense(h)(x))
            x = nn.Dropout(0.1, deterministic=not train)(x)
        return nn.Dense(12)(x)


model = Diag()
params = model.init(jax.random.PRNGKey(0), jp.zeros((1, D * len(LAGS))))
opt = optax.adamw(1e-3, weight_decay=1e-4)
ost = opt.init(params)


@jax.jit
def train_step(params, ost, x, y, key):
    def loss_fn(p):
        lg = model.apply(p, x, train=True, rngs={"dropout": key})
        l = optax.sigmoid_binary_cross_entropy(lg, y)
        return (l * jp.where(y > 0, pw, 1.0)).mean()
    l, g = jax.value_and_grad(loss_fn)(params)
    up, ost = opt.update(g, ost, params)
    return optax.apply_updates(params, up), ost, l


predict = jax.jit(lambda p, x: jax.nn.sigmoid(model.apply(p, x)))
BS, EPOCHS = 2048, int(args.get("epochs", 10))
key = jax.random.PRNGKey(1)
for ep in range(EPOCHS):
    perm = np.random.default_rng(ep).permutation(len(Etr))
    ls = []
    for i in range(0, len(perm) - BS + 1, BS):
        b = perm[i:i + BS]
        key, kk = jax.random.split(key)
        params, ost, l = train_step(params, ost, jp.asarray(gather(Etr[b], Ttr[b]), jp.float32),
                                    jp.asarray(Y[Etr[b], Ttr[b]], jp.float32), kk)
        ls.append(float(l))
    print(f"epoch {ep}: loss {np.mean(ls):.4f}  {time.time() - t0:.0f}s", flush=True)
pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "mean": mean, "std": std, "lags": LAGS,
             "keep": KEEP, "policy": POLICY, "input": "imu_cmd"}, open(os.path.join(OUT, "diag_params.pkl"), "wb"))

# ------------------------------------------------------------------ held-out evaluation
NV = EPIS - n_tr
P = np.zeros((NV, TS, 12), np.float32)
for i in range(NV):
    P[i] = np.asarray(predict(params, jp.asarray(gather(np.full(TS, n_tr + i), np.arange(TS)), jp.float32)))
Yv, Vv, Kv, Ov = Y[n_tr:], valid[n_tr:], KND[n_tr:], ONS[n_tr:] // KEEP
score = P.max(-1)
faulty = Yv.any(-1)


def auroc(a, b):
    s = np.concatenate([a, b]); r = s.argsort().argsort() + 1
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


res = {"policy": POLICY, "scene": scene, "sensors": cfg["config"]["sensors"], "episodes": EPIS, "heldout": NV}
res["detect_auroc"] = auroc(score[Vv & faulty], score[Vv & ~faulty])
t_idx = np.arange(TS)[None]
nf = Yv.sum(-1)
last_on = np.where(Kv > 0, Ov, -1).max(1)[:, None]
settled = Vv & (t_idx >= last_on + 10)                          # >= 1 s after the last onset
one = settled & (nf == 1)
true1 = Yv.argmax(-1)
res["right_servo_pct"] = float((P.argmax(-1) == true1)[one].mean() * 100)
order = np.argsort(-P, -1)
for k in (1, 2, 3):
    m = settled & (nf == k)
    if m.any():
        topk = np.zeros_like(Yv)
        np.put_along_axis(topk, order[..., :k], True, -1)
        res[f"all_found_{k}_pct"] = float((topk == Yv).all(-1)[m].mean() * 100)
kind_t = np.take_along_axis(np.broadcast_to(Kv[:, None, :], Yv.shape), true1[..., None], -1)[..., 0]
res["right_servo_by_kind"] = {nm: float((P.argmax(-1) == true1)[one & (kind_t == k)].mean() * 100)
                              for k, nm in ((1, "dead"), (2, "stuck"), (3, "signal loss"), (4, "worn gears"))
                              if (one & (kind_t == k)).any()}
thr = float(np.quantile(score[Vv & ~faulty], 0.995))
above = (score > thr) & Vv
run3 = above & np.roll(above, 1, 1) & np.roll(above, 2, 1)
run3[:, :2] = False
pre = Vv & ~faulty
starts = run3 & ~np.roll(run3, 1, 1)
res["false_alarms_per_min"] = float((starts & pre).sum() / (pre.sum() * env.dt * KEEP / 60))
first_on = np.where(Kv > 0, Ov, 10 ** 6).min(1)
delays = []
for i in range(NV):
    o = int(first_on[i])
    if o >= TS:
        continue
    o = max(o, FULL)
    hit = np.nonzero(run3[i, o:])[0]
    delays.append(hit[0] * env.dt * KEEP if len(hit) else np.nan)
delays = np.array(delays)
res["detected_pct"] = float(np.isfinite(delays).mean() * 100)
res["median_delay_s"] = float(np.nanmedian(delays))
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
print("done", time.time() - t0)

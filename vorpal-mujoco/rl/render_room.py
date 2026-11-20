"""
GIF of one room episode: chase view, overhead mini-map with the path and the trail, the body camera,
the injected faults, and (with diag=) the live diagnosis of which servos are faulty.

    bash wslrun.sh python render_room.py policy=room_both kind=dead,stuck servo=knee1,hip4 onset=40 \
        seed=3 [diag=diag_room_both] [env=room_dyn] [seconds=25] out=media/room_dead_knee1.gif

kind: dead | stuck | signal | worn (several: comma list, the k-th fault starts at onset + 40k steps);
kind=none for a healthy run. onset=0 means the faults are there from the start.
"""
import json
import os
import pickle
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import jax
import jax.numpy as jp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fault_rollout as R
import scenes
import room_env as E
import room_view as V

args = dict(a.split("=", 1) for a in sys.argv[1:])
POLICY = args.get("policy", "room_both")
cfg = json.load(open(os.path.join(R.ROOT, "runs", POLICY, "config.json")))
scene = args.get("env", cfg.get("env", "room"))
cls = scenes.env_class(scene)
env = cls(config_overrides={"sensors": cfg["config"]["sensors"]})
KIND_ID = {"none": 0, "dead": 1, "stuck": 2, "signal": 3, "worn": 4}
SIZE = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.7, 4: float(np.radians(15))}
kinds = [KIND_ID[k] for k in args.get("kind", "dead").split(",")]
servos = [int(v[-1]) * 2 + (0 if v.startswith("hip") else 1) for v in args.get("servo", "knee1").split(",")]
if kinds == [0]:
    servos = [-1]
kinds = (kinds * len(servos))[:len(servos)]
ONSET = int(args.get("onset", 40))
onsets = [ONSET + 40 * k for k in range(len(servos))]
STEPS = int(float(args.get("seconds", 25)) / env.dt)
OUT = os.path.join(R.ROOT, args.get("out", "media/room_episode.gif"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)

run = R.make_runner(env, R.load_policy(POLICY, env), steps=STEPS, record_obs=True, record_mocap=True)
key = jax.random.PRNGKey(int(args.get("seed", 0)))
tr = run(key[None], jp.array([kinds]), jp.array([servos]), jp.array([onsets]),
         jp.array([[SIZE[k] for k in kinds]], jp.float32))
tr = jax.tree_util.tree_map(lambda a: np.asarray(a)[0], tr)

probs = None
if "diag" in args:
    import flax.linen as nn
    z = pickle.load(open(os.path.join(R.ROOT, "runs", args["diag"], "diag_params.pkl"), "rb"))

    class Diag(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = (x - z["mean"]) / z["std"]
            for h in (512, 512, 256):
                x = nn.gelu(nn.Dense(h)(x))
            return nn.Dense(12)(x)
    keep, lags = int(z["keep"]), np.asarray(z["lags"]) * int(z["keep"])
    t = np.arange(STEPS)
    src = tr["diag_x"] if z.get("input") == "imu_cmd" else tr["obs"]
    x = np.concatenate([src[np.maximum(t - l, 0)] for l in lags], -1)
    probs = np.asarray(jax.nn.sigmoid(Diag().apply(z["params"], jp.asarray(x, jp.float32))))

W, H = 720, 460
vs = scenes.view_scene(scene)
big, mini, ego = V.RoomView(W, H, vs), V.RoomView(200, 200, vs), V.RoomView(240, 160, vs)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    small = ImageFont.truetype("DejaVuSans.ttf", 12)
except OSError:
    font = small = ImageFont.load_default()
names = [f"{['front', 'front-left', 'rear-left', 'rear', 'rear-right', 'front-right'][s // 2]} {'hip' if s % 2 == 0 else 'knee'}"
         for s in range(12)]
path = tr["path"][:int(tr["goal_idx"]) + 1]
KN = {1: "dead", 2: "stuck", 3: "signal loss", 4: "worn gears"}
frames = []
end = STEPS
reached = np.nonzero(tr["remaining"] < 0.03)[0]
if len(reached):
    end = min(STEPS, reached[0] + 30)
for t in range(0, end, 2):
    mp, mq = tr["mocap_pos"][t], tr["mocap_quat"][t]
    red = {s // 2 for s, on in zip(servos, onsets) if s >= 0 and t >= on}
    trail = tr["xy"][:t + 1]
    for v in (big, mini, ego):
        v.set_state(tr["qpos"][t], mp, mq)
    img = Image.fromarray(big.render("chase", path=path, trail=trail[-300:], red_legs=red))
    img.paste(Image.fromarray(mini.render("top", path=path, trail=trail, red_legs=red)), (W - 206, 34))
    img.paste(Image.fromarray(ego.render("ego")), (6, 34))
    dr = ImageDraw.Draw(img)
    dr.rectangle([5, 33, 247, 195], outline=(255, 255, 255), width=2)
    dr.rectangle([W - 207, 33, W - 5, 235], outline=(255, 255, 255), width=2)
    dr.rectangle([5, 196, 100, 214], fill=(0, 0, 0))
    dr.text((10, 198), "body camera", fill=(255, 255, 255), font=small)
    act = [f"{names[s]} {KN[k]}" for s, k, on in zip(servos, kinds, onsets) if s >= 0 and t >= on]
    dr.rectangle([0, 0, W, 26], fill=(0, 0, 0))
    dr.text((8, 4), f"t = {t * env.dt:4.1f} s   faults: {', '.join(act) if act else 'none'}", fill=(255, 255, 255), font=font)
    rem = max(0.0, float(tr["remaining"][t]))
    dr.rectangle([W - 206, 238, W - 6, 276], fill=(0, 0, 0))
    dr.text((W - 200, 241), f"to goal  {rem * 100:5.1f} cm", fill=(255, 255, 255), font=small)
    dr.text((W - 200, 257), f"off path {float(tr['cross_track'][t]) * 1000:5.0f} mm", fill=(255, 255, 255), font=small)
    if probs is not None:
        p = probs[t]
        flagged = [i for i in np.argsort(-p)[:3] if p[i] >= 0.5]
        txt = "diagnosis: all servos working" if not flagged else \
            "diagnosis: faulty " + ", ".join(f"{names[i]} ({p[i] * 100:.0f}%)" for i in flagged)
        dr.rectangle([0, H - 26, W, H], fill=(0, 0, 0))
        dr.text((8, H - 22), txt, fill=(120, 255, 120) if not flagged else (255, 150, 90), font=font)
    frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=160))
frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
print("wrote", OUT, len(frames), "frames;", "ended early (fell / off path / bumped)" if tr["done"][-1] else "",
      f"reached goal at {reached[0] * env.dt:.1f} s" if len(reached) else "goal not reached")

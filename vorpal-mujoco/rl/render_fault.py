"""
Render one episode as a GIF: the full-mesh robot following its path, the faulty leg in
red after the fault starts, and (if a diagnosis model is given) its live read-out.

    bash wslrun.sh python render_fault.py policy=fault_v1 diag=diag_v1 kind=limp servo=knee2 \
        onset=60 seed=3 out=media/limp_knee2.gif [seconds=20]

The episode runs in the MJX training model; the recorded poses are replayed in
hexapod_physics.xml (meshes) for the picture.
"""
import os
import pickle
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import fault_path_env as E
import fault_rollout as R

args = dict(a.split("=", 1) for a in sys.argv[1:])
POLICY = args.get("policy", "fault_v1")
# several faults: kind=limp,frozen servo=knee2,hip4 (the k-th fault starts at onset + 40k steps)
KINDS_ = [E.KINDS.index(k) for k in args.get("kind", "limp").split(",")]
SERVOS = [int(v[-1]) * 2 + (0 if v.startswith("hip") else 1) for v in args.get("servo", "knee2").split(",")]
if KINDS_ == [0]:
    SERVOS = [-1]
KINDS_ = (KINDS_ * len(SERVOS))[:len(SERVOS)]
ONSET = int(args.get("onset", 60))
ONSETS = [ONSET + 40 * k for k in range(len(SERVOS))]
SEED = int(args.get("seed", 0))
OUT = os.path.join(R.ROOT, args.get("out", "media/episode.gif"))
STEPS = int(float(args.get("seconds", 20)) / E.C.CTRL_DT)
W, H = 640, 400
os.makedirs(os.path.dirname(OUT), exist_ok=True)

env = E.FaultPath()
GOAL = int(args["goal"]) if "goal" in args else None    # path index, e.g. 190 = 1.9 m
run = R.make_runner(env, R.load_policy(POLICY, env), steps=STEPS, record_obs=True, goal_idx=GOAL)
tr = run(jax.random.PRNGKey(SEED)[None], jp.array([KINDS_]), jp.array([SERVOS]), jp.array([ONSETS]),
         jp.array([[R.EVAL_SIZE[k] for k in KINDS_]], jp.float32))
tr = jax.tree_util.tree_map(lambda a: np.asarray(a)[0], tr)

diag = None
if "diag" in args:
    import diagnose_model as DM
    diag = DM.load(os.path.join(R.ROOT, "runs", args["diag"], "diag_params.pkl"))
    probs = diag(tr["obs"])                                          # (T, 12, NK)

ROOT_XML = os.path.join(os.path.dirname(R.ROOT), "hexapod_physics.xml")
m = mujoco.MjModel.from_xml_path(ROOT_XML)
d = mujoco.MjData(m)
m.vis.headlight.ambient[:] = 0.45
m.vis.headlight.diffuse[:] = 0.65
ren = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 0.85, -45, 120

# geoms of each faulty leg (descendants of that leg's hip body)
leg_geoms = []
for sv_ in SERVOS:
    gs = []
    if sv_ >= 0:
        hip_body = m.jnt_bodyid[m.joint(f"hip{sv_ // 2}").id]
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            while b not in (0, hip_body):
                b = m.body_parentid[b]
            if b == hip_body:
                gs.append(g)
    leg_geoms.append(gs)
rgba0 = m.geom_rgba.copy()

try:
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    small = ImageFont.truetype("DejaVuSans.ttf", 12)
except OSError:
    font = small = ImageFont.load_default()

path, goal = tr["path"], int(tr["goal_idx"])
names = [f"{j}{k}" for k in range(6) for j in ("hip", "knee")]
frames = []
for t in range(0, STEPS, 2):
    d.qpos[:] = tr["qpos"][t]
    d.qpos[2] += 0.002
    mujoco.mj_forward(m, d)
    m.geom_rgba[:] = rgba0
    for gs, sv_, on in zip(leg_geoms, SERVOS, ONSETS):
        if sv_ >= 0 and t >= on:
            for g in gs:
                m.geom_rgba[g] = [0.9, 0.15, 0.1, 1]
    cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.03]
    ren.update_scene(d, cam)
    sc = ren.scene
    for i in range(0, goal + 1, 2):                                  # path dots, goal ring
        if sc.ngeom >= sc.maxgeom - 2:
            break
        mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.003, 0, 0]),
                            np.array([path[i, 0], path[i, 1], 0.002]), np.eye(3).reshape(-1),
                            np.array([0.1, 0.6, 1.0, 1.0], np.float32))
        sc.ngeom += 1
    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.03, 0.001, 0]),
                        np.array([path[goal, 0], path[goal, 1], 0.001]), np.eye(3).reshape(-1),
                        np.array([0.1, 0.85, 0.3, 0.8], np.float32))
    sc.ngeom += 1
    img = Image.fromarray(ren.render())
    dr = ImageDraw.Draw(img)
    act = [f"{names[sv_]} {E.KINDS[k]}" for sv_, k, on in zip(SERVOS, KINDS_, ONSETS) if sv_ >= 0 and t >= on]
    fault_txt = ", ".join(act) if act else "none yet"
    dr.rectangle([0, 0, W, 26], fill=(0, 0, 0))
    dr.text((8, 4), f"t = {t*E.C.CTRL_DT:4.1f} s   injected fault: {fault_txt}", fill=(255, 255, 255), font=font)
    if diag is not None:
        p = probs[t]
        bad = 1 - p[:, 0]
        flagged = [k for k in np.argsort(-bad)[:3] if bad[k] >= 0.5]
        ok = not flagged
        parts = []
        for k in flagged:
            kk = int(p[k, 1:].argmax()) + 1
            parts.append(f"{names[k]} {E.KINDS[kk]} ({bad[k]*100:.0f}%)")
        txt = "diagnosis: all servos healthy" if ok else "diagnosis: " + ", ".join(parts)
        dr.rectangle([0, H - 26, W, H], fill=(0, 0, 0))
        dr.text((8, H - 22), txt, fill=(120, 255, 120) if ok else (255, 140, 90), font=font)
    rem = max(0.0, float(tr["remaining"][t]))
    dr.rectangle([W - 150, 30, W - 6, 66], fill=(0, 0, 0))
    dr.text((W - 144, 33), f"to goal  {rem*100:5.1f} cm", fill=(255, 255, 255), font=small)
    dr.text((W - 144, 49), f"off path {float(tr['cross_track'][t])*1000:5.0f} mm", fill=(255, 255, 255), font=small)
    frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
print("wrote", OUT, len(frames), "frames;", "fell" if tr["done"][-1] else "",
      "reached goal" if (tr["remaining"] < 0.03).any() else "goal not reached")

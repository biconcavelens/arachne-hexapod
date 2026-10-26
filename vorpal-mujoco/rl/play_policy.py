"""
Run an exported policy (runs/<name>/policy.npz) in the FULL physics model
(hexapod_physics.xml: meshes, hinge/knee ground contact) with plain MuJoCo +
NumPy, the same inference the Uno Q will do. Sim-to-sim check of the lean
training model, and a GIF.

    python play_policy.py name=imu cmd=0.06,0,0 seconds=20 [video=1] [noise=0]
"""
import math
import os
import sys

import mujoco
import numpy as np

import common as C

ROOT = os.path.dirname(os.path.abspath(__file__))
args = dict(a.split("=", 1) for a in sys.argv[1:])
NAME = args.get("name", "imu")
CMD = np.array([float(v) for v in args.get("cmd", "0.06,0,0").split(",")])
SECONDS = float(args.get("seconds", 20))
NOISE = float(args.get("noise", 0))
VIDEO = args.get("video", "0") == "1"
USE_IMU = args.get("imu", "1") == "1"


class Policy:
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.std = z["obs_mean"], z["obs_std"]
        n = sum(k.startswith("W") for k in z.files)
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]

    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = x / (1.0 + np.exp(-x))                 # swish
        return np.tanh(x[:C.ACT_CENTER.size])              # first half of the output = mean


def run(policy, cmd, seconds, noise=0.0, use_imu=True, video=False, rng=np.random.default_rng(0)):
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "..", "hexapod_physics.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    base = m.body("base").id
    floor = m.geom("floor").id
    legs = {m.geom(f"leg{k}").id for k in range(6)}
    last, dragging, log, frames = C.STAND_ACTION.copy(), set(), [], []
    ren = mujoco.Renderer(m, height=360, width=560) if video else None
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.55, 125, -22
    n = int(seconds / C.CTRL_DT)
    for step in range(n):
        R = d.xmat[base].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.qvel[3:6].copy()
        if noise:
            grav = grav + rng.uniform(-1, 1, 3) * 0.03 * noise
            gyro = gyro + rng.uniform(-1, 1, 3) * 0.1 * noise
        if not use_imu:
            grav, gyro = np.zeros(3), np.zeros(3)
        a = policy(C.build_obs(grav, gyro, cmd, last, step))
        d.ctrl[:] = C.action_to_target(a)
        for _ in range(C.N_SUBSTEPS):
            mujoco.mj_step(m, d)
        last = a
        for c in d.contact[:d.ncon]:
            other = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
            if other is not None and other not in legs:
                dragging.add(m.geom(other).name)
        q = d.qpos[3:7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        up = R[2, 2]
        log.append((step * C.CTRL_DT, *d.qpos[:3], yaw, up, *(R.T @ d.qvel[:3])))
        if up < 0.5:
            break                                          # fell
        if video and step % 1 == 0:
            cam.lookat[:] = [d.xpos[base][0], d.xpos[base][1], 0.05]
            ren.update_scene(d, camera=cam)
            frames.append(ren.render())
    return np.array(log), dragging, frames


def summarise(log, cmd, dragging, skip_s=3.0):
    t = log[:, 0]
    w = log[t >= skip_s]
    if len(w) < 5:
        return {"fell": True, "survived_s": float(t[-1])}
    dur = w[-1, 0] - w[0, 0]
    dyaw = (w[-1, 4] - w[0, 4] + math.pi) % (2 * math.pi) - math.pi
    vloc = w[:, 6:9].mean(axis=0)
    return {"fell": t[-1] < SECONDS - 2 * C.CTRL_DT, "survived_s": float(t[-1]),
            "v_forward_mm_s": vloc[0] * 1000, "v_left_mm_s": vloc[1] * 1000,
            "yaw_rate_deg_s": math.degrees(dyaw) / dur, "cmd": cmd,
            "tilt_max_deg": float(np.degrees(np.arccos(np.clip(log[:, 5].min(), -1, 1)))),
            "height_mm": float(w[:, 3].mean() * 1000), "non_foot_contacts": sorted(dragging) or "none"}


if __name__ == "__main__":
    pol = Policy(os.path.join(ROOT, "runs", NAME, "policy.npz"))
    log, dragging, frames = run(pol, CMD, SECONDS, NOISE, USE_IMU, VIDEO)
    s = summarise(log, CMD, dragging)
    print(f"policy {NAME}, command fwd/left/yaw = {CMD}")
    for k, v in s.items():
        print(f"  {k}: {v if not isinstance(v, float) else round(v, 2)}")
    if VIDEO and frames:
        from PIL import Image
        out = os.path.join(ROOT, "runs", NAME, "play.gif")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(C.CTRL_DT * 1000), loop=0, optimize=True)
        print("  video:", out)

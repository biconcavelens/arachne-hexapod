"""
Run a gait open-loop in the physics model and report how well it walks.

Plays the exact gait file the web page plays (servo-web-control/assets/gaits/
<name>.json, sim joint angles), through torque-limited position servos, and
logs forward distance, drift, heading, tilt and anything other than a foot
touching the ground. Writes a following-camera GIF.

Usage: python sim_walk.py [gait=forward] [seconds=8] [speed=1.0] [steer=0]
  steer: gaits.steered() trim, + turns right; ~-0.15 holds heading in physics
"""
import json
import math
import os
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
args = dict(a.split("=", 1) for a in sys.argv[1:])
GAIT_NAME = args.get("gait", "forward")
SECONDS = float(args.get("seconds", 8))
SPEED = float(args.get("speed", 1.0))
STEER = float(args.get("steer", 0.0))
RAMP_S = 0.6
VIDEO_FPS = 20

gait = json.load(open(os.path.join(ROOT, "..", "servo-web-control", "assets", "gaits", f"{GAIT_NAME}.json")))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "hexapod_physics.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)

# gait pose [hip0..5, knee0..5] (deg) -> ctrl in actuator order (rad)
act_order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
idx = {j: i for i, j in enumerate(gait["joints"])}
to_ctrl = lambda pose: np.radians([pose[idx[a]] for a in act_order])

n = len(gait["frames"])
frames = [np.array(f) for f in gait["frames"]]
rest = np.array(gait["rest"])
# steer trim, same as gaits.steered(): side legs' hip swing x (1 + steer * side)
steer_scale = np.array([1 + STEER * sd for sd in gait["hip_side"]] + [1.0] * 6)


def pose_at(t):
    """Time -> commanded gait pose: settle, ramp rest->frame0, cycle, (no stop)."""
    if t < 1.0:
        return rest
    t -= 1.0
    if t < RAMP_S:
        u = 0.5 - 0.5 * math.cos(math.pi * t / RAMP_S)
        return rest + (frames[0] - rest) * u
    f = ((t - RAMP_S) * gait["fps"] * SPEED) % n
    i = int(f)
    return (frames[i] + (frames[(i + 1) % n] - frames[i]) * (f - i)) * steer_scale


base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
leg_geoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"leg{k}") for k in range(6)}
VIDEO = args.get("video", "1") == "1"
renderer = mujoco.Renderer(m, height=360, width=560) if VIDEO else None
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 0.55, 125, -22

images, log = [], []
dragging = set()
steps = int(SECONDS / m.opt.timestep)
frame_every = int(1 / (VIDEO_FPS * m.opt.timestep))
start_xy = None
for s in range(steps):
    t = s * m.opt.timestep
    d.ctrl[:] = to_ctrl(pose_at(t))
    mujoco.mj_step(m, d)
    if start_xy is None and t >= 1.0:
        start_xy = d.qpos[:2].copy(), d.qpos[3:7].copy()
    for c in d.contact[:d.ncon]:
        other = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else None
        if other is not None and other not in leg_geoms:
            dragging.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other))
    if s % frame_every == 0:
        q = d.qpos[3:7]
        roll = math.degrees(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        pitch = math.degrees(math.asin(max(-1, min(1, 2 * (q[0] * q[2] - q[3] * q[1])))))
        yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        log.append((t, *d.qpos[:3], roll, pitch, yaw))
        if VIDEO:
            cam.lookat[:] = [d.xpos[base][0], d.xpos[base][1], 0.05]
            renderer.update_scene(d, camera=cam)
            images.append(Image.fromarray(renderer.render()))

L = np.array(log)
walk = L[L[:, 0] >= 1.0 + RAMP_S]
dur = walk[-1, 0] - walk[0, 0]
dx, dy = walk[-1, 1] - walk[0, 1], walk[-1, 2] - walk[0, 2]
dyaw = (walk[-1, 6] - walk[0, 6] + 180) % 360 - 180
print(f"gait '{gait['name']}', speed x{SPEED}, steer {STEER:+.3f}, walking for {dur:.1f}s")
print(f"  forward (x): {dx*1000:+.0f} mm  ->  {dx/dur*1000:+.1f} mm/s")
print(f"  sideways (y): {dy*1000:+.0f} mm, heading change {dyaw:+.1f} deg")
print(f"  body height: min {walk[:, 3].min()*1000:.0f} mm, max {walk[:, 3].max()*1000:.0f} mm")
print(f"  tilt: max |roll| {np.abs(walk[:, 4]).max():.1f} deg, max |pitch| {np.abs(walk[:, 5]).max():.1f} deg")
print(f"  non-foot ground contact: {sorted(dragging) or 'none'}")
print(f"RESULT dx_mm={dx*1000:.1f} dy_mm={dy*1000:.1f} dyaw_deg={dyaw:.2f}")
if not VIDEO:
    sys.exit()
out = os.path.join(ROOT, f"sim_{GAIT_NAME}.gif")
images[0].save(out, save_all=True, append_images=images[1::], duration=int(1000 / VIDEO_FPS), loop=0, optimize=True)
print(f"  video: {out} ({len(images)} frames)")

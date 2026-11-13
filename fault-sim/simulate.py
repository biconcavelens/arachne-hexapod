"""
One simulated episode: gait -> servo faults -> physics -> sensors -> features and labels.

egomotion="flow": render the body camera, optical flow, plane ego-motion (Stage A).
                  Needs an OpenGL context (MUJOCO_GL=egl on a headless GPU machine).
egomotion="gt":   no rendering. The true body twist plus noise and a per-episode scale
                  error (as in fault-pilot), and the flow grid predicted from it. Runs anywhere.

Per control step (20 Hz) the episode records:
  feat  (38)  ego twist (6) + flow grid (32)          what Stage B sees from the camera
  imu   (9)   gyro, gravity direction, specific force (g)
  cmd   (12)  commanded angles (deg), what the gait asked for
  target(12)  what the servos held after the faults (deg)
  q     (12)  executed joint angles (rad), for the auxiliary head
  phase (2)   sin, cos of the gait phase
  types (12)  fault type id per servo (0 none, 1 loss, 2 sag, 3 deadband), sizes (12)
  pose  (3)   base x, y (m) and yaw (rad), for walking metrics
"""
import copy
import math

import mujoco
import numpy as np

import config as C
import environment
from egomotion import PlaneEgomotion, StageA, camera_axes_in_body, grid_from_twist, intrinsics
from faults import ServoBank
from gaits_lib import Gait

BODY_HEIGHT = 0.095          # base height when standing (lean model, measured in build_mjx.py)
_BASE_MODEL = None


def base_model():
    global _BASE_MODEL
    if _BASE_MODEL is None:
        _BASE_MODEL = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    return _BASE_MODEL


def floor_texture(rng, w, h):
    """Rug-like procedural texture: noise at several scales, random colour and contrast."""
    import cv2
    img = np.zeros((h, w), np.float32)
    for cells in (4, 16, 64, 256):
        if cells <= w:
            g = rng.random((cells, cells)).astype(np.float32)
            interp = cv2.INTER_LINEAR if cells < 64 else cv2.INTER_NEAREST
            img += cv2.resize(g, (w, h), interpolation=interp) * rng.uniform(0.2, 1.0)
    img = (img - img.min()) / (np.ptp(img) + 1e-6)
    img = img ** rng.uniform(0.6, 1.6)
    colour = rng.uniform(0.25, 0.9, 3)
    return (np.clip(img[..., None] * colour * rng.uniform(0.7, 1.3) + rng.uniform(0, 0.15), 0, 1) * 255).astype(np.uint8)


def randomize(m, rng):
    """Per-episode physics and appearance randomisation (in place)."""
    feet = [m.geom(f"foot{k}").id for k in range(6)]
    m.geom_friction[feet, 0] = rng.uniform(0.6, 1.2)
    m.body_mass[m.body("base").id] *= rng.uniform(0.95, 1.05)
    kp = m.actuator_gainprm[:, 0] * rng.uniform(0.8, 1.2, m.nu)          # servo strength varies per unit
    m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
    if C.STAGE:                                       # room, lights (and objects): environment.py
        environment.randomize(m, rng)
    else:
        t = m.texture("floor_tex").id
        w, h, adr = m.tex_width[t], m.tex_height[t], m.tex_adr[t]
        nch = int(m.tex_nchannel[t]) if hasattr(m, "tex_nchannel") else 3
        tex = floor_texture(rng, w, h)
        if nch == 4:
            tex = np.concatenate([tex, np.full((h, w, 1), 255, np.uint8)], axis=-1)
        m.tex_data[adr:adr + w * h * nch] = tex.reshape(-1)
        m.light_diffuse[:] *= rng.uniform(0.6, 1.2)
    cam = m.camera("ego").id
    m.cam_pos[cam] += rng.normal(0, C.CAM_POS_JITTER, 3)
    axis = rng.normal(size=3)
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), math.radians(rng.normal(0, C.CAM_ROT_JITTER_DEG)))
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, m.cam_quat[cam].copy(), dq)
    m.cam_quat[cam] = q


def movable_servos(gait):
    return gait.travel() >= C.MIN_SERVO_TRAVEL_DEG


def run_episode(seed, faults, gait=None, seconds=C.EPISODE_S, egomotion="gt", flow_method="dis",
                controller=None, bank=None, randomize_scene=True, keep_frames=False, on_step=None):
    """Simulate one episode. controller (optional) supplies commands and sees every step (closed loop).
    on_step(m, d, t) (optional) is called after each control step's physics, e.g. to render videos."""
    rng = np.random.default_rng([seed, 2])
    m = copy.deepcopy(base_model())
    if randomize_scene:
        randomize(m, rng)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    gait = gait or Gait.random(rng)
    bank = bank or ServoBank(faults, np.random.default_rng([seed, 3]))   # own stream: same noise with or without faults
    pose = C.camera_pose()
    cam_h = BODY_HEIGHT + pose["pos_m"][2] + C.CAP_Z_OFFSET
    render = egomotion == "flow" or keep_frames
    renderer = mujoco.Renderer(m, C.IMG_H, C.IMG_W) if render else None
    stage_a = StageA(pose, cam_h, flow_method) if egomotion == "flow" else None
    pe_gt = PlaneEgomotion(intrinsics(), camera_axes_in_body(pose), pose["pos_m"], cam_h)
    scale_err = rng.lognormal(0, C.GT_SCALE_SD)
    if controller is not None:
        controller.attach(gait, bank, rng)

    n = int(round(seconds / C.CTRL_DT))
    rec = {k: np.zeros((n, w), np.float32) for k, w in
           (("feat", C.FEAT_EGO), ("imu", C.FEAT_IMU), ("cmd", 12), ("target", 12), ("q", 12),
            ("phase", 2), ("sizes", 12), ("pose", 3), ("twist_true", 6))}
    rec["types"] = np.zeros((n, 12), np.int64)
    frames = []
    base = m.body("base").id
    v_prev = d.qvel[0:3].copy()
    for k in range(n):
        t = k * C.CTRL_DT
        cmd = controller.command(t) if controller is not None else gait.command(t)
        target = bank.step(cmd, t)
        d.ctrl[:] = np.radians(target)
        for _ in range(C.N_SUBSTEPS):
            mujoco.mj_step(m, d)
        if on_step is not None:
            on_step(m, d, t)

        R = d.xmat[base].reshape(3, 3)
        v = d.qvel[0:3].copy()
        w_b = d.qvel[3:6].copy()
        twist = np.r_[R.T @ v, w_b]
        grav = R.T @ np.array([0, 0, -1.0])
        acc = R.T @ ((v - v_prev) / C.CTRL_DT + np.array([0, 0, 9.81])) / 9.81
        v_prev = v
        imu = np.r_[w_b + rng.normal(0, C.GYRO_NOISE, 3), grav + rng.normal(0, C.GRAV_NOISE, 3),
                    acc + rng.normal(0, C.ACC_NOISE_G, 3)]

        gray = None
        if render:
            renderer.update_scene(d, camera="ego")
            gray = renderer.render().mean(axis=2).astype(np.uint8)
            if keep_frames:
                frames.append(gray)
        if egomotion == "flow":
            feat = stage_a.update(gray)
        else:
            noisy = np.r_[twist[:3] * scale_err + rng.normal(0, C.GT_LIN_NOISE, 3), twist[3:] + rng.normal(0, C.GT_ANG_NOISE, 3)]
            feat = np.r_[noisy, grid_from_twist(pe_gt, noisy, C.CTRL_DT)]

        ph = 2 * math.pi * (controller.phase(t) if controller is not None else gait.phase(t))
        types, sizes = bank.labels(t)
        q4 = d.qpos[3:7]
        yaw = math.atan2(2 * (q4[0] * q4[3] + q4[1] * q4[2]), 1 - 2 * (q4[2] ** 2 + q4[3] ** 2))
        row = dict(feat=feat, imu=imu, cmd=cmd, target=target, q=d.qpos[7:], phase=(math.sin(ph), math.cos(ph)),
                   sizes=sizes, pose=(d.qpos[0], d.qpos[1], yaw), twist_true=twist)
        for key, val in row.items():
            rec[key][k] = val
        rec["types"][k] = types
        if controller is not None:
            controller.observe(t, {key: rec[key][k] for key in ("feat", "imu", "cmd", "phase")})
        if R[2, 2] < 0.3:                      # fell over: stop and keep only what happened before
            rec["fell_at"] = np.array(t)
            for key in [k_ for k_, v_ in rec.items() if isinstance(v_, np.ndarray) and v_.ndim and len(v_) == n]:
                rec[key] = rec[key][:k + 1]
            break
    if renderer is not None:
        renderer.close()
    if keep_frames:
        rec["frames"] = np.array(frames)
    return rec

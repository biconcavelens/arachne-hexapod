"""
Fault-tolerant path following in a furnished room, with the body camera (PLAN.md, stage "room").

Each episode: a random path from the robot (straight pieces and arcs, 2 m), a goal 1.0-1.9 m along
it, and the room's 23 household objects placed anew around it. Furniture keeps a clearance from the
path (the path is what a person drew around it on the overhead view); rugs and cables may lie across
it. Walking into a piece of furniture ends the episode like a fall.

Faults (0-3 servos per episode, each present from the start or starting at any time):
  1 limp      dead: no torque (signal or power lost)
  2 frozen    stuck: holds the angle it had when it failed
  3 loss      intermittent signal: each command is dropped with probability p (keeps the old one)
  4 deadband  worn gears: command changes smaller than db are ignored
The policy is never told which servo failed or how; it must keep to the path with what still works.
The diagnosis head (diagnose_room.py) names the faulty servos, not the kind.

Sensors (config.sensors): "imu" (gravity + gyro), "cam" (body camera), "both". Every variant also
gets its last commands and the path features (on the real robot: the overhead camera that tracks the
robot and the drawn path). No sensor noise for now (config.noise = 0).

Body camera: a 24 x 16 grey image ray-cast from the camera-cap lens pose (mjx.ray; MJX's own renderer
needs Warp and cannot run inside the vmapped training loop). Grey level = object shade x a
procedural texture of the hit point x distance falloff, so the floor and walls carry texture and the
image moves when the robot does. The policy sees the frames from now, 0.05 s and 0.2 s ago.

Reward: following the path first (tracking term, episode ends beyond 15 cm off the path). Speed:
any speed at or above half the target earns most of the progress reward, so a robot slowed down by a
dead servo is not punished for it; below half the target the reward falls off.
"""
import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env

import common as C
import room_scene as RS
from fault_path_env import DS, LOOKAHEAD, N_PATH, make_path

if not os.path.exists(RS.OUT):
    RS.build()
XML = RS.OUT
FOOT_R = 0.005

KINDS = ("none", "limp", "frozen", "loss", "deadband")
NK = len(KINDS)
HIST = 20                                              # control steps of history (1 s)
CAM_FRAMES = (0, 1, 4)                                 # camera frames given to the policy: now, -0.05 s, -0.2 s
CAM_BUF = max(CAM_FRAMES) + 1
HEAD_LOOK = 10                                         # path points (cm) ahead used for the heading target
NPIX = RS.CAM_W * RS.CAM_H
ROBOT_R = 0.12                                         # m, body + legs, for bumping into furniture

# object footprints (m) for the bump check; None = may be walked over (rugs, cables)
FOOT = {"table": 0.55, "chair": 0.25, "sofa": 0.95, "shelf": 0.45, "cardbox": 0.22, "bin": 0.13,
        "ball": 0.1, "plant": 0.12, "lamp": 0.15, "book": 0.12, "shoe": 0.14, "rug": None, "cable": None}


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=C.CTRL_DT,
      sim_dt=C.SIM_DT,
      episode_length=600,                              # 30 s: a dead servo may halve the speed
      v_des=0.12,                                      # m/s target speed along the path
      sensors="both",                                  # imu | cam | both
      noise=0.0,                                       # no sensor noise until the real robot is measured
      path_config=config_dict.create(kappa_max=3.0, p_straight=0.4, goal_min=1.0, goal_max=1.9),
      room_config=config_dict.create(p_present=0.75, clearance=0.2, p_cross=0.5, candidates=16),
      fault_config=config_dict.create(
          p_count=[0.15, 0.3, 0.3, 0.25],              # 0, 1, 2, 3 faulty servos
          p_kind=[0.35, 0.35, 0.15, 0.15],             # limp, frozen, loss, deadband
          p_from_start=0.3,                            # a fault is there before the robot sets off
          onset_max=400,                               # otherwise it starts uniformly in 0-20 s
          loss_p=[0.3, 0.9], deadband_deg=[5.0, 20.0],
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              progress=1.5, cross_track=2.0, at_goal=2.0,
              heading=1.0,                             # face along the path: the body camera looks where it walks
              orientation=-2.0, ang_vel_xy=-0.02, yaw_rate=-0.05,
              action_rate=-0.02, energy=-0.01, feet_slip=-0.5, feet_air_time=0.5,
          ),
          sigma_track=0.02,                            # m
          sigma_heading=0.35,                          # rad (20 deg)
          slow_frac=0.5,                               # speeds >= this fraction of v_des are not penalised
          air_time_min=0.15,
          termination=-5.0,
      ),
      max_cross_track=0.15,
      start_yaw_spread=1.5708,                         # start heading: path direction +- this (rad)
      impl="jax",
  )


def sample_faults(rng, cfg):
  fc = cfg.fault_config
  k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
  nmax = len(fc.p_count) - 1
  n = jax.random.choice(k1, nmax + 1, p=jp.array(fc.p_count))
  order = jax.random.permutation(k2, 12)
  chosen = jp.zeros(12, bool).at[order[:nmax]].set(jp.arange(nmax) < n)
  kind = jax.random.choice(k3, jp.arange(1, NK), (12,), p=jp.array(fc.p_kind)) * chosen
  u = jax.random.uniform(k4, (12,))
  lo = jp.array([0.0, 0.0, 0.0, fc.loss_p[0], jp.radians(fc.deadband_deg[0])])
  hi = jp.array([0.0, 0.0, 0.0, fc.loss_p[1], jp.radians(fc.deadband_deg[1])])
  size = lo[kind] + u * (hi[kind] - lo[kind])
  onset = jax.random.randint(k5, (12,), 0, fc.onset_max)
  onset = jp.where(jax.random.uniform(k6, (12,)) < fc.p_from_start, 0, onset)
  return kind.astype(jp.int32), size, onset


class FaultRoom(mjx_env.MjxEnv):
  XML = XML                                      # room_dyn_env overrides it (scene with movers)
  OBJECTS = RS.OBJECTS                           # (kind, clearance, body name); terrain_env has its own
  FOOTPRINT = FOOT
  WORLD_LIM = 1e3                                # objects are kept within +-WORLD_LIM (terrain edge)
  START_LIFT = 0.0                               # extra start height (terrain: clear of stones)

  def __init__(self, config: config_dict.ConfigDict = default_config(),
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_path(self.XML)
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    m = self._mj_model
    self._init_q = jp.array(m.keyframe("stand").qpos)
    self._default_pose = jp.array(C.STAND_QPOS)
    self._base_id = m.body("base").id
    self._foot_ids = jp.array([m.geom(f"foot{k}").id for k in range(6)])
    self._center, self._half = jp.array(C.ACT_CENTER), jp.array(C.ACT_HALF)
    self._look = jp.array(LOOKAHEAD)
    self._cam_id = m.camera("ego").id
    self._floor_id = m.geom("floor").id
    # camera rays in the camera frame (MuJoCo cameras look along -z)
    th = np.tan(np.radians(RS.FOVY_DEG / 2))
    u = (np.arange(RS.CAM_W) + 0.5) / RS.CAM_W * 2 - 1
    v = (np.arange(RS.CAM_H) + 0.5) / RS.CAM_H * 2 - 1
    uu, vv = np.meshgrid(u * th * RS.CAM_W / RS.CAM_H, -v * th)
    d = np.stack([uu, vv, -np.ones_like(uu)], -1).reshape(-1, 3)
    self._rays = jp.array(d / np.linalg.norm(d, axis=1, keepdims=True))
    rgba = m.geom_rgba
    self._lum = jp.array(0.3 * rgba[:, 0] + 0.59 * rgba[:, 1] + 0.11 * rgba[:, 2])
    # objects: mocap index, footprint, clearance from the path
    kinds = [k for k, _, _ in self.OBJECTS]
    self._n_obj = len(kinds)
    assert self._n_obj <= m.nmocap                      # extra mocap bodies (movers) come after the objects
    mocap_bodies = [b for b in range(m.nbody) if m.body_mocapid[b] >= 0]
    self._mocap_pos0 = jp.array(m.body_pos[mocap_bodies])
    self._mocap_quat0 = jp.array(m.body_quat[mocap_bodies])
    self._obj_foot = jp.array([self.FOOTPRINT[k] or 0.0 for k in kinds])
    self._obj_cross = jp.array([self.FOOTPRINT[k] is None for k in kinds])
    self._obj_clear = self._obj_foot + self._config.room_config.clearance + ROBOT_R * 0.5
    s = self._config.sensors
    self._use_imu, self._use_cam = s in ("imu", "both"), s in ("cam", "both")
    self._obs_step = (6 if self._use_imu else 0) + 12 + 8 + 2

  # ---------------------------------------------------------------- helpers
  def _rot(self, data):
    return data.xmat[self._base_id].reshape(3, 3)

  def _gravity(self, data):
    return self._rot(data).T @ jp.array([0.0, 0.0, -1.0])

  def _yaw(self, data):
    r = self._rot(data)
    return jp.arctan2(r[1, 0], r[0, 0])

  def _ground(self, xy):
    """Ground height under points xy (..., 2); flat floor here, terrain_env overrides."""
    return jp.zeros(xy.shape[:-1])

  def _start(self, rng):
    """Where the robot starts (x, y); the room starts at the origin, terrain_env picks a spot."""
    return jp.zeros(2)

  def _ray_extra(self, o, dirs, dist, gid):
    """Hook for surfaces mjx.ray cannot hit (terrain_env: the heightfield)."""
    return dist, gid

  def _floor_texture(self, p, info):
    along = p[:, 0] * info["floor_dir"][0] + p[:, 1] * info["floor_dir"][1]
    across = -p[:, 0] * info["floor_dir"][1] + p[:, 1] * info["floor_dir"][0]
    plank = jp.floor(across / info["floor_period"])
    seam = (jp.abs(across / info["floor_period"] - plank - 0.5) > 0.46)
    grain = 0.88 + 0.12 * jp.sin(2 * jp.pi * along / 0.07 + 3.7 * plank) * jp.sin(1.3 * plank + 0.5)
    return jp.where(seam, 0.55, grain) * (0.85 + 0.15 * jp.sin(5.1 * plank))

  def _miss_shade(self, dirs):
    return jp.zeros(dirs.shape[0])

  def _feet(self, data):
    p = data.geom_xpos[self._foot_ids]
    return p, (p[:, 2] - FOOT_R - self._ground(p[:, :2])) < 0.0015

  def _track(self, data, info):
    pts, goal = info["path"], info["goal_idx"]
    xy = data.qpos[0:2]
    cand = jp.clip(info["path_idx"] + jp.arange(-5, 25), 0, goal)
    i = cand[jp.argmin(jp.sum(jp.square(pts[cand] - xy), -1))]
    th = info["path_theta"][i]
    t = jp.array([jp.cos(th), jp.sin(th)])
    rel = xy - pts[i]
    s = jp.clip(i * DS + jp.dot(rel, t), 0.0, goal * DS)
    d = t[0] * rel[1] - t[1] * rel[0]
    yaw = self._yaw(data)
    c, sn = jp.cos(yaw), jp.sin(yaw)
    ahead = pts[jp.clip(i + self._look, 0, goal)] - xy
    body = jp.stack([c * ahead[:, 0] + sn * ahead[:, 1], -sn * ahead[:, 0] + c * ahead[:, 1]], -1)
    remaining = goal * DS - s
    feats = jp.concatenate([body.reshape(-1) * 10.0, jp.array([d * 10.0, jp.minimum(remaining, 0.3) * 10.0])])
    return i, s, d, remaining, feats

  def _heading_err(self, data, info):
    """Body heading (where the camera looks) minus the path direction 10 cm ahead, wrapped to +-pi."""
    th = info["path_theta"][jp.clip(info["path_idx"] + HEAD_LOOK, 0, info["goal_idx"])]
    e = self._yaw(data) - th
    return jp.arctan2(jp.sin(e), jp.cos(e))

  def camera(self, data, info):
    """Grey 24 x 16 body-camera image in [0, 1] (row-major, top row first)."""
    o = data.cam_xpos[self._cam_id]
    R = data.cam_xmat[self._cam_id].reshape(3, 3)
    dirs = self._rays @ R.T
    ray = lambda v: mjx.ray(self._mjx_model, data, o, v, geomgroup=(1, 0, 0, 0, 0, 0))
    dist, gid = jax.lax.map(jax.vmap(ray), dirs.reshape(4, -1, 3))
    dist, gid = dist.reshape(-1), gid.reshape(-1)
    dist, gid = self._ray_extra(o, dirs, dist, gid)
    hit = gid >= 0
    dist = jp.where(hit, dist, 10.0)
    p = o + dist[:, None] * dirs
    g = jp.maximum(gid, 0)
    # texture: floor planks along a random direction + fine grain; everything else a 5 cm checker
    floor_tex = self._floor_texture(p, info)
    cell = jp.floor(p / 0.05).sum(-1)
    other_tex = 0.9 + 0.1 * (jp.mod(cell, 2) * 2 - 1)
    tex = jp.where(g == self._floor_id, floor_tex, other_tex)
    img = self._lum[g] * info["tint"][g] * tex * info["bright"] / (1 + 0.15 * dist)
    return jp.clip(jp.where(hit, img, self._miss_shade(dirs)), 0.0, 1.0)

  def _place_objects(self, rng, pts, goal_idx, center=None):
    """Mocap positions and yaws: furniture off the path, rugs/cables sometimes across it."""
    rc = self._config.room_config
    n, K = self._n_obj, rc.candidates
    k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
    lim = RS.ROOM_HALF - 0.35
    cand = jax.random.uniform(k1, (n, K, 2), minval=-lim, maxval=lim)
    if center is not None:
      cand = cand + center
    cand = jp.clip(cand, -self.WORLD_LIM, self.WORLD_LIM)
    used = jp.arange(N_PATH) <= goal_idx + 15
    def place(carry, i):                       # one object at a time, apart from those already placed
      xy, r = carry
      # distance to the path per object inside the scan: an (n, K, N_PATH) reduce outside it made the
      # terrain reset take > 7 GB and many minutes to compile on the GPU
      d2 = jp.sum(jp.square(cand[i][:, None, :] - pts[None]), -1)               # (K, N_PATH)
      ok_path = jp.sqrt(jp.min(jp.where(used, d2, 1e6), -1)) > self._obj_clear[i]
      gap = jp.linalg.norm(cand[i][:, None] - xy[None], axis=-1) - r[None] - self._obj_foot[i]
      ok = ok_path & jp.all(gap > 0.05, -1)
      j = jp.argmax(ok)
      good = ok.any()
      xy = xy.at[i].set(jp.where(good, cand[i, j], xy[i]))
      r = r.at[i].set(jp.where(good, self._obj_foot[i], 0.0))
      return (xy, r), good

    far = jp.stack([50.0 + jp.arange(n), jp.zeros(n)], -1)
    (pos, _), placed = jax.lax.scan(place, (far, jp.zeros(n)), jp.arange(n))
    # rugs / cables: half of them across the path at a random point along it
    on_path = pts[jax.random.randint(k2, (n,), 10, jp.maximum(goal_idx, 11))]
    cross = self._obj_cross & (jax.random.uniform(k3, (n,)) < rc.p_cross)
    pos = jp.where(cross[:, None], on_path, pos)
    placed = placed | self._obj_cross
    present = placed & (jax.random.uniform(k4, (n,)) < rc.p_present)
    park = jp.stack([50.0 + jp.arange(n), jp.zeros(n)], -1)
    pos = jp.where(present[:, None], pos, park)
    z = jp.where(present, self._ground(pos), -1.0)
    yaw = jax.random.uniform(k5, (n,), minval=-jp.pi, maxval=jp.pi)
    quat = jp.stack([jp.cos(yaw / 2), jp.zeros(n), jp.zeros(n), jp.sin(yaw / 2)], -1)
    return jp.concatenate([pos, z[:, None]], -1), quat, present

  # ---------------------------------------------------------------- reset / step
  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 8)
    qpos = self._init_q
    pts, theta, goal_idx = make_path(k3, self._config)
    spread = self._config.start_yaw_spread
    yaw = theta[0] + jax.random.uniform(k1, (), minval=-spread, maxval=spread)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)))
    qpos = qpos.at[0:2].add(jax.random.uniform(k4, (2,), minval=-0.01, maxval=0.01))
    qpos = qpos.at[7:].add(jax.random.uniform(k2, (12,), minval=-0.05, maxval=0.05))
    start = self._start(jax.random.fold_in(k3, 11))
    pts = pts + start
    qpos = qpos.at[0:2].add(start)
    qpos = qpos.at[2].add(self._ground(qpos[None, 0:2])[0] + self.START_LIFT)
    mpos, mquat, present = self._place_objects(k6, pts, goal_idx, center=start)
    data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=jp.zeros(self.mjx_model.nv), ctrl=qpos[7:],
                             impl=self.mjx_model.impl.value)
    data = data.replace(mocap_pos=self._mocap_pos0.at[:self._n_obj].set(mpos),
                        mocap_quat=self._mocap_quat0.at[:self._n_obj].set(mquat))
    data = mjx.forward(self.mjx_model, data)
    kind, size, onset = sample_faults(k5, self._config)
    feet, contact = self._feet(data)
    kt = jax.random.split(k7, 4)
    a = jax.random.uniform(kt[0], (), minval=-jp.pi, maxval=jp.pi)
    info = {
        "rng": rng,
        "t": jp.zeros((), jp.int32),
        "path": pts, "path_theta": theta, "goal_idx": goal_idx, "path_idx": jp.zeros((), jp.int32),
        "s": jp.zeros(()), "reached": jp.zeros((), bool),
        "obj_xy": mpos[:, :2], "obj_present": present,
        "fault_kind": kind, "fault_size": size, "fault_onset": onset,
        "held": qpos[7:], "anchor": qpos[7:], "was_active": jp.zeros(12, bool),
        "last_act": jp.asarray(C.STAND_ACTION),
        "feet_air_time": jp.zeros(6), "last_contact": contact, "last_foot_xy": feet[:, :2],
        "hist": jp.zeros((HIST, self._obs_step)),
        "tint": jax.random.uniform(kt[1], (self._mj_model.ngeom,), minval=0.6, maxval=1.4),
        "floor_dir": jp.array([jp.cos(a), jp.sin(a)]),
        "floor_period": jax.random.uniform(kt[2], (), minval=0.08, maxval=0.2),
        "bright": jax.random.uniform(kt[3], (), minval=0.7, maxval=1.3),
    }
    img = self.camera(data, info) if self._use_cam else jp.zeros(NPIX)
    info["cam"] = jp.broadcast_to(img, (CAM_BUF, NPIX))
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    metrics.update({"cross_track": jp.zeros(()), "heading_err": jp.zeros(()), "progress": jp.zeros(()), "at_goal": jp.zeros(()),
                    "success": jp.zeros(()), "bumped": jp.zeros(()), "faulty": jp.zeros(())})
    i, s, d, rem, feats = self._track(data, info)
    info["path_idx"], info["s"] = i, s
    obs = self._get_obs(data, info, feats)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  def fault_active(self, info):
    return (info["fault_kind"] > 0) & (info["t"] >= info["fault_onset"])

  def apply_faults(self, info, target, rng):
    """Command -> the target each servo actually holds. Returns (held, limp mask, info)."""
    kind, size = info["fault_kind"], info["fault_size"]
    active = self.fault_active(info)
    anchor = jp.where(active & ~info["was_active"], info["held"], info["anchor"])
    held = info["held"]
    drop = jax.random.uniform(rng, (12,)) < size
    by_kind = jp.stack([
        target,                                                        # none
        target,                                                        # limp (handled in the substeps)
        anchor,                                                        # frozen
        jp.where(drop, held, target),                                  # loss
        jp.where(jp.abs(target - held) > size, target, held),          # deadband
    ])
    new = jp.where(active, by_kind[kind, jp.arange(12)], target)
    info["anchor"], info["was_active"] = anchor, active
    return new, active & (kind == 1), info

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = state.info
    action = jp.clip(action, -1.0, 1.0)
    info["rng"], kf = jax.random.split(info["rng"])
    held, limp, info = self.apply_faults(info, self._center + self._half * action, kf)

    def sub(data, _):
      ctrl = jp.where(limp, data.qpos[7:], held)
      return mjx.step(self.mjx_model, data.replace(ctrl=ctrl)), None
    data = jax.lax.scan(sub, state.data, (), self.n_substeps)[0]
    info["held"] = jp.where(limp, data.qpos[7:], held)

    feet, contact = self._feet(data)
    first_contact = (info["feet_air_time"] > 0.0) * (contact | info["last_contact"])
    info["feet_air_time"] = info["feet_air_time"] + self.dt
    foot_vel_xy = (feet[:, :2] - info["last_foot_xy"]) / self.dt

    s_old = info["s"]
    i, s, d, rem, feats = self._track(data, info)
    info["path_idx"], info["s"] = i, s
    head_err = self._heading_err(data, info)
    grav = self._gravity(data)
    gyro = data.qvel[3:6]
    gap = jp.linalg.norm(info["obj_xy"] - data.qpos[None, 0:2], axis=-1) - self._obj_foot - ROBOT_R
    bumped = jp.any(info["obj_present"] & ~self._obj_cross & (gap < 0)) | self._extra_bump(data, info)
    clear = data.qpos[2] - self._ground(data.qpos[None, 0:2])[0]
    done = (grav[2] > -0.5) | (clear < 0.045) | (jp.abs(d) > self._config.max_cross_track) | bumped

    rc = self._config.reward_config
    v_des = self._config.v_des
    at_goal = rem < 0.03
    first_reach = at_goal & ~info["reached"]
    info["reached"] = info["reached"] | at_goal
    moving = ~at_goal
    v = (s - s_old) / self.dt
    progress = 0.75 * jp.clip(v / (rc.slow_frac * v_des), -1, 1) + 0.25 * jp.clip(v / v_des, -1, 1)
    cost = {
        "progress": progress * moving,
        "cross_track": jp.exp(-jp.square(d / rc.sigma_track)),
        "at_goal": at_goal.astype(jp.float32),
        "heading": jp.exp(-jp.square(head_err / rc.sigma_heading)),
        "orientation": jp.sum(jp.square(grav[:2])),
        "ang_vel_xy": jp.sum(jp.square(gyro[:2])),
        "yaw_rate": jp.square(gyro[2]),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
        "feet_slip": jp.sum(jp.sum(jp.square(foot_vel_xy), -1) * contact) * moving,
        "feet_air_time": jp.sum((info["feet_air_time"] - rc.air_time_min) * first_contact) * moving,
    }
    rewards = {k: v_ * rc.scales[k] for k, v_ in cost.items()}
    reward = sum(rewards.values()) * self.dt + rc.termination * done

    info["feet_air_time"] = info["feet_air_time"] * ~contact
    info["last_contact"] = contact
    info["last_foot_xy"] = feet[:, :2]
    info["last_act"] = action
    info["t"] = info["t"] + 1
    if self._use_cam:
      info["cam"] = jp.concatenate([self.camera(data, info)[None], info["cam"][:-1]], 0)

    metrics = dict(state.metrics)
    for k, v_ in rewards.items():
      metrics[f"reward/{k}"] = v_
    metrics.update({"cross_track": jp.abs(d), "heading_err": jp.abs(head_err), "progress": s, "at_goal": at_goal.astype(jp.float32),
                    "success": first_reach.astype(jp.float32), "bumped": bumped.astype(jp.float32),
                    "faulty": jp.any(self.fault_active(info)).astype(jp.float32)})
    obs = self._get_obs(data, info, feats)
    return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), metrics=metrics)

  def _get_obs(self, data, info, feats):
    grav, gyro = self._gravity(data), data.qvel[3:6]
    ph = 2 * jp.pi * C.CLOCK_HZ * C.CTRL_DT * info["t"]
    parts = ([grav, gyro] if self._use_imu else []) + [info["last_act"], feats, jp.stack([jp.sin(ph), jp.cos(ph)])]
    now = jp.concatenate(parts)
    hist = jp.concatenate([now[None], info["hist"][:-1]], 0)
    info["hist"] = hist
    state = [hist.reshape(-1)]
    if self._use_cam:
      state.append(info["cam"][jp.array(CAM_FRAMES)].reshape(-1))
    feet, contact = self._feet(data)
    active = self.fault_active(info)
    onehot = jax.nn.one_hot(info["fault_kind"] * active, NK)
    privileged = jp.concatenate([
        feats, info["last_act"],
        self._rot(data).T @ data.qvel[0:3], grav, gyro,
        data.qpos[7:] - self._default_pose, data.qvel[6:], data.actuator_force,
        contact.astype(jp.float32), jp.array([data.qpos[2] - self._ground(data.qpos[None, 0:2])[0]]),
        onehot.reshape(-1), info["fault_size"] * active, info["held"] - data.qpos[7:],
    ])
    return {"state": jp.concatenate(state), "privileged_state": privileged}

  def _extra_bump(self, data, info):
    return jp.zeros((), bool)

  def fault_label(self, info):
    """Per-servo label for the diagnosis head: 1 = faulty now (any kind)."""
    return self.fault_active(info).astype(jp.int32)

  # ---------------------------------------------------------------- accessors
  @property
  def xml_path(self) -> str:
    return self.XML

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

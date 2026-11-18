"""
Fault-tolerant path following for the Arachne hexapod (PLAN.md).

Each episode gives the robot a random path (straight pieces and arcs) and a goal
0.4-0.8 m along it. Zero, one or two servos fail at a random moment mid-run. The
policy must keep following the path and reach the goal, using only what the
real robot has: IMU gravity + gyro, its own past commands, and the path relative
to its pose (on the real robot: the overhead AprilTag tracker). It sees the last
HIST control steps (1 s), so it can notice a fault and adapt without retraining.
The critic also sees the true fault state, joint angles, contacts and velocity.

Fault layer, between the command and the servo (as in fault-sim/faults.py):
  1 loss      each command is dropped with probability p (the servo keeps the old one)
  2 deadband  changes smaller than db are ignored
  3 sag       the servo only moves within +-span of where it was when the fault began
  4 frozen    dead, holds the angle it had when it died (PCA9685 still sends the old pulse)
  5 limp      dead, no torque (signal or power lost; only the gear damping is left)
1-3 can be partly worked around on that servo; 4-5 cannot, the robot must walk
with the other servos.
"""
import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env

import common as C

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hexapod_mjx.xml")
FOOT_R = 0.005

KINDS = ("none", "loss", "deadband", "sag", "frozen", "limp")
NK = len(KINDS)
CORRECTABLE = jp.array([0, 1, 1, 1, 0, 0])          # 1 = can be worked around on the servo itself

N_PATH, DS = 200, 0.01                                 # path: 200 points, 1 cm apart (2 m)
LOOKAHEAD = (3, 8, 15)                                 # path points ahead, in cm
HIST = 20                                              # control steps of history (1 s)
OBS_STEP = 3 + 3 + 12 + 8 + 2                          # one step of actor input


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=C.CTRL_DT,
      sim_dt=C.SIM_DT,
      episode_length=400,                              # 20 s
      v_des=0.12,                                      # m/s along the path (v2: 0.05 capped 20 s at 1 m < goals)
      noise_config=config_dict.create(level=1.0, scales=config_dict.create(gyro=0.1, gravity=0.03)),
      path_config=config_dict.create(
          kappa_max=3.0,                               # 1/m: tightest arc radius 33 cm
          p_straight=0.4,                              # probability a 20 cm piece is straight
          # v2 (2026-11-18): goals 1.0-1.9 m. The policy walks ~0.19 m/s, so v1's 0.4-0.8 m goals
          # were reached in ~3 s, before most faults started.
          goal_min=1.0, goal_max=1.9,                  # m along the path
      ),
      fault_config=config_dict.create(
          p_count=[0.2, 0.35, 0.3, 0.15],              # 0, 1, 2, 3 faulty servos per episode (v1: 0-2)
          p_kind=[0.15, 0.15, 0.15, 0.25, 0.3],        # loss, deadband, sag, frozen, limp
          onset_min=10, onset_max=160,                 # control steps (0.5-8 s), i.e. mid-walk
          loss_p=[0.5, 0.95], deadband_deg=[10.0, 20.0], sag_deg=[2.0, 5.0],
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              progress=1.5, cross_track=1.0, at_goal=2.0,
              orientation=-2.0, ang_vel_xy=-0.02, yaw_rate=-0.05,
              action_rate=-0.02, energy=-0.01, feet_slip=-0.5, feet_air_time=0.5,
          ),
          sigma_track=0.03,                            # m
          air_time_min=0.15,
          termination=-5.0,                            # added once, not scaled by dt
      ),
      max_cross_track=0.25,                            # m: further off the path ends the episode
      impl="jax",
  )


def make_path(rng, cfg):
  """Random path from the origin: 10 pieces of 20 cm, each straight or a constant arc."""
  k1, k2, k3, k4 = jax.random.split(rng, 4)
  pc = cfg.path_config
  theta0 = jax.random.uniform(k1, (), minval=-jp.pi, maxval=jp.pi)
  kap = jax.random.uniform(k2, (10,), minval=-pc.kappa_max, maxval=pc.kappa_max)
  kap = kap * (jax.random.uniform(k3, (10,)) > pc.p_straight)
  kap = jp.repeat(kap, N_PATH // 10)
  theta = theta0 + jp.cumsum(kap * DS) - kap * DS
  steps = DS * jp.stack([jp.cos(theta), jp.sin(theta)], -1)
  pts = jp.concatenate([jp.zeros((1, 2)), jp.cumsum(steps, 0)[:-1]], 0)
  goal_idx = jp.round(jax.random.uniform(k4, (), minval=pc.goal_min, maxval=pc.goal_max) / DS).astype(jp.int32)
  return pts, theta, goal_idx


def sample_faults(rng, cfg):
  fc = cfg.fault_config
  k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
  nmax = len(fc.p_count) - 1
  n = jax.random.choice(k1, nmax + 1, p=jp.array(fc.p_count))
  order = jax.random.permutation(k2, 12)
  chosen = jp.zeros(12, bool).at[order[:nmax]].set(jp.arange(nmax) < n)
  kind = jax.random.choice(k3, jp.arange(1, NK), (12,), p=jp.array(fc.p_kind)) * chosen
  u = jax.random.uniform(k4, (12,))
  lo = jp.array([0.0, fc.loss_p[0], jp.radians(fc.deadband_deg[0]), jp.radians(fc.sag_deg[0]), 0.0, 0.0])
  hi = jp.array([0.0, fc.loss_p[1], jp.radians(fc.deadband_deg[1]), jp.radians(fc.sag_deg[1]), 0.0, 0.0])
  size = lo[kind] + u * (hi[kind] - lo[kind])
  onset = jax.random.randint(k5, (12,), fc.onset_min, fc.onset_max)
  return kind.astype(jp.int32), size, onset


class FaultPath(mjx_env.MjxEnv):

  def __init__(self, config: config_dict.ConfigDict = default_config(),
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_path(XML)
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    m = self._mj_model
    self._init_q = jp.array(m.keyframe("stand").qpos)
    self._default_pose = jp.array(C.STAND_QPOS)
    self._base_id = m.body("base").id
    self._foot_ids = jp.array([m.geom(f"foot{k}").id for k in range(6)])
    self._center, self._half = jp.array(C.ACT_CENTER), jp.array(C.ACT_HALF)
    self._look = jp.array(LOOKAHEAD)

  # ---------------------------------------------------------------- helpers
  def _rot(self, data):
    return data.xmat[self._base_id].reshape(3, 3)

  def _gravity(self, data):
    return self._rot(data).T @ jp.array([0.0, 0.0, -1.0])

  def _yaw(self, data):
    r = self._rot(data)
    return jp.arctan2(r[1, 0], r[0, 0])

  def _feet(self, data):
    p = data.geom_xpos[self._foot_ids]
    return p, (p[:, 2] - FOOT_R) < 0.0015

  def _noise(self, rng, x, scale):
    return x + (2 * jax.random.uniform(rng, x.shape) - 1) * self._config.noise_config.level * scale

  def _track(self, data, info):
    """Arc length s along the path, signed cross-track error d, and the path features."""
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

  # ---------------------------------------------------------------- reset / step
  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
    qpos = self._init_q
    yaw = jax.random.uniform(k1, (1,), minval=-3.14, maxval=3.14)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)))
    qpos = qpos.at[0:2].add(jax.random.uniform(k4, (2,), minval=-0.01, maxval=0.01))
    qpos = qpos.at[7:].add(jax.random.uniform(k2, (12,), minval=-0.05, maxval=0.05))
    qvel = jp.zeros(self.mjx_model.nv)
    data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:], impl=self.mjx_model.impl.value)
    data = mjx.forward(self.mjx_model, data)
    pts, theta, goal_idx = make_path(k3, self._config)
    kind, size, onset = sample_faults(k5, self._config)
    feet, contact = self._feet(data)
    info = {
        "rng": rng,
        "t": jp.zeros((), jp.int32),
        "path": pts, "path_theta": theta, "goal_idx": goal_idx, "path_idx": jp.zeros((), jp.int32),
        "s": jp.zeros(()),
        "fault_kind": kind, "fault_size": size, "fault_onset": onset,
        "held": qpos[7:], "anchor": qpos[7:], "was_active": jp.zeros(12, bool),
        "last_act": jp.asarray(C.STAND_ACTION),
        "feet_air_time": jp.zeros(6),
        "last_contact": contact,
        "last_foot_xy": feet[:, :2],
        "hist": jp.zeros((HIST, OBS_STEP)),
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    metrics.update({"cross_track": jp.zeros(()), "progress": jp.zeros(()), "at_goal": jp.zeros(()),
                    "faulty": jp.zeros(())})
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
        jp.where(drop, held, target),                                  # loss
        jp.where(jp.abs(target - held) > size, target, held),          # deadband
        jp.clip(target, anchor - size, anchor + size),                 # sag
        anchor,                                                        # frozen
        target,                                                        # limp (handled in the substeps)
    ])
    new = jp.where(active, by_kind[kind, jp.arange(12)], target)
    info["anchor"], info["was_active"] = anchor, active
    return new, active & (kind == 5), info

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = state.info
    action = jp.clip(action, -1.0, 1.0)
    info["rng"], kf = jax.random.split(info["rng"])
    held, limp, info = self.apply_faults(info, self._center + self._half * action, kf)

    def sub(data, _):
      ctrl = jp.where(limp, data.qpos[7:], held)                       # limp: no position error, no torque
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
    grav = self._gravity(data)
    gyro = data.qvel[3:6]
    done = (grav[2] > -0.5) | (data.qpos[2] < 0.045) | (jp.abs(d) > self._config.max_cross_track)

    rc = self._config.reward_config
    v_des = self._config.v_des
    at_goal = rem < 0.03
    moving = ~at_goal
    cost = {
        "progress": jp.clip((s - s_old) / self.dt, -v_des, v_des) / v_des * moving,
        "cross_track": jp.exp(-jp.square(d / rc.sigma_track)),
        "at_goal": at_goal.astype(jp.float32),
        "orientation": jp.sum(jp.square(grav[:2])),
        "ang_vel_xy": jp.sum(jp.square(gyro[:2])),
        "yaw_rate": jp.square(gyro[2]),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
        "feet_slip": jp.sum(jp.sum(jp.square(foot_vel_xy), -1) * contact) * moving,
        "feet_air_time": jp.sum((info["feet_air_time"] - rc.air_time_min) * first_contact) * moving,
    }
    rewards = {k: v * rc.scales[k] for k, v in cost.items()}
    reward = sum(rewards.values()) * self.dt + rc.termination * done

    info["feet_air_time"] = info["feet_air_time"] * ~contact
    info["last_contact"] = contact
    info["last_foot_xy"] = feet[:, :2]
    info["last_act"] = action
    info["t"] = info["t"] + 1

    metrics = dict(state.metrics)
    for k, v in rewards.items():
      metrics[f"reward/{k}"] = v
    metrics.update({"cross_track": jp.abs(d), "progress": s, "at_goal": at_goal.astype(jp.float32),
                    "faulty": jp.any(self.fault_active(info)).astype(jp.float32)})
    obs = self._get_obs(data, info, feats)
    return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), metrics=metrics)

  def _get_obs(self, data, info, feats):
    rng, k1, k2 = jax.random.split(info["rng"], 3)
    info["rng"] = rng
    ns = self._config.noise_config.scales
    grav, gyro = self._gravity(data), data.qvel[3:6]
    ph = 2 * jp.pi * C.CLOCK_HZ * C.CTRL_DT * info["t"]
    now = jp.concatenate([self._noise(k1, grav, ns.gravity), self._noise(k2, gyro, ns.gyro), info["last_act"],
                          feats, jp.stack([jp.sin(ph), jp.cos(ph)])])
    hist = jp.concatenate([now[None], info["hist"][:-1]], 0)
    info["hist"] = hist
    feet, contact = self._feet(data)
    active = self.fault_active(info)
    onehot = jax.nn.one_hot(info["fault_kind"] * active, NK)
    privileged = jp.concatenate([
        now,
        self._rot(data).T @ data.qvel[0:3], grav, gyro,
        data.qpos[7:] - self._default_pose, data.qvel[6:], data.actuator_force,
        contact.astype(jp.float32), jp.array([data.qpos[2]]),
        onehot.reshape(-1), info["fault_size"] * active, info["held"] - data.qpos[7:],
    ])
    return {"state": hist.reshape(-1), "privileged_state": privileged}

  # ---------------------------------------------------------------- accessors
  @property
  def xml_path(self) -> str:
    return XML

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

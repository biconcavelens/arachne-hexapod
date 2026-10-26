"""
MuJoCo Playground / MJX environment for the Arachne hexapod: track a body-frame
velocity command (forward, left, yaw rate) on flat ground.

"Perfect motor" version: the actuators are the ideal torque-limited position
servos of hexapod_mjx.xml (no delay, deadband or rate limit). This is the kill
test's policy: how much does it lose on the real, sealed servos?

The actor only gets what the real robot can provide (see common.py); the critic
also gets joint angles/velocities, base velocity and foot contacts.
use_imu=False zeroes the gravity/gyro inputs, giving a policy that can run
before the IMU is wired.
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
BASE_HEIGHT = 0.095
FOOT_R = 0.005


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=C.CTRL_DT,
      sim_dt=C.SIM_DT,
      episode_length=400,                  # 20 s
      use_imu=True,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(gyro=0.1, gravity=0.03),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              orientation=-2.0,
              ang_vel_xy=-0.02,
              base_height=-20.0,
              pose=0.5,
              stand_still=-1.0,
              termination=-1.0,
              action_rate=-0.02,
              energy=-0.01,
              feet_slip=-1.0,
              feet_air_time=1.0,
          ),
          sigma_lin=0.001,                 # (m/s)^2: 0.03 m/s error -> reward 0.37
          sigma_ang=0.1,                   # (rad/s)^2
          air_time_min=0.2,                # s a foot must be up to count as a step
      ),
      command_config=config_dict.create(
          a=[float(v) for v in C.CMD_MAX],
          b=[0.9, 0.5, 0.5],               # probability each component is non-zero
      ),
      impl="jax",
  )


class Hexapod(mjx_env.MjxEnv):

  def __init__(self, config: config_dict.ConfigDict = default_config(),
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
    super().__init__(config, config_overrides)
    self._mj_model = mujoco.MjModel.from_xml_path(XML)
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    m = self._mj_model
    assert [m.actuator(i).name for i in range(m.nu)] == [f"{j}{k}" for k in range(6) for j in ("hip", "knee")]
    self._init_q = jp.array(m.keyframe("stand").qpos)
    self._default_pose = jp.array(C.STAND_QPOS)
    self._base_id = m.body("base").id
    self._foot_ids = jp.array([m.geom(f"foot{k}").id for k in range(6)])
    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)
    self._center, self._half = jp.array(C.ACT_CENTER), jp.array(C.ACT_HALF)

  # ---------------------------------------------------------------- helpers
  def _rot(self, data):
    return data.xmat[self._base_id].reshape(3, 3)          # body -> world

  def _gravity(self, data):
    return self._rot(data).T @ jp.array([0.0, 0.0, -1.0])

  def _noise(self, rng, x, scale):
    return x + (2 * jax.random.uniform(rng, x.shape) - 1) * self._config.noise_config.level * scale

  def _feet(self, data):
    p = data.geom_xpos[self._foot_ids]
    return p, (p[:, 2] - FOOT_R) < 0.0015                   # positions, contact

  def sample_command(self, rng, cmd):
    r1, r2, r3 = jax.random.split(rng, 3)
    new = jax.random.uniform(r1, (3,), minval=-self._cmd_a, maxval=self._cmd_a)
    new = new * jax.random.bernoulli(r2, self._cmd_b, (3,))
    keep = jax.random.bernoulli(r3, 0.5, (3,))
    return cmd - keep * (cmd - new)

  # ---------------------------------------------------------------- reset / step
  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
    qpos = self._init_q
    yaw = jax.random.uniform(k1, (1,), minval=-3.14, maxval=3.14)
    qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)))
    qpos = qpos.at[7:].add(jax.random.uniform(k2, (12,), minval=-0.05, maxval=0.05))
    qvel = jp.zeros(self.mjx_model.nv).at[0:6].set(jax.random.uniform(k3, (6,), minval=-0.02, maxval=0.02))
    data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:], impl=self.mjx_model.impl.value)
    data = mjx.forward(self.mjx_model, data)
    feet, contact = self._feet(data)
    info = {
        "rng": rng,
        "t": jp.zeros((), jp.int32),
        "command": jax.random.uniform(k4, (3,), minval=-self._cmd_a, maxval=self._cmd_a),
        "steps_until_next_cmd": jp.round(jax.random.exponential(k5) * 3.0 / self.dt).astype(jp.int32),
        "last_act": jp.asarray(C.STAND_ACTION),
        "last_last_act": jp.asarray(C.STAND_ACTION),
        "feet_air_time": jp.zeros(6),
        "last_contact": contact,
        "last_foot_xy": feet[:, :2],
    }
    metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
    metrics["speed_error"] = jp.zeros(())
    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action = jp.clip(action, -1.0, 1.0)
    data = mjx_env.step(self.mjx_model, state.data, self._center + self._half * action, self.n_substeps)
    info = state.info
    feet, contact = self._feet(data)
    first_contact = (info["feet_air_time"] > 0.0) * (contact | info["last_contact"])
    info["feet_air_time"] = info["feet_air_time"] + self.dt
    foot_vel_xy = (feet[:, :2] - info["last_foot_xy"]) / self.dt

    grav = self._gravity(data)
    v_local = self._rot(data).T @ data.qvel[0:3]
    gyro = data.qvel[3:6]
    done = (grav[2] > -0.5) | (data.qpos[2] < 0.06)         # >60 deg tilt, or belly on the floor

    rc = self._config.reward_config
    cmd = info["command"]
    qj = data.qpos[7:]
    cost = {
        "tracking_lin_vel": jp.exp(-jp.sum(jp.square(cmd[:2] - v_local[:2])) / rc.sigma_lin),
        "tracking_ang_vel": jp.exp(-jp.square(cmd[2] - gyro[2]) / rc.sigma_ang),
        "orientation": jp.sum(jp.square(grav[:2])),
        "ang_vel_xy": jp.sum(jp.square(gyro[:2])),
        "base_height": jp.square(data.qpos[2] - BASE_HEIGHT),
        "pose": jp.exp(-jp.sum(jp.square(qj - self._default_pose) * jp.tile(jp.array([0.5, 0.25]), 6))),
        "stand_still": jp.sum(jp.abs(qj - self._default_pose)) * (jp.linalg.norm(cmd) < 0.005),
        "termination": done.astype(jp.float32),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
        "feet_slip": jp.sum(jp.sum(jp.square(foot_vel_xy), axis=-1) * contact) * (jp.linalg.norm(cmd) > 0.005),
        "feet_air_time": jp.sum((info["feet_air_time"] - rc.air_time_min) * first_contact) * (jp.linalg.norm(cmd) > 0.005),
    }
    rewards = {k: v * rc.scales[k] for k, v in cost.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    info["feet_air_time"] = info["feet_air_time"] * ~contact
    info["last_contact"] = contact
    info["last_foot_xy"] = feet[:, :2]
    info["last_last_act"], info["last_act"] = info["last_act"], action
    info["t"] = info["t"] + 1
    info["steps_until_next_cmd"] = info["steps_until_next_cmd"] - 1
    info["rng"], k1, k2 = jax.random.split(info["rng"], 3)
    resample = info["steps_until_next_cmd"] <= 0
    info["command"] = jp.where(resample, self.sample_command(k1, cmd), cmd)
    info["steps_until_next_cmd"] = jp.where(
        resample, jp.round(jax.random.exponential(k2) * 3.0 / self.dt).astype(jp.int32), info["steps_until_next_cmd"])

    metrics = dict(state.metrics)
    for k, v in rewards.items():
      metrics[f"reward/{k}"] = v
    metrics["speed_error"] = jp.linalg.norm(cmd[:2] - v_local[:2])
    obs = self._get_obs(data, info)
    return state.replace(data=data, obs=obs, reward=reward, done=done.astype(reward.dtype), metrics=metrics)

  def _get_obs(self, data, info):
    rng, k1, k2 = jax.random.split(info["rng"], 3)
    info["rng"] = rng
    ns = self._config.noise_config.scales
    imu = jp.asarray(1.0 if self._config.use_imu else 0.0)
    grav = self._gravity(data)
    gyro = data.qvel[3:6]
    state = C.build_obs(self._noise(k1, grav, ns.gravity) * imu, self._noise(k2, gyro, ns.gyro) * imu,
                        info["command"], info["last_act"], info["t"], xp=jp)
    feet, contact = self._feet(data)
    privileged = jp.concatenate([
        state,
        self._rot(data).T @ data.qvel[0:3],            # base linear velocity (body frame)
        grav, gyro,
        data.qpos[7:] - self._default_pose,
        data.qvel[6:],
        data.actuator_force,
        contact.astype(jp.float32),
        jp.array([data.qpos[2]]),
    ])
    return {"state": state, "privileged_state": privileged}

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

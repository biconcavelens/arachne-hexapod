"""
The furnished room (room_env) plus things that move by themselves: a robot vacuum, two rolling balls
and a person walking. Each is present with probability p_mover and crosses the drawn path once, in a
straight line, timed to reach the crossing point roughly when the robot does (from 3 s early to 3 s
late at the robot's nominal speed). The vacuum and the person stop while the robot is in their way
(0.35 m ahead of them) and go on once it has passed; balls do not stop. Walking into a mover, i.e.
touching one within 60 deg of the robot's heading while closing the gap at least as fast as the
mover, ends the episode like a fall, so the robot has to slow down or wait for what is crossing in
front of it. Any other touch (hit from the side or behind, a ball rolling at it) is not the robot's
fault: it does not end the episode and is counted in the "struck" metric.

Movers are kinematic (mocap) and not solid in the physics, like the furniture (see room_scene.py).
The critic additionally sees each mover's position and velocity relative to the robot.
"""
import jax
import jax.numpy as jp
from ml_collections import config_dict

import room_env as RE
import room_scene as RS
from fault_path_env import DS

if not __import__("os").path.exists(RS.DYN_OUT):
    RS.build(movers=True)

N_MOV = len(RS.MOVERS)
MOV_FOOT = jp.array([f for _, f, _ in RS.MOVERS])
MOV_SPEED = jp.array([[0.2, 0.3], [0.3, 0.6], [0.3, 0.6], [0.6, 1.0]])      # m/s per mover
MOV_ROLL_R = jp.array([0.0, 0.08, 0.06, 0.0])                              # balls roll
MOV_YIELD = jp.array([True, False, False, True])                           # stops for the robot
YIELD_AHEAD = 0.35                         # m of free space a yielding mover keeps in front of it
FRONT_COS = 0.5                            # within 60 deg of the heading: the robot walked into it


def default_config() -> config_dict.ConfigDict:
  c = RE.default_config()
  c.mover_config = config_dict.create(p_mover=0.6, v_nominal=0.1, early_late_s=3.0, cross_from=0.25, cross_to=0.85)
  return c


class FaultRoomDyn(RE.FaultRoom):
  XML = RS.DYN_OUT
  MOV_FOOT = MOV_FOOT
  MOV_SPEED = MOV_SPEED
  MOV_ROLL_R = MOV_ROLL_R
  MOV_YIELD = MOV_YIELD
  MOV_LIM = RS.ROOM_HALF - 0.1                             # movers leave the scene beyond this

  def __init__(self, config=None, config_overrides=None):
    super().__init__(config or default_config(), config_overrides)

  def _movers(self, info):
    """Mover positions (N_MOV, 3), quaternions and which are in the scene, from info["mov_s"]
    (distance travelled along each mover's line, 0 at the crossing point)."""
    sp = jp.linalg.norm(info["mov_v"], axis=-1)
    u = info["mov_v"] / jp.maximum(sp, 1e-6)[:, None]
    xy = info["mov_p"] + u * info["mov_s"][:, None]
    inside = jp.all(jp.abs(xy) < self.MOV_LIM, -1) & info["mov_on"]
    park = jp.stack([80.0 + jp.arange(N_MOV), jp.zeros(N_MOV)], -1)
    xy = jp.where(inside[:, None], xy, park)
    z = jp.where(inside, self._ground(xy), -1.0)        # on the ground
    head = jp.arctan2(u[:, 1], u[:, 0])
    # yaw to the heading; balls also spin about the axis across their motion
    roll = jp.where(self.MOV_ROLL_R > 0, info["mov_s"] / jp.maximum(self.MOV_ROLL_R, 1e-3), 0.0)
    qy = jp.stack([jp.cos(head / 2), jp.zeros(N_MOV), jp.zeros(N_MOV), jp.sin(head / 2)], -1)
    qr = jp.stack([jp.cos(roll / 2), jp.zeros(N_MOV), jp.sin(roll / 2), jp.zeros(N_MOV)], -1)
    from mujoco.mjx._src import math
    quat = jax.vmap(math.quat_mul)(qy, qr)
    return jp.concatenate([xy, z[:, None]], -1), quat, inside

  def _advance_movers(self, info, robot_xy):
    """One control step of mover motion; yielding movers wait while the robot is in their way."""
    pos, _, inside = self._movers(info)
    sp = jp.linalg.norm(info["mov_v"], axis=-1)
    u = info["mov_v"] / jp.maximum(sp, 1e-6)[:, None]
    rel = robot_xy[None] - pos[:, :2]
    along = jp.sum(rel * u, -1)
    lat = jp.abs(rel[:, 0] * u[:, 1] - rel[:, 1] * u[:, 0])
    reach = self.MOV_FOOT + RE.ROBOT_R
    blocked = self.MOV_YIELD & inside & (along > 0) & (along < reach + YIELD_AHEAD) & (lat < reach + 0.05)
    info = dict(info)
    info["mov_s"] = info["mov_s"] + sp * self.dt * ~blocked
    info["mov_go"] = (~blocked).astype(jp.float32)
    return info

  def _mover_contact(self, data, info):
    """(walked into one, struck from the side or behind) for the current mover positions."""
    pos, _, inside = self._movers(info)
    rel = pos[:, :2] - data.qpos[None, 0:2]
    dist = jp.linalg.norm(rel, axis=-1)
    touch = inside & (dist - self.MOV_FOOT - RE.ROBOT_R < 0)
    yaw = self._yaw(data)
    front = (rel @ jp.array([jp.cos(yaw), jp.sin(yaw)])) > FRONT_COS * dist
    # the robot's fault only if it closed the gap at least as fast as the mover did (a ball rolling
    # straight at it is not)
    n = rel / jp.maximum(dist, 1e-6)[:, None]
    robot_close = n @ data.qvel[0:2]
    mover_close = -jp.sum(n * info["mov_v"] * info["mov_go"][:, None], -1)
    fault = front & (robot_close >= mover_close)
    return jp.any(touch & fault), jp.any(touch & ~fault)

  def _set_movers(self, data, info):
    pos, quat, inside = self._movers(info)
    n = self._n_obj
    data = data.replace(mocap_pos=data.mocap_pos.at[n:n + N_MOV].set(pos),
                        mocap_quat=data.mocap_quat.at[n:n + N_MOV].set(quat))
    return data

  def reset(self, rng):
    rng, k = jax.random.split(rng)
    st = super().reset(rng)
    mc = self._config.mover_config
    k1, k2, k3, k4, k5, k6 = jax.random.split(k, 6)
    info = dict(st.info)
    g = info["goal_idx"]
    idx = (g * jax.random.uniform(k1, (N_MOV,), minval=mc.cross_from, maxval=mc.cross_to)).astype(jp.int32)
    p = info["path"][idx]
    side = jp.where(jax.random.uniform(k2, (N_MOV,)) < 0.5, 1.0, -1.0)
    ang = info["path_theta"][idx] + side * jp.pi / 2 + jax.random.uniform(k3, (N_MOV,), minval=-0.6, maxval=0.6)
    sp = self.MOV_SPEED[:, 0] + jax.random.uniform(k4, (N_MOV,)) * (self.MOV_SPEED[:, 1] - self.MOV_SPEED[:, 0])
    info["mov_p"] = p
    info["mov_v"] = sp[:, None] * jp.stack([jp.cos(ang), jp.sin(ang)], -1)
    info["mov_tc"] = idx * DS / mc.v_nominal + jax.random.uniform(k5, (N_MOV,), minval=-mc.early_late_s,
                                                                    maxval=mc.early_late_s)
    info["mov_on"] = jax.random.uniform(k6, (N_MOV,)) < mc.p_mover
    info["mov_s"] = -sp * info["mov_tc"]                   # reaches the crossing point at t = mov_tc
    info["mov_go"] = jp.ones(N_MOV)
    data = self._set_movers(st.data, info)
    from mujoco import mjx
    data = mjx.forward(self.mjx_model, data)
    if self._use_cam:
      info["cam"] = jp.broadcast_to(self.camera(data, info), info["cam"].shape)
    info["hist"] = jp.zeros_like(info["hist"])
    feats = self._track(data, info)[4]
    obs = self._get_obs_extra(self._get_obs(data, info, feats), data, info)
    metrics = dict(st.metrics)
    metrics["struck"] = jp.zeros(())
    return st.replace(data=data, info=info, obs=obs, metrics=metrics)

  def step(self, state, action):
    info = self._advance_movers(state.info, state.data.qpos[0:2])
    data = self._set_movers(state.data, info)
    st = super().step(state.replace(data=data, info=info), action)
    metrics = dict(st.metrics)
    metrics["struck"] = self._mover_contact(st.data, st.info)[1].astype(jp.float32)
    return st.replace(obs=self._get_obs_extra(st.obs, st.data, st.info), metrics=metrics)

  def _extra_bump(self, data, info):                       # only walking into a mover ends the episode
    return self._mover_contact(data, info)[0]

  def _get_obs_extra(self, obs, data, info):
    pos, _, inside = self._movers(info)
    rel = (pos[:, :2] - data.qpos[None, 0:2]) * inside[:, None]
    vel = info["mov_v"] * (inside * info["mov_go"])[:, None]
    return {"state": obs["state"],
            "privileged_state": jp.concatenate([obs["privileged_state"], rel.reshape(-1), vel.reshape(-1)])}

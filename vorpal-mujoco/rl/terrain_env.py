"""
Fault-tolerant path following on rough outdoor ground (third environment stage).

The room task (room_env.FaultRoom) on terrain_scene's heightfield: hills, bumps and stones, more in
some areas than others. Each episode starts at a random spot and heading inside the 8 x 8 m field,
with the path drawn from there; rocks, bushes, logs, trees and crates are placed off the path.
Ground height is a bilinear lookup in the same grid MJX collides the feet with; the body camera
ray-marches that grid (mjx.ray cannot hit a heightfield) and shades it by the slope towards the sun.
"""
import os

import jax
import jax.numpy as jp
import numpy as np

import room_env as RE
import terrain_scene as TS

if not os.path.exists(TS.NPZ):
    TS.build()

_Z = np.load(TS.NPZ)
H = jp.asarray(_Z["H"])                         # H[row = y, col = x], metres
CELL = 2 * TS.HALF / (TS.N - 1)
SUN = jp.array([0.3, -0.3, 1.0]) / jp.linalg.norm(jp.array([0.3, -0.3, 1.0]))
MARCH = jp.concatenate([jp.linspace(0.02, 0.6, 30), jp.linspace(0.65, 4.0, 34)])   # ray samples (m)


def ground(xy):
    """Bilinear ground height at xy (..., 2)."""
    u = jp.clip((xy[..., 0] + TS.HALF) / CELL, 0, TS.N - 1.001)
    v = jp.clip((xy[..., 1] + TS.HALF) / CELL, 0, TS.N - 1.001)
    c, r = jp.floor(u).astype(jp.int32), jp.floor(v).astype(jp.int32)
    fu, fv = u - c, v - r
    return ((1 - fu) * (1 - fv) * H[r, c] + fu * (1 - fv) * H[r, c + 1]
            + (1 - fu) * fv * H[r + 1, c] + fu * fv * H[r + 1, c + 1])


def _normal(xy):
    e = CELL
    gx = (ground(xy + jp.array([e, 0.0])) - ground(xy - jp.array([e, 0.0]))) / (2 * e)
    gy = (ground(xy + jp.array([0.0, e])) - ground(xy - jp.array([0.0, e]))) / (2 * e)
    n = jp.stack([-gx, -gy, jp.ones_like(gx)], -1)
    return n / jp.linalg.norm(n, axis=-1, keepdims=True)


class FaultTerrain(RE.FaultRoom):
  XML = TS.OUT
  OBJECTS = TS.OBJECTS
  FOOTPRINT = TS.FOOT
  WORLD_LIM = TS.HALF - 0.3
  START_LIFT = 0.02

  def __init__(self, config=None, config_overrides=None):
    super().__init__(config or RE.default_config(), config_overrides)
    m = TS.load_model(self.XML)                  # same model with the heightfield data filled in
    m.opt.timestep = self._config.sim_dt
    from mujoco import mjx
    self._mj_model = m
    self._mjx_model = mjx.put_model(m, impl=self._config.impl)

  def _ground(self, xy):
    return ground(xy)

  def _start(self, rng):
    lim = TS.HALF - 2.3                          # a 2 m path from here stays on the field
    return jax.random.uniform(rng, (2,), minval=-lim, maxval=lim)

  def _ray_extra(self, o, dirs, dist, gid):
    """First crossing of each ray below the ground, refined between the two samples around it."""
    # sequential march (a loop, not a rays x samples array: that blew up the XLA compile past 7 GB)
    n = dirs.shape[0]

    def body(i, c):
      t_prev, h_prev, t_hit, hit = c
      t1 = MARCH[i]
      p = o + t1 * dirs
      h1 = p[:, 2] - ground(p[:, :2])
      cross = (~hit) & (h1 < 0)
      t_sec = jp.where(i == 0, t1, t_prev + (t1 - t_prev) * h_prev / jp.maximum(h_prev - h1, 1e-6))
      return jp.broadcast_to(t1, h1.shape), h1, jp.where(cross, t_sec, t_hit), hit | cross

    z = jp.zeros(n)
    _, _, t, hit = jax.lax.fori_loop(0, MARCH.shape[0], body, (z, z, z, jp.zeros(n, bool)))
    closer = hit & ((gid < 0) | (t < dist))
    return jp.where(closer, t, dist), jp.where(closer, self._floor_id, gid)

  def _floor_texture(self, p, info):
    """Grass and dirt: patchy albedo, fine grain, lit by the sun from the local slope."""
    xy = p[:, :2]
    patch = 0.8 + 0.2 * jp.sin(xy[:, 0] * 3.1 + jp.sin(xy[:, 1] * 2.3)) * jp.sin(xy[:, 1] * 2.7)
    grain = 0.85 + 0.15 * jp.sin(xy[:, 0] * 83.0) * jp.sin(xy[:, 1] * 71.0 + xy[:, 0] * 13.0)
    light = 0.35 + 0.75 * jp.clip(_normal(xy) @ SUN, 0, 1)
    return patch * grain * light

  def _miss_shade(self, dirs):
    return 0.7 + 0.25 * jp.clip(dirs[:, 2], 0, 1)                         # sky, brighter higher up

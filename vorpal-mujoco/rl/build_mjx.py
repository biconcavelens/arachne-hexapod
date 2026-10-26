"""
Lean training model for MJX: hexapod_physics.xml with the meshes replaced by
what contact needs, and the dynamics kept.

  - bodies, joints, ranges, damping/armature, position actuators: unchanged
  - masses/inertias: read from the compiled physics model, written as <inertial>
  - ground contact: one sphere per foot (at the leg mesh's lowest point in the
    stand pose) + a thin box under the base. Hinges/knees do NOT collide with
    the floor here; sim_walk.py-style checks in the full model catch that.
  - no meshes at all, so MJX never touches the 100k-triangle base.

Foot radius is a guess for the hot-glue blob (FOOT_R); measure and update.
Writes hexapod_mjx.xml next to hexapod_physics.xml.
"""
import os
import re

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "hexapod_physics.xml")
OUT = os.path.join(ROOT, "hexapod_mjx.xml")
FOOT_R = 0.005          # m, hot-glue blob (unmeasured)

m = mujoco.MjModel.from_xml_path(SRC)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)
nm = lambda t, i: mujoco.mj_id2name(m, t, i)
B, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM


def verts_world(g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T


def to_body(b, p):
    return d.xmat[b].reshape(3, 3).T @ (p - d.xpos[b])


f = lambda a: " ".join(f"{x:.6g}" for x in a)
foot = {}
for k in range(6):
    g = mujoco.mj_name2id(m, G, f"leg{k}")
    w = verts_world(g)
    low = w[w[:, 2] < w[:, 2].min() + 0.0015].mean(axis=0)            # cluster at the lowest point
    foot[k] = to_body(m.geom_bodyid[g], low + np.array([0, 0, FOOT_R]))
    print(f"leg{k} foot (world, stand): x={low[0]*1000:+.1f} y={low[1]*1000:+.1f} z={low[2]*1000:+.1f} mm")

bg = mujoco.mj_name2id(m, G, "base_geom")
bv = verts_world(bg) - d.xpos[1]                                       # body 1 = base
lo, hi = bv.min(axis=0), bv.max(axis=0)
box_half = np.array([(hi[0] - lo[0]) * 0.4, (hi[1] - lo[1]) * 0.4, 0.004])
box_pos = np.array([(hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, lo[2] + 0.004])
print(f"base: bottom {lo[2]*1000:.0f} mm below its origin, box half-extents {box_half*1000} mm")


def inertial(b):
    return (f'<inertial pos="{f(m.body_ipos[b])}" quat="{f(m.body_iquat[b])}" '
            f'mass="{m.body_mass[b]:.6g}" diaginertia="{f(m.body_inertia[b])}"/>')


xml = open(SRC).read()
# 1. no meshes: drop the mesh assets and every mesh geom, add inertials + contact geoms per body
xml = re.sub(r'  <asset>\s*<mesh .*?</asset>\n', '', xml, flags=re.S)
xml = re.sub(r'\s*<geom name="(?:base_geom|hingeA\d|hingeB\d)"[^>]*/>', '', xml)
xml = xml.replace('<compiler angle="degree" meshdir="meshes"/>',
                  '<compiler angle="degree" inertiafromgeom="false"/>')
for b in range(1, m.nbody):
    name = nm(B, b)
    pat = re.compile(rf'(<body name="{name}"[^>]*>)')
    extra = "\n        " + inertial(b)
    if name == "base":
        extra += (f'\n        <geom name="base_box" type="box" size="{f(box_half)}" pos="{f(box_pos)}" '
                  f'rgba="0.3 0.35 0.8 1"/>')
    if name.startswith("knee"):
        k = int(name[4:])
        extra += (f'\n          <geom name="foot{k}" type="sphere" size="{FOOT_R}" pos="{f(foot[k])}" '
                  f'rgba="0.95 0.3 0.3 1" friction="0.9 0.01 0.001"/>')
    xml = pat.sub(lambda mo: mo.group(1) + extra, xml, count=1)
xml = re.sub(r'\s*<geom name="leg\d"[^>]*/>', '', xml)
# MJX-friendly solver settings
xml = xml.replace('<option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>',
                  '<option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast" iterations="10" ls_iterations="10"/>')
open(OUT, "w").write(xml)

# sanity: loads, same mass, stands (feet carry the weight)
m2 = mujoco.MjModel.from_xml_path(OUT)
d2 = mujoco.MjData(m2)
mujoco.mj_resetDataKeyframe(m2, d2, 0)
for _ in range(1500):
    mujoco.mj_step(m2, d2)
print(f"lean model: {m2.nbody-1} bodies, {m2.ngeom-1} geoms, mass {m2.body_mass.sum()*1000:.0f} g "
      f"(full {m.body_mass.sum()*1000:.0f} g); after 3 s standing: base z {d2.qpos[2]*1000:.1f} mm, "
      f"tilt {np.degrees(np.arccos(np.clip(1-2*(d2.qpos[4]**2+d2.qpos[5]**2),-1,1))):.2f} deg")
print("written:", OUT)

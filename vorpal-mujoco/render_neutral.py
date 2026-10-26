"""Render hexapod_neutral.xml: top / side / iso. Usage: render_neutral.py out.png [joint=deg,...]"""
import sys
import mujoco
import numpy as np
from PIL import Image

m = mujoco.MjModel.from_xml_path(r"C:\projects\arachne\vorpal-mujoco\hexapod_neutral.xml")
d = mujoco.MjData(m)
if len(sys.argv) > 2:
    for pair in sys.argv[2].split(","):
        jn, deg = pair.split("=")
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = np.radians(float(deg))
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=600, width=800)
imgs = []
for az, el in [(90, -89), (90, -4), (135, -25)]:
    c = mujoco.MjvCamera(); c.lookat = [0, 0, -0.03]; c.distance = 0.42
    c.azimuth, c.elevation = az, el
    r.update_scene(d, camera=c); imgs.append(r.render())
blank = np.zeros_like(imgs[0])
Image.fromarray(np.vstack([np.hstack(imgs[:2]), np.hstack([imgs[2], blank])])).save(sys.argv[1])
mujoco.mj_collision(m, d)
pairs = sorted({tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1),
                              mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2))))
                for i in range(d.ncon)})
print(f"saved {sys.argv[1]}  contacts={d.ncon} {pairs}")

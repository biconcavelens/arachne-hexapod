import sys, mujoco, numpy as np
from PIL import Image
m = mujoco.MjModel.from_xml_path(r"C:\projects\arachne\vorpal-mujoco\hexapod.xml")
d = mujoco.MjData(m)
if len(sys.argv) > 2:
    for pair in sys.argv[2].split(","):
        jn, deg = pair.split("=")
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        d.qpos[m.jnt_qposadr[jid]] = np.radians(float(deg))
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=600, width=800)
imgs = []
for az, el in [(90, -89), (135, -25), (90, -10)]:
    c = mujoco.MjvCamera(); c.lookat = [-0.074, 0.005, -0.01]; c.distance = 0.42
    c.azimuth, c.elevation = az, el
    r.update_scene(d, camera=c); imgs.append(r.render())
Image.fromarray(np.vstack([np.hstack(imgs[:2]), np.hstack([imgs[2], np.zeros_like(imgs[2])])])).save(sys.argv[1])
mujoco.mj_collision(m, d)
print(f"saved {sys.argv[1]}  ncon={d.ncon}")

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(r"C:\projects\arachne\vorpal-mujoco\hexapod.xml")
data = mujoco.MjData(model)

for delta0, delta1 in [(89, -89), (89, 0), (0, -89), (89, 89), (-89,-89)]:
    for ch, deg in [(0, delta0), (1, delta1)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hip{ch}_joint")
        data.qpos[model.jnt_qposadr[jid]] = np.radians(deg)
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    print(f"delta0={delta0:+4d} delta1={delta1:+4d}: ncon={data.ncon}")

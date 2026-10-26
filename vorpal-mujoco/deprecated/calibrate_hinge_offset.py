"""
Sweeps HINGE_OFFSET_MM (see generate_mjcf.py) to find the value that makes
MuJoCo's real mesh collision reproduce the one real hardware touch point we
have: legs 0/1 touching at servo0=112 (delta +22), servo1=46 (delta -44).
"""
import subprocess

import mujoco
import numpy as np

GEN = r"C:\projects\arachne\vorpal-mujoco\generate_mjcf.py"
XML = r"C:\projects\arachne\vorpal-mujoco\hexapod.xml"
PY = r"C:\Users\trish\AppData\Local\Programs\Python\Python313\python.exe"


def legs_collide(offset_mm, delta0_deg, delta1_deg):
    subprocess.run([PY, GEN, str(offset_mm)], check=True, capture_output=True)
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    for ch, deg in [(0, delta0_deg), (1, delta1_deg)]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hip{ch}_joint")
        data.qpos[model.jnt_qposadr[jid]] = np.radians(deg)
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    names = set()
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        names.add((g1, g2))
    return data.ncon > 0, names


TOUCH_DELTA0, TOUCH_DELTA1 = 22, -44

print("Sanity check: does the touch point collide at offset=0 (no hinge extension)?")
c, n = legs_collide(0, TOUCH_DELTA0, TOUCH_DELTA1)
print(f"  offset=0mm: collide={c} {n}")

print("\nSweeping offset...")
last_state = None
for offset in range(0, 41, 2):
    collide, names = legs_collide(offset, TOUCH_DELTA0, TOUCH_DELTA1)
    print(f"  offset={offset:3d}mm: collide={collide} {names if collide else ''}")
    if collide and not last_state:
        print(f"  --> first collision onset near offset={offset}mm")
    last_state = collide

print("\nSanity check: does NEUTRAL pose (0,0) collide at a few candidate offsets?")
for offset in [10, 15, 20, 25, 30]:
    c, n = legs_collide(offset, 0, 0)
    print(f"  offset={offset}mm, neutral pose: collide={c} {n if c else ''}")

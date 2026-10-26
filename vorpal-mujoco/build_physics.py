"""
Physics version of the hexapod for gait testing: hexapod_neutral.xml (the
verified kinematic twin) + gravity, a free-floating body, ground contact,
masses, and servos as torque-limited position actuators.

Estimates (to be replaced by real measurements):
  - total mass ~450 g: body 300 g (base, cap, Uno Q, PCA9685, 6 hip servos,
    wiring), per leg ~25 g (leg part + knee servo) + 2 x 6 g hinges
  - MG90S stall torque ~2.2 kg.cm = 0.216 N.m  -> forcerange +-0.2 N.m
  - position gain kp 2.0 N.m/rad: torque saturates at ~6 deg of error
Contacts: robot parts collide with the ground only (gaits are already
clearance-checked for leg-leg contact in gaits.py).
"""
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "hexapod_neutral.xml")
OUT = os.path.join(ROOT, "hexapod_physics.xml")

MASS = {"base_geom": 0.300, "hinge": 0.006, "leg": 0.025}
SERVO = {"kp": 2.0, "kv": 0.03, "force": 0.2}
STAND_KNEE_DEG = -30.0          # same standing pose as gaits.py
STAND_BASE_Z = 0.098            # base origin height when standing on its feet (m)

x = open(SRC).read()

x = x.replace('<option gravity="0 0 0"/>',
              '<option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>\n'
              '  <default>\n'
              '    <geom contype="1" conaffinity="0" friction="0.9 0.01 0.001" solref="0.004 1"/>\n'
              '    <joint damping="0.02" armature="0.0005"/>\n'
              '  </default>')
x = x.replace('<visual><global offwidth="1600" offheight="1200"/></visual>',
              '<visual><global offwidth="1600" offheight="1200"/><quality shadowsize="4096"/></visual>\n'
              '  <asset>\n'
              '    <texture name="grid" type="2d" builtin="checker" rgb1=".18 .2 .24" rgb2=".24 .26 .3" width="512" height="512"/>\n'
              '    <material name="grid" texture="grid" texrepeat="40 40" reflectance="0.1"/>\n'
              '  </asset>')

# ground + free-floating body
x = x.replace('    <body name="base">',
              '    <geom name="floor" type="plane" size="5 5 0.1" material="grid" contype="0" conaffinity="1"/>\n'
              f'    <body name="base" pos="0 0 {STAND_BASE_Z}">\n'
              '      <freejoint name="root"/>')

# base was visual-only; it must hit the ground if the body sags
x = x.replace('rgba="0.3 0.35 0.8 1" contype="0" conaffinity="0"/>',
              f'rgba="0.3 0.35 0.8 1" mass="{MASS["base_geom"]}"/>')
x = re.sub(r'(<geom name="hinge[AB]\d"[^/]*?)(/>)', rf'\1 mass="{MASS["hinge"]}"\2', x)
x = re.sub(r'(<geom name="leg\d"[^/]*?)(/>)', rf'\1 mass="{MASS["leg"]}"\2', x)

acts = []
for k in range(6):
    for j in ("hip", "knee"):
        acts.append(f'    <position name="{j}{k}" joint="{j}{k}" kp="{SERVO["kp"]}" kv="{SERVO["kv"]}" '
                    f'forcelimited="true" forcerange="-{SERVO["force"]} {SERVO["force"]}"/>')
qpos_stand = ["0", "0", f"{STAND_BASE_Z}", "1", "0", "0", "0"]
for k in range(6):
    qpos_stand += ["0", f"{STAND_KNEE_DEG * 3.141592653589793 / 180:.6f}"]   # hip, knee per leg (tree order)
ctrl_stand = []
for k in range(6):
    ctrl_stand += ["0", f"{STAND_KNEE_DEG * 3.141592653589793 / 180:.6f}"]
x = x.replace('</worldbody>',
              '</worldbody>\n  <actuator>\n' + "\n".join(acts) + '\n  </actuator>\n'
              f'  <keyframe>\n    <key name="stand" qpos="{" ".join(qpos_stand)}" ctrl="{" ".join(ctrl_stand)}"/>\n  </keyframe>')

open(OUT, "w").write(x)
print(f"written: {OUT}")

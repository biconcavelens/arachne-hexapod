"""
Neutral-pose MuJoCo model for the Vorpal hexapod (servos at 90/90).

Why this exists: the OnShape assembly is a POSED display model - each leg sits
at a different angle and Leg 6 / Legs-11 are mis-mated - so its transforms are
not a neutral stance. This builder takes only what the CAD gets right and
defines neutral from the real robot (photos/, 2026-10-04, all servos at 90):

  From CAD (onshape_assembly.json):
    - hip axes: 6 vertical revolutes on the base, exact (57.9mm, 60deg apart)
    - the hip->knee link: two interlocked hinges, rigid, taken from Leg 10
      (cleanest leg) and cross-checked against Legs 7, 8, 9 below
    - knee axis: the knee mate on the leg (leg-local -Y)
  From the real robot at 90/90 (photos):
    - hip neutral: leg points radially out from the body
    - knee neutral: the leg's servo box is level (box axis = leg-local +Z),
      foot section bending down (leg-local +X -> world down)

Chain per leg: base -[hip, vertical]- {hinge A + hinge B} -[knee]- leg.
All 6 legs are copies of one template, placed on the 6 CAD hip axes.
"""
import json
import sys

import numpy as np

ROOT = r"C:\projects\arachne\vorpal-mujoco"
J = json.load(open(rf"{ROOT}\onshape_assembly.json"))
MM = 0.001


def mat(v):
    m = np.eye(4)
    m[:3, :] = np.array(v, float).reshape(3, 4)
    return m


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def fmt(v):
    return " ".join(f"{x:.6f}" for x in v)


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


occ = {k: mat(v) for k, v in J["occurrences"].items()}
Tb_inv = np.linalg.inv(occ["Base <1>"])
bl = {k: Tb_inv @ T for k, T in occ.items()}          # everything in base-local frame
HIP = {k: np.array(v) for k, v in J["hip_revolutes_base_local"].items() if not k.startswith("_")}

# The four mechanically consistent CAD legs: leg -> (its hip revolute, its two
# hinges, knee mate in that leg's own local frame). Hinge pairs from the
# centroid-based assignment in build_from_onshape.py.
GOOD = {
    "Leg 10 <1>": ("Revolute 1", (2, 4), [0.136487, 0.024793, -0.051058]),
    "Leg 7 <1>":  ("Revolute 4", (9, 10), [0.116487, 0.024793, -0.051058]),
    "Leg 8 <1>":  ("Revolute 3", (7, 8), [0.116487, 0.056793, -0.051058]),
    "Leg 9 <1>":  ("Revolute 2", (5, 6), [0.136487, -0.007207, -0.051058]),
}
TEMPLATE = "Leg 10 <1>"
Z = np.array([0, 0, 1.0])

# Leg orientation at neutral, in the hip frame H (x radial out, z up):
# leg-local +Z (box axis) -> +x, leg-local +X (toward foot bend) -> -z.
R_LEG_NEUTRAL = np.column_stack([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])


def neutral_link(legname):
    """Express one CAD leg's link (hinges + knee) in its neutral hip frame."""
    hipname, (ha, hb), knee_local = GOOD[legname]
    p_hip = HIP[hipname]
    T_leg = bl[legname]
    knee = (T_leg @ np.array([*knee_local, 1]))[:3]
    axis = T_leg[:3, :3] @ np.array([0, -1.0, 0])
    # link frame L: origin at hip, x = horizontal hip->knee, z up
    d = knee - p_hip
    x = np.array([d[0], d[1], 0]) / np.hypot(d[0], d[1])
    T_L = np.eye(4)
    T_L[:3, :3] = np.column_stack([x, np.cross(Z, x), Z])
    T_L[:3, 3] = p_hip
    L_inv = np.linalg.inv(T_L)
    aL = L_inv[:3, :3] @ axis
    kL = (L_inv @ np.array([*knee, 1]))[:3]
    # leg box direction when level: horizontal and perpendicular to knee axis,
    # pointing away from the hip
    b = np.cross(aL, Z)
    b /= np.linalg.norm(b)
    if b @ np.array([kL[0], kL[1], 0]) < 0:
        b = -b
    yaw0 = np.arctan2(b[1], b[0])
    R0 = rotz(-yaw0)                                   # L -> neutral hip frame H
    out = {
        "knee": (R0 @ np.array([*kL, 1]))[:3],
        "axis": R0[:3, :3] @ aL,
        "hinges": [R0 @ L_inv @ bl[f"LEG_Hinge_TowerPro <{h}>"] for h in (ha, hb)],
        "R_leg_cad": R0[:3, :3] @ L_inv[:3, :3] @ T_leg[:3, :3],
        "knee_local": np.array(knee_local),
    }
    # CAD pose of this leg, for information: hip yaw off radial, knee angle
    radial = np.array([p_hip[0], p_hip[1], 0]) / np.hypot(p_hip[0], p_hip[1])
    b_base = T_L[:3, :3] @ b
    out["cad_hip_deg"] = np.degrees(np.arctan2(np.cross(radial, b_base)[2], radial @ b_base))
    R_rel = R_LEG_NEUTRAL.T @ out["R_leg_cad"]         # leg-local frame: CAD vs neutral
    out["cad_knee_deg"] = np.degrees(np.arctan2(R_rel[0, 2], R_rel[2, 2]))
    out["off_axis_deg"] = rot_angle_deg(R_rel @ np.array(  # residual not explained by a pure
        [[np.cos(np.radians(out["cad_knee_deg"])), 0, -np.sin(np.radians(out["cad_knee_deg"]))],
         [0, 1, 0],
         [np.sin(np.radians(out["cad_knee_deg"])), 0, np.cos(np.radians(out["cad_knee_deg"]))]]))
    return out


links = {lg: neutral_link(lg) for lg in GOOD}
tpl = links[TEMPLATE]

if "--check" in sys.argv:
    print("knee point & axis in neutral hip frame (mm), per CAD leg:")
    for lg, L in links.items():
        print(f"  {lg:11s} knee=({L['knee'][0]*1000:6.1f},{L['knee'][1]*1000:6.1f},{L['knee'][2]*1000:6.1f})"
              f"  axis=({L['axis'][0]:+.3f},{L['axis'][1]:+.3f},{L['axis'][2]:+.3f})"
              f"  CAD pose: hip {L['cad_hip_deg']:+6.1f}deg, knee {L['cad_knee_deg']:+6.1f}deg,"
              f" off-axis residual {L['off_axis_deg']:.2f}deg")
    print("\nhinge pair vs template (Leg 10), after removing hip/knee angle:")
    for lg, L in links.items():
        if lg == TEMPLATE:
            continue
        for i in (0, 1):
            Ta, Tt = L["hinges"][i], tpl["hinges"][i]
            dp = np.linalg.norm(Ta[:3, 3] - Tt[:3, 3]) * 1000
            dr = rot_angle_deg(Tt[:3, :3].T @ Ta[:3, :3])
            print(f"  {lg:11s} hinge {'AB'[i]}: position diff {dp:5.2f} mm, rotation diff {dr:5.2f} deg")
    sys.exit()

# ---------------- build the MJCF ----------------
# Consensus geometry. Orientations agree across all 4 legs to 0.00deg, but each
# leg has one slightly mis-mated part (Leg 10's hinge A is 5mm off, Leg 7's
# knee+hinge B are 2.8mm off; see --check). The per-part median across the 4
# legs drops the outlier every time.
def _median_pos(get):
    return np.median(np.array([get(L) for L in links.values()]), axis=0)


tpl = dict(tpl)
tpl["knee"] = _median_pos(lambda L: L["knee"])
tpl["hinges"] = [h.copy() for h in tpl["hinges"]]
for i in (0, 1):
    tpl["hinges"][i][:3, 3] = _median_pos(lambda L, i=i: L["hinges"][i][:3, 3])

knee_local = tpl["knee_local"]
t_leg = tpl["knee"] - R_LEG_NEUTRAL @ knee_local     # puts the knee mate on the knee point
T_leg_H = np.eye(4)
T_leg_H[:3, :3] = R_LEG_NEUTRAL
T_leg_H[:3, 3] = t_leg

# knee body frame: at the knee point, axes aligned with H
T_knee_H = np.eye(4)
T_knee_H[:3, 3] = tpl["knee"]
T_leg_K = np.linalg.inv(T_knee_H) @ T_leg_H
knee_axis = tpl["axis"] / np.linalg.norm(tpl["axis"])

# hip order: by angle around the base, starting at the +X side
hip_order = sorted(HIP, key=lambda n: np.degrees(np.arctan2(HIP[n][1], HIP[n][0])) % 360)

x = ['<mujoco model="vorpal_hexapod_neutral">',
     '  <compiler angle="degree" meshdir="meshes"/>',
     '  <option gravity="0 0 0"/>',
     '  <visual><global offwidth="1600" offheight="1200"/></visual>',
     '  <asset>',
     '    <mesh name="base_mesh" file="base.stl" scale="0.001 0.001 0.001"/>',
     '    <mesh name="hinge_mesh" file="hinge.stl" scale="0.001 0.001 0.001"/>',
     '    <mesh name="leg_mesh" file="leg10.stl" scale="0.001 0.001 0.001"/>',
     '  </asset>',
     '  <worldbody>',
     '    <light diffuse=".9 .9 .9" pos="0.2 -0.2 0.6" dir="-0.3 0.3 -1"/>',
     '    <light diffuse=".4 .4 .4" pos="-0.3 0.2 0.4" dir="0.4 -0.3 -1"/>',
     '    <body name="base">',
     '      <geom name="base_geom" type="mesh" mesh="base_mesh" rgba="0.3 0.35 0.8 1" contype="0" conaffinity="0"/>']
for k, hn in enumerate(hip_order):
    p = HIP[hn]
    yaw = np.degrees(np.arctan2(p[1], p[0]))
    x.append(f'      <body name="hip{k}" pos="{fmt(p)}" euler="0 0 {yaw:.4f}">')
    x.append(f'        <joint name="hip{k}" type="hinge" axis="0 0 1" range="-90 90"/>')
    for i, T in enumerate(tpl["hinges"]):
        x.append(f'        <geom name="hinge{"AB"[i]}{k}" type="mesh" mesh="hinge_mesh" '
                 f'pos="{fmt(T[:3,3])}" quat="{fmt(quat(T[:3,:3]))}" rgba="0.95 0.75 0.1 1"/>')
    x.append(f'        <body name="knee{k}" pos="{fmt(tpl["knee"])}">')
    x.append(f'          <joint name="knee{k}" type="hinge" axis="{fmt(knee_axis)}" range="-90 90"/>')
    x.append(f'          <geom name="leg{k}" type="mesh" mesh="leg_mesh" '
             f'pos="{fmt(T_leg_K[:3,3])}" quat="{fmt(quat(T_leg_K[:3,:3]))}" rgba="0.25 0.45 0.9 1"/>')
    x.append('        </body>')
    x.append('      </body>')
x += ['    </body>', '  </worldbody>', '</mujoco>']

out = rf"{ROOT}\hexapod_neutral.xml"
open(out, "w").write("\n".join(x) + "\n")
print(f"written: {out}")

# Same model for the browser twin (servo-web-control/assets/sim3d.js), so the
# web view and MuJoCo can never disagree on geometry. Units: meters, base frame
# (z up). 4x4 matrices row-major.
twin = {
    "_source": "build_neutral.py - do not hand-edit",
    "meshes": {"base": "meshes/base.stl", "hinge": "meshes/hinge.stl", "leg": "meshes/leg10.stl"},
    "mesh_scale": 0.001,
    "hips": [{"pos": [float(v) for v in HIP[hn]],
              "yaw_deg": float(np.degrees(np.arctan2(HIP[hn][1], HIP[hn][0])))} for hn in hip_order],
    "hinges_in_hip": [T.flatten().tolist() for T in tpl["hinges"]],
    "knee_in_hip": [float(v) for v in tpl["knee"]],
    "knee_axis": [float(v) for v in knee_axis],
    "leg_in_knee": T_leg_K.flatten().tolist(),
}
twin_out = r"C:\projects\arachne\servo-web-control\assets\twin_model.json"
json.dump(twin, open(twin_out, "w"), indent=1)
print(f"written: {twin_out}")
print("hip order (sim leg k -> CAD hip revolute, angle):",
      [(k, hn, round(float(np.degrees(np.arctan2(HIP[hn][1], HIP[hn][0])) % 360), 1)) for k, hn in enumerate(hip_order)])

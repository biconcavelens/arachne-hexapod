"""
Builds the MuJoCo model from the OnShape assembly ground truth.

MODE "static": every part placed at its exact OnShape world transform, no
joints. Used to VERIFY the frames are right (render should reproduce the
OnShape assembly view).

MODE "tree": proper kinematic tree with hip + knee joints, positioned so
qpos=0 reproduces the OnShape pose.

Ground truth in onshape_assembly.json - see that file for provenance.
"""
import json
import sys

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment

ROOT = r"C:\projects\arachne\vorpal-mujoco"
J = json.load(open(rf"{ROOT}\onshape_assembly.json"))
MODE = sys.argv[1] if len(sys.argv) > 1 else "static"


def mat(v):
    m = np.eye(4)
    m[:3, :] = np.array(v, dtype=float).reshape(3, 4)
    return m


def quat(R):
    """rotation matrix -> mujoco wxyz quaternion"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def fmt(v):
    return " ".join(f"{x:.6f}" for x in v)


occ = {k: mat(v) for k, v in J["occurrences"].items()}
T_base = occ["Base <1>"]

# Hinge mesh's own local centroid (mm) - see BUG FIX note above for why this
# matters more than the mesh's local origin.
_hinge_centroid_local_mm = np.array(trimesh.load(rf"{ROOT}\meshes\hinge.stl").centroid)
_hinge_centroid_local_m = _hinge_centroid_local_mm / 1000.0


def hinge_world_centroid(i):
    T = occ[f"LEG_Hinge_TowerPro <{i}>"]
    return (T @ np.array([*_hinge_centroid_local_m, 1.0]))[:3]

# leg part name -> mesh basename
LEGS = {"Leg 6 <1>": "leg6", "Leg 7 <1>": "leg7", "Leg 8 <1>": "leg8",
        "Leg 9 <1>": "leg9", "Leg 10 <1>": "leg10", "Legs-11 <1>": "leg11"}

# BUG FIX (2026-09-12): the original PROXIMAL/DISTAL split below was guessed
# from eyeballing a render and classified hinges by comparing OCCURRENCE
# ORIGIN (where the mesh's local (0,0,0) lands) to hip/knee positions. But
# hinge.stl's local origin sits ~190mm from its own mesh centroid, so after
# rotation the origin can land 150-300mm from anything real - nowhere near
# the ~22-66mm range every hinge is ACTUALLY at once you use the mesh
# CENTROID (transformed the same way) instead. That bug is why one leg's
# hinge was missing/floating. Classification is now done properly below via
# capacitated bipartite assignment (2 hinges per hip) on centroid distance.

# knee joint: origin in that leg's own local frame, axis -Y local
KNEE_MATE = {
    "Leg 6 <1>": [0.116487, -0.007207, -0.051058],
    "Leg 7 <1>": [0.116487, 0.024793, -0.051058],
    "Leg 8 <1>": [0.116487, 0.056793, -0.051058],
    "Leg 9 <1>": [0.136487, -0.007207, -0.051058],
    "Leg 10 <1>": [0.136487, 0.024793, -0.051058],
    "Legs-11 <1>": [0.136487, 0.056793, -0.051058],
}
HIP_MATE = {k: v for k, v in J["hip_revolutes_base_local"].items() if not k.startswith("_")}

x = []
x.append('<mujoco model="vorpal_hexapod_onshape">')
x.append('  <compiler angle="degree" meshdir="meshes"/>')
x.append('  <option gravity="0 0 0"/>')
x.append('  <visual><global offwidth="1600" offheight="1200"/></visual>')
x.append('  <asset>')
x.append('    <mesh name="base_mesh" file="base.stl" scale="0.001 0.001 0.001"/>')
x.append('    <mesh name="hinge_mesh" file="hinge.stl" scale="0.001 0.001 0.001"/>')
for nm in LEGS.values():
    x.append(f'    <mesh name="{nm}_mesh" file="{nm}.stl" scale="0.001 0.001 0.001"/>')
x.append('  </asset>')
x.append('  <worldbody>')
x.append('    <light diffuse=".9 .9 .9" pos="0.2 -0.2 0.6" dir="-0.3 0.3 -1"/>')
x.append('    <light diffuse=".4 .4 .4" pos="-0.3 0.2 0.4" dir="0.4 -0.3 -1"/>')

# ---- base ----
x.append(f'    <body name="base" pos="{fmt(T_base[:3,3])}" quat="{fmt(quat(T_base[:3,:3]))}">')
x.append('      <geom type="mesh" mesh="base_mesh" rgba="0.75 0.35 0.2 1" contype="0" conaffinity="0"/>')

if MODE == "static":
    # everything pinned at its exact OnShape world transform, as children of
    # base (so poses are expressed relative to base). Placement uses the
    # full transform (always correct); only the debug color below used the
    # buggy classification, now removed - all hinges drawn the same color.
    Tb_inv = np.linalg.inv(T_base)
    for i in range(1, 13):
        T = Tb_inv @ occ[f"LEG_Hinge_TowerPro <{i}>"]
        x.append(f'      <geom name="hinge{i}" type="mesh" mesh="hinge_mesh" '
                 f'pos="{fmt(T[:3,3])}" quat="{fmt(quat(T[:3,:3]))}" rgba="0.95 0.75 0.1 1"/>')
    for legname, meshnm in LEGS.items():
        T = Tb_inv @ occ[legname]
        x.append(f'      <geom name="{meshnm}" type="mesh" mesh="{meshnm}_mesh" '
                 f'pos="{fmt(T[:3,3])}" quat="{fmt(quat(T[:3,:3]))}" rgba="0.25 0.45 0.85 1"/>')
    x.append('    </body>')
else:
    # ---- kinematic tree ----
    hip_names = list(HIP_MATE.keys())
    hipworld = {n: (T_base @ np.array([*HIP_MATE[n], 1.0]))[:3] for n in hip_names}

    # knee joint (world) per leg, from that leg's own local mate
    kneeinfo = {}
    for legname in LEGS:
        T = occ[legname]
        o = np.array(KNEE_MATE[legname], float)
        kneeinfo[legname] = {
            "pos": (T @ np.array([*o, 1.0]))[:3],
            "axis": T[:3, :3] @ np.array([0, -1, 0], float),
        }

    def assign(items, targets, item_pos, target_pos):
        """Proper 1-1 assignment (Hungarian algorithm) minimizing total
        distance - NOT naive per-item argmin, which lets two items fight
        over the same target and silently orphans another target (the bug
        that left one leg with no hinge in the first version of this).
        Returns {item: target}."""
        cost = np.array([[np.linalg.norm(item_pos[i] - target_pos[t]) for t in targets] for i in items])
        rows, cols = linear_sum_assignment(cost)
        result = {items[rows[k]]: targets[cols[k]] for k in range(len(rows))}
        maxd = max(cost[rows[k], cols[k]] for k in range(len(rows)))
        print(f"  assignment max distance: {maxd*1000:.2f} mm", file=sys.stderr)
        return result

    # leg -> its hip joint (one-to-one over all 6 legs / 6 hips at once)
    hip_for_leg = assign(
        list(kneeinfo.keys()), hip_names,
        {lg: ki["pos"] for lg, ki in kneeinfo.items()}, hipworld)
    leg_for_hip = {hip: lg for lg, hip in hip_for_leg.items()}

    # All 12 hinges -> the 6 hips, CAPACITY 2 EACH (every leg's mechanism
    # uses two hinges in series, matching the assembly guide's "two
    # identical U-shaped halves"). Done via centroid distance (see BUG FIX
    # note), with each hip duplicated into 2 slots so plain 1-1 Hungarian
    # assignment gives a capacitated 2-per-hip result.
    all_hinges = list(range(1, 13))
    hinge_centroid = {h: hinge_world_centroid(h) for h in all_hinges}
    hip_slots = [n for n in hip_names for _ in range(2)]
    slot_pos = {i: hipworld[hip_slots[i]] for i in range(len(hip_slots))}
    slot_assign = assign(all_hinges, list(range(len(hip_slots))), hinge_centroid, slot_pos)
    hinges_for_hip = {n: [] for n in hip_names}
    for h, slot in slot_assign.items():
        hinges_for_hip[hip_slots[slot]].append(h)

    # Within each hip's pair, the one closer to the HIP itself rides on the
    # hip body (rotates with hip only); the one closer to that leg's KNEE
    # joint rides on the knee body (rotates with hip+knee).
    prox_for_hip, dist_for_leg = {}, {}
    for hip, pair in hinges_for_hip.items():
        legname = leg_for_hip[hip]
        assert len(pair) == 2, f"{hip} got {len(pair)} hinges, expected 2: {pair}"
        d_hip = {h: np.linalg.norm(hinge_centroid[h] - hipworld[hip]) for h in pair}
        d_knee = {h: np.linalg.norm(hinge_centroid[h] - kneeinfo[legname]["pos"]) for h in pair}
        prox, dist = sorted(pair, key=lambda h: d_hip[h] - d_knee[h])
        prox_for_hip[hip] = prox
        dist_for_leg[legname] = dist
        print(f"  {legname:14s} hip={hip:10s}: prox=H{prox}({d_hip[prox]*1000:.1f}mm to hip) "
              f"dist=H{dist}({d_knee[dist]*1000:.1f}mm to knee)", file=sys.stderr)

    Tb_inv = np.linalg.inv(T_base)
    for idx, (legname, meshnm) in enumerate(LEGS.items()):
        hipn = hip_for_leg[legname]
        # hip body frame: at the hip joint, axis-aligned with base
        T_hip_w = np.eye(4)
        T_hip_w[:3, :3] = T_base[:3, :3]
        T_hip_w[:3, 3] = hipworld[hipn]
        T_hip_rel = Tb_inv @ T_hip_w
        Thw_inv = np.linalg.inv(T_hip_w)
        x.append(f'      <body name="hip{idx}" pos="{fmt(T_hip_rel[:3,3])}" quat="{fmt(quat(T_hip_rel[:3,:3]))}">')
        x.append(f'        <joint name="hip{idx}_joint" type="hinge" axis="0 0 1" pos="0 0 0" range="-90 90"/>')
        ph = prox_for_hip.get(hipn)
        if ph is not None:
            T = Thw_inv @ occ[f"LEG_Hinge_TowerPro <{ph}>"]
            x.append(f'        <geom name="proxhinge{idx}" type="mesh" mesh="hinge_mesh" '
                     f'pos="{fmt(T[:3,3])}" quat="{fmt(quat(T[:3,:3]))}" rgba="0.95 0.75 0.1 1"/>')

        # knee body: distal hinge + the leg, rigid together
        ki = kneeinfo[legname]
        T_knee_w = np.eye(4)
        T_knee_w[:3, :3] = T_base[:3, :3]
        T_knee_w[:3, 3] = ki["pos"]
        T_knee_rel = Thw_inv @ T_knee_w
        Tkw_inv = np.linalg.inv(T_knee_w)
        axis_in_knee = Tkw_inv[:3, :3] @ ki["axis"]
        x.append(f'        <body name="knee{idx}" pos="{fmt(T_knee_rel[:3,3])}" quat="{fmt(quat(T_knee_rel[:3,:3]))}">')
        x.append(f'          <joint name="knee{idx}_joint" type="hinge" axis="{fmt(axis_in_knee)}" pos="0 0 0" range="-90 90"/>')
        dh = dist_for_leg.get(legname)
        if dh is not None:
            T = Tkw_inv @ occ[f"LEG_Hinge_TowerPro <{dh}>"]
            x.append(f'          <geom name="disthinge{idx}" type="mesh" mesh="hinge_mesh" '
                     f'pos="{fmt(T[:3,3])}" quat="{fmt(quat(T[:3,:3]))}" rgba="0.2 0.75 0.3 1"/>')
        T = Tkw_inv @ occ[legname]
        x.append(f'          <geom name="{meshnm}" type="mesh" mesh="{meshnm}_mesh" '
                 f'pos="{fmt(T[:3,3])}" quat="{fmt(quat(T[:3,:3]))}" rgba="0.25 0.45 0.85 1"/>')
        x.append('        </body>')
        x.append('      </body>')
    x.append('    </body>')

x.append('  </worldbody>')
x.append('</mujoco>')

out = rf"{ROOT}\hexapod.xml"
open(out, "w").write("\n".join(x) + "\n")
print(f"written: {out}  (mode={MODE})")

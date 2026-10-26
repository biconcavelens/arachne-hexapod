"""
Works out the hexapod's real kinematic structure from the OnShape assembly
ground truth (onshape_assembly.json), so the MJCF can be built from measured
transforms instead of guesses.
"""
import json
import numpy as np
import trimesh

J = json.load(open(r"C:\projects\arachne\vorpal-mujoco\onshape_assembly.json"))


def mat(v):
    """12-element row-major [R|t] -> 4x4."""
    m = np.eye(4)
    m[:3, :] = np.array(v, dtype=float).reshape(3, 4)
    return m


occ = {k: mat(v) for k, v in J["occurrences"].items()}

# --- world positions of the hip joints (mate origins are Base-local) ---
T_base = occ["Base <1>"]
hips = {}
for name, o in J["hip_revolutes_base_local"].items():
    if name.startswith("_"):
        continue
    hips[name] = (T_base @ np.array([*o, 1.0]))[:3]
hip_axis_world = (T_base[:3, :3] @ np.array(J["hip_revolutes_base_local"]["_axis"], float))

print("=== HIP joints (world) ===")
for n, p in hips.items():
    r = np.hypot(p[0] - T_base[0, 3], p[1] - T_base[1, 3])
    ang = np.degrees(np.arctan2(p[1] - T_base[1, 3], p[0] - T_base[0, 3])) % 360
    print(f"  {n:12s} world={np.round(p,4)}  r={r*1000:6.2f}mm  ang={ang:7.2f}deg")
print(f"  axis (world) = {np.round(hip_axis_world,4)}")

# --- world positions of the knee joints (mate origins are Leg-local) ---
legmap = {
    "Revolute 7  (Leg 6)": "Leg 6 <1>",
    "Revolute 8  (Leg 7)": "Leg 7 <1>",
    "Revolute 9  (Leg 8)": "Leg 8 <1>",
    "Revolute 10 (Leg 9)": "Leg 9 <1>",
    "Revolute 11 (Leg 10)": "Leg 10 <1>",
    "Revolute 12 (Legs-11)": "Legs-11 <1>",
}
knee_axis_local = np.array(J["knee_revolutes_leg_local"]["_axis"], float)
knees = {}
for rev, legname in legmap.items():
    o = np.array(J["knee_revolutes_leg_local"][rev], float)
    T = occ[legname]
    knees[legname] = {
        "pos": (T @ np.array([*o, 1.0]))[:3],
        "axis": T[:3, :3] @ knee_axis_local,
    }

print("\n=== KNEE joints (world) ===")
for n, d in knees.items():
    print(f"  {n:14s} world={np.round(d['pos'],4)}  axis={np.round(d['axis'],3)}")

# --- world centroids of hinges (using the canonical part-frame mesh) ---
hinge_mesh = trimesh.load(r"C:\projects\arachne\vorpal-mujoco\meshes\hinge.stl")
hinge_c_local = np.array(hinge_mesh.centroid) / 1000.0  # mm -> m

leg_mesh = trimesh.load(r"C:\projects\arachne\vorpal-mujoco\meshes\leg.stl")
leg_c_local = np.array(leg_mesh.centroid) / 1000.0

print("\n=== HINGE association (nearest joint to each hinge centroid) ===")
rows = []
for k, T in occ.items():
    if not k.startswith("LEG_Hinge"):
        continue
    cw = (T @ np.array([*hinge_c_local, 1.0]))[:3]
    dh = {n: np.linalg.norm(cw - p) for n, p in hips.items()}
    dk = {n: np.linalg.norm(cw - d["pos"]) for n, d in knees.items()}
    bh = min(dh, key=dh.get)
    bk = min(dk, key=dk.get)
    kind = "HIP " if dh[bh] < dk[bk] else "KNEE"
    near = bh if dh[bh] < dk[bk] else bk
    rows.append((k, kind, near, min(dh[bh], dk[bk])))
    print(f"  {k:26s} centroid={np.round(cw,4)}  -> {kind} {near:14s} d={min(dh[bh],dk[bk])*1000:6.2f}mm")

print("\n=== LEG centroids (world) vs their hip ===")
for legname in legmap.values():
    T = occ[legname]
    cw = (T @ np.array([*leg_c_local, 1.0]))[:3]
    dh = {n: np.linalg.norm(cw - p) for n, p in hips.items()}
    bh = min(dh, key=dh.get)
    print(f"  {legname:14s} centroid={np.round(cw,4)} nearest hip={bh} d={dh[bh]*1000:6.2f}mm")

print("\n=== summary ===")
n_hip = sum(1 for r in rows if r[1] == "HIP ")
print(f"  hinges classified as HIP-side : {n_hip}")
print(f"  hinges classified as KNEE-side: {len(rows)-n_hip}")

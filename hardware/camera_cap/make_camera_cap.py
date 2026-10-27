"""
Vorpal cap with a slide-in USB-camera mount, plus the camera pose for the simulator.

The mount sits on the cap's knob (a flat 27 mm disk on top of the dome): a flared
pedestal, then two C-shaped side rails that hold the camera board by its edges.
The board slides in from the top; its front and back stay open (lens and cable).
The rails tilt forward so the camera looks TILT_DEG below horizontal, which keeps
the legs out of view. A slot through the dome behind the knob takes the cable.

    python make_camera_cap.py

Writes, next to this file:
  cap_with_camera_mount.stl   the whole cap to print (rim down, like the original)
  camera_mount_only.stl       just the mount, to glue onto an existing cap's knob
  camera_pose.json            lens position and axes in the robot base frame (read by fault-sim)
  preview.png                 side and front views

All dimensions in mm, in the robot base frame (x toward the front leg L0, z up).
The original cap STL is already in this frame: its bounds match the twin's base.
"""
import json
import math
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CAP_IN = os.path.join(ROOT, "Vorpal needed", "files", "Hexapod_V1r8a_Cap_VARIATION_NoSkeletonize.stl")

# ---- the camera (measure yours; these are the numbers you gave)
BOARD_W = 30.0      # board width
BOARD_H = 25.0      # board height
BOARD_T = 1.6       # board thickness (a typical PCB; MEASURE IT)
CLEAR = 0.4         # fit clearance added to width and thickness
GRIP = 2.0          # how deep the board edges sit inside the rails (your "2 mm")
LENS_FWD = 4.0      # lens front ahead of the board face (only for the sim pose)

# ---- the mount
LIP = 1.6           # thickness of the front and back lips of each rail
WALL = 2.4          # side wall thickness
TILT_DEG = 20.0     # optical axis below horizontal
PED_H = 8.0         # pedestal height above the knob
PED_DEPTH = 12.0    # pedestal top, front to back
SINK = 3.0          # rails and pedestal overlap this much so they fuse

# ---- the cap (measured from the STL)
KNOB_TOP = 84.15    # z of the flat knob top
KNOB_R = 13.4       # radius of the flat knob top
CABLE_X = -24.0     # cable slot centre, behind the knob
CABLE_SLOT = (7.0, 12.0)   # x by y size of the cable slot

WP = BOARD_W + CLEAR       # slot width
G = BOARD_T + CLEAR        # slot gap (front to back)
Z_TOP = KNOB_TOP + PED_H   # where the rails start


def box(extents, center):
    return trimesh.creation.box(extents=extents, transform=trimesh.transformations.translation_matrix(center))


def pedestal():
    """Flares from a disk on the knob to the rail footprint; sides under 35 degrees, prints without support."""
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    r = KNOB_R - 1.0
    bottom = np.c_[r * np.cos(ang), r * np.sin(ang), np.full_like(ang, KNOB_TOP - 1.0)]
    hx, hy = PED_DEPTH / 2, WP / 2 + WALL
    top = np.array([[sx * hx, sy * hy, Z_TOP] for sx in (-1, 1) for sy in (-1, 1)])
    return trimesh.convex.convex_hull(np.r_[bottom, top])


def rails():
    """C-channel rails plus a bottom bar, built at the origin, then tilted and lifted."""
    z0, z1 = -SINK, BOARD_H + 1.0
    h, zc = z1 - z0, (z0 + z1) / 2
    parts = [box((G + 2 * LIP, WP + 2 * WALL, SINK), (0, 0, -SINK / 2))]          # bottom bar the board rests on
    for s in (1, -1):
        parts.append(box((G + 2 * LIP, WALL, h), (0, s * (WP / 2 + WALL / 2), zc)))  # side wall
        for fx in (1, -1):                                                         # front and back lips
            parts.append(box((LIP, GRIP + 0.2, h), (fx * (G / 2 + LIP / 2), s * (WP / 2 - GRIP / 2 + 0.1), zc)))
    m = trimesh.boolean.union(parts, engine="manifold")
    tilt = trimesh.transformations.rotation_matrix(math.radians(TILT_DEG), [0, 1, 0])   # +x tips down
    m.apply_transform(tilt)
    m.apply_translation((0, 0, Z_TOP))
    return m, tilt[:3, :3]


def camera_pose(R):
    """Lens position and axes in the base frame (metres) for a MuJoCo <camera>."""
    fwd, up = R @ [1, 0, 0], R @ [0, 0, 1]            # optical axis, image up
    right = np.cross(fwd, up)                          # image right
    board_c = np.array([0, 0, Z_TOP]) + up * (BOARD_H / 2)
    lens = board_c + fwd * (G / 2 + LENS_FWD)
    return {
        "pos_m": (lens / 1000).round(5).tolist(),
        "forward": fwd.round(5).tolist(),
        "up": up.round(5).tolist(),
        "mujoco_xyaxes": np.r_[right, up].round(5).tolist(),   # camera x = right, y = up, looks along -z
        "tilt_deg": TILT_DEG,
        "note": "base-body frame; MuJoCo cameras look along their -z axis",
    }


def preview(mesh, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(11, 5))
    for i, (elev, azim, title) in enumerate(((0, -90, "side (front is right)"), (10, 0, "front"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        tri = mesh.vertices[mesh.faces]
        ax.add_collection3d(Poly3DCollection(tri, facecolor=(0.3, 0.4, 0.9), edgecolor="none", alpha=1.0))
        lo, hi = mesh.bounds
        c, span = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set(xlim=(c[0] - span, c[0] + span), ylim=(c[1] - span, c[1] + span), zlim=(c[2] - span, c[2] + span), title=title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    if not os.path.exists(CAP_IN):
        raise SystemExit(f"original cap STL not found: {CAP_IN} (download Vorpal from Thingiverse 2513566)")
    cap = trimesh.load(CAP_IN)
    rail, R = rails()
    mount = trimesh.boolean.union([pedestal(), rail], engine="manifold")
    slot = box((CABLE_SLOT[0], CABLE_SLOT[1], 50.0), (CABLE_X, 0, 65.0))
    full = trimesh.boolean.difference([trimesh.boolean.union([cap, mount], engine="manifold"), slot], engine="manifold")

    full.export(os.path.join(HERE, "cap_with_camera_mount.stl"))
    mount.export(os.path.join(HERE, "camera_mount_only.stl"))
    pose = camera_pose(R)
    json.dump(pose, open(os.path.join(HERE, "camera_pose.json"), "w"), indent=1)
    preview(full, os.path.join(HERE, "preview.png"))

    print(f"cap+mount: watertight {full.is_watertight}, bodies {len(full.split(only_watertight=False))}, "
          f"height {full.bounds[1][2] - full.bounds[0][2]:.1f} mm (original {cap.extents[2]:.1f})")
    print(f"mount only: watertight {mount.is_watertight}, {mount.extents.round(1)} mm")
    print(f"camera lens at {np.round(np.array(pose['pos_m']) * 1000, 1)} mm in the base frame, looking {TILT_DEG} deg down")

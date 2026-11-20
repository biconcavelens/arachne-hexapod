"""
Builds the MJX room scene: hexapod_mjx.xml plus
  - the body camera "ego" at the camera-cap lens pose (hardware/camera_cap/camera_pose.json)
  - a 6 x 6 m room (floor, four walls)
  - household objects as mocap bodies, placed anew every episode by room_env.py: tables, chairs,
    a sofa, shelves, boxes, a bin, balls, plants, a floor lamp, books, shoes (all kept off the
    path), and rugs and cables (allowed across the path: seen, not felt)
Only primitives that mjx.ray can hit (box, capsule, sphere, plane): no cylinders, no meshes.

Collision: the robot touches only the floor. Objects are seen by the camera but not solid in the
physics: MJX checks every foot against every solid geom in every substep, and 32 solid parts made
the simulation 10x slower (even the 4 rugs and cables alone halved it). room_env.py instead ends
the episode when the robot enters a furniture footprint; rugs and cables are flat enough (6 and
8 mm) to leave out of the physics in this stage. Solid obstacles come with the rough-ground stage. Robot geoms are in geom group 1 so the camera rays skip them.

    python room_scene.py            # writes room_scene.xml next to this file and prints counts
"""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_XML = os.path.join(os.path.dirname(HERE), "hexapod_mjx.xml")
CAM_JSON = os.path.join(os.path.dirname(os.path.dirname(HERE)), "hardware", "camera_cap", "camera_pose.json")
OUT = os.path.join(HERE, "room_scene.xml")
DYN_OUT = os.path.join(HERE, "room_dyn_scene.xml")                          # with the movers
RENDER_XML = os.path.join(os.path.dirname(HERE), "hexapod_physics.xml")       # full meshes, for pictures
RENDER_OUT = os.path.join(os.path.dirname(HERE), "room_render.xml")         # next to meshes/
TOP_H, TOP_FOVY = 7.0, 48.0          # overhead camera: height (m) and field of view (deg)

ROOM_HALF = 3.0                     # m; the path stays within ~2.1 m of the start
WALL_H, WALL_T = 1.2, 0.05
FOVY_DEG = 45.0
CAM_W, CAM_H = map(int, os.environ.get("ARACHNE_CAM", "24x16").split("x"))   # rays per image (columns, rows)

# object kinds: name, count, clearance radius from the path (m; None = may cross the path),
# and parts: (type, half sizes, position in the object frame, rgba, solid)
def _legs4(dx, dy, h, r, rgba):
    return [("capsule", (r, h / 2), (sx * dx, sy * dy, h / 2), rgba, True) for sx in (-1, 1) for sy in (-1, 1)]


WOOD, DARK, GREY, WHITE = (0.55, 0.38, 0.22, 1), (0.25, 0.2, 0.17, 1), (0.5, 0.5, 0.52, 1), (0.85, 0.85, 0.82, 1)
KINDS = [
    ("table", 1, 0.55, [("box", (0.5, 0.35, 0.015), (0, 0, 0.72), WOOD, False)] + _legs4(0.45, 0.3, 0.7, 0.02, WOOD)),
    ("chair", 2, 0.35, [("box", (0.2, 0.2, 0.02), (0, 0, 0.45), DARK, False),
                        ("box", (0.02, 0.2, 0.2), (-0.19, 0, 0.67), DARK, False)] + _legs4(0.17, 0.17, 0.44, 0.012, DARK)),
    ("sofa", 1, 0.75, [("box", (0.9, 0.4, 0.2), (0, 0, 0.2), (0.35, 0.4, 0.55, 1), True),
                       ("box", (0.9, 0.1, 0.25), (0, 0.35, 0.45), (0.33, 0.38, 0.52, 1), False)]),
    ("shelf", 2, 0.45, [("box", (0.4, 0.17, 0.6), (0, 0, 0.6), WOOD, True)]),
    ("cardbox", 3, 0.3, [("box", (0.18, 0.14, 0.12), (0, 0, 0.12), (0.7, 0.55, 0.35, 1), True)]),
    ("bin", 1, 0.2, [("capsule", (0.12, 0.12), (0, 0, 0.24), GREY, True)]),
    ("ball", 2, 0.15, [("sphere", (0.09,), (0, 0, 0.09), (0.85, 0.25, 0.2, 1), True)]),
    ("plant", 2, 0.3, [("capsule", (0.1, 0.06), (0, 0, 0.16), (0.6, 0.35, 0.25, 1), True),
                       ("sphere", (0.22,), (0, 0, 0.45), (0.2, 0.5, 0.22, 1), False)]),
    ("lamp", 1, 0.25, [("box", (0.13, 0.13, 0.015), (0, 0, 0.015), DARK, True),
                       ("capsule", (0.012, 0.7), (0, 0, 0.72), DARK, False),
                       ("sphere", (0.16,), (0, 0, 1.45), WHITE, False)]),
    ("book", 2, 0.15, [("box", (0.1, 0.07, 0.012), (0, 0, 0.012), (0.2, 0.3, 0.6, 1), True)]),
    ("shoe", 2, 0.15, [("box", (0.13, 0.05, 0.04), (0, 0, 0.04), (0.15, 0.15, 0.15, 1), True)]),
    ("rug", 2, None, [("box", (0.6, 0.4, 0.003), (0, 0, 0.003), (0.6, 0.2, 0.25, 1), True)]),
    ("cable", 2, None, [("capsule", (0.004, 0.45), (0, 0, 0.004), (0.1, 0.1, 0.1, 1), True)]),
]


# things that move by themselves (room_dyn_env.py): name, footprint radius (m), parts as in KINDS
MOVERS = [
    ("vacuum", 0.17, [("ellipsoid", (0.17, 0.17, 0.045), (0, 0, 0.045), (0.15, 0.15, 0.17, 1), False)]),
    ("ball_a", 0.08, [("sphere", (0.08,), (0, 0, 0.08), (0.2, 0.45, 0.85, 1), False)]),
    ("ball_b", 0.06, [("sphere", (0.06,), (0, 0, 0.06), (0.95, 0.75, 0.15, 1), False)]),
    ("person", 0.2, [("capsule", (0.06, 0.38), (0.0, 0.1, 0.44), (0.2, 0.22, 0.3, 1), False),
                     ("capsule", (0.06, 0.38), (0.0, -0.1, 0.44), (0.2, 0.22, 0.3, 1), False),
                     ("box", (0.13, 0.05, 0.04), (0.04, 0.1, 0.04), (0.1, 0.1, 0.1, 1), False),
                     ("box", (0.13, 0.05, 0.04), (0.04, -0.1, 0.04), (0.1, 0.1, 0.1, 1), False)]),
]


def camera_pose():
    return json.load(open(CAM_JSON))


def build(movers=False):
    tree = ET.parse(ROBOT_XML)
    root = tree.getroot()
    root.set("model", "arachne_room")
    world = root.find("worldbody")
    for g in world.iter("geom"):
        if g.get("name") != "floor":
            g.set("group", "1")                                    # robot: invisible to the camera rays
    floor = world.find("geom[@name='floor']")
    floor.set("size", f"{ROOM_HALF + 1} {ROOM_HALF + 1} 0.1")
    floor.set("rgba", "0.6 0.45 0.3 1")
    floor.attrib.pop("material", None)

    pose = camera_pose()
    base = world.find("body[@name='base']")
    ET.SubElement(base, "camera", name="ego", pos=" ".join(f"{x:.5f}" for x in pose["pos_m"]),
                  xyaxes=" ".join(f"{x:.5f}" for x in pose["mujoco_xyaxes"]), fovy=f"{FOVY_DEG}")

    objects = _add_room(world, movers=movers)
    ET.indent(tree, "  ")
    tree.write(DYN_OUT if movers else OUT)
    return objects


def _add_room(world, wall_mat=None, movers=False):
    """Walls and the mocap objects (same order in the physics and the render model)."""
    for i, (x, y, sx, sy) in enumerate([(ROOM_HALF, 0, WALL_T, ROOM_HALF), (-ROOM_HALF, 0, WALL_T, ROOM_HALF),
                                        (0, ROOM_HALF, ROOM_HALF, WALL_T), (0, -ROOM_HALF, ROOM_HALF, WALL_T)]):
        w = ET.SubElement(world, "geom", name=f"wall{i}", type="box", size=f"{sx} {sy} {WALL_H / 2}",
                          pos=f"{x} {y} {WALL_H / 2}", rgba="0.82 0.8 0.74 1", contype="0", conaffinity="0", group="0")
        if wall_mat:
            w.set("material", wall_mat)
    objects = []
    for kind, count, clear, parts in KINDS:
        for k in range(count):
            name = f"{kind}{k}"
            b = ET.SubElement(world, "body", name=name, mocap="true", pos=f"{50 + len(objects)} 0 -1")
            for j, (typ, half, pos, rgba, solid) in enumerate(parts):
                ET.SubElement(b, "geom", name=f"{name}_{j}", type=typ, size=" ".join(f"{v:g}" for v in half),
                              pos=" ".join(f"{v:g}" for v in pos), rgba=" ".join(f"{v:g}" for v in rgba),
                              contype="0", conaffinity="0", group="0")
            objects.append((kind, clear, name))
    if movers:                                    # after the objects, so object mocap indices stay the same
        for i, (name, foot, parts) in enumerate(MOVERS):
            b = ET.SubElement(world, "body", name=name, mocap="true", pos=f"{80 + i} 0 -1")
            for j, (typ, half, pos, rgba, solid) in enumerate(parts):
                ET.SubElement(b, "geom", name=f"{name}_{j}", type=typ, size=" ".join(f"{v:g}" for v in half),
                              pos=" ".join(f"{v:g}" for v in pos), rgba=" ".join(f"{v:g}" for v in rgba),
                              contype="0", conaffinity="0", group="0")
    return objects


def build_render():
    """The same room around the full-mesh robot, with lights, a wood floor and an overhead camera."""
    tree = ET.parse(RENDER_XML)
    root = tree.getroot()
    root.set("model", "arachne_room_render")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="wood", type="2d", builtin="checker", rgb1="0.52 0.36 0.22",
                  rgb2="0.46 0.31 0.19", width="512", height="512", mark="edge", markrgb="0.3 0.2 0.12")
    ET.SubElement(asset, "material", name="wood", texture="wood", texrepeat="40 8", reflectance="0.05")
    ET.SubElement(asset, "material", name="wallpaint", rgba="0.86 0.84 0.78 1")
    world = root.find("worldbody")
    for li in list(world.findall("light")):
        world.remove(li)
    for x, y in ((-1.2, -1.2), (1.2, 1.2), (-1.2, 1.2), (1.2, -1.2)):
        ET.SubElement(world, "light", pos=f"{x} {y} 2.4", dir="0 0 -1", diffuse="0.45 0.44 0.4",
                      specular="0.05 0.05 0.05", castshadow="false")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", diffuse="0.35 0.35 0.35", castshadow="true")
    floor = world.find("geom[@name='floor']")
    floor.set("size", f"{ROOM_HALF + 1} {ROOM_HALF + 1} 0.1")
    floor.set("material", "wood")
    ET.SubElement(world, "camera", name="top", pos=f"0 0 {TOP_H}", xyaxes="1 0 0 0 1 0", fovy=f"{TOP_FOVY}")
    pose = camera_pose()
    base = world.find("body[@name='base']")
    ET.SubElement(base, "camera", name="ego", pos=" ".join(f"{x:.5f}" for x in pose["pos_m"]),
                  xyaxes=" ".join(f"{x:.5f}" for x in pose["mujoco_xyaxes"]), fovy=f"{FOVY_DEG}")
    _add_room(world, wall_mat="wallpaint", movers=True)
    vis = root.find("visual")
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.25 0.25 0.25")
    # parked objects sit 50-72 m away; without this the auto extent pushes the near clip plane to ~0.7 m
    ET.SubElement(vis, "map", znear="0.005", zfar="30")
    ET.SubElement(root, "statistic", extent="4", center="0 0 0.3")
    ET.indent(tree, "  ")
    tree.write(RENDER_OUT)


OBJECTS = [(k, c, f"{k}{i}") for k, n, c, _ in KINDS for i in range(n)]

if __name__ == "__main__":
    objs = build()
    build(movers=True)
    build_render()
    import mujoco
    m = mujoco.MjModel.from_xml_path(OUT)
    print("wrote", OUT, "| bodies", m.nbody, "geoms", m.ngeom, "mocap", m.nmocap, "qpos", m.nq,
          "| solid object geoms", int(((m.geom_conaffinity & 2) > 0).sum()))
    r = mujoco.MjModel.from_xml_path(RENDER_OUT)
    print("render model", RENDER_OUT, "| qpos", r.nq, "mocap", r.nmocap, "cameras", [r.camera(i).name for i in range(r.ncam)])

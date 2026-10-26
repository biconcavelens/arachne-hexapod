"""Render the generated Vorpal model and report mesh contacts.

Examples:
  python render_snapshot.py
  python render_snapshot.py hip0=22,knee0=-20 --view=front
  python render_snapshot.py 0=22,1=-44 --view=top  # legacy hip shorthand
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from PIL import Image


XML_PATH = r"C:\projects\arachne\vorpal-mujoco\hexapod.xml"
OUT_PATH = r"C:\projects\arachne\vorpal-mujoco\snapshot.png"

VIEWS = {
    "top": dict(lookat=(0, 0, -0.035), distance=0.43, azimuth=90, elevation=-89),
    "front": dict(lookat=(0, 0, -0.060), distance=0.38, azimuth=0, elevation=-8),
    "iso": dict(lookat=(0, 0, -0.055), distance=0.40, azimuth=35, elevation=-20),
}


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, name: str, degrees: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Unknown joint: {name}")
    data.qpos[model.jnt_qposadr[joint_id]] = np.radians(degrees)


def parse_args(argv: list[str], model: mujoco.MjModel, data: mujoco.MjData) -> str:
    view = "iso"
    for arg in argv:
        if arg.startswith("--view="):
            view = arg.partition("=")[2].lower()
            continue
        for pair in arg.split(","):
            if not pair:
                continue
            channel, raw_degrees = pair.split("=", 1)
            degrees = float(raw_degrees)
            if channel.isdigit():  # Preserve the original script's hip shorthand.
                joint_name = f"hip{channel}_joint"
            elif channel.startswith("hip"):
                joint_name = f"{channel}_joint"
            elif channel.startswith("knee"):
                joint_name = f"{channel}_joint"
            else:
                raise ValueError(f"Expected hipN, kneeN, or a bare hip number; got {channel!r}")
            set_joint(model, data, joint_name, degrees)
    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}; choose one of {', '.join(VIEWS)}")
    return view


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
camera_view = parse_args(sys.argv[1:], model, data)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=900, width=1280)
camera = mujoco.MjvCamera()
for field, value in VIEWS[camera_view].items():
    setattr(camera, field, value)
renderer.update_scene(data, camera=camera)
Image.fromarray(renderer.render()).save(OUT_PATH)
renderer.close()
print(f"written: {OUT_PATH} ({camera_view} view)")

mujoco.mj_collision(model, data)
print(f"ncon: {data.ncon}")
for index in range(data.ncon):
    contact = data.contact[index]
    geom_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    geom_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
    print(f"  contact: {geom_a} <-> {geom_b}")

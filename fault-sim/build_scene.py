"""
Builds hexapod_camera.xml: the lean physics model (vorpal-mujoco/hexapod_mjx.xml) plus
  - the full twin's visual meshes (base, hinges, legs), with no collisions
  - the camera cap (hardware/camera_cap/cap_with_camera_mount.stl), with no collisions
  - the body camera "ego" at the lens pose from hardware/camera_cap/camera_pose.json
  - a textured floor (simulate.py re-randomises the texture every episode)

    python build_scene.py            # write the XML
    python build_scene.py --check    # also render the camera: how much of the robot does it see?
                                     # writes camera_view.png (needs an OpenGL context)
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

import config as C


def build():
    tree = ET.parse(C.MJX_XML)
    root = tree.getroot()
    phys = ET.parse(C.PHYS_XML).getroot()
    out_dir = os.path.dirname(C.SCENE_XML)
    root.find("compiler").set("meshdir", os.path.relpath(C.MESH_DIR, out_dir).replace("\\", "/"))

    asset = root.find("asset")
    for mesh in phys.iter("mesh"):                        # base_mesh, hinge_mesh, leg_mesh
        asset.append(mesh)
    ET.SubElement(asset, "mesh", name="cap_mesh", scale="0.001 0.001 0.001",
                  file=os.path.relpath(C.CAP_STL, C.MESH_DIR).replace("\\", "/"))
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="flat", width="512", height="512", rgb1=".5 .5 .5")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="8 8", reflectance="0.05")
    root.find(".//geom[@name='floor']").set("material", "floor_mat")

    bodies = {b.get("name"): b for b in root.iter("body")}
    for pb in phys.iter("body"):
        for g in pb.findall("geom"):
            if g.get("type") == "mesh":
                attrs = {k: v for k, v in g.attrib.items() if k != "mass"}
                attrs.update(name=g.get("name") + "_vis", contype="0", conaffinity="0", group="1")
                ET.SubElement(bodies[pb.get("name")], "geom", attrs)

    base = bodies["base"]
    ET.SubElement(base, "geom", name="cap_vis", type="mesh", mesh="cap_mesh", pos=f"0 0 {C.CAP_Z_OFFSET}",
                  contype="0", conaffinity="0", group="1", rgba="0.3 0.35 0.8 1")
    pose = C.camera_pose()
    p = np.array(pose["pos_m"]) + np.array([0, 0, C.CAP_Z_OFFSET])
    ET.SubElement(base, "camera", name="ego", pos=" ".join(f"{x:.5f}" for x in p),
                  xyaxes=" ".join(f"{x:.5f}" for x in pose["mujoco_xyaxes"]), fovy=f"{C.FOVY_DEG}")
    tree.write(C.SCENE_XML)
    return C.SCENE_XML


def check():
    """Render the body camera while standing and walking; report the share of pixels showing the robot."""
    import mujoco
    from PIL import Image

    from gaits_lib import Gait
    m = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    ren = mujoco.Renderer(m, C.IMG_H * 2, C.IMG_W * 2)
    floor = m.geom("floor").id
    gait, worst, snap = Gait(), 0.0, None
    for k in range(int(4.0 / C.CTRL_DT)):
        t = k * C.CTRL_DT
        d.ctrl[:] = np.radians(gait.command(t))
        for _ in range(C.N_SUBSTEPS):
            mujoco.mj_step(m, d)
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera="ego")
        seg = ren.render()
        ren.disable_segmentation_rendering()
        robot = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] != floor) & (seg[..., 0] >= 0)
        worst = max(worst, robot.mean())
        if snap is None and t > 2.0:
            ren.update_scene(d, camera="ego")
            snap = ren.render()
    Image.fromarray(snap).save(os.path.join(C.HERE, "camera_view.png"))
    print(f"robot pixels in the camera view: worst {worst * 100:.2f}% over 4 s of walking "
          f"(fovy {C.FOVY_DEG} deg, tilt {C.camera_pose()['tilt_deg']} deg) -> camera_view.png")


if __name__ == "__main__":
    print("written:", build())
    if "--check" in sys.argv:
        check()

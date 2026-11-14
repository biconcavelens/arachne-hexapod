"""
Builds hexapod_camera.xml: the lean physics model (vorpal-mujoco/hexapod_mjx.xml) plus
  - the full twin's visual meshes (base, hinges, legs), with no collisions
  - the camera cap (hardware/camera_cap/cap_with_camera_mount.stl), with no collisions
  - the body camera "ego" at the lens pose from hardware/camera_cap/camera_pose.json
  - a textured floor (simulate.py re-randomises the texture every episode)
  - for stage 1 and up (config.STAGE), a lit room and, from stage 2, objects (environment.py)

    python build_scene.py            # write the XML for the current stage
    python build_scene.py --check    # also render the camera: how much of the robot does it see?
                                     # writes camera_view.png, or camera_view_stage<N>.png (needs OpenGL)
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

import config as C
import environment


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
    if C.STAGE == 0:
        ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="flat", width="512", height="512", rgb1=".5 .5 .5")
        ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="8 8", reflectance="0.05")
    else:
        n = str(environment.FLOOR_TEX)
        ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="flat", width=n, height=n, rgb1=".5 .5 .5")
        ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texuniform="true",
                      texrepeat=f"{1 / environment.FLOOR_M} {1 / environment.FLOOR_M}")
        environment.add_to_xml(root, C.STAGE)
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


def check(seeds=(0, 1, 2, 3)):
    """Walk 4 s in a few randomised scenes. Reports the share of the body camera's pixels showing the
    robot, and saves one row per scene: body camera, a camera behind the robot, an overview of the room."""
    import copy

    import mujoco
    from PIL import Image

    from gaits_lib import Gait
    from simulate import randomize
    base = mujoco.MjModel.from_xml_path(C.SCENE_XML)
    robot_root = base.body_rootid[base.body("base").id]
    robot_geoms = np.flatnonzero(base.body_rootid[base.geom_bodyid] == robot_root)
    rows, worst = [], 0.0
    for seed in seeds:
        m = copy.deepcopy(base)
        animate = randomize(m, np.random.default_rng(seed))   # the scene as the episodes see it
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        ren = mujoco.Renderer(m, 360, 480)
        cams = {"behind": (1.0, 20.0, -18.0), "overview": (5.0, 30.0, -50.0)}   # distance, azimuth, elevation
        gait, shots = Gait(), None
        walk_s = 9.0 if animate else 4.0                     # movers start at 1-10 s: snapshot late enough to see some
        for k in range(int(walk_s / C.CTRL_DT)):
            t = k * C.CTRL_DT
            d.ctrl[:] = np.radians(gait.command(t))
            if animate:
                animate(d, t)
            for _ in range(C.N_SUBSTEPS):
                mujoco.mj_step(m, d)
            if seed == seeds[0] and worst is not None:
                ren.enable_segmentation_rendering()
                ren.update_scene(d, camera="ego")
                try:
                    seg = ren.render()
                    robot = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], robot_geoms)
                    worst = max(worst, robot.mean())
                except IndexError:                           # MuJoCo's segment lookup overflows in big scenes (stage 3+)
                    worst = None                             # and leaves the scene in segment colours: switch them off
                    ren.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = False
                    ren.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = False
                ren.disable_segmentation_rendering()
            if shots is None and t > walk_s - 1.0:
                ren.update_scene(d, camera="ego")
                shots = [ren.render()]
                for dist, az, el in cams.values():
                    cam = mujoco.MjvCamera()
                    cam.distance, cam.azimuth, cam.elevation = dist, az, el
                    cam.lookat[:] = d.xpos[m.body("base").id] + (0.6 * (dist > 2), 0, 0.1)
                    ren.update_scene(d, camera=cam)
                    shots.append(ren.render())
        ren.close()
        rows.append(np.concatenate(shots, axis=1))
    name = f"camera_view_stage{C.STAGE}.png" if C.STAGE else "camera_view.png"
    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(C.HERE, name))
    robot = "not measured" if worst is None else f"worst {worst * 100:.2f}%"
    print(f"stage {C.STAGE}: robot pixels in the camera view: {robot} while walking "
          f"(fovy {C.FOVY_DEG} deg, tilt {C.camera_pose()['tilt_deg']} deg) -> {name}")


if __name__ == "__main__":
    print("written:", build())
    if "--check" in sys.argv:
        check()

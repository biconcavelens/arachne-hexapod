"""
Rough outdoor ground for the third environment stage: one 8 x 8 m heightfield (3 cm cells) with
  - hills: slopes up to about 10 deg (12 deg at the 99th percentile before the difficulty scaling)
  - bumps: uneven ground, a few millimetres to about 1 cm
  - stones: small mounds 0.6-2 cm high (the gait lifts a foot 17-23 mm)
all scaled by a smooth "difficulty" map, so some areas are nearly flat and some are rough. Each
episode starts at a random spot and heading, so the robot meets a different patch every time.
Plus outdoor objects (rocks, bushes, logs, trees, crates) as no-go footprints, like the furniture.

    python terrain_scene.py          # writes terrain.npz, terrain_scene.xml (MJX), ../terrain_render.xml

MJX collides the robot's feet (5 mm spheres) with the heightfield; the body box does not collide
(the episode ends when the body comes within 4.5 cm of the ground instead). mjx.ray cannot hit a
heightfield, so terrain_env ray-marches the body camera over the height grid itself.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

import room_scene as RS

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "terrain.npz")
OUT = os.path.join(HERE, "terrain_scene.xml")
DYN_OUT = os.path.join(HERE, "terrain_dyn_scene.xml")                  # with the movers (terrain_dyn_env)
RENDER_OUT = os.path.join(os.path.dirname(HERE), "terrain_render.xml")
HALF = 4.0                     # m
N = 267                        # grid points per side (3.0 cm cells)

BROWN, GREEN, GREY = (0.45, 0.35, 0.25, 1), (0.25, 0.45, 0.2, 1), (0.5, 0.5, 0.48, 1)
KINDS = [
    ("rock", 6, 0.3, [("ellipsoid", (0.15, 0.12, 0.08), (0, 0, 0.03), GREY, False)]),
    ("bush", 4, 0.4, [("sphere", (0.25,), (0, 0, 0.18), GREEN, False)]),
    ("log", 2, 0.7, [("capsule", (0.07, 0.5), (0, 0, 0.06), BROWN, False)]),
    ("tree", 2, 0.3, [("capsule", (0.08, 0.8), (0, 0, 0.8), BROWN, False),
                      ("sphere", (0.6,), (0, 0, 1.9), (0.2, 0.4, 0.18, 1), False)]),
    ("crate", 2, 0.4, [("box", (0.2, 0.2, 0.2), (0, 0, 0.2), (0.6, 0.48, 0.3, 1), False)]),
]
FOOT = {"rock": 0.17, "bush": 0.27, "log": 0.55, "tree": 0.12, "crate": 0.29}
OBJECTS = [(k, c, f"{k}{i}") for k, n, c, _ in KINDS for i in range(n)]

# things that move by themselves outdoors (terrain_dyn_env): name, footprint radius (m), parts
DOG = (0.55, 0.4, 0.3, 1)
MOVERS = [
    ("dog", 0.25, [("ellipsoid", (0.22, 0.08, 0.08), (0, 0, 0.25), DOG, False),
                   ("sphere", (0.07,), (0.24, 0, 0.33), DOG, False),
                   ("capsule", (0.025, 0.09), (0.15, 0.05, 0.1), DOG, False),
                   ("capsule", (0.025, 0.09), (0.15, -0.05, 0.1), DOG, False),
                   ("capsule", (0.025, 0.09), (-0.15, 0.05, 0.1), DOG, False),
                   ("capsule", (0.025, 0.09), (-0.15, -0.05, 0.1), DOG, False)]),
    ("football", 0.11, [("sphere", (0.11,), (0, 0, 0.11), (0.92, 0.92, 0.9, 1), False)]),
    ("hiker", 0.2, [("capsule", (0.06, 0.38), (0.0, 0.1, 0.44), (0.3, 0.32, 0.22, 1), False),
                    ("capsule", (0.06, 0.38), (0.0, -0.1, 0.44), (0.3, 0.32, 0.22, 1), False),
                    ("box", (0.14, 0.055, 0.05), (0.04, 0.1, 0.05), (0.25, 0.15, 0.08, 1), False),
                    ("box", (0.14, 0.055, 0.05), (0.04, -0.1, 0.05), (0.25, 0.15, 0.08, 1), False)]),
    ("mower", 0.24, [("box", (0.24, 0.19, 0.07), (0, 0, 0.08), (0.85, 0.4, 0.1, 1), False),
                     ("ellipsoid", (0.16, 0.13, 0.04), (0, 0, 0.16), (0.15, 0.15, 0.15, 1), False)]),
]
MOV_SPEED = [[0.4, 0.9], [0.3, 0.6], [0.6, 1.0], [0.2, 0.35]]       # m/s
MOV_ROLL_R = [0.0, 0.11, 0.0, 0.0]                                   # the football rolls


def _noise(rng, n, lo, hi):
    """Band-limited noise on an n x n grid: wavelengths between lo and hi grid cells, unit std."""
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    r = np.hypot(fx, fy)
    band = (r > 1 / hi) & (r < 1 / lo)
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * band
    z = np.real(np.fft.ifft2(spec))
    return z / (z.std() + 1e-12)


def heights(seed=0):
    rng = np.random.default_rng(seed)
    cell = 2 * HALF / (N - 1)
    diff = _noise(rng, N, 60, 200)                                   # 1.8-6 m patches
    diff = np.clip((diff - diff.min()) / (diff.max() - diff.min()) * 1.3 - 0.15, 0, 1)
    hills = _noise(rng, N, 50, 160)
    gy, gx = np.gradient(hills, cell)
    hills *= np.tan(np.radians(12)) / np.percentile(np.hypot(gx, gy), 99)
    hills *= 0.4 + 0.6 * diff
    bumps = _noise(rng, N, 4, 16) * 0.004 * (0.2 + 0.8 * diff)        # 12-48 cm wavelengths
    stones = np.zeros((N, N))
    xs = np.arange(N) * cell
    for _ in range(2500):
        i, j = rng.integers(0, N, 2)
        if rng.random() > diff[i, j]:
            continue
        R = rng.uniform(0.02, 0.06)
        h = rng.uniform(0.006, 0.02)
        k = int(np.ceil(R / cell)) + 1
        i0, i1, j0, j1 = max(i - k, 0), min(i + k + 1, N), max(j - k, 0), min(j + k + 1, N)
        yy, xx = np.meshgrid(xs[i0:i1] - xs[i], xs[j0:j1] - xs[j], indexing="ij")
        mound = h * np.clip(1 - (xx ** 2 + yy ** 2) / R ** 2, 0, None) ** 0.6
        stones[i0:i1, j0:j1] = np.maximum(stones[i0:i1, j0:j1], mound)
    H = hills + bumps + stones
    H -= H[N // 2, N // 2]
    return H.astype(np.float32), diff.astype(np.float32)             # H[row = y, col = x]


def _robot_xml(path, render=False):
    tree = ET.parse(path)
    root = tree.getroot()
    world = root.find("worldbody")
    world.remove(world.find("geom[@name='floor']"))
    return tree, root, world


def _hfield(root, world, H, material=None):
    lo, hi = float(H.min()), float(H.max())
    asset = root.find("asset")
    ET.SubElement(asset, "hfield", name="terrain", nrow=str(N), ncol=str(N),
                  size=f"{HALF} {HALF} {hi - lo:.5f} 0.1")
    g = ET.SubElement(world, "geom", name="floor", type="hfield", hfield="terrain", pos=f"0 0 {lo:.5f}",
                      contype="0", conaffinity="1", group="0", rgba="0.42 0.4 0.3 1")
    if material:
        g.set("material", material)
        g.attrib.pop("rgba")
    world.remove(g)
    world.insert(0, g)
    return lo, hi


def _movers(world):
    for i, (name, foot, parts) in enumerate(MOVERS):                     # after the objects
        b = ET.SubElement(world, "body", name=name, mocap="true", pos=f"{80 + i} 0 -1")
        for j, (typ, half, pos, rgba, solid) in enumerate(parts):
            ET.SubElement(b, "geom", name=f"{name}_{j}", type=typ, size=" ".join(f"{v:g}" for v in half),
                          pos=" ".join(f"{v:g}" for v in pos), rgba=" ".join(f"{v:g}" for v in rgba),
                          contype="0", conaffinity="0", group="0")


def _objects(world):
    for kind, count, clear, parts in KINDS:
        for k in range(count):
            name = f"{kind}{k}"
            b = ET.SubElement(world, "body", name=name, mocap="true", pos=f"{50 + len(world)} 0 -1")
            for j, (typ, half, pos, rgba, solid) in enumerate(parts):
                ET.SubElement(b, "geom", name=f"{name}_{j}", type=typ, size=" ".join(f"{v:g}" for v in half),
                              pos=" ".join(f"{v:g}" for v in pos), rgba=" ".join(f"{v:g}" for v in rgba),
                              contype="0", conaffinity="0", group="0")


def build():
    H, diff = heights()
    np.savez(NPZ, H=H, diff=diff, half=HALF)
    # physics model
    tree, root, world = _robot_xml(RS.ROBOT_XML)
    root.set("model", "arachne_terrain")
    for g in world.iter("geom"):
        g.set("group", "1")
        if g.get("name") == "base_box":
            g.set("contype", "0")                                  # body never collides with the ground
    lo, hi = _hfield(root, world, H)
    pose = RS.camera_pose()
    base = world.find("body[@name='base']")
    ET.SubElement(base, "camera", name="ego", pos=" ".join(f"{x:.5f}" for x in pose["pos_m"]),
                  xyaxes=" ".join(f"{x:.5f}" for x in pose["mujoco_xyaxes"]), fovy=f"{RS.FOVY_DEG}")
    _objects(world)
    ET.indent(tree, "  ")
    tree.write(OUT)
    _movers(world)
    tree.write(DYN_OUT)
    # render model
    tree, root, world = _robot_xml(RS.RENDER_XML)
    root.set("model", "arachne_terrain_render")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="ground", type="2d", builtin="flat", rgb1="0.38 0.41 0.24",
                  width="512", height="512", mark="random", markrgb="0.3 0.27 0.18", random="0.35")
    ET.SubElement(asset, "material", name="ground", texture="ground", texrepeat="12 12", reflectance="0")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.55 0.7 0.9",
                  rgb2="0.9 0.93 0.97", width="256", height="256")
    _hfield(root, world, H, material="ground")
    for li in list(world.findall("light")):
        world.remove(li)
    ET.SubElement(world, "light", pos="2 -2 6", dir="-0.3 0.3 -1", diffuse="0.8 0.78 0.72", castshadow="true",
                  directional="true")
    ET.SubElement(world, "camera", name="top", pos=f"0 0 {RS.TOP_H}", xyaxes="1 0 0 0 1 0", fovy=f"{RS.TOP_FOVY}")
    base = world.find("body[@name='base']")
    ET.SubElement(base, "camera", name="ego", pos=" ".join(f"{x:.5f}" for x in pose["pos_m"]),
                  xyaxes=" ".join(f"{x:.5f}" for x in pose["mujoco_xyaxes"]), fovy=f"{RS.FOVY_DEG}")
    _objects(world)
    _movers(world)                       # parked unless terrain_dyn places them
    vis = root.find("visual")
    ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.2 0.2 0.2")
    ET.SubElement(vis, "map", znear="0.005", zfar="30")
    ET.SubElement(root, "statistic", extent="4", center="0 0 0.3")
    ET.indent(tree, "  ")
    tree.write(RENDER_OUT)
    return H, lo, hi


def ground_np(xy):
    """Bilinear ground height (numpy), for pictures."""
    H = _height_cache()
    xy = np.asarray(xy, np.float64)
    cell = 2 * HALF / (N - 1)
    u = np.clip((xy[..., 0] + HALF) / cell, 0, N - 1.001)
    v = np.clip((xy[..., 1] + HALF) / cell, 0, N - 1.001)
    c, r = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - c, v - r
    return ((1 - fu) * (1 - fv) * H[r, c] + fu * (1 - fv) * H[r, c + 1]
            + (1 - fu) * fv * H[r + 1, c] + fu * fv * H[r + 1, c + 1])


_H = None


def _height_cache():
    global _H
    if _H is None:
        _H = np.load(NPZ)["H"]
    return _H


def load_model(path):
    """MjModel with the heightfield data filled in (MuJoCo leaves it flat when there is no file)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(path)
    H = np.load(NPZ)["H"]
    lo, hi = float(H.min()), float(H.max())
    m.hfield_data[:] = ((H - lo) / (hi - lo)).ravel()
    return m


if __name__ == "__main__":
    H, lo, hi = build()
    cell = 2 * HALF / (N - 1)
    gy, gx = np.gradient(H, cell)
    print(f"terrain {N}x{N}, cell {cell * 100:.1f} cm, heights {lo * 100:.1f} to {hi * 100:.1f} cm, "
          f"slope p50/p95/max {np.percentile(np.degrees(np.arctan(np.hypot(gx, gy))), [50, 95, 100]).round(1)} deg")
    import mujoco
    for p in (OUT, DYN_OUT, RENDER_OUT):
        m = load_model(p)
        print(os.path.basename(p), "qpos", m.nq, "mocap", m.nmocap, "geoms", m.ngeom)

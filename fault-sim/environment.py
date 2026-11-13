"""
Rooms for the fault simulator, in stages (config.STAGE, set with the env var FAULTSIM_STAGE):
  0  the original textured plane (no change; not built here)
  1  a room: painted walls with skirting, a door and a window, a wood / tile / carpet / concrete
     floor, 1-4 ceiling spotlights with shadows and dim ambient light; all re-drawn every episode
  2  stage 1 plus static objects: boxes, cabinets against the walls, bins, balls, tables, chairs, rugs

The robot starts at the origin facing +x. Nothing solid is placed on the strip it walks along
(PATH), so the robot never bumps into anything; the room and objects only change what the
camera sees. Textures are procedural and tile without seams (FFT noise), written into the
model's texture memory before the renderer uploads them.
"""
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import config as C

FLOOR_TEX, WALL_TEX, OBJ_TEX = 1024, 512, 256
FLOOR_M = 2.0                    # metres of floor one texture covers
WALL_H, WALL_T = 2.4, 0.1
ROOM_FRONT, ROOM_BACK, ROOM_SIDE = (1.8, 3.5), (0.8, 2.0), (1.0, 2.5)   # distance from the robot to each wall
ROOM_YAW_DEG = 15.0
N_LIGHTS = 4
# No solid object within KEEP_R of the start or on the strip ahead (x from, x to, half width). In 20 s
# the robot walks up to ~1.4 m and, steering, drifts up to ~0.65 m sideways (Kaggle run 9 found bumps
# with a 0.55 m strip and no disc).
KEEP_R = 1.5
PATH = (-0.45, 2.0, 0.9)
OBJ_TEXTURES = ("cardboard", "wood", "fabric", "plain", "metal")
OBJECTS = {"box": 10, "cyl": 6, "ball": 3, "table": 2, "chair": 3, "rug": 2}       # stage 2 pool
P_ACTIVE = {"box": 0.6, "cyl": 0.5, "ball": 0.4, "table": 0.4, "chair": 0.5, "rug": 0.5}
RGB = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)


# ---------------------------------------------------------------- procedural textures (numpy only)
def noise(rng, h, w, beta=2.0, fmax=None):
    """Tileable fractal noise, zero mean, unit sd: random phases, power ~ 1/f^beta."""
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.rfftfreq(w)[None, :] * w
    f = np.hypot(fx, fy)
    amp = np.where(f > 0, np.maximum(f, 1) ** (-beta / 2), 0.0)
    if fmax:
        amp[f > fmax] = 0
    n = np.fft.irfft2(amp * np.exp(2j * np.pi * rng.random(amp.shape)), s=(h, w))
    return ((n - n.mean()) / (n.std() + 1e-9)).astype(np.float32)


def _rgb(base, shade):
    """base colour (0-255) x per-pixel shade (h, w) -> uint8 image."""
    return np.clip(np.asarray(base, np.float32)[None, None, :] * shade[..., None], 0, 255).astype(np.uint8)


def wood_planks(rng, n=FLOOR_TEX):
    px = n / FLOOR_M                                                   # pixels per metre
    widths = int(round(n / (rng.uniform(0.09, 0.2) * px)))           # planks across the texture
    pw = n / widths
    base = np.array([(186, 140, 92), (110, 72, 45), (215, 180, 135), (150, 140, 128), (150, 85, 55)][rng.integers(5)], np.float32)
    v, u = np.mgrid[0:n, 0:n].astype(np.float32)
    plank = (v // pw).astype(int)
    shade = np.ones((n, n), np.float32)
    seam = np.zeros((n, n), bool)
    for p in range(widths):                                           # end joints, staggered; wrap-around safe
        joints = np.sort(rng.uniform(0, n, int(rng.integers(1, 3))))
        rows = plank == p
        seg = np.searchsorted(joints, u[rows], side="right") % len(joints)
        tone = rng.normal(1.0, 0.08, len(joints))
        shade[rows] *= tone[seg]
        seam[rows] |= np.min(np.abs(u[rows][:, None] - joints[None, :]), 1) < 1.5
    warp = noise(rng, n, n, 3.0) * rng.uniform(4, 10)
    grain = np.sin(2 * np.pi * (v + warp) / rng.uniform(3, 7))
    shade *= 0.88 + 0.07 * grain + 0.04 * noise(rng, n, n, 1.0)
    seam |= (v % pw) < 1.5
    shade[seam] *= 0.5
    return _rgb(base, shade)


def tiles(rng, n=FLOOR_TEX):
    k = int(rng.choice([4, 5, 6, 8, 10]))                             # tiles across (0.2-0.5 m each)
    tw, grout = n / k, rng.uniform(2, 4)
    base = np.array([(230, 230, 225), (215, 200, 175), (150, 150, 150), (175, 95, 65), (80, 85, 90)][rng.integers(5)], np.float32)
    v, u = np.mgrid[0:n, 0:n].astype(np.float32)
    tone = rng.normal(1.0, 0.04, (k, k))[(v // tw).astype(int), (u // tw).astype(int)]
    shade = tone * (1 + 0.05 * noise(rng, n, n, 2.5))
    g = ((v % tw) < grout) | ((u % tw) < grout)
    img = _rgb(base, shade)
    img[g] = np.clip(base.mean() * rng.choice([0.55, 1.15]) + rng.normal(0, 4, (g.sum(), 1)), 0, 255)
    return img


def carpet(rng, n=FLOOR_TEX):
    base = [(120, 110, 100), (90, 100, 120), (140, 60, 60), (80, 90, 70), (170, 160, 145)][rng.integers(5)]
    shade = 1 + 0.06 * noise(rng, n, n, 2.0) + 0.12 * rng.standard_normal((n, n)).astype(np.float32)
    if rng.random() < 0.3:                                             # striped carpet
        shade *= 1 + 0.08 * np.sign(np.sin(2 * np.pi * np.arange(n) / rng.uniform(20, 60)))[None, :]
    return _rgb(base, shade)


def concrete(rng, n=FLOOR_TEX):
    g = rng.uniform(120, 175)
    shade = 1 + 0.08 * noise(rng, n, n, 2.2) + 0.05 * noise(rng, n, n, 0.5)
    return _rgb((g, g * 0.98, g * 0.95), shade)


def plaster(rng, n=WALL_TEX):
    base = [(235, 232, 225), (225, 215, 195), (200, 205, 210), (190, 210, 225), (200, 220, 200), (230, 210, 200)][rng.integers(6)]
    shade = 1 + 0.025 * noise(rng, n, n, 2.0) + 0.015 * noise(rng, n, n, 0.8)
    if rng.random() < 0.25:                                            # striped wallpaper
        shade *= 1 + 0.05 * (np.sin(2 * np.pi * np.arange(n) / rng.uniform(16, 64)) > 0)[None, :]
    return _rgb(base, shade)


def cardboard(rng, n=OBJ_TEX):
    shade = 1 + 0.06 * noise(rng, n, n, 2.0) + 0.02 * np.sin(np.arange(n) / 1.5)[None, :]
    img = _rgb((170, 130, 85), shade)
    a = int(n * rng.uniform(0.4, 0.5))
    img[a:a + n // 8] = np.clip(img[a:a + n // 8] * np.float32(1.15), 0, 255)          # packing tape
    return img


def fabric(rng, n=OBJ_TEX):
    c1, c2 = rng.uniform(40, 220, 3), rng.uniform(40, 220, 3)
    v, u = np.mgrid[0:n, 0:n]
    p = rng.uniform(8, 32)
    mask = ((u // p + v // p) % 2 == 0) if rng.random() < 0.5 else (np.sin(2 * np.pi * u / p) > 0)
    img = np.where(mask[..., None], c1, c2) * (1 + 0.1 * rng.standard_normal((n, n, 1)))
    return np.clip(img, 0, 255).astype(np.uint8)


def rug(rng, n=WALL_TEX):
    pal = rng.uniform(30, 220, (3, 3)) * 0.6 + rng.uniform(70, 150) * 0.4       # muted, like a real rug
    v, u = np.mgrid[0:n, 0:n] / n
    d = np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v))         # distance to the edge
    band = np.digitize(d, [0.06, 0.1])                                  # border, inner border, field
    img = pal[band] * (1 + 0.12 * np.sin(2 * np.pi * (u + v) * rng.integers(4, 12)))[..., None]
    img *= 1 + 0.1 * rng.standard_normal((n, n, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def plain(rng, n=OBJ_TEX):                                             # tinted by the material colour
    return _rgb((235, 235, 235), 1 + 0.03 * noise(rng, n, n, 1.5))


def metal(rng, n=OBJ_TEX):
    streak = np.repeat(rng.standard_normal((n, 1)).astype(np.float32), n, 1)
    return _rgb((170, 172, 175), 1 + 0.06 * streak + 0.02 * noise(rng, n, n, 1.0))


def wood_board(rng, n=OBJ_TEX):
    v = np.mgrid[0:n, 0:n][0].astype(np.float32)
    grain = np.sin(2 * np.pi * (v + noise(rng, n, n, 3.0) * 6) / rng.uniform(4, 9))
    base = [(186, 140, 92), (110, 72, 45), (215, 180, 135)][rng.integers(3)]
    return _rgb(base, 0.9 + 0.08 * grain + 0.03 * noise(rng, n, n, 1.0))


FLOORS = {"wood": (wood_planks, 0.08), "tiles": (tiles, 0.12), "carpet": (carpet, 0.0), "concrete": (concrete, 0.02)}
OBJ_GEN = {"cardboard": cardboard, "wood": wood_board, "fabric": fabric, "plain": plain, "metal": metal}


# ---------------------------------------------------------------- scene XML (build_scene.py)
def add_to_xml(root, stage):
    """Room (and, for stage 2, the object pool) on top of the lean model. Sizes and places are placeholders."""
    asset, world = root.find("asset"), root.find("worldbody")
    for light in world.findall("light"):                              # the model's two test lights
        world.remove(light)
    vis = root.find("visual")
    ET.SubElement(vis, "headlight", ambient=".15 .15 .15", diffuse="0 0 0", specular="0 0 0")
    ET.SubElement(vis, "map", znear="0.005", shadowclip="3")

    def tex(name, n):
        ET.SubElement(asset, "texture", name=name, type="2d", builtin="flat", width=str(n), height=str(n), rgb1=".7 .7 .7")

    def mat(name, texture, rep=1.0, **kw):
        ET.SubElement(asset, "material", name=name, texture=texture, texuniform="true", texrepeat=f"{rep} {rep}", **kw)

    room = ET.SubElement(world, "body", name="room")                  # geoms of the world body itself never move
    tex("wall_tex", WALL_TEX)
    tex("door_tex", OBJ_TEX)
    mat("wall_mat", "wall_tex", 1.0)
    mat("door_mat", "door_tex", 2.0)
    ET.SubElement(asset, "material", name="skirting_mat", rgba=".9 .9 .88 1", specular=".3")
    ET.SubElement(asset, "material", name="window_mat", rgba=".8 .9 1 1", emission="0.8")
    for i in range(4):
        ET.SubElement(room, "geom", name=f"wall{i}", type="box", size=".1 .1 .1", material="wall_mat",
                      contype="0", conaffinity="1")
        ET.SubElement(room, "geom", name=f"skirting{i}", type="box", size=".1 .1 .1", material="skirting_mat",
                      contype="0", conaffinity="0")
    ET.SubElement(room, "geom", name="door", type="box", size=".1 .1 .1", material="door_mat", contype="0", conaffinity="0")
    ET.SubElement(room, "geom", name="window", type="box", size=".1 .1 .1", material="window_mat", contype="0", conaffinity="0")
    for i in range(N_LIGHTS):
        ET.SubElement(room, "light", name=f"ceil{i}", pos="0 0 2.3", dir="0 0 -1", cutoff="70", exponent="2",
                      attenuation="1 0.05 0.02", castshadow="true" if i < 2 else "false", diffuse=".5 .5 .5", specular=".15 .15 .15")
    if stage < 2:
        return
    for name in OBJ_TEXTURES + ("rug",):
        tex(f"obj_{name}_tex", WALL_TEX if name == "rug" else OBJ_TEX)
    for kind, count in OBJECTS.items():
        for i in range(count):
            name = f"obj_{kind}{i}"
            ET.SubElement(asset, "material", name=f"{name}_mat", texture="obj_plain_tex", texuniform="true", texrepeat="3 3")
            b = ET.SubElement(world, "body", name=name, pos="0 0 -5")
            g = dict(material=f"{name}_mat", contype="0", conaffinity="1")
            if kind == "box":
                ET.SubElement(b, "geom", name=f"{name}_g", type="box", size=".1 .1 .1", **g)
            elif kind == "cyl":
                ET.SubElement(b, "geom", name=f"{name}_g", type="cylinder", size=".1 .1", **g)
            elif kind == "ball":
                ET.SubElement(b, "geom", name=f"{name}_g", type="sphere", size=".1", **g)
            elif kind == "rug":
                ET.SubElement(b, "geom", name=f"{name}_g", type="box", size=".5 .4 .002", **dict(g, conaffinity="0"))
            elif kind == "table":
                ET.SubElement(b, "geom", type="box", size=".5 .35 .02", pos="0 0 .72", **g)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        ET.SubElement(b, "geom", type="box", size=".02 .02 .35", pos=f"{.45 * sx} {.3 * sy} .35", **g)
            elif kind == "chair":
                ET.SubElement(b, "geom", type="box", size=".2 .2 .02", pos="0 0 .45", **g)
                ET.SubElement(b, "geom", type="box", size=".015 .2 .22", pos="-.185 0 .69", **g)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        ET.SubElement(b, "geom", type="box", size=".015 .015 .215", pos=f"{.17 * sx} {.17 * sy} .215", **g)


# ---------------------------------------------------------------- per-episode randomisation (simulate.randomize)
def _id(m, kind, name):
    return mujoco.mj_name2id(m, kind, name)


def _put_texture(m, name, img):
    t = _id(m, mujoco.mjtObj.mjOBJ_TEXTURE, name)
    w, h, adr, nch = m.tex_width[t], m.tex_height[t], m.tex_adr[t], m.tex_nchannel[t]
    if nch == 4:
        img = np.concatenate([img, np.full((h, w, 1), 255, np.uint8)], -1)
    m.tex_data[adr:adr + w * h * nch] = img.reshape(-1)


def _yaw_quat(a):
    return np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])


def _rot(p, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])


def _box(m, gid, center_xy, z, half, yaw):
    """Place and resize a box geom (keeps the collision bounds right)."""
    m.geom_pos[gid] = (*center_xy, z)
    m.geom_quat[gid] = _yaw_quat(yaw)
    m.geom_size[gid] = half
    m.geom_aabb[gid] = (0, 0, 0, *half)
    m.geom_rbound[gid] = float(np.linalg.norm(half))


def _colour_temp(rng):
    """Warm (2700 K) to cool (6500 K) white."""
    t = rng.random()
    return np.array([1.0, 0.82 + 0.16 * t, 0.66 + 0.33 * t])


def randomize(m, rng):
    # Geoms compiled at their body's origin are flagged "same frame", and MuJoCo then copies the
    # body pose and ignores geom_pos. Ours move every episode, so clear the flag.
    ours = [b for b in range(m.nbody) if m.body(b).name == "room" or m.body(b).name.startswith("obj_")]
    m.geom_sameframe[np.isin(m.geom_bodyid, ours)] = 0
    room = _room(m, rng)
    if _id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_box0") >= 0:
        _objects(m, rng, room)


def _room(m, rng):
    front, back = rng.uniform(*ROOM_FRONT), rng.uniform(*ROOM_BACK)
    left, right = rng.uniform(*ROOM_SIDE), rng.uniform(*ROOM_SIDE)
    yaw = math.radians(rng.uniform(-ROOM_YAW_DEG, ROOM_YAW_DEG))
    xc, yc, hx, hy = (front - back) / 2, (left - right) / 2, (front + back) / 2, (left + right) / 2
    G = mujoco.mjtObj.mjOBJ_GEOM
    walls = [((front + WALL_T / 2, yc), (WALL_T / 2, hy + WALL_T, WALL_H / 2), (-1, 0)),
             ((-back - WALL_T / 2, yc), (WALL_T / 2, hy + WALL_T, WALL_H / 2), (1, 0)),
             ((xc, left + WALL_T / 2), (hx + WALL_T, WALL_T / 2, WALL_H / 2), (0, -1)),
             ((xc, -right - WALL_T / 2), (hx + WALL_T, WALL_T / 2, WALL_H / 2), (0, 1))]
    for i, (c, half, inward) in enumerate(walls):
        _box(m, _id(m, G, f"wall{i}"), _rot(c, yaw), WALL_H / 2, half, yaw)
        sk = (0.006, half[1] - WALL_T, 0.04) if inward[0] else (half[0] - WALL_T, 0.006, 0.04)
        inner = (c[0] + inward[0] * (WALL_T / 2 + 0.006), c[1] + inward[1] * (WALL_T / 2 + 0.006))
        _box(m, _id(m, G, f"skirting{i}"), _rot(inner, yaw), 0.04, sk, yaw)
    for name, w, h, z in (("door", 0.45, 1.0, 1.0), ("window", rng.uniform(0.5, 0.8), rng.uniform(0.5, 0.65), rng.uniform(1.3, 1.5))):
        c, half, inward = walls[rng.integers(4)]
        along = rng.uniform(-1, 1) * (max(half[0], half[1]) - WALL_T - w - 0.1)
        x = c[0] + inward[0] * (WALL_T / 2 + 0.01) + (0 if inward[0] else along)
        y = c[1] + inward[1] * (WALL_T / 2 + 0.01) + (along if inward[0] else 0)
        _box(m, _id(m, G, name), _rot((x, y), yaw), z, (0.012, w, h) if inward[0] else (w, 0.012, h), yaw)

    kind = list(FLOORS)[rng.integers(len(FLOORS))]
    gen, refl = FLOORS[kind]
    _put_texture(m, "floor_tex", gen(rng))
    fm = _id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_mat")
    m.mat_texrepeat[fm] = 1 / FLOOR_M
    m.mat_reflectance[fm] = refl * rng.uniform(0.5, 1.5)
    m.mat_specular[fm], m.mat_shininess[fm] = rng.uniform(0.1, 0.5), rng.uniform(0.1, 0.6)
    _put_texture(m, "wall_tex", plaster(rng))
    _put_texture(m, "door_tex", wood_board(rng))

    n_on = int(rng.choice([1, 2, 2, 3, 3, 4]))
    total = rng.uniform(0.9, 1.6)
    for i in range(N_LIGHTS):
        lid = _id(m, mujoco.mjtObj.mjOBJ_LIGHT, f"ceil{i}")
        on = i < n_on
        m.light_active[lid] = on
        p = _rot((rng.uniform(-back + 0.3, front - 0.3), rng.uniform(-right + 0.3, left - 0.3)), yaw)
        m.light_pos[lid] = (*p, rng.uniform(2.2, 2.35))
        d = np.array([rng.normal(0, 0.15), rng.normal(0, 0.15), -1.0])
        m.light_dir[lid] = d / np.linalg.norm(d)
        m.light_cutoff[lid], m.light_exponent[lid] = rng.uniform(55, 85), rng.uniform(0.5, 4)
        m.light_diffuse[lid] = _colour_temp(rng) * total / n_on
        m.light_specular[lid] = 0.15 * m.light_diffuse[lid]
        m.light_castshadow[lid] = on and i < 2
    m.vis.headlight.ambient[:] = rng.uniform(0.06, 0.2) * _colour_temp(rng)
    return dict(front=front, back=back, left=left, right=right, yaw=yaw)


def _in_way(p, r):
    """Could an object of radius r at p touch the walking robot?"""
    return math.hypot(*p) < KEEP_R + r or (PATH[0] - r < p[0] < PATH[1] + r and abs(p[1]) < PATH[2] + r)


def _free_spot(rng, room, r, placed, keep_path=True, tries=60):
    for _ in range(tries):
        q = (rng.uniform(-room["back"] + r, room["front"] - r), rng.uniform(-room["right"] + r, room["left"] - r))
        p = _rot(q, room["yaw"])
        if keep_path and _in_way(p, r):
            continue
        if any(np.hypot(*(p - p2)) < r + r2 + 0.05 for p2, r2 in placed):
            continue
        placed.append((p, r))
        return p
    return None


def _objects(m, rng, room):
    B, G, MAT = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_MATERIAL
    for name in OBJ_TEXTURES:
        _put_texture(m, f"obj_{name}_tex", OBJ_GEN[name](rng))
    _put_texture(m, "obj_rug_tex", rug(rng))
    tex_id = {n: _id(m, mujoco.mjtObj.mjOBJ_TEXTURE, f"obj_{n}_tex") for n in OBJ_TEXTURES + ("rug",)}
    placed = []
    for kind, count in OBJECTS.items():
        for i in range(count):
            name = f"obj_{kind}{i}"
            b = _id(m, B, name)
            m.body_pos[b] = (0, 0, -5)                                 # parked under the floor unless placed
            if rng.random() > P_ACTIVE[kind]:
                continue
            mid = _id(m, MAT, f"{name}_mat")
            look = "rug" if kind == "rug" else str(rng.choice(OBJ_TEXTURES, p=[.3, .25, .1, .25, .1] if kind == "box" else None))
            m.mat_texid[mid, RGB] = tex_id[look]
            m.mat_texuniform[mid] = look != "rug"                      # a rug shows its pattern once
            m.mat_texrepeat[mid] = (1, 1) if look == "rug" else (3, 3)
            m.mat_rgba[mid] = (*(rng.uniform(0.25, 1.0, 3) if look == "plain" else rng.uniform(0.85, 1.0, 3)), 1)
            m.mat_specular[mid], m.mat_shininess[mid] = rng.uniform(0.05, 0.6), rng.uniform(0.1, 0.8)
            yaw = rng.uniform(0, 2 * math.pi)
            gid = _id(m, G, f"{name}_g")
            if kind == "box" and rng.random() < 0.35:                  # cabinet / shelf against a wall
                depth, width, height = rng.uniform(0.15, 0.3), rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.9)
                wall = rng.integers(4)
                ax = [(room["front"] - depth, None), (-room["back"] + depth, None), (None, room["left"] - depth), (None, -room["right"] + depth)][wall]
                along = rng.uniform(-1, 1)
                q = (ax[0] if ax[0] is not None else along * (room["front"] + room["back"]) / 2 - (room["back"] - room["front"]) / 2,
                     ax[1] if ax[1] is not None else along * (room["left"] + room["right"]) / 2 - (room["right"] - room["left"]) / 2)
                p = _rot(q, room["yaw"])
                r = math.hypot(depth, width)
                if _in_way(p, r) or any(np.hypot(*(p - p2)) < r + r2 for p2, r2 in placed):
                    continue
                placed.append((p, r))
                half = (depth, width, height) if wall < 2 else (width, depth, height)
                _box(m, gid, (0, 0), height, half, 0)
                m.body_pos[b], m.body_quat[b] = (*p, 0), _yaw_quat(room["yaw"])
                continue
            if kind == "box":
                half = tuple(rng.uniform(0.04, 0.25, 3))
                _box(m, gid, (0, 0), half[2], half, 0)
                r = math.hypot(half[0], half[1])
            elif kind == "cyl":
                rad, hh = rng.uniform(0.03, 0.2), rng.uniform(0.04, 0.4)
                m.geom_size[gid], m.geom_pos[gid] = (rad, hh, 0), (0, 0, hh)
                m.geom_aabb[gid], m.geom_rbound[gid] = (0, 0, 0, rad, rad, hh), math.hypot(rad, hh)
                r = rad
            elif kind == "ball":
                rad = rng.uniform(0.04, 0.14)
                m.geom_size[gid], m.geom_pos[gid] = (rad, 0, 0), (0, 0, rad)
                m.geom_aabb[gid], m.geom_rbound[gid] = (0, 0, 0, rad, rad, rad), rad
                r = rad
            elif kind == "rug":
                half = (rng.uniform(0.4, 1.2), rng.uniform(0.3, 0.9), 0.002)
                _box(m, gid, (0, 0), 0.002 + 0.001 * i, half, 0)
                p = _rot((rng.uniform(-room["back"] + 0.5, room["front"] - 0.5), rng.uniform(-room["right"] + 0.5, room["left"] - 0.5)), room["yaw"])
                m.body_pos[b], m.body_quat[b] = (*p, 0), _yaw_quat(yaw)
                continue                                               # visual only, may lie under the robot
            else:
                r = 0.62 if kind == "table" else 0.3
            p = _free_spot(rng, room, r, placed)
            if p is not None:
                m.body_pos[b], m.body_quat[b] = (*p, 0), _yaw_quat(yaw)

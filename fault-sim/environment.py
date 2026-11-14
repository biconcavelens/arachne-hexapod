"""
Environments for the fault simulator, in stages (config.STAGE, set with the env var FAULTSIM_STAGE):
  0  the original textured plane (no change; not built here)
  1  a room: painted walls with skirting, a door and a window, a wood / tile / carpet / concrete
     floor, 1-4 ceiling spotlights with shadows and dim ambient light; all re-drawn every episode
  2  stage 1 plus static objects: boxes, cabinets against the walls, bins, balls, tables, chairs, rugs
  3  denser clutter: about twice the objects, plus bookshelves, sofas, plants, floor lamps, shoes,
     papers on the floor, pictures on the walls; dimmer, brighter and tinted lighting
  4  stage 3 plus things that move by themselves: a robot vacuum, rolling balls, a sliding box, a
     person's legs walking by (camera motion the robot did not cause)
  5  outdoors on gentle hills: grass / dirt / gravel ground, sky, sun with shadows, rocks, bushes, trees
  6  stage 5 made rough: bumps and stones up to about half the foot lift (4-12 mm; the gait lifts a
     foot 17-23 mm), so the healthy robot still walks
  7  rougher still: steeper hills (7-11 deg) and stones of 8-25 mm, as tall as the foot lift; the
     current gait mostly stalls on it (healthy median speed 12 mm/s, against 43 on stage 5)

Up to stage 4 the floor is flat and nothing solid is within the robot's reach (KEEP_R, PATH), so
the physics are identical to stage 0 and only the camera image changes; the movers are visual
only. From stage 5 the ground itself changes, so the walking does too. Textures are procedural and
tile without seams (FFT noise), written into the model before the renderer uploads them.
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
OBJECTS3 = dict(OBJECTS, box=18, cyl=8, ball=5, table=3, chair=5, rug=3,                 # stage 3+ pool
                shelf=2, sofa=1, plant=3, lamp=2, shoe=6, paper=8, picture=4)
P_ACTIVE3 = dict(P_ACTIVE, box=0.7, cyl=0.6, ball=0.5, shelf=0.6, sofa=0.5, plant=0.6, lamp=0.5,
                 shoe=0.6, paper=0.7, picture=0.7)
FIXED_LOOK = {"shelf": "wood", "sofa": "fabric", "plant": "plain", "lamp": "metal", "shoe": "plain",
              "paper": "plain", "picture": "rug"}
MOVERS = ("vac", "ball0", "ball1", "box", "legL", "legR")                               # stage 4+
TERR_N, TERR_R = 251, 2.5               # outdoor near heightfield: 2 cm cells over +-2.5 m (detail)
TERR_N_FAR, TERR_R_FAR = 251, 10.0      # far heightfield: 8 cm cells over +-10 m (hills to the horizon)
TERR_ZMAX = 1.0                         # height range either can hold (m)
RGB = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)


# ---------------------------------------------------------------- procedural textures (numpy only)
def noise(rng, h, w, beta=2.0, fmax=None, fmin=None):
    """Tileable fractal noise, zero mean, unit sd: random phases, power ~ 1/f^beta (f in cycles per image)."""
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.rfftfreq(w)[None, :] * w
    f = np.hypot(fx, fy)
    amp = np.where(f > 0, np.maximum(f, 1) ** (-beta / 2), 0.0)
    if fmax:
        amp[f > fmax] = 0
    if fmin:
        amp[f < fmin] = 0
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


def grass(rng, n=FLOOR_TEX):
    base = np.array([rng.uniform(55, 95), rng.uniform(95, 140), rng.uniform(35, 60)])
    shade = 1 + 0.12 * noise(rng, n, n, 1.8) + 0.18 * rng.standard_normal((n, n)).astype(np.float32)
    img = _rgb(base, shade)
    dry = noise(rng, n, n, 3.0) > rng.uniform(0.6, 1.4)                # yellowed / bare patches
    img[dry] = np.clip(img[dry] * np.float32([1.25, 1.05, 0.8]), 0, 255)
    return img


def dirt(rng, n=FLOOR_TEX):
    base = (rng.uniform(110, 150), rng.uniform(85, 110), rng.uniform(55, 75))
    shade = 1 + 0.1 * noise(rng, n, n, 2.2) + 0.08 * rng.standard_normal((n, n)).astype(np.float32)
    img = _rgb(base, shade)
    pebbles = rng.random((n, n)) < 0.01
    img[pebbles] = np.clip(rng.uniform(90, 200, (pebbles.sum(), 1)), 0, 255)
    return img


def gravel(rng, n=FLOOR_TEX):
    g = rng.uniform(110, 160)
    shade = 1 + 0.25 * noise(rng, n, n, 0.3, fmin=150) + 0.05 * noise(rng, n, n, 2.5)   # stone-sized blobs
    return _rgb((g, g * 0.97, g * 0.92), shade)


def mixed_ground(rng, n=FLOOR_TEX):
    mask = (noise(rng, n, n, 3.5) > rng.uniform(-0.3, 0.5))[..., None]
    return np.where(mask, grass(rng, n), dirt(rng, n))


def rock_tex(rng, n=OBJ_TEX):
    g = rng.uniform(95, 150)
    shade = 1 + 0.15 * noise(rng, n, n, 2.0) + 0.06 * noise(rng, n, n, 0.5)
    return _rgb((g, g * 0.96, g * 0.9), shade)


def bark(rng, n=OBJ_TEX):
    streak = np.repeat(noise(rng, 1, n, 1.0), n, 0)                     # vertical fibres
    return _rgb((95, 70, 50), 1 + 0.2 * streak + 0.06 * noise(rng, n, n, 1.5))


def foliage(rng, n=OBJ_TEX):
    base = (rng.uniform(35, 70), rng.uniform(80, 130), rng.uniform(30, 55))
    return _rgb(base, 1 + 0.25 * noise(rng, n, n, 0.8) + 0.1 * noise(rng, n, n, 2.5))


def _smooth(w):
    w = np.clip(w, 0, 1)
    return w * w * (3 - 2 * w)


def _bilinear(h, x, y, half):
    """Heights of grid h (rows along y, columns along x, spanning +-half) at points x, y."""
    n = h.shape[0]
    dx = 2 * half / (n - 1)
    j, i = np.clip((np.asarray(x) + half) / dx, 0, n - 1.001), np.clip((np.asarray(y) + half) / dx, 0, n - 1.001)
    j0, i0 = j.astype(int), i.astype(int)
    fj, fi = j - j0, i - i0
    return (h[i0, j0] * (1 - fj) + h[i0, j0 + 1] * fj) * (1 - fi) + (h[i0 + 1, j0] * (1 - fj) + h[i0 + 1, j0 + 1] * fj) * fi


def terrain(rng, level, n=TERR_N, half=TERR_R, n_far=TERR_N_FAR, half_far=TERR_R_FAR):
    """Two height grids in metres (rows along y, columns along x), 0 at the origin and flat under the robot:
    near, a detailed patch of +-half around the start, and far, the smooth hills out to +-half_far.
    level 1: hills (slopes of 5-9 deg, wavelengths 2-8 m); level 2: the same hills plus bumps of 2-5 mm and
    stones of 4-12 mm; level 3: steeper hills (7-11 deg), bumps of 5-10 mm and stones of 8-25 mm.
    One fine grid over the whole area renders ~40x slower (1001^2: 89 ms a frame on a T4, 251^2: 2 ms),
    so the details live only where the robot walks and the camera can resolve them. They fade out at the
    patch edge, and under the patch the far grid is lowered so it never shows through."""
    dxf = 2 * half_far / (n_far - 1)
    hills = noise(rng, n_far, n_far, 4.0, fmax=2 * half_far / 2.0, fmin=2 * half_far / 8.0)
    slope = rng.uniform(5, 9) if level < 3 else rng.uniform(7, 11)
    hills *= math.tan(math.radians(slope)) / np.percentile(np.hypot(*np.gradient(hills, dxf)), 95)
    h0 = float(_bilinear(hills, 0.0, 0.0, half_far))

    dx = 2 * half / (n - 1)
    y, x = (np.mgrid[0:n, 0:n] - (n - 1) / 2) * dx
    detail = np.zeros((n, n), np.float32)
    if level >= 2:
        bump, stone = ((0.002, 0.005), (0.004, 0.012)) if level == 2 else ((0.005, 0.01), (0.008, 0.025))
        detail += rng.uniform(*bump) * noise(rng, n, n, 1.5, fmax=2 * half / 0.08, fmin=2 * half / 0.5)
        fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(n)[None, :]
        for r in (0.025, 0.04, 0.06):                                  # stones: impulses blurred to bumps of radius r
            imp = np.zeros((n, n), np.float32)
            k = int(rng.uniform(0.5, 1.5) * (2 * half) ** 2)            # about 1 per square metre per size
            imp[rng.integers(0, n, k), rng.integers(0, n, k)] = rng.uniform(*stone, k)
            blur = np.exp(-2 * (np.pi * (r / dx)) ** 2 * (fx ** 2 + fy ** 2))   # Gaussian with sigma r, peak kept
            detail += np.fft.irfft2(np.fft.rfft2(imp) * blur, s=(n, n)) * (2 * np.pi * (r / dx) ** 2)
    edge = lambda xx, yy: _smooth((half - np.maximum(abs(xx), abs(yy))) / 0.5)   # 1 inside, 0 at the patch edge
    start = lambda xx, yy: _smooth((np.hypot(xx, yy) - 0.15) / 0.3)             # flat start patch
    near = (_bilinear(hills, x, y, half_far) - h0 + detail * edge(x, y)) * start(x, y)
    yf, xf = (np.mgrid[0:n_far, 0:n_far] - (n_far - 1) / 2) * dxf
    far = (hills - h0) * start(xf, yf) - (0.003 + 0.05 * edge(xf, yf))
    return near.astype(np.float32), far.astype(np.float32)


FLOORS = {"wood": (wood_planks, 0.08), "tiles": (tiles, 0.12), "carpet": (carpet, 0.0), "concrete": (concrete, 0.02)}
GROUNDS = {"grass": grass, "dirt": dirt, "gravel": gravel, "mixed": mixed_ground}
OBJ_GEN = {"cardboard": cardboard, "wood": wood_board, "fabric": fabric, "plain": plain, "metal": metal}


# ---------------------------------------------------------------- scene XML (build_scene.py)
def add_to_xml(root, stage):
    """Room, objects and movers (or, from stage 5, the outdoor terrain) on top of the lean model.
    Sizes and places are placeholders; randomize() sets them every episode."""
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

    if stage >= 5:
        _outdoor_xml(root, asset, world, tex, mat)
        return
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
    if stage >= 3:
        ET.SubElement(asset, "material", name="lamp_shade_mat", rgba="1 .92 .8 1", emission="0.6")
    if stage >= 4:
        _movers_xml(asset, world)
    book = np.random.default_rng(7)                                    # fixed book colours
    for kind, count in (OBJECTS if stage == 2 else OBJECTS3).items():
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
            # stage 3 kinds; wall-standing ones face +x (their back is at -x)
            elif kind == "shelf":
                ET.SubElement(b, "geom", type="box", size=".01 .4 .8", pos="-.14 0 .8", **g)
                for sy in (-1, 1):
                    ET.SubElement(b, "geom", type="box", size=".15 .01 .8", pos=f"0 {.39 * sy} .8", **g)
                for z in (.02, .42, .82, 1.22, 1.58):
                    ET.SubElement(b, "geom", type="box", size=".15 .38 .01", pos=f"0 0 {z}", **g)
                for z in (.03, .43, .83, 1.23):                         # books, no material: rgba shows
                    y = -.36
                    while y < .3:
                        t, hh = book.uniform(.012, .03), book.uniform(.09, .17)
                        ET.SubElement(b, "geom", type="box", size=f".11 {t:.3f} {hh:.3f}", pos=f"0 {y + t:.3f} {z + hh:.3f}",
                                      rgba=" ".join(f"{c:.2f}" for c in (*book.uniform(.1, .9, 3), 1)), contype="0", conaffinity="0")
                        y += 2 * t + .004
            elif kind == "sofa":
                ET.SubElement(b, "geom", type="box", size=".4 .9 .2", pos="0 0 .2", **g)
                ET.SubElement(b, "geom", type="box", size=".1 .9 .25", pos="-.3 0 .55", **g)
                for sy in (-1, 1):
                    ET.SubElement(b, "geom", type="box", size=".4 .1 .15", pos=f"0 {.85 * sy} .45", **g)
            elif kind == "plant":
                ET.SubElement(b, "geom", type="cylinder", size=".12 .12", pos="0 0 .12", **g)
                for x, y, z, r in ((0, 0, .42, .22), (.1, .08, .6, .16), (-.08, -.06, .62, .15)):
                    ET.SubElement(b, "geom", type="sphere", size=str(r), pos=f"{x} {y} {z}", contype="0", conaffinity="0",
                                  rgba=f"{book.uniform(.15, .3):.2f} {book.uniform(.4, .6):.2f} {book.uniform(.12, .25):.2f} 1")
            elif kind == "lamp":
                ET.SubElement(b, "geom", type="cylinder", size=".15 .015", pos="0 0 .015", **g)
                ET.SubElement(b, "geom", type="cylinder", size=".012 .75", pos="0 0 .78", **g)
                ET.SubElement(b, "geom", type="cylinder", size=".17 .12", pos="0 0 1.5", material="lamp_shade_mat",
                              contype="0", conaffinity="0")
                ET.SubElement(b, "light", name=f"{name}_light", pos="0 0 1.38", dir="0 0 -1", cutoff="80", exponent="1",
                              attenuation="1 0.3 0.3", castshadow="false", diffuse=".35 .3 .25", specular="0 0 0")
            elif kind == "shoe":
                ET.SubElement(b, "geom", name=f"{name}_g", type="box", size=".13 .05 .045", pos="0 0 .045", **g)
            elif kind == "paper":
                ET.SubElement(b, "geom", name=f"{name}_g", type="box", size=".105 .15 .0005", **dict(g, conaffinity="0"))
            elif kind == "picture":
                ET.SubElement(b, "geom", name=f"{name}_g", type="box", size=".01 .3 .2", **dict(g, conaffinity="0"))


def _movers_xml(asset, world):
    """Stage 4: mocap bodies moved along paths every step. Visual only (no contacts)."""
    vis = dict(contype="0", conaffinity="0")
    for name in MOVERS:
        ET.SubElement(asset, "material", name=f"mv_{name}_mat", texture="obj_plain_tex" if name != "box" else "obj_cardboard_tex",
                      texuniform="true", texrepeat="3 3")
        b = ET.SubElement(world, "body", name=f"mv_{name}", mocap="true", pos="0 0 -5")
        g = dict(vis, material=f"mv_{name}_mat")
        if name == "vac":
            ET.SubElement(b, "geom", type="cylinder", size=".17 .045", pos="0 0 .045", **g)
            ET.SubElement(b, "geom", type="cylinder", size=".05 .005", pos=".08 0 .095", rgba=".9 .9 .9 1", **vis)
        elif name.startswith("ball"):
            ET.SubElement(b, "geom", name=f"mv_{name}_g", type="sphere", size=".1", **g)
        elif name == "box":
            ET.SubElement(b, "geom", type="box", size=".2 .15 .12", pos="0 0 .12", **g)
        else:                                                          # a leg from the hip down, with a shoe
            ET.SubElement(b, "geom", type="capsule", fromto="0 0 0 0 0 -.8", size=".06", **g)
            ET.SubElement(b, "geom", type="box", size=".13 .05 .04", pos=".05 0 -.84", rgba=".15 .15 .15 1", **vis)


def _outdoor_xml(root, asset, world, tex, mat):
    """Stages 5-7: heightfield ground in a body (so it can be shifted), sky, sun, rocks / bushes / trees."""
    root.find(".//geom[@name='floor']").set("pos", "0 0 -5")           # the flat floor goes out of the way
    ET.SubElement(asset, "hfield", name="terrain", nrow=str(TERR_N), ncol=str(TERR_N), size=f"{TERR_R} {TERR_R} {TERR_ZMAX} 0.2")
    ET.SubElement(asset, "hfield", name="terrain_far", nrow=str(TERR_N_FAR), ncol=str(TERR_N_FAR),
                  size=f"{TERR_R_FAR} {TERR_R_FAR} {TERR_ZMAX} 0.2")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1=".4 .6 .85", rgb2=".92 .95 1",
                  width="512", height="3072")
    tex("ground_tex", FLOOR_TEX)
    mat("ground_mat", "ground_tex", 1 / FLOOR_M)
    for name in ("rock", "bark", "foliage"):
        tex(f"{name}_tex", OBJ_TEX)
        mat(f"{name}_mat", f"{name}_tex", 2.0)
    for name in ("terrain", "terrain_far"):
        ground = ET.SubElement(world, "body", name="ground" if name == "terrain" else "ground_far")
        ET.SubElement(ground, "geom", name=name, type="hfield", hfield=name, material="ground_mat", contype="0", conaffinity="1")
    sky = ET.SubElement(world, "body", name="sky_body")
    ET.SubElement(sky, "light", name="sun", type="directional", pos="0 0 10", dir="0 0 -1", castshadow="true",
                  diffuse=".8 .8 .8", specular=".2 .2 .2")
    for kind, count, m_ in (("rock", 14, "rock_mat"), ("bush", 6, "foliage_mat"), ("tree", 4, "bark_mat")):
        for i in range(count):
            b = ET.SubElement(world, "body", name=f"out_{kind}{i}", pos="0 0 -5")
            if kind == "rock":
                ET.SubElement(b, "geom", name=f"out_{kind}{i}_g", type="ellipsoid", size=".2 .15 .1", material=m_, contype="0", conaffinity="1")
            elif kind == "bush":
                ET.SubElement(b, "geom", name=f"out_{kind}{i}_g", type="sphere", size=".3", material=m_, contype="0", conaffinity="1")
            else:
                ET.SubElement(b, "geom", name=f"out_{kind}{i}_g", type="cylinder", size=".12 1.2", pos="0 0 1.2", material=m_,
                              contype="0", conaffinity="1")
                ET.SubElement(b, "geom", name=f"out_{kind}{i}_c", type="ellipsoid", size="1 1 .8", pos="0 0 2.6",
                              material="foliage_mat", contype="0", conaffinity="0")


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
    """Re-draw the environment in place. Returns animate(d, t), which moves the stage 4 movers, or None."""
    # Geoms compiled at their body's origin are flagged "same frame", and MuJoCo then copies the
    # body pose and ignores geom_pos. Ours move every episode, so clear the flag.
    ours = [b for b in range(m.nbody) if m.body(b).name in ("room", "ground") or m.body(b).name.startswith(("obj_", "out_", "mv_"))]
    m.geom_sameframe[np.isin(m.geom_bodyid, ours)] = 0
    B = mujoco.mjtObj.mjOBJ_BODY
    if _id(m, B, "ground") >= 0:
        _outdoor(m, rng)
        return None
    room = _room(m, rng)
    if _id(m, B, "obj_box0") >= 0:
        _objects(m, rng, room)
    if _id(m, B, "mv_vac") >= 0:
        return _movers(m, rng, room)
    return None


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
    if C.STAGE >= 3:                                                   # dimmer, brighter, sometimes tinted
        tint = rng.uniform(0.7, 1.0, 3) if rng.random() < 0.15 else np.ones(3)
        m.light_diffuse[:] *= rng.uniform(0.35, 1.6) * tint
        m.vis.headlight.ambient[:] *= rng.uniform(0.3, 1.8)
    return dict(front=front, back=back, left=left, right=right, yaw=yaw)


def _against_wall(rng, room, depth, width):
    """Centre and yaw for something standing against a random wall, its front (local +x) facing in."""
    wall = int(rng.integers(4))
    if wall < 2:
        q = (room["front"] - depth if wall == 0 else -room["back"] + depth, rng.uniform(-room["right"] + width, room["left"] - width))
        face = math.pi if wall == 0 else 0.0
    else:
        q = (rng.uniform(-room["back"] + width, room["front"] - width), room["left"] - depth if wall == 2 else -room["right"] + depth)
        face = -math.pi / 2 if wall == 2 else math.pi / 2
    return _rot(q, room["yaw"]), room["yaw"] + face


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
    pool, p_active = (OBJECTS, P_ACTIVE) if C.STAGE == 2 else (OBJECTS3, P_ACTIVE3)
    for kind, count in pool.items():
        for i in range(count):
            name = f"obj_{kind}{i}"
            b = _id(m, B, name)
            m.body_pos[b] = (0, 0, -5)                                 # parked under the floor unless placed
            if kind == "lamp":
                lamp_light = _id(m, mujoco.mjtObj.mjOBJ_LIGHT, f"{name}_light")
                m.light_active[lamp_light] = False
            if rng.random() > p_active[kind]:
                continue
            mid = _id(m, MAT, f"{name}_mat")
            look = FIXED_LOOK.get(kind) or ("rug" if kind == "rug" else
                                            str(rng.choice(OBJ_TEXTURES, p=[.3, .25, .1, .25, .1] if kind == "box" else None)))
            m.mat_texid[mid, RGB] = tex_id[look]
            m.mat_texuniform[mid] = look != "rug"                      # a rug shows its pattern once
            m.mat_texrepeat[mid] = (1, 1) if look == "rug" else (3, 3)
            m.mat_rgba[mid] = (*(rng.uniform(0.25, 1.0, 3) if look == "plain" else rng.uniform(0.85, 1.0, 3)), 1)
            m.mat_specular[mid], m.mat_shininess[mid] = rng.uniform(0.05, 0.6), rng.uniform(0.1, 0.8)
            yaw = rng.uniform(0, 2 * math.pi)
            gid = _id(m, G, f"{name}_g")
            if kind == "paper":
                m.mat_rgba[mid] = (0.95, 0.95, 0.92, 1)
            if kind in ("shelf", "sofa"):
                depth, width = (0.15, 0.4) if kind == "shelf" else (0.4, 1.0)
                p, face = _against_wall(rng, room, depth, width)
                r = math.hypot(depth, width)
                if not (_in_way(p, r) or any(np.hypot(*(p - p2)) < r + r2 for p2, r2 in placed)):
                    placed.append((p, r))
                    m.body_pos[b], m.body_quat[b] = (*p, 0), _yaw_quat(face)
                continue
            if kind == "picture":                                       # on a wall, visual only
                half = (0.01, rng.uniform(0.2, 0.5), rng.uniform(0.15, 0.4))
                p, face = _against_wall(rng, room, 0.012, half[1])
                _box(m, gid, (0, 0), 0, half, 0)
                m.body_pos[b], m.body_quat[b] = (*p, rng.uniform(1.0, 1.7)), _yaw_quat(face)
                continue
            if kind == "paper":                                         # flat, visual only, anywhere
                q = (rng.uniform(-room["back"] + 0.2, room["front"] - 0.2), rng.uniform(-room["right"] + 0.2, room["left"] - 0.2))
                m.geom_pos[gid] = (0, 0, 0.0006 + 0.0002 * i)
                m.body_pos[b], m.body_quat[b] = (*_rot(q, room["yaw"]), 0), _yaw_quat(yaw)
                continue
            if kind in ("plant", "lamp", "shoe"):
                p = _free_spot(rng, room, {"plant": 0.25, "lamp": 0.2, "shoe": 0.14}[kind], placed)
                if p is not None:
                    m.body_pos[b], m.body_quat[b] = (*p, 0), _yaw_quat(yaw)
                    if kind == "lamp":
                        m.light_active[lamp_light] = True
                continue
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


# ---------------------------------------------------------------- stage 4: things that move by themselves
def _movers(m, rng, room):
    """Plan paths for a few movers; returns animate(d, t), which places them (mocap) at time t.
    Paths cross the view beyond the robot's reach (x >= 1.6 m) or pass it at |y| >= 1 m."""
    B = mujoco.mjtObj.mjOBJ_BODY
    mocap = {n: m.body_mocapid[_id(m, B, f"mv_{n}")] for n in MOVERS}
    plain = _id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "obj_plain_tex")
    for n in ("vac", "ball0", "ball1", "legL", "legR"):
        mid = _id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"mv_{n}_mat")
        m.mat_texid[mid, RGB] = plain
        m.mat_rgba[mid] = (*(rng.uniform(0.05, 0.25, 3) if n == "vac" else rng.uniform(0.15, 1.0, 3)), 1)
    kinds = [k for k in ("vac", "ball0", "ball1", "box", "person") if rng.random() < 0.5] or [str(rng.choice(["vac", "ball0", "person"]))]
    speed = {"vac": (0.25, 0.35), "ball0": (0.4, 1.2), "ball1": (0.4, 1.2), "box": (0.15, 0.3), "person": (0.8, 1.3)}
    plans = []
    for k in kinds:
        if k.startswith("ball"):
            rad = rng.uniform(0.05, 0.12)
            gid = _id(m, mujoco.mjtObj.mjOBJ_GEOM, f"mv_{k}_g")
            m.geom_size[gid], m.geom_rbound[gid] = (rad, 0, 0), rad
        else:
            rad = 0.0
        side = room["left"] if rng.random() < 0.5 else -room["right"]
        if rng.random() < 0.6 or abs(side) < 1.35:                       # cross the view
            xc = rng.uniform(min(1.6, room["front"] - 0.4), room["front"] - 0.4)
            a, b = np.array([xc, -room["right"] + 0.35]), np.array([xc, room["left"] - 0.35])
        else:                                                           # pass alongside, towards or away
            y0 = math.copysign(rng.uniform(1.0, abs(side) - 0.35), side)
            a, b = np.array([room["front"] - 0.4, y0]), np.array([-room["back"] + 0.4, y0])
        if rng.random() < 0.5:
            a, b = b, a
        plans.append(dict(kind=k, a=_rot(a, room["yaw"]), b=_rot(b, room["yaw"]), v=rng.uniform(*speed[k]),
                          t0=rng.uniform(1.0, 10.0), rad=rad, back_and_forth=k in ("vac", "person")))

    def animate(d, t):
        for n in MOVERS:
            d.mocap_pos[mocap[n]] = (0, 0, -5)
        for p in plans:
            length = float(np.linalg.norm(p["b"] - p["a"]))
            s = max(0.0, t - p["t0"]) * p["v"]
            if p["back_and_forth"]:
                s %= 2 * length
                forward = s <= length
                s = s if forward else 2 * length - s
            else:
                forward, s = True, min(s, length)
            u = (p["b"] - p["a"]) / max(length, 1e-6)
            pos = p["a"] + u * s
            heading = math.atan2(*(u if forward else -u)[::-1])
            if p["kind"] == "person":
                swing = 0.45 * math.sin(2 * math.pi * s / 1.3) if 0 < s < length else 0.0
                for leg, sgn in (("legL", 1), ("legR", -1)):
                    side = np.array([-math.sin(heading), math.cos(heading)]) * 0.1 * sgn
                    q = np.zeros(4)
                    mujoco.mju_mulQuat(q, _yaw_quat(heading), np.array([math.cos(sgn * swing / 2), 0, math.sin(sgn * swing / 2), 0]))
                    d.mocap_pos[mocap[leg]] = (*(pos + side), 0.88)
                    d.mocap_quat[mocap[leg]] = q
            else:
                d.mocap_pos[mocap[p["kind"]]] = (*pos, p["rad"])
                if p["rad"]:                                            # roll: turn about the horizontal axis across the path
                    q = np.zeros(4)
                    mujoco.mju_axisAngle2Quat(q, np.array([-u[1], u[0], 0.0]), s / p["rad"])
                    d.mocap_quat[mocap[p["kind"]]] = q
                else:
                    d.mocap_quat[mocap[p["kind"]]] = _yaw_quat(heading)
    return animate


# ---------------------------------------------------------------- stages 5-7: outdoors
def _outdoor(m, rng):
    B, G, MAT = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_MATERIAL
    near, far = terrain(rng, C.STAGE - 4)
    for name, body, h in (("terrain", "ground", near), ("terrain_far", "ground_far", far)):
        hid = _id(m, mujoco.mjtObj.mjOBJ_HFIELD, name)
        n, adr = int(m.hfield_nrow[hid]), m.hfield_adr[hid]
        hmin = float(h.min())
        m.hfield_data[adr:adr + n * n] = np.clip((h - hmin) / TERR_ZMAX, 0, 1).ravel()
        m.body_pos[_id(m, B, body)] = (0, 0, hmin)

    def height(x, y):
        inside = max(abs(x), abs(y)) < TERR_R
        return float(_bilinear(near, x, y, TERR_R) if inside else _bilinear(far, x, y, TERR_R_FAR))

    _put_texture(m, "ground_tex", GROUNDS[list(GROUNDS)[rng.integers(len(GROUNDS))]](rng))
    gm = _id(m, MAT, "ground_mat")
    m.mat_texrepeat[gm], m.mat_specular[gm], m.mat_reflectance[gm] = 1 / FLOOR_M, rng.uniform(0.0, 0.2), 0.0
    for name, gen in (("rock", rock_tex), ("bark", bark), ("foliage", foliage)):
        _put_texture(m, f"{name}_tex", gen(rng))
    sun = _id(m, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
    el, az = math.radians(rng.uniform(20, 75)), rng.uniform(0, 2 * math.pi)
    m.light_dir[sun] = (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))
    m.light_diffuse[sun] = _colour_temp(rng) * rng.uniform(0.6, 1.0)
    m.vis.headlight.ambient[:] = rng.uniform(0.25, 0.45) * np.array([0.85, 0.9, 1.0])

    placed = []
    for kind, count, p_on, r_min in (("tree", 4, 0.6, 3.0), ("rock", 14, 0.7, 1.6), ("bush", 6, 0.6, 1.6)):
        for i in range(count):
            b = _id(m, B, f"out_{kind}{i}")
            m.body_pos[b] = (0, 0, -5)
            if rng.random() > p_on:
                continue
            g = _id(m, G, f"out_{kind}{i}_g")
            if kind == "rock":
                half = (rng.uniform(0.05, 0.4), rng.uniform(0.05, 0.3), rng.uniform(0.04, 0.25))
                m.geom_size[g], m.geom_aabb[g], m.geom_rbound[g] = half, (0, 0, 0, *half), max(half)
                r, z_off = max(half[:2]), -0.3 * half[2]
            elif kind == "bush":
                rad = rng.uniform(0.15, 0.5)
                m.geom_size[g], m.geom_aabb[g], m.geom_rbound[g] = (rad, 0, 0), (0, 0, 0, rad, rad, rad), rad
                r, z_off = rad, 0.6 * rad
            else:
                tr, th = rng.uniform(0.06, 0.2), rng.uniform(0.6, 1.5)
                m.geom_size[g], m.geom_pos[g], m.geom_aabb[g], m.geom_rbound[g] = (tr, th, 0), (0, 0, th), (0, 0, 0, tr, tr, th), math.hypot(tr, th)
                c = _id(m, G, f"out_{kind}{i}_c")
                cr, cz = rng.uniform(0.6, 1.5), rng.uniform(0.5, 1.2)
                m.geom_size[c], m.geom_pos[c] = (cr, cr, cz), (0, 0, 2 * th + 0.6 * cz)
                r, z_off = tr, -0.05
            for _ in range(40):
                dist, ang = rng.uniform(r_min, 9.0), rng.uniform(0, 2 * math.pi)
                p = np.array([dist * math.cos(ang), dist * math.sin(ang)])
                if not _in_way(p, r) and all(np.hypot(*(p - p2)) > r + r2 + 0.1 for p2, r2 in placed):
                    placed.append((p, r))
                    m.body_pos[b] = (*p, height(*p) + z_off)
                    m.body_quat[b] = _yaw_quat(rng.uniform(0, 2 * math.pi))
                    break

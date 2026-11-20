"""
Pictures of the room scene with the full-mesh robot (room_render.xml): the overhead view used to draw
paths, the body camera, and a chase view. Physics runs in room_env (MJX); this only replays poses.

    v = RoomView(640, 480)
    v.set_state(qpos, mocap_pos, mocap_quat)
    img = v.render("chase", path=pts[:goal+1], trail=xy_so_far, red_legs={2})
    u, vpx = v.world_to_top(x, y); x, y = v.top_to_world(u, vpx)      # overhead pixel <-> floor metres
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

import room_scene as RS

if not os.path.exists(RS.RENDER_OUT):
    RS.build_render()


class RoomView:

    def __init__(self, w=640, h=480, scene="room"):
        if scene == "terrain":                       # heightfield ground, outdoor objects
            import terrain_scene as TS
            self.m = TS.load_model(TS.RENDER_OUT)
            self.ground = TS.ground_np
        else:                                        # room and room_dyn share the render model
            self.m = mujoco.MjModel.from_xml_path(RS.RENDER_OUT)
            self.ground = None
        self.d = mujoco.MjData(self.m)
        self.w, self.h = w, h
        self.ren = mujoco.Renderer(self.m, h, w)
        self.rgba0 = self.m.geom_rgba.copy()
        self.leg_geoms = []
        for leg in range(6):
            hb = self.m.jnt_bodyid[self.m.joint(f"hip{leg}").id]
            gs = []
            for g in range(self.m.ngeom):
                b = self.m.geom_bodyid[g]
                while b not in (0, hb):
                    b = self.m.body_parentid[b]
                if b == hb:
                    gs.append(g)
            self.leg_geoms.append(gs)
        self.chase = mujoco.MjvCamera()
        self.chase.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.chase.distance, self.chase.elevation, self.chase.azimuth = 0.9, -38, 0
        hh = RS.TOP_H * np.tan(np.radians(RS.TOP_FOVY / 2))
        self.top_half = (hh * w / h, hh)                    # metres of floor from the image centre

    def set_state(self, qpos, mocap_pos=None, mocap_quat=None):
        self.d.qpos[:] = qpos
        if mocap_pos is not None:                       # a static-room state has fewer rows: movers stay parked
            n = len(mocap_pos)
            self.d.mocap_pos[:n] = mocap_pos
            self.d.mocap_quat[:n] = mocap_quat
            if n < self.m.nmocap:
                self.d.mocap_pos[n:] = [[80 + i, 0, -1] for i in range(self.m.nmocap - n)]
        mujoco.mj_forward(self.m, self.d)

    def world_to_top(self, x, y):
        hw, hh = self.top_half
        return (np.asarray(x) / hw + 1) * self.w / 2, (1 - np.asarray(y) / hh) * self.h / 2

    def top_to_world(self, u, v):
        hw, hh = self.top_half
        return (np.asarray(u) / (self.w / 2) - 1) * hw, (1 - np.asarray(v) / (self.h / 2)) * hh

    def _dots(self, sc, pts, rgba, r, z=0.003, every=1):
        pts = np.asarray(pts)[::every]
        zs = z + (self.ground(pts) if self.ground is not None else np.zeros(len(pts)))
        for p, zz in zip(pts, zs):
            if sc.ngeom >= sc.maxgeom - 2:
                return
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
                                np.array([p[0], p[1], zz]), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
            sc.ngeom += 1

    def render(self, view="chase", path=None, trail=None, red_legs=(), azimuth=None):
        m = self.m
        m.geom_rgba[:] = self.rgba0
        for leg in red_legs:
            for g in self.leg_geoms[leg]:
                m.geom_rgba[g] = [0.9, 0.15, 0.1, 1]
        if view == "chase":
            yaw = np.degrees(np.arctan2(self.d.xmat[1][3], self.d.xmat[1][0]))
            self.chase.azimuth = yaw + 180 + 25 if azimuth is None else azimuth
            self.chase.lookat[:] = [self.d.qpos[0], self.d.qpos[1], 0.05]
            self.ren.update_scene(self.d, self.chase)
        else:
            self.ren.update_scene(self.d, camera=view)
        sc = self.ren.scene
        big = view == "top"
        if path is not None and len(path) and view != "ego":
            self._dots(sc, path, (0.1, 0.55, 1.0, 1), 0.02 if big else 0.004, every=3 if big else 2)
            g = path[-1]
            gz = 0.002 + (float(self.ground(np.asarray(g))) if self.ground is not None else 0.0)
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                                np.array([0.08 if big else 0.03, 0.002, 0]), np.array([g[0], g[1], gz]),
                                np.eye(3).reshape(-1), np.array([0.1, 0.85, 0.3, 0.9], np.float32))
            sc.ngeom += 1
        if trail is not None and len(trail) and view != "ego":
            self._dots(sc, trail, (1.0, 0.55, 0.1, 1), 0.015 if big else 0.003, every=2, z=0.004)
        return self.ren.render()

"""
Interactive demo: draw a path on the overhead view, watch the robot follow it, kill servos live.

    bash wslrun.sh python sim_app.py [policy=room_both] [port=8765] [seed=0]
    then open http://127.0.0.1:8765

The overhead camera is used only to draw the path (and, on the real robot, to track where the robot
is). The policy runs on the robot's own senses: its body camera and/or IMU, its past commands, and
where the drawn path lies relative to it. Physics: room_env (MJX); pictures: room_view (full meshes).
"""
import io
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("MUJOCO_GL", "egl")
import jax
import jax.numpy as jp
import numpy as np
from PIL import Image

import fault_rollout as R
import room_env as E
import room_scene as RS
import room_view as V

args = dict(a.split("=", 1) for a in sys.argv[1:])
POLICY = args.get("policy", "room_both")
PORT = int(args.get("port", 8765))
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "runs", POLICY, "config.json")))
SENSORS = cfg["config"].get("sensors", "both")
FAULT_NAMES = {"dead": 1, "stuck": 2, "signal": 3, "worn": 4, "repair": 0}
FAULT_SIZE = {1: 0.0, 2: 0.0, 3: 0.7, 4: float(np.radians(15)), 0: 0.0}
LEGS = ["front", "front-left", "rear-left", "rear", "rear-right", "front-right"]
WIN = E.N_PATH                                  # the policy sees a 2 m window of the drawn path

env = E.FaultRoom(config_overrides={"sensors": SENSORS})
policy = jax.jit(R.load_policy(POLICY, env))
reset, step = jax.jit(env.reset), jax.jit(env.step)
view = small = None                              # created in the sim thread (EGL contexts are per thread)

cmds = queue.Queue()
lock = threading.Lock()
frames = {}                                      # name -> jpeg bytes
status = {}


def jpeg(img, q=82):
    b = io.BytesIO()
    Image.fromarray(img).save(b, "JPEG", quality=q)
    return b.getvalue()


class Sim:
    def __init__(self, seed):
        self.new_room(seed)

    def new_room(self, seed):
        self.seed = seed
        self.st = reset(jax.random.PRNGKey(seed))
        info = dict(self.st.info)                 # no faults until the user picks one
        info["fault_kind"] = jp.zeros(12, jp.int32)
        info["fault_onset"] = jp.full(12, 10 ** 6, jp.int32)
        self.st = self.st.replace(info=info)
        self.full = None                          # the drawn path, 1 cm spacing (M, 2)
        self.offset = 0                           # index of the window start in the full path
        self.running = False
        self.trail = []
        self.t0 = 0
        self.recording, self.rec = False, []
        self.faults = {}                          # servo -> name
        self.msg = "Draw a path from the robot on the overhead view, then press Go."

    # ------------------------------------------------------------------ path handling
    def set_path(self, pts_px):
        xy = np.stack(view.top_to_world(np.array([p[0] for p in pts_px]), np.array([p[1] for p in pts_px])), -1)
        robot = np.asarray(self.st.data.qpos[:2])
        if np.linalg.norm(xy[0] - robot) > 0.02:
            xy = np.concatenate([robot[None], xy])
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-4])
        xy = xy[keep]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        if s[-1] < 0.1:
            self.msg = "Path too short: draw at least 10 cm."
            return
        n = int(s[-1] / E.DS) + 1
        q = np.arange(n) * E.DS
        full = np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], -1)
        self.full = full
        self.offset = 0
        self.trail = []
        self._load_window(0)
        self.msg = f"Path {s[-1]:.2f} m. Press Go."

    def _load_window(self, start):
        full = self.full
        win = full[start:start + WIN]
        if len(win) < WIN:                        # pad past the goal along the last direction
            d = win[-1] - win[-2] if len(win) > 1 else np.array([E.DS, 0])
            d = d / (np.linalg.norm(d) + 1e-9) * E.DS
            win = np.concatenate([win, win[-1] + d * np.arange(1, WIN - len(win) + 1)[:, None]])
        dd = np.diff(win, axis=0)
        th = np.arctan2(dd[:, 1], dd[:, 0])
        th = np.concatenate([th, th[-1:]])
        goal = min(WIN - 1, len(full) - 1 - start)
        info = dict(self.st.info)
        shift = start - self.offset
        info["path"] = jp.asarray(win, jp.float32)
        info["path_theta"] = jp.asarray(th, jp.float32)
        info["goal_idx"] = jp.asarray(goal, jp.int32)
        info["path_idx"] = jp.asarray(max(0, int(info["path_idx"]) - shift) if shift else 0, jp.int32)
        info["s"] = jp.asarray(max(0.0, float(info["s"]) - shift * E.DS) if shift else 0.0, jp.float32)
        info["reached"] = jp.zeros((), bool)
        self.offset = start
        self.st = self.st.replace(info=info)

    # ------------------------------------------------------------------ faults
    def set_fault(self, servo, name):
        k = FAULT_NAMES[name]
        info = dict(self.st.info)
        t = int(info["t"])
        info["fault_kind"] = info["fault_kind"].at[servo].set(k)
        info["fault_size"] = info["fault_size"].at[servo].set(FAULT_SIZE[k])
        info["fault_onset"] = info["fault_onset"].at[servo].set(t if k else 10 ** 6)
        info["was_active"] = info["was_active"].at[servo].set(False)
        self.st = self.st.replace(info=info)
        nm = f"{LEGS[servo // 2]} {'hip' if servo % 2 == 0 else 'knee'}"
        if k:
            self.faults[servo] = name
            self.msg = f"{nm} servo: {name}."
        else:
            self.faults.pop(servo, None)
            self.msg = f"{nm} servo repaired."

    # ------------------------------------------------------------------ loop
    def tick(self):
        if not self.running or self.full is None:
            return
        a = policy(self.st.obs)
        self.st = step(self.st, a)
        info = self.st.info
        if int(info["path_idx"]) > 120 and self.offset + WIN < len(self.full):
            self._load_window(self.offset + int(info["path_idx"]) - 20)
        self.trail.append(np.asarray(self.st.data.qpos[:2]))
        rem = (len(self.full) - 1) * E.DS - (self.offset * E.DS + float(self.st.info["s"]))
        if rem < 0.03:
            self.msg = f"Goal reached in {(int(info['t']) - self.t0) * E.C.CTRL_DT:.1f} s."
        if self.recording:
            self.rec.append(Image.fromarray(small.render("chase", path=self._path_ahead(), trail=self.trail[-200:],
                                                         red_legs=self._red())))

    def _red(self):
        return {s // 2 for s in self.faults}

    def _path_ahead(self):
        if self.full is None:
            return None
        return self.full

    def render(self):
        q = np.asarray(self.st.data.qpos)
        mp, mq = np.asarray(self.st.data.mocap_pos), np.asarray(self.st.data.mocap_quat)
        for v in (view, small):
            v.set_state(q, mp, mq)
        path, red = self._path_ahead(), self._red()
        out = {"top": jpeg(view.render("top", path=path, trail=self.trail, red_legs=red)),
               "chase": jpeg(small.render("chase", path=path, trail=self.trail[-300:], red_legs=red)),
               "ego": jpeg(small.render("ego"))}
        if E.FaultRoom is not None and SENSORS in ("cam", "both"):
            img = np.asarray(self.st.info["cam"][0]).reshape(RS.CAM_H, RS.CAM_W)
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            out["policycam"] = jpeg(np.asarray(Image.fromarray(img).resize((240, 160), Image.NEAREST)))
        return out

    def status(self):
        info = self.st.info
        t = (int(info["t"]) - self.t0) * E.C.CTRL_DT
        d = None
        rem = None
        if self.full is not None:
            rem = (len(self.full) - 1) * E.DS - (self.offset * E.DS + float(info["s"]))
            pts = self.full
            xy = np.asarray(self.st.data.qpos[:2])
            d = float(np.min(np.linalg.norm(pts - xy, axis=1)))
        speed = 0.0
        if len(self.trail) > 10:
            speed = float(np.linalg.norm(self.trail[-1] - self.trail[-11]) / (10 * E.C.CTRL_DT))
        return {"t": round(t, 2), "running": self.running, "remaining_cm": None if rem is None else round(max(rem, 0) * 100, 1),
                "off_path_mm": None if d is None else round(d * 1000), "speed_cms": round(speed * 100, 1),
                "faults": {str(k): v for k, v in self.faults.items()}, "msg": self.msg, "policy": POLICY,
                "sensors": SENSORS, "recording": self.recording, "has_path": self.full is not None,
                "tilt_deg": round(float(np.degrees(np.arccos(np.clip(-np.asarray(env._gravity(self.st.data))[2], -1, 1)))), 1)}


def sim_thread():
    global view, small
    view, small = V.RoomView(560, 560), V.RoomView(400, 300)
    sim = Sim(int(args.get("seed", 0)))
    last_render = 0.0
    while True:
        t_start = time.time()
        try:
            while True:
                c = cmds.get_nowait()
                kind = c[0]
                if kind == "new":
                    sim.new_room(c[1] if c[1] is not None else sim.seed + 1)
                elif kind == "path":
                    sim.running = False
                    sim.set_path(c[1])
                elif kind == "go":
                    if sim.full is None:
                        sim.msg = "Draw a path first."
                    else:
                        if not sim.running:
                            sim.t0 = int(sim.st.info["t"]) if not sim.trail else sim.t0
                        sim.running = True
                        sim.msg = "Walking."
                elif kind == "pause":
                    sim.running = False
                    sim.msg = "Paused."
                elif kind == "fault":
                    sim.set_fault(c[1], c[2])
                elif kind == "record":
                    if sim.recording and sim.rec:
                        os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
                        fn = os.path.join(HERE, "media", time.strftime("demo_%Y%m%d_%H%M%S.gif"))
                        fr = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in sim.rec[::2]]
                        fr[0].save(fn, save_all=True, append_images=fr[1:], duration=100, loop=0, optimize=True)
                        sim.msg = f"Saved {os.path.relpath(fn, HERE)} ({len(fr)} frames)."
                        sim.rec = []
                    sim.recording = not sim.recording
        except queue.Empty:
            pass
        sim.tick()
        if time.time() - last_render > 0.1 or not sim.running:
            f = sim.render()
            s = sim.status()
            with lock:
                frames.update(f)
                status.clear()
                status.update(s)
            last_render = time.time()
        dt = time.time() - t_start
        time.sleep(max(0.0, (E.C.CTRL_DT if sim.running else 0.1) - dt))


PAGE = os.path.join(HERE, "sim_app.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self._send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
        elif p == "/state":
            with lock:
                body = json.dumps(status).encode()
            self._send(200, body, "application/json")
        elif p.startswith("/img/"):
            with lock:
                b = frames.get(p[5:])
            if b is None:
                self._send(404, b"", "text/plain")
            else:
                self._send(200, b, "image/jpeg")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        p = self.path
        if p == "/path":
            cmds.put(("path", body["points"]))
        elif p == "/go":
            cmds.put(("go",))
        elif p == "/pause":
            cmds.put(("pause",))
        elif p == "/new":
            cmds.put(("new", body.get("seed")))
        elif p == "/fault":
            cmds.put(("fault", int(body["servo"]), body["kind"]))
        elif p == "/record":
            cmds.put(("record",))
        else:
            self._send(404, b"{}", "application/json")
            return
        self._send(200, b"{}", "application/json")


if __name__ == "__main__":
    threading.Thread(target=sim_thread, daemon=True).start()
    print(f"policy {POLICY} (sensors: {SENSORS}); open http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

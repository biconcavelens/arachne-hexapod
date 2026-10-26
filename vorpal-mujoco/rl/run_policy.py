"""
Run a trained policy on the REAL robot through the web app (laptop in the loop):
IMU -> policy -> 12 servo angles, 20 Hz, one HTTP round trip per tick.

    python run_policy.py name=imu cmd=0.06,0,0 seconds=15            # DRY RUN: reads the IMU, sends nothing
    python run_policy.py name=imu cmd=0.06,0,0 seconds=15 send=1     # moves the robot

Needs: `adb forward tcp:7000 tcp:7000`, the IMU firmware flashed and the IMU's
axes checked (tools/imu_log.py live), gyro calibrated with the robot still.

With send=1: eases from the current pose to the standing pose (2 s), ramps the
command up over 2 s, runs, then eases back to standing, also on Ctrl-C. Keep the
legs clear. The first time, hold the robot in the air; then put it on the floor.
Actions are limited to the collision-checked ranges in common.py, and servo
angles are clipped to 5..175 deg.

Logs every tick to runs/<name>/real_<tag>.csv (obs, action, servo commands,
IMU timestamp, loop latency) for the sim-to-real comparison.
"""
import csv
import http.client
import json
import math
import os
import sys
import time

import numpy as np

import common as C
from play_policy import Policy

ROOT = os.path.dirname(os.path.abspath(__file__))
args = dict(a.split("=", 1) for a in sys.argv[1:])
NAME = args.get("name", "imu")
CMD = np.array([float(v) for v in args.get("cmd", "0.06,0,0").split(",")])
SECONDS = float(args.get("seconds", 15))
SEND = args.get("send", "0") == "1"
HZ = float(args.get("hz", 1 / C.CTRL_DT))
TAG = args.get("tag", time.strftime("%H%M%S"))
RAMP_S = 2.0

conn = http.client.HTTPConnection("127.0.0.1", 7000, timeout=10)


def req(method, path):
    conn.request(method, path)
    return json.loads(conn.getresponse().read())


# ---- joint -> servo mapping, same formula as the web page: servo = 90 + (sim_deg - offset) / sign
CONFIRMED = {0: (0, 6), 1: (3, 9), 2: (4, 10), 3: (5, 11), 4: (2, 8), 5: (1, 7)}   # leg -> (hip ch, knee ch)
mapping = {}
saved = req("GET", "/mapping").get("mapping")
if saved:
    mapping = {e["joint"]: (ch, e["sign"], e["offset"]) for ch, e in enumerate(saved) if e["joint"] != "none"}
for k, (h, kn) in CONFIRMED.items():
    mapping.setdefault(f"hip{k}", (h, -1, 0))
    mapping.setdefault(f"knee{k}", (kn, 1, 0))
ORDER = [f"{j}{k}" for k in range(6) for j in ("hip", "knee")]       # = policy action order


def to_servos(target_rad):
    s = np.zeros(12)
    for name, t in zip(ORDER, target_rad):
        ch, sign, off = mapping[name]
        s[ch] = 90 + (math.degrees(t) - off) / sign
    return np.clip(np.round(s), 5, 175)


def tick(servos):
    """Set the servos and get the IMU back (dry run: IMU only)."""
    if SEND:
        r = req("POST", "/policy_step?a=" + ",".join(str(int(v)) for v in servos))
        return r["imu"]
    return req("GET", "/imu")["imu"]


def parse_imu(s):
    t, gx, gy, gz, ax, ay, az, dx, dy, dz = map(float, s.split(","))
    return t, np.array([dx, dy, dz]), np.array([gx, gy, gz])


def ease(servos_from, servos_to, seconds):
    n = max(1, int(seconds * HZ))
    for i in range(1, n + 1):
        u = 0.5 - 0.5 * math.cos(math.pi * i / n)
        tick(np.round(servos_from + (servos_to - servos_from) * u))
        time.sleep(1 / HZ)


pol = Policy(os.path.join(ROOT, "runs", NAME, "policy.npz"))
stand = to_servos(C.STAND_QPOS)
if tick_imu := req("GET", "/imu").get("imu"):
    print("IMU ok:", tick_imu)
else:
    sys.exit("no IMU data: flash the IMU firmware, check wiring (tools/imu_log.py status)")
print(f"{'SENDING to the robot' if SEND else 'DRY RUN (no servo commands)'}; policy {NAME}; command fwd/left/yaw = {CMD}; {SECONDS:.0f} s at {HZ:.0f} Hz")
print("stand pose servos:", stand.astype(int).tolist())

rows, last = [], C.STAND_ACTION.copy()
current = np.array(req("GET", "/state")["angles"], dtype=float)
try:
    if SEND:
        ease(current, stand, RAMP_S)
    t0 = time.time()
    n, overruns = int(SECONDS * HZ), 0
    for step in range(n):
        t_start = time.time()
        cmd = CMD * min(1.0, step / (RAMP_S * HZ))
        _, grav, gyro = parse_imu(req("GET", "/imu")["imu"]) if step == 0 else (None, grav, gyro)
        obs = C.build_obs(grav, gyro, cmd, last, step)
        a = pol(obs)
        servos = to_servos(C.action_to_target(a))
        t_imu, grav, gyro = parse_imu(tick(servos))
        rows.append([step, time.time() - t0, t_imu, *cmd, *obs[:6], *a, *servos, (time.time() - t_start) * 1000])
        last = a
        sleep = t_start + 1 / HZ - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            overruns += 1
finally:
    if SEND:
        print("easing back to standing...")
        ease(np.array([r for r in rows[-1][-13:-1]]) if rows else stand, stand, RAMP_S)
    out = os.path.join(ROOT, "runs", NAME, f"real_{TAG}.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "t_s", "imu_t_us", "cmd_fwd", "cmd_left", "cmd_yaw", "grav_x", "grav_y", "grav_z",
                        "gyro_x", "gyro_y", "gyro_z"] + [f"a{i}" for i in range(12)] + [f"servo{i}" for i in range(12)] + ["loop_ms"])
            w.writerows(rows)
        lat = np.array([r[-1] for r in rows])
        print(f"{len(rows)} ticks, loop {lat.mean():.0f} ms mean / {np.percentile(lat, 95):.0f} ms p95 / {lat.max():.0f} ms max, "
              f"{overruns} ticks over the {1000/HZ:.0f} ms budget -> {out}")

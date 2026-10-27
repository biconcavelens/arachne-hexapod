"""
Quick checks of the pieces that do not need MuJoCo, rendering or torch (a few seconds, CPU).

    python selftest.py
"""
import numpy as np

import config as C
from egomotion import PlaneEgomotion, camera_axes_in_body, intrinsics
from faults import Servo
from gaits_lib import PROBE_PRE_S, REST, T_RAMP, T_SETTLE, DeadbandWorkaround, Gait, probe_command
from recovery import estimate_deadband


def test_faults():
    s = Servo("deadband", 17.0, onset=0.0)
    assert s.step(0, 0) == 0                      # first command always lands
    assert s.step(10, 0.05) == 0                  # inside the dead band: ignored
    assert s.step(20, 0.10) == 20                 # bigger than it: goes through
    s = Servo("sag", 3.0, onset=1.0)
    s.step(0, 0.0)
    assert s.step(10, 0.5) == 10                  # before onset: healthy
    assert s.step(-20, 1.0) == 7 and s.step(30, 1.05) == 13      # +-3 deg around the onset angle
    s = Servo("loss", 1.0, onset=0.0, rng=np.random.default_rng(1))
    s.step(5, 0)
    assert all(s.step(c, 0.1) == 5 for c in range(-20, 20))       # frozen
    arrived = lambda copies: np.mean([_arrive(Servo("loss", 0.6, 0, np.random.default_rng(i)), copies) for i in range(400)])
    assert arrived(3) > arrived(1) + 0.2          # 1 - 0.6^3 vs 1 - 0.6: resending gets through more often


def _arrive(s, copies):
    s.copies = copies
    s.step(0, 0)
    return s.step(10, 0.1) == 10


def test_deadband_workaround():
    g = Gait()
    j = int(np.argmax(g.travel()))
    w = DeadbandWorkaround(g, j, db_deg=17.0)
    assert w.usable
    ts = np.arange(T_SETTLE + T_RAMP, 6, C.CTRL_DT)
    vals = {round(float(w.command(t)[j]), 6) for t in ts}
    assert vals == {round(w.lo, 6), round(w.hi, 6)}                       # only the two extremes
    others = [k for k in range(12) if k != j]
    assert all(np.allclose(w.command(t)[others], g.command(t)[others]) for t in ts)
    assert not DeadbandWorkaround(g, j, db_deg=g.travel()[j]).usable       # too wide to work around


def test_probe_and_estimate():
    j = 1                                                                  # knee0
    offs = []
    for t in np.arange(0, PROBE_PRE_S + C.PROBE_S, C.CTRL_DT):
        c, off = probe_command(REST, j, t)
        assert (off is None) == (t < PROBE_PRE_S)
        assert np.allclose(np.delete(c, j), np.delete(REST, j))           # only servo j moves
        if off is not None:
            offs.append(off)
    offs = np.array(offs)
    assert np.all(np.diff(offs) >= 0) and offs[-1] <= C.PROBE_RAMP_DEG
    rng = np.random.default_rng(0)
    for db in (15.0, 17.5, 20.0):
        ang = rng.normal(0.02, 0.01, len(offs)) + 0.5 * (offs > db)       # body jumps once the offset clears db
        est = estimate_deadband(offs, ang, rng.normal(0.01, 0.005, len(offs)))
        assert abs(est - db) <= C.PROBE_RAMP_DEG / (C.PROBE_S / C.CTRL_DT) + 1e-6, (db, est)


def test_egomotion_roundtrip():
    pose = C.camera_pose()
    pe = PlaneEgomotion(intrinsics(), camera_axes_in_body(pose), pose["pos_m"], 0.095 + pose["pos_m"][2])
    twist = np.array([0.06, 0.01, 0.0, 0.0, 0.0, 0.3])                     # walking forward, turning
    du, dv = pe.flow_from_twist(twist, C.CTRL_DT)
    flow = np.zeros((C.IMG_H, C.IMG_W, 2))
    flow[pe.v, pe.u, 0], flow[pe.v, pe.u, 1] = du, dv
    assert np.allclose(pe.solve(flow, C.CTRL_DT), twist, atol=1e-6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

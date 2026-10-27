"""
Servo fault layer: sits between the gait (what we command) and the actuators (what
the servo actually holds). Angles in degrees.

  loss      each command is dropped with probability p (p = 1: the servo is frozen)
  sag       the servo still moves, but only within +-span (2-5 deg) of where it was
            when the sag began; like a loss with a little movement
  deadband  a commanded change smaller than db (15-20 deg) is ignored; bigger ones go through

Recovery actions change the servo's state the way the config says they would on the
real robot: resend (several copies per command), channel reset, pause-and-resend.
"""
import numpy as np

import config as C


class Servo:
    def __init__(self, kind="none", size=0.0, onset=np.inf, rng=None):
        self.kind, self.size, self.onset = kind, float(size), float(onset)
        self.rng = rng or np.random.default_rng(0)
        self.target = None       # what the servo is holding (deg)
        self.cleared = False     # a recovery action removed the fault
        self.copies = 1          # commands are sent this many times
        self.anchor = None       # sag: where the servo was when the sag began

    def active(self, t):
        return self.kind != "none" and t >= self.onset and not self.cleared

    def step(self, cmd, t):
        """One command (deg) at time t -> the target the servo now holds."""
        if self.target is None or not self.active(t):
            self.target = float(cmd)
            return self.target
        if self.kind == "loss":
            if self.rng.random() >= self.size ** self.copies:        # at least one copy arrived
                self.target = float(cmd)
        elif self.kind == "sag":
            if self.anchor is None:
                self.anchor = self.target
            self.target = float(np.clip(cmd, self.anchor - self.size, self.anchor + self.size))
        elif self.kind == "deadband":
            if abs(cmd - self.target) > self.size:
                self.target = float(cmd)
        return self.target

    # ---- recovery actions; each returns True if the fault is gone
    def resend(self):
        if self.kind == "loss":
            self.copies = C.RESEND_COPIES
            if self.size >= 0.999 and self.rng.random() < C.LINK_RESET_P:   # frozen: a channel reset may fix it
                self.cleared = True
        elif self.kind == "sag" and self.rng.random() < C.SAG_CLEAR_P:
            self.cleared, self.anchor = True, None
        return self.cleared


class ServoBank:
    """The 12 servos in actuator order (hip0, knee0, hip1, ...)."""

    def __init__(self, faults=(), rng=None):
        rng = rng or np.random.default_rng(0)
        self.servos = [Servo(rng=np.random.default_rng(rng.integers(1 << 31))) for _ in range(12)]
        for f in faults:
            self.servos[f["servo"]] = Servo(f["kind"], f["size"], f["onset"], np.random.default_rng(rng.integers(1 << 31)))

    def step(self, cmd_deg, t):
        return np.array([s.step(c, t) for s, c in zip(self.servos, cmd_deg)])

    def labels(self, t):
        """Per-servo fault type id and size, as true at time t."""
        types = np.array([C.TYPE_ID[s.kind] if s.active(t) else 0 for s in self.servos], dtype=np.int64)
        sizes = np.array([s.size if s.active(t) else 0.0 for s in self.servos], dtype=np.float32)
        return types, sizes


def sample_faults(rng, movable, n=None, kinds=("loss", "sag", "deadband")):
    """Random faults on 1 or 2 of the servos the gait actually moves."""
    n = n if n is not None else int(rng.integers(C.N_FAULTY[0], C.N_FAULTY[1] + 1))
    servos = rng.choice(np.flatnonzero(movable), size=n, replace=False)
    out = []
    for j in servos:
        kind = str(rng.choice(kinds))
        size = {"loss": rng.uniform(*C.LOSS_P_DROP),
                "sag": rng.uniform(*C.SAG_SPAN_DEG),
                "deadband": rng.uniform(*C.DEADBAND_DEG)}[kind]
        if kind == "loss" and rng.random() < 0.3:
            size = 1.0                                   # some losses are a fully frozen channel
        out.append({"servo": int(j), "kind": kind, "size": float(size), "onset": float(rng.uniform(*C.ONSET_S))})
    return out

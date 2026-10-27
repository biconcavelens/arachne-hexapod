"""
Online diagnosis and recovery, for the closed-loop simulation (closed_loop.py).

Diagnosis every STRIDE steps on the last WINDOW steps. An alarm needs CONSEC windows in
a row agreeing on the same servo and fault type.
  mode "model"   the trained Stage B network (train.py)
  mode "oracle"  the true fault labels: the best any diagnosis could do
  mode "none"    no diagnosis, no recovery

Recovery per diagnosed (servo, type), at most MAX_ATTEMPTS per servo:
  loss      resend: every command goes out RESEND_COPIES times, plus a channel reset
  sag       stand still for PAUSE_S, then resend
  deadband  stand, ramp that servo slowly (probe), take the command offset at which the
            body first jumps (Stage A twist) as the dead band, then switch that servo to the
            jump-between-extremes gait (gaits_lib.DeadbandWorkaround)
"""
from collections import deque

import numpy as np

import config as C
from gaits_lib import PROBE_PRE_S, REST, DeadbandWorkaround, probe_command


def estimate_deadband(offsets, ang_rate, lin_speed, base_frac=0.2, k=6.0):
    """Command offset (deg) at which the body first moves during a probe ramp.

    The baseline is the first part of the ramp, where the servo has not moved yet."""
    n0 = max(3, int(len(offsets) * base_frac))
    for sig in (np.asarray(ang_rate), np.asarray(lin_speed)):
        mu, sd = sig[:n0].mean(), max(sig[:n0].std(), 1e-3)
        hits = np.flatnonzero(sig[n0:] > mu + k * sd)
        if hits.size:
            first = n0 + hits[0]
            return float(offsets[max(first - 1, 0)])
    return float(offsets[-1])            # never moved: at least the full ramp


class ModelDiagnosis:
    def __init__(self, ckpt, device="cpu"):
        import torch

        import models as M
        self.torch, self.M = torch, M
        self.healthy, self.diag, self.norm, self.cfg = M.load(ckpt, device)
        self.device = device

    def __call__(self, rows):
        """rows: the last WINDOW + HIST - 1 steps -> (12, n_types) probabilities."""
        M = self.M
        cat = {k: np.stack([r[k] for r in rows]) for k in ("feat", "imu", "cmd", "phase")}
        step = M.step_inputs(cat["feat"], cat["imu"], cat["cmd"], cat["phase"])
        hin = M.healthy_inputs(cat["cmd"], cat["phase"])
        x = M.window_inputs(step, hin, self.healthy, self.norm, self.device)[-C.WINDOW:]
        with self.torch.no_grad():
            out = self.diag(self.torch.from_numpy(x)[None].to(self.device))
        return self.torch.softmax(out["types"][0], -1).cpu().numpy()


class RecoveryController:
    """Supplies the commands each step and watches the robot's own signals (see simulate.run_episode)."""

    def __init__(self, mode="model", diagnosis=None, threshold=0.5, consec=2):
        self.mode, self.diagnosis, self.thr, self.consec = mode, diagnosis, threshold, consec

    def attach(self, gait, bank, rng):
        from models import HIST
        self.gait, self.bank = gait, bank
        self.rows = deque(maxlen=C.WINDOW + HIST - 1)
        self.state, self.k = "walk", 0
        self.streak, self.attempts, self.events = {}, {}, []

    def phase(self, t):
        return self.gait.phase(t) if self.state == "walk" else 0.0

    # ---- commands
    def command(self, t):
        if self.state == "pause":
            if t < self.until:
                return REST.copy()
            ok = self.bank.servos[self.pending].resend()
            self.events.append({"t": t, "action": "resend after pause", "servo": self.pending, "fixed": ok})
            self._resume()
        if self.state == "probe":
            c, self._probe_now = probe_command(REST, self.pending, t - self.t0)
            if t - self.t0 >= PROBE_PRE_S + C.PROBE_S:
                self._finish_probe(t)
            return c
        return self.gait.command(t)

    def _resume(self):
        self.state = "walk"
        self.rows.clear()
        self.streak = {}

    # ---- observations
    def observe(self, t, row):
        if self.state == "probe":
            if self._probe_now is not None:                       # only during the ramp
                self.probe_off.append(self._probe_now)
                self.probe_ang.append(float(np.linalg.norm(row["feat"][3:6])))
                self.probe_lin.append(float(np.linalg.norm(row["feat"][0:3])))
            return
        if self.state != "walk" or self.mode == "none" or t < C.WARMUP_S:   # training skips the warm-up too
            return
        self.rows.append(row)
        self.k += 1
        if len(self.rows) < self.rows.maxlen or self.k % C.STRIDE:
            return
        p = self._diagnose(t)
        for j in range(12):
            p_fault = 1 - p[j, 0]
            if p_fault <= self.thr:
                self.streak[j] = (0, 0)
                continue
            ty = 1 + int(p[j, 1:].argmax())
            prev_ty, n = self.streak.get(j, (0, 0))
            n = n + 1 if prev_ty == ty else 1
            self.streak[j] = (ty, n)
            if n >= self.consec:
                self._alarm(t, j, ty)
                return                                     # one action at a time

    def _diagnose(self, t):
        if self.mode == "oracle":
            types, _ = self.bank.labels(t)
            p = np.zeros((12, len(C.FAULT_TYPES)))
            p[np.arange(12), types] = 1.0
            return p
        return self.diagnosis(list(self.rows))

    # ---- actions
    def _alarm(self, t, j, ty):
        name = C.FAULT_TYPES[ty]
        self.events.append({"t": t, "alarm": name, "servo": j})
        if self.attempts.get(j, 0) >= C.MAX_ATTEMPTS:
            return
        self.attempts[j] = self.attempts.get(j, 0) + 1
        self.pending = j
        if name == "loss":
            ok = self.bank.servos[j].resend()
            self.events.append({"t": t, "action": "resend", "servo": j, "fixed": ok})
            self._resume()
        elif name == "sag":
            self.state, self.until = "pause", t + C.PAUSE_S
        elif name == "deadband":
            self.state, self.t0, self._probe_now = "probe", t, None
            self.probe_off, self.probe_ang, self.probe_lin = [], [], []

    def _finish_probe(self, t):
        j = self.pending
        db = (estimate_deadband(np.array(self.probe_off), self.probe_ang, self.probe_lin)
              if len(self.probe_off) >= 10 else C.PROBE_RAMP_DEG)
        wrapped = DeadbandWorkaround(self.gait, j, db)
        if wrapped.usable:
            self.gait = wrapped
        self.events.append({"t": t, "action": "deadband workaround" if wrapped.usable else "deadband too wide to work around",
                            "servo": j, "estimate_deg": round(db, 1)})
        self._resume()

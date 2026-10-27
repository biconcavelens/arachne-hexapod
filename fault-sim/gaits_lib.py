"""
Commands for the fault simulator: the clearance-checked tripod gait with random
variations, the dead-band workaround gait, and the probe used to measure a dead band.
Angles in degrees, actuator order (hip0, knee0, hip1, ...).
"""
import json
import math

import numpy as np

import config as C

_G = json.load(open(C.GAIT_JSON))
FRAMES = np.array(_G["frames"], dtype=float)[:, C.GAIT_TO_ACT]   # one cycle, actuator order
REST = np.array(_G["rest"], dtype=float)[C.GAIT_TO_ACT]
FPS = _G["fps"]
HIP_SIDE = np.array(_G["hip_side"], dtype=float)                 # +1 left legs, -1 right, 0 front/back
STANCE_KNEE = _G["params"]["stance_knee_deg"]
IS_HIP = np.array([n.startswith("hip") for n in C.JOINT_NAMES])
T_SETTLE, T_RAMP = 1.0, 0.6


class Gait:
    """The tripod gait, scaled. command(t) -> 12 angles."""

    def __init__(self, steer=0.0, speed=1.0, hip_scale=1.0, knee_scale=1.0):
        f = FRAMES.copy()
        side = np.repeat(HIP_SIDE, 2)                          # per actuator: its leg's side
        f[:, IS_HIP] *= hip_scale * (1 + steer * side[IS_HIP])
        f[:, ~IS_HIP] = STANCE_KNEE + (f[:, ~IS_HIP] - STANCE_KNEE) * knee_scale
        self.frames, self.speed = f, speed
        self.params = dict(steer=steer, speed=speed, hip_scale=hip_scale, knee_scale=knee_scale)

    @classmethod
    def random(cls, rng):
        return cls(rng.uniform(*C.STEER), rng.uniform(*C.SPEED), rng.uniform(*C.HIP_AMP_SCALE), rng.uniform(*C.KNEE_LIFT_SCALE))

    def phase(self, t):
        """Gait phase in [0, 1) (0 before walking starts)."""
        return 0.0 if t < T_SETTLE + T_RAMP else ((t - T_SETTLE - T_RAMP) * FPS * self.speed / len(self.frames)) % 1.0

    def command(self, t):
        if t < T_SETTLE:
            return REST.copy()
        if t < T_SETTLE + T_RAMP:
            u = 0.5 - 0.5 * math.cos(math.pi * (t - T_SETTLE) / T_RAMP)
            return REST + (self.frames[0] - REST) * u
        f = self.phase(t) * len(self.frames)
        i = int(f)
        return self.frames[i] + (self.frames[(i + 1) % len(self.frames)] - self.frames[i]) * (f - i)

    def travel(self):
        """Peak-to-peak commanded travel per servo over one cycle (deg)."""
        return self.frames.max(0) - self.frames.min(0)


class DeadbandWorkaround:
    """Wraps a gait: servo j jumps between its cycle extremes instead of moving smoothly.

    Every jump is the servo's full travel, so it clears a dead band smaller than that
    travel. The switch happens where the smooth trajectory crosses its midpoint, so the
    timing of the gait is kept. If the travel is not bigger than the dead band, the
    servo cannot be moved reliably; usable is then False and the caller walks around it.
    """

    def __init__(self, gait, servo, db_deg, margin=C.DEADBAND_MARGIN_DEG):
        self.gait, self.j = gait, servo
        self.frames, self.params = gait.frames, dict(gait.params, deadband_workaround=servo)   # stackable
        lo, hi = gait.frames[:, servo].min(), gait.frames[:, servo].max()
        self.lo, self.hi, self.mid = lo, hi, (lo + hi) / 2
        self.usable = (hi - lo) > db_deg + margin

    def phase(self, t):
        return self.gait.phase(t)

    def command(self, t):
        c = self.gait.command(t)
        if self.usable and t >= T_SETTLE + T_RAMP:
            c[self.j] = self.hi if c[self.j] >= self.mid else self.lo
        return c


PROBE_LIMITS = {"hip": (-24.0, 24.0), "knee": (-54.0, 10.0)}   # inside the collision-checked ranges
PROBE_PRE_S = 1.8


def probe_command(stand, servo, t, ramp_deg=C.PROBE_RAMP_DEG, seconds=C.PROBE_S):
    """Standing pose with one servo ramped slowly, to find where it starts to move.

    With no feedback we do not know what angle a dead-banded servo is holding, so it is
    first forced to a known angle: to one limit, then the other (that jump is bigger than
    any dead band, whatever the servo held), then back to the stand angle; then the body
    settles and the slow ramp starts. Knees ramp down (the foot pushes, the body tilts),
    hips toward +. Returns (command, offset in deg, or None before the ramp)."""
    c = stand.copy()
    kind = "knee" if C.JOINT_NAMES[servo].startswith("knee") else "hip"
    lo, hi = PROBE_LIMITS[kind]
    if t < 0.4:
        c[servo] = lo
        return c, None
    if t < 0.8:
        c[servo] = hi
        return c, None
    if t < PROBE_PRE_S:
        return c, None
    direction = -1.0 if kind == "knee" else 1.0
    off = direction * ramp_deg * min(1.0, (t - PROBE_PRE_S) / seconds)
    c[servo] += off
    return c, abs(off)

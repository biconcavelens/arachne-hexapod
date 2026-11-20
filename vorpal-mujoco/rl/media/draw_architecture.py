"""Model architecture figure: asymmetric actor-critic PPO policy + fault-diagnosis head.
    ~/rl/bin/python media/draw_architecture.py   -> media/architecture.png / .svg
Numbers match room_env.py (HIST=20, CAM_FRAMES=(0,1,4), 24x16 camera) and train_fault.py layers."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, RULE = "#15181d", "#4b5260", "#c9ced6"
C_IN = "#e8f0fb"      # observation
C_ACT = "#2a78d6"     # actor
C_CRI = "#eb6834"     # critic (training only)
C_DIA = "#1baf7a"     # diagnosis
C_ENV = "#f1f2f4"

fig = plt.figure(figsize=(16, 9.4), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 94); ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"


def box(x, y, w, h, title, lines=(), fc="white", ec=RULE, tc=INK, lw=1.2, title_size=10.5, size=8.6, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw,
                       ls=(0, (4, 3)) if dashed else "-")
    ax.add_patch(p)
    ax.text(x + 1.4, y + h - 1.6, title, ha="left", va="top", fontsize=title_size, fontweight="bold", color=tc)
    for k, l in enumerate(lines):
        ax.text(x + 1.4, y + h - 4.6 - k * 2.55, l, ha="left", va="top", fontsize=size, color=INK2 if tc == INK else tc)
    return p


def layer_stack(x, y, w, sizes, color, label, sub):
    """Dense layers drawn as bars whose height follows the width."""
    ax.text(x, y + 15.5, label, fontsize=10.5, fontweight="bold", color=color, va="bottom")
    ax.text(x, y + 13.6, sub, fontsize=8.3, color=INK2, va="bottom")
    gap = w / len(sizes)
    for k, n in enumerate(sizes):
        h = 2.0 + 10.5 * n / 512
        cx = x + gap * k + gap / 2
        ax.add_patch(FancyBboxPatch((cx - 1.6, y + 6.5 - h / 2), 3.2, h, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=color, ec="none", alpha=0.9))
        ax.text(cx, y + 6.5 - h / 2 - 1.2, str(n), ha="center", va="top", fontsize=8.2, color=INK)
        if k:
            ax.annotate("", (cx - 1.8, y + 6.5), (cx - gap + 1.8, y + 6.5),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))


def arrow(a, b, color=INK2, lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))


# title
ax.text(3, 91, "Arachne: fault-tolerant path following, model architecture", fontsize=16, fontweight="bold", color=INK, va="top")
ax.text(3, 87.4, "Asymmetric actor-critic PPO (Brax, MuJoCo MJX). One policy per environment stage and sensor set; "
        "the policy is never told which servo failed or how.", fontsize=9.5, color=INK2, va="top")

# ---------------------------------------------------------------- observations (left)
ax.text(3, 82.5, "ACTOR OBSERVATION  (what the robot has)", fontsize=9, color=INK2, fontweight="bold")
box(3, 62.5, 34, 18, "Per control step (20 Hz)", [
    "IMU: gravity vector 3 + gyro 3      (imu, both)",
    "last servo commands 12",
    "path features 8  (3 points ahead, offset,",
    "   distance left; from the overhead camera)",
    "gait clock 2  (sin, cos)"], fc=C_IN, ec=C_IN)
box(3, 52.5, 34, 8, "History  x 20 steps (1 s)", [
    "28 x 20 = 560 values   (cam-only: 22 x 20 = 440)"], fc=C_IN, ec=C_IN)
box(3, 37.5, 34, 13.5, "Body camera  (cam, both)", [
    "24 x 16 grey, ray-cast from the camera cap,",
    "looks along the body's +x, 20 deg down",
    "frames: now, -0.05 s, -0.2 s",
    "3 x 384 = 1,152 values"], fc=C_IN, ec=C_IN)
# little pixel grid icon
for i in range(3):
    gx, gy = 29.5 + i * 1.2, 46.8 + i * 0.7
    ax.add_patch(FancyBboxPatch((gx, gy), 3.9, 2.4, boxstyle="square,pad=0", fc="#9fb4cf", ec="white", lw=1.0, alpha=0.6 + 0.13 * i))
box(3, 25.5, 34, 9.5, "Actor input  'state'", ["both 1,712   |   imu 560   |   cam 1,592",
                                              "running mean / std normalisation"], fc="white", ec=C_ACT, lw=1.6)
arrow((20, 62.5), (20, 60.5))
ax.plot([37, 39.5, 39.5], [56.5, 56.5, 30], color=INK2, lw=1.4)
ax.plot([37, 39.5], [44, 44], color=INK2, lw=1.4)
arrow((39.5, 30), (37.2, 30))

# ---------------------------------------------------------------- actor
layer_stack(48, 42, 27, [512, 256, 128], C_ACT, "Actor  (policy)", "MLP, swish activations")
arrow((37, 33.5), (49.5, 46), color=C_ACT, rad=-0.25)
box(78, 42, 20, 13, "Tanh-normal", ["mean + std, 24 outputs", "deterministic at test:", "mean only"], fc="white", ec=C_ACT)
arrow((74.5, 48.5), (78, 48.5), color=C_ACT)
box(102, 43, 21, 11, "12 servo targets", ["6 legs x (hip, knee)", "20 Hz position commands"], fc="white", ec=C_ACT)
arrow((98, 48.5), (102, 48.5), color=C_ACT)

# ---------------------------------------------------------------- environment
box(127, 27, 30, 34, "Simulation", [
    "fault layer (0-3 servos, any time):",
    "   dead  ·  stuck  ·  signal loss  ·  worn gears",
    "MuJoCo MJX hexapod, 4,096 in parallel",
    "stages: room · room + movers ·",
    "   rough terrain · terrain + movers",
    "",
    "reward: stay on path (cross-track),",
    "   progress (no penalty above 50% speed),",
    "   face along path, reach goal,",
    "   upright, smooth; fall / bump ends it"], fc=C_ENV, ec=C_ENV, size=8.2)
arrow((123, 48.5), (127, 48.5), color=C_ACT)
# loop back: observations
ax.plot([142, 142, 40], [61, 78.5, 78.5], color=INK2, lw=1.2)
arrow((40, 78.5), (37.2, 78.5), lw=1.2)
ax.text(90, 79.4, "next observation", fontsize=8.5, color=INK2, ha="center")

# ---------------------------------------------------------------- critic
box(46, 6, 30, 19, "Critic input  'privileged_state'", [
    "only in training; sees what the robot can't:",
    "body velocity, joint angles, speeds, torques,",
    "foot contacts, height, path features,",
    "fault kind one-hot + size per servo,",
    "mover positions / velocities (dyn stages)",
    "156 values  (172 with movers)"], fc="#fdf0ea", ec="#fdf0ea", size=8.2)
layer_stack(82, 7.5, 26, [512, 256, 256], C_CRI, "Critic  (value)", "MLP, swish; training only")
box(112, 13, 11, 7, "V(s)", ["1 value"], fc="white", ec=C_CRI)
arrow((76, 15), (83, 14), color=C_CRI)
arrow((108.5, 14), (112, 16), color=C_CRI)
ax.plot([127, 61], [28.5, 28.5], color=C_CRI, lw=1.4, ls=(0, (4, 3)))
arrow((61, 28.5), (61, 25.2), color=C_CRI)
ax.text(100, 29.4, "privileged state (training only)", fontsize=8.3, color=C_CRI, ha="center")
box(127, 6, 30, 15, "PPO", ["40M steps, 4,096 envs, unroll 20", "32 minibatches x 4 epochs",
                             "gamma 0.97, lr 3e-4, entropy 0.01", "RTX 4060 8 GB, ~2.5 h per policy"],
    fc="white", ec=RULE, size=8.2)

# ---------------------------------------------------------------- diagnosis
box(46, 61, 30, 13, "Diagnosis input  (separate net)", [
    "actor input at now, -1 s, -2 s",
    "3 x 1,712 values, standardised",
    "trained after the policy, frozen policy"], fc="#e7f6f0", ec="#e7f6f0", size=8.2)
layer_stack(82, 59, 26, [512, 512, 256], C_DIA, "Fault diagnosis", "MLP, GELU, dropout 0.1")
box(111, 62, 14, 10, "12 sigmoids", ["P(servo i", "is faulty)"], fc="white", ec=C_DIA, size=8.2)
arrow((76, 67.5), (83, 66), color=C_DIA)
arrow((108.5, 65.5), (111, 67), color=C_DIA)
arrow((39.5, 52), (46, 64), color=C_DIA, ls=(0, (4, 3)))
ax.text(127, 71.5, "names the servo,\nnot the fault kind;\nnot used for control", fontsize=8.2, color=INK2, va="top")

# legend
for k, (c, t) in enumerate([(C_ACT, "runs on the robot"), (C_CRI, "training only"), (C_DIA, "diagnosis (offline head)")]):
    ax.add_patch(FancyBboxPatch((3 + k * 30, 2.2), 2.4, 2.4, boxstyle="round,pad=0,rounding_size=0.4", fc=c, ec="none"))
    ax.text(6.4 + k * 30, 3.4, t, fontsize=8.8, color=INK2, va="center")

fig.savefig(os.path.join(HERE, "architecture.png"), facecolor="white")
fig.savefig(os.path.join(HERE, "architecture.svg"), facecolor="white")
print("saved", os.path.join(HERE, "architecture.png"))

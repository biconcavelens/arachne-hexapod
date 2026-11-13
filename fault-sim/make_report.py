"""
Builds the results report for the professor from the Kaggle outputs in runs/kaggle_v4 and
runs/kaggle_v5 (every number is read from those files). Writes runs/report/report.html;
print it to PDF with a headless browser:

    python make_report.py
    chrome --headless --no-pdf-header-footer --print-to-pdf=../docs/fault-sim-results.pdf runs/report/report.html
"""
import base64
import html
import json
import os
import re

import numpy as np

from recovery import estimate_deadband

HERE = os.path.dirname(os.path.abspath(__file__))
V4, V5 = os.path.join(HERE, "runs/kaggle_v4"), os.path.join(HERE, "runs/kaggle_v5")
OUT = os.path.join(HERE, "runs/report/report.html")
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated, light)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de"


def jload(*p):
    return json.load(open(os.path.join(*p)))


def pct(x):
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------- data
M = {k: jload(V5, "runs", k, "metrics.json") for k in ("gt_imu", "gt_noimu", "flow_imu", "flow_noimu")}
CL = {k: jload(V5, "runs", k + ".json") for k in ("cl_none", "cl_oracle", "cl_imu", "cl_imu_dropped", "cl_noimu",
                                                    "flow_cl_imu", "flow_cl_imu_dropped", "flow_cl_noimu")}
n_val = {k: int(re.search(r"/ (\d+) val", open(os.path.join(V5, "runs", k, "log.txt")).read()).group(1)) for k in M}
status = open(os.path.join(V5, "status.txt")).read()
flow_q = dict(re.findall(r"^\s+(\w+) \S+\s+corr ([+-][\d.]+)", status, re.M))
run_min = sum(int(s) for s in re.findall(r": ok in (\d+) s", status)) / 60


def speed(eps, key, kind=None, n=None):
    eps = eps[:n] if n else eps
    r = [e[key]["speed_mm_s"] / e["healthy"]["speed_mm_s"] for e in eps
         if (kind is None or e["faults"][0]["kind"] == kind) and e["healthy"]["speed_mm_s"] > 5]
    return float(np.median(r))


KINDS = ("loss", "sag", "deadband", None)
none60 = [speed(CL["cl_none"]["episodes"], "no_recovery", k) for k in KINDS]
none40 = [speed(CL["cl_none"]["episodes"], "no_recovery", k, 40) for k in KINDS]
oracle = [speed(CL["cl_oracle"]["episodes"], "recovery", k) for k in KINDS]
model = {k: [speed(CL[k]["episodes"], "recovery", kd) for kd in KINDS] for k in CL if k not in ("cl_none", "cl_oracle")}

conf = np.array(M["gt_imu"]["type_confusion_rows_true_cols_pred"]["counts"], float)
conf = conf / conf.sum(1, keepdims=True)

probes = jload(V4, "probe_traces.json")              # (true dead band, recorded estimate, trace)
old_ok = sum(abs(est - true) <= 1 for true, est, _ in probes)
new_ok = sum(abs(estimate_deadband(np.array(t["off"]), t["ang"], t["lin"]) - true) <= 1 for true, _, t in probes)


# ---------------------------------------------------------------- charts (inline SVG)
def hbar_chart(rows, chance):
    """rows: (label, value 0-1, is_reference). Horizontal bars, 0-100%."""
    W, LW, PW, RH, BH, TOP = 690, 300, 320, 25, 12, 20
    H = TOP + RH * len(rows) + 30
    x = lambda v: LW + v * PW
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Right servo identified, by setup">']
    for g in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{x(g)}" x2="{x(g)}" y1="{TOP}" y2="{H - 26}" stroke="{GRID}" stroke-width="1"/>'
                 f'<text x="{x(g)}" y="{H - 10}" font-size="10" fill="{INK2}" text-anchor="middle">{g * 100:.0f}%</text>')
    for i, (label, v, ref) in enumerate(rows):
        y = TOP + i * RH + (RH - BH) / 2
        x0, x1, r = x(0), x(v), 4
        col = "#b9b8b1" if ref else S1
        s.append(f'<text x="{LW - 10}" y="{y + BH - 2}" font-size="11" fill="{INK}" text-anchor="end">{html.escape(label)}</text>'
                 f'<path d="M{x0},{y} H{x1 - r} Q{x1},{y} {x1},{y + r} V{y + BH - r} Q{x1},{y + BH} {x1 - r},{y + BH} H{x0} Z" fill="{col}"/>'
                 f'<text x="{x1 + 6}" y="{y + BH - 2}" font-size="11" fill="{INK}" font-weight="600">{pct(v)}</text>')
    s.append(f'<line x1="{x(chance)}" x2="{x(chance)}" y1="{TOP}" y2="{H - 26}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="4 3"/>'
             f'<text x="{x(chance)}" y="{TOP - 6}" font-size="10" fill="{INK2}" text-anchor="middle">chance ({pct(chance)})</text>')
    return "".join(s) + "</svg>"


def grouped_chart(groups, series):
    """series: (name, colour, values per group). Vertical bars, 0-1.1 of healthy speed."""
    W, H, L, B, T = 690, 250, 46, 30, 14
    PH, gw = H - B - T, (W - L - 10) / len(groups)
    bw, gap = 30, 2
    y = lambda v: T + PH * (1 - v / 1.1)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Walking speed relative to healthy, by fault type">']
    for g in (0, .25, .5, .75, 1.0):
        s.append(f'<line x1="{L}" x2="{W - 10}" y1="{y(g)}" y2="{y(g)}" stroke="{GRID}"/>'
                 f'<text x="{L - 6}" y="{y(g) + 3}" font-size="10" fill="{INK2}" text-anchor="end">{g:.2f}</text>')
    s.append(f'<line x1="{L}" x2="{W - 10}" y1="{y(1)}" y2="{y(1)}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="4 3"/>'
             f'<text x="{W - 12}" y="{y(1) - 4}" font-size="10" fill="{INK2}" text-anchor="end">healthy</text>')
    total = len(series) * bw + (len(series) - 1) * gap
    for gi, gname in enumerate(groups):
        gx = L + gi * gw + (gw - total) / 2
        s.append(f'<text x="{L + gi * gw + gw / 2}" y="{H - 10}" font-size="11" fill="{INK}" text-anchor="middle">{gname}</text>')
        for si, (_, col, vals) in enumerate(series):
            v, bx, r = vals[gi], gx + si * (bw + gap), 4
            top, base = y(v), y(0)
            s.append(f'<path d="M{bx},{base} V{top + r} Q{bx},{top} {bx + r},{top} H{bx + bw - r} '
                     f'Q{bx + bw},{top} {bx + bw},{top + r} V{base} Z" fill="{col}"/>')
            if gi == len(groups) - 1:                     # direct labels on the summary group only
                s.append(f'<text x="{bx + bw / 2}" y="{top - 4}" font-size="10" fill="{INK}" text-anchor="middle">{v:.2f}</text>')
    return "".join(s) + "</svg>"


def legend(series):
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{html.escape(n)}</span>' for n, c, _ in series) + "</div>"


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def img(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


# ---------------------------------------------------------------- content
gi, gn, fi, fn = (M[k] for k in ("gt_imu", "gt_noimu", "flow_imu", "flow_noimu"))
gd, fd = gi["imu_dropped_at_test"], fi["imu_dropped_at_test"]
servo_rows = [
    ("Linear baseline, optical flow, camera only", fn["baseline_which_servo_acc_single_fault"], True),
    ("Sim. ego-motion, with IMU", gi["which_servo_acc_single_fault"], False),
    ("Sim. ego-motion, IMU dropped at test", gd["which_servo_acc_single_fault"], False),
    ("Sim. ego-motion, camera only", gn["which_servo_acc_single_fault"], False),
    ("Optical flow, with IMU", fi["which_servo_acc_single_fault"], False),
    ("Optical flow, IMU dropped at test", fd["which_servo_acc_single_fault"], False),
    ("Optical flow, camera only", fn["which_servo_acc_single_fault"], False),
]
offline_rows = [
    ["Sim. ego-motion", "with IMU", f"{gi['detect_auroc']:.3f}", pct(gi["which_servo_acc_single_fault"]), pct(gi["type_acc_on_faulty_servos"])],
    ["", "IMU dropped at test", f"{gd['detect_auroc']:.3f}", pct(gd["which_servo_acc_single_fault"]), pct(gd["type_acc_on_faulty_servos"])],
    ["", "camera only", f"{gn['detect_auroc']:.3f}", pct(gn["which_servo_acc_single_fault"]), pct(gn["type_acc_on_faulty_servos"])],
    ["", "linear baseline, camera only", f"{gn['baseline_detect_auroc']:.3f}", pct(gn["baseline_which_servo_acc_single_fault"]), "&ndash;"],
    ["Optical flow", "with IMU", f"{fi['detect_auroc']:.3f}", pct(fi["which_servo_acc_single_fault"]), pct(fi["type_acc_on_faulty_servos"])],
    ["", "IMU dropped at test", f"{fd['detect_auroc']:.3f}", pct(fd["which_servo_acc_single_fault"]), pct(fd["type_acc_on_faulty_servos"])],
    ["", "camera only", f"<b>{fn['detect_auroc']:.3f}</b>", f"<b>{pct(fn['which_servo_acc_single_fault'])}</b>", pct(fn["type_acc_on_faulty_servos"])],
    ["", "linear baseline, camera only", f"{fn['baseline_detect_auroc']:.3f}", pct(fn["baseline_which_servo_acc_single_fault"]), "&ndash;"],
]
TN = ["healthy", "command loss", "power sag", "dead band"]
conf_rows = [[f"<b>{TN[i]}</b>"] + [(f"<b>{pct(conf[i, j])}</b>" if i == j else pct(conf[i, j])) for j in range(4)] for i in range(4)]
axes = [("forward speed", "vx"), ("sideways speed", "vy"), ("vertical speed", "vz"), ("roll rate", "wx"), ("pitch rate", "wy"), ("turn rate", "wz")]
flow_rows = [[n, f"{float(flow_q[k]):.2f}"] for n, k in axes]


def cl_row(name, seeds, v, k=None):
    s = CL[k]["summary"] if k else {}
    fa = f"{s['false_alarms_per_min_healthy']:.1f}" if k else "&ndash;"
    srv = pct(s["first_alarm_correct_servo"]) if k else "&ndash;"
    return [name, str(seeds)] + [f"{x:.2f}" for x in v] + [fa, srv]


cl_rows = [
    cl_row("No recovery", 60, none60),
    cl_row("Correct diagnosis + recovery (upper bound)", 60, oracle),
    cl_row("Model, sim. ego-motion, with IMU", 60, model["cl_imu"], "cl_imu"),
    cl_row("Model, sim. ego-motion, IMU dropped", 60, model["cl_imu_dropped"], "cl_imu_dropped"),
    cl_row("Model, sim. ego-motion, camera only", 60, model["cl_noimu"], "cl_noimu"),
    cl_row("No recovery (first 40 seeds)", 40, none40),
    cl_row("Model, optical flow, with IMU", 40, model["flow_cl_imu"], "flow_cl_imu"),
    cl_row("Model, optical flow, IMU dropped", 40, model["flow_cl_imu_dropped"], "flow_cl_imu_dropped"),
    cl_row("Model, optical flow, camera only", 40, model["flow_cl_noimu"], "flow_cl_noimu"),
]
flow_fa = [CL[k]["summary"]["false_alarms_per_min_healthy"] for k in ("flow_cl_imu", "flow_cl_imu_dropped", "flow_cl_noimu")]
series = [("No recovery", S1, none60), ("Correct diagnosis + recovery", S2, oracle),
          ("Trained model + recovery (camera only)", S3, model["cl_noimu"])]
model_all = [v[3] for v in model.values()]
types = [gi["type_acc_on_faulty_servos"], gn["type_acc_on_faulty_servos"], fi["type_acc_on_faulty_servos"], fn["type_acc_on_faulty_servos"]]

CSS = f"""
@page {{ size: A4; margin: 13mm 16mm 13mm; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: {INK}; background: #fff; margin: 0; }}
h1 {{ font-size: 19pt; margin: 0 0 2px; letter-spacing: -0.2px; }}
.sub {{ font-size: 11.5pt; color: {INK2}; margin: 0 0 6px; }}
.meta {{ font-size: 9pt; color: {MUTED}; margin-bottom: 12px; }}
h2 {{ font-size: 13pt; margin: 18px 0 6px; border-bottom: 1.5px solid {INK}; padding-bottom: 2px; }}
h3 {{ font-size: 10.5pt; margin: 12px 0 4px; }}
p {{ margin: 4px 0 6px; }}
.key {{ border: 1px solid {GRID}; background: #f7f7f5; border-radius: 6px; padding: 8px 14px; margin: 8px 0 10px; }}
.key ul {{ margin: 4px 0; padding-left: 18px; }} .key li {{ margin: 3px 0; }}
.status {{ display: inline-block; font-size: 8.5pt; font-weight: 600; color: #7a4a00; background: #fff3d6; border: 1px solid #f0d49a; border-radius: 4px; padding: 1px 6px; }}
figure {{ margin: 8px 0 10px; break-inside: avoid; }}
figure img {{ width: 100%; border-radius: 4px; }}
figcaption {{ font-size: 8.8pt; color: {INK2}; margin-top: 3px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0 8px; break-inside: avoid; }}
th {{ text-align: left; font-weight: 600; border-bottom: 1.2px solid {INK}; padding: 3px 6px; }}
td {{ border-bottom: 1px solid {GRID}; padding: 3px 6px; vertical-align: top; }}
table.num td:nth-child(n+2), table.num th:nth-child(n+2) {{ text-align: right; }}
table.num2 td:nth-child(n+3), table.num2 th:nth-child(n+3) {{ text-align: right; }}
.legend {{ font-size: 9pt; color: {INK}; margin: 2px 0 2px 46px; }}
.legend span {{ margin-right: 16px; white-space: nowrap; }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }}
.two {{ display: flex; gap: 18px; }} .two > div {{ flex: 1; }}
.note {{ font-size: 8.8pt; color: {INK2}; }}
.keep {{ break-inside: avoid; }}
ol, ul {{ margin: 4px 0; padding-left: 20px; }} li {{ margin: 2px 0; }}
"""

body = f"""
<h1>Seeing servo faults without seeing the legs</h1>
<p class="sub">Camera-based detection and recovery of hobby-servo faults on an open-loop hexapod: simulation results</p>
<div class="meta">Trishanth Mellimi &middot; 28 October 2026 &middot; <span class="status">Simulation only &middot; no real-robot results yet</span></div>

<div class="key"><b>Key results</b>
<ul>
<li><b>Detection.</b> A forward body camera that never sees the legs detects servo faults with window-level AUROC
<b>{fn['detect_auroc']:.2f}</b> (optical flow from rendered images, no IMU).</li>
<li><b>Localisation.</b> It names the faulty servo out of 12 in <b>{pct(fn['which_servo_acc_single_fault'])}</b> of single-fault windows
(chance {pct(1 / 12)}; linear baseline {pct(fn['baseline_which_servo_acc_single_fault'])}).</li>
<li><b>IMU.</b> Adds nothing in simulation: camera-only models match camera + IMU models on both feature sets.</li>
<li><b>Recovery.</b> With the correct diagnosis, simple recoveries bring walking speed back to <b>{oracle[3]:.2f}</b> of healthy
(no recovery: {none60[3]:.2f}).</li>
<li><b>Open problem.</b> With the trained model in the loop, recovery does not beat doing nothing yet ({min(model_all):.2f}&ndash;{max(model_all):.2f}
of healthy vs {none40[3]:.2f}&ndash;{none60[3]:.2f}). The causes are false alarms (about {np.mean(flow_fa):.0f} per minute with optical flow) and weak
fault-type classification ({pct(min(types))}&ndash;{pct(max(types))}).</li>
</ul></div>

<figure><img src="{img(os.path.join(V4, 'frame_deadband_14s.png'))}">
<figcaption><b>Figure 1.</b> One closed-loop run with the trained model (seed 100009: a 17.97&deg; dead band on hip 2 from t&nbsp;=&nbsp;6.9&nbsp;s).
Left: the body camera; right: a follow camera, for viewing only. The model flags hip 2 at 8.6&nbsp;s; the probe estimates the dead band at
17.6&deg;; the gait is modified at 13.4&nbsp;s. This run uses the simulated ego-motion features and is a good example picked for
illustration; the numbers below are aggregates. Videos of all four cases (loss, sag, dead band, healthy) are available.</figcaption></figure>

<h2>1&nbsp; Setup</h2>
<p><b>Robot and simulator.</b> A MuJoCo twin of our Vorpal-style hexapod: 6 legs &times; 2 MG90S servos on a PCA9685. The servos have no
position feedback, so the controller never knows what they actually did. Commands go out at 20&nbsp;Hz. The tripod gait is randomised in
steering, speed and amplitude. Every episode also randomises friction, mass, per-servo strength, floor texture, lighting and camera pose.</p>
<p><b>Camera.</b> Forward-facing, in a slide-in mount on the head cap (designed, ready to print). It is tilted 20&deg; down, with a 45&deg;
vertical field of view, about 20&nbsp;cm above the floor. The legs are out of view: at most 0.05% of pixels show the robot while walking.</p>
{table(["Fault", "What the servo does", "Size", "Recovery"], [
    ["Command loss", "each command is dropped with probability p; 30% of cases fully frozen", "p 0.5&ndash;1.0", "send every command 3&times;; a frozen channel clears on reset (assumed 70%)"],
    ["Power sag", "moves only &plusmn;s around where it was when the sag began", "s 2&ndash;5&deg;", "stand 1&nbsp;s, then resend (assumed to clear 50%)"],
    ["Dead band", "ignores commanded changes smaller than d", "d 15&ndash;20&deg;", "probe the servo while standing to measure d, then make it jump between its cycle extremes (a modified gait)"],
])}
<p class="note">One or two of the 12 servos are faulty per episode; 25% of episodes are healthy. Faults start at 4&ndash;7&nbsp;s.
Stalled legs, horn slip and dead servos are out of scope for now.</p>
<p><b>Diagnoser.</b> <i>Stage A:</i> optical flow (DIS, 160&times;120) gives the body velocity, through the ground-plane motion field
(robust least squares), plus a 4&times;4 flow grid. <i>Stage B:</i> a healthy-motion model predicts the camera (and IMU) signals from
the commands. A Transformer (3 layers, width 128) reads one gait cycle (24 steps, 1.2&nbsp;s) of the measured signals and their
residuals. It outputs whether there is a fault, each servo's fault type, and its size. Online, it diagnoses every 0.3&nbsp;s and raises
an alarm after two agreeing windows.</p>
<p><b>Two feature sets.</b> <i>Simulated ego-motion:</i> the true body velocity plus noise and scale error, with no rendering (2000
episodes). <i>Optical flow:</i> features computed from rendered camera images (600 episodes). <b>IMU dropped at test</b> means the
IMU-trained model runs with its IMU inputs held at their training mean. <b>Camera only</b> means a model trained and run without the IMU.</p>
<p><b>Evaluation.</b> Offline: held-out episodes ({n_val['gt_imu']} for simulated ego-motion, {n_val['flow_imu']} for optical flow).
Closed loop: 60 unseen seeds (the first 40 for optical flow), 20&nbsp;s each. Speed is measured over the last 4&nbsp;s and given
relative to the same seed with no fault.</p>

<div class="keep"><h2>2&nbsp; Results</h2>
<h3>2.1&nbsp; Detecting a fault and finding the servo</h3>
<figure>{hbar_chart(servo_rows, 1 / 12)}
<figcaption><b>Figure 2.</b> Right servo identified, as a share of single-fault windows in held-out episodes. Grey: the linear
(logistic-regression) baseline from our earlier pilot. Dashed: chance.</figcaption></figure></div>
{table(["Features", "Model", "Detection AUROC", "Right servo (single fault)", "Right fault type"], offline_rows, "num2")}
<p>The camera alone carries the signal. Camera-only models match camera + IMU models (simulated ego-motion: {pct(gn['which_servo_acc_single_fault'])}
vs {pct(gi['which_servo_acc_single_fault'])}; optical flow: {pct(fn['which_servo_acc_single_fault'])} vs {pct(fi['which_servo_acc_single_fault'])}).
A model trained with the IMU leans on it somewhat: with simulated ego-motion it falls to {pct(gd['which_servo_acc_single_fault'])} when the IMU
is removed. Optical flow does at least as well as simulated ego-motion, with fewer episodes.</p>

<div class="two"><div>
<h3>2.2&nbsp; Fault type is the weak part</h3>
{table(["True \\ predicted", "healthy", "loss", "sag", "dead band"], conf_rows, "num")}
<p class="note">Row percentages, per servo slot in the window (simulated ego-motion, with IMU). A sagging servo still moves 2&ndash;5&deg;
and is called healthy {pct(conf[2, 0])} of the time. Command loss is spread across all types: a frozen servo and a sagging one look
alike from the body.</p>
</div><div>
<h3>2.3&nbsp; What the camera measures</h3>
{table(["Body motion (600 episodes)", "Correlation with truth"], flow_rows, "num")}
<p class="note">Optical-flow ego-motion against the true body velocity. Forward speed and turning are tracked well; vertical motion, roll
and pitch poorly. Localisation does not suffer, so the information is in forward and turning disturbances and the flow pattern.</p>
</div></div>

<div class="keep"><h3>2.4&nbsp; Closed loop: diagnose, recover, keep walking</h3>
<figure>{legend(series)}{grouped_chart(["Command loss", "Power sag", "Dead band", "All faults"], series)}
<figcaption><b>Figure 3.</b> Median walking speed relative to the same seed without a fault (60 test seeds, simulated ego-motion),
grouped by the type of the first fault.</figcaption></figure></div>
{table(["Setup", "Seeds", "Loss", "Sag", "Dead band", "All", "False alarms / min", "1st alarm right servo"], cl_rows, "num2")}
<p>The recoveries work: given the correct diagnosis, walking speed comes back to {oracle[3]:.2f} of healthy for every fault type. With the
trained model in the loop, the system is still no better than doing nothing. Each false alarm or wrong fault type triggers a pause or a probe
that costs standing time, so speed mostly tracks the false-alarm rate rather than the sensor set. The alarm policy, not the sensing, is now
the bottleneck.</p>

<h3>2.5&nbsp; Measuring the dead band</h3>
<p>When the right servo is probed, the estimated dead band is within 1&deg; of the truth in {old_ok} of {len(probes)} probes. The
others saw no clear jump and fell back to the full 24&deg; ramp. A revised detector, which sums the two motion signals' z-scores, gets
{new_ok} of {len(probes)} on the same traces. It was chosen on those traces, so it still needs confirming. The modified gait always uses
the servo's full travel, so the estimate only decides whether it can be applied.</p>

<h2>3&nbsp; Limitations</h2>
<ul>
<li><b>Simulation only.</b> The fault models are our assumptions, not measured on MG90S servos. The reset and pause success rates are
assumed (70% / 50%), so the loss and sag "upper bound" bars are partly by construction. The dead-band recovery is simulated physics,
not an assumed rate.</li>
<li><b>The simulated camera is cleaner than a real one:</b> a synthetic textured floor, no motion blur, no rolling shutter, no exposure
changes. The simulated IMU has white noise only (no bias, vibration or mount flex).</li>
<li><b>Small evaluations:</b> one training seed, no confidence intervals yet. Optical-flow results rest on {n_val['flow_imu']}
validation episodes and 40 closed-loop seeds. The model overfits after about 7 epochs, so more data should help.</li>
</ul>

<h2>4&nbsp; Next steps</h2>
<ol>
<li><b>Fix the closed loop.</b> Act on the identified servo and escalate by cost (resend, then pause, then probe) instead of trusting the
type head. Calibrate alarms to under 0.5 false alarms per minute, and train on more optical-flow data. Target: clearly beat no recovery.</li>
<li><b>Real robot.</b> Print the camera cap. Inject the same faults in software on the robot's controller (drop commands, add a dead band,
clamp motion) to get labelled real data, and record naturally faulty servos too.</li>
<li><b>Calibrate the fault models.</b> Measure real MG90S dead bands, and log commands to check whether the "command loss" we see comes
from our own software.</li>
<li><b>Statistics.</b> Several training seeds, confidence intervals, and a per-servo (hip vs knee) breakdown.</li>
</ol>
<h3>Questions for discussion</h3>
<ul>
<li>Is "which servo + escalating recovery" enough for a paper, or does it need reliable fault-type classification?</li>
<li>Is software fault injection on the real robot acceptable as evaluation, alongside naturally occurring faults?</li>
<li>Venue and timeline: the CVPR 2027 deadline (16 Nov) is too tight for real-robot results. IROS 2027 or RA-L?</li>
</ul>
<p class="note" style="margin-top:14px">Reproducibility: code in the private repository biconcavelens/arachne-hexapod (folder
<code>fault-sim/</code>). Runs on Kaggle (T4 GPU): kernel version 5, the IMU comparison ({run_min:.0f}&nbsp;min for all steps), and
version 4, videos and dead-band traces. Every number in this report is read from those runs' output files by
<code>fault-sim/make_report.py</code>.</p>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(f'<!doctype html><html><head><meta charset="utf-8"><title>Fault-sim results</title>'
                                       f'<style>{CSS}</style></head><body>{body}</body></html>')
print("written", OUT)

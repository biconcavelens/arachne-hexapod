"""
Report for the professor on the environment stages (1-7) against the plane. Every number is read from the
Kaggle outputs: runs/kaggle_v5 (plane), v9 (stage 1), v10 (stage 2), v13 (stages 3, 4, 5, 7), v14 (stage 6).
Writes runs/report/stages.html; print it to PDF as in make_report.py.
"""
import html
import io
import json
import os
import re

import numpy as np
from PIL import Image

import make_report as R

RUNS = os.path.join(R.HERE, "runs")
OUT = os.path.join(RUNS, "report", "stages.html")
KINDS = ("loss", "sag", "deadband", None)

# stage: (run folder, file prefix, label of its block in status.txt; v13 ran stage 7 as "stage 6" before the split)
SRC = {1: ("kaggle_v9", "s1", "stage 1"), 2: ("kaggle_v10", "s2", "stage 2"), 3: ("kaggle_v13", "s3", "stage 3"),
       4: ("kaggle_v13", "s4", "stage 4"), 5: ("kaggle_v13", "s5", "stage 5"), 6: ("kaggle_v14", "s6", "stage 6"),
       7: ("kaggle_v13", "s7", "stage 6")}
NAMES = {0: "plane", 1: "lit room", 2: "room + objects", 3: "dense clutter", 4: "clutter + moving objects",
         5: "outdoor hills", 6: "stones up to half a step", 7: "stones as tall as a step"}


def flow_corr(status, label=None):
    """Correlations (vx vy vz wx wy wz) of flow ego-motion with the truth, from a status.txt."""
    if label:
        status = re.search(rf"flow quality {label} over .*?\n((?:\s+\w+ .*\n){{6}})", status).group(1)
    return [float(x) for x in re.findall(r"corr ([+-][\d.]+)", status)[:6]]


def stage(n):
    if n == 0:
        d = os.path.join(RUNS, "kaggle_v5")
        cam, imu, cl, none = R.M["flow_noimu"], R.M["flow_imu"], R.CL["flow_cl_noimu"], R.CL["cl_none"]["episodes"][:40]
        transfer, flow = None, flow_corr(R.status)
    else:
        folder, p, label = SRC[n]
        d = os.path.join(RUNS, folder)
        j = lambda *f: R.jload(d, "runs", *f)
        cam, imu, transfer = j(f"{p}_cam", "metrics.json"), j(f"{p}_imu", "metrics.json"), j(f"{p}_plane_model", "metrics.json")
        cl, none = j(f"{p}_cl_cam.json"), j(f"{p}_cl_none.json")["episodes"]
        flow = flow_corr(open(os.path.join(d, "status.txt")).read(), label)
    healthy = np.array([e["healthy"]["speed_mm_s"] for e in none])
    return dict(cam=cam, imu=imu, transfer=transfer, cl=cl, flow=flow, healthy=float(np.median(healthy)),
                stalled=int((healthy < 5).sum()), n_cl=len(none),
                none=[R.speed(none, "no_recovery", k) for k in KINDS],
                model=[R.speed(cl["episodes"], "recovery", k) for k in KINDS])


S = {n: stage(n) for n in range(8)}
pct, f3 = R.pct, (lambda x: f"{x:.3f}")
servo = lambda n, k="cam": S[n][k]["which_servo_acc_single_fault"]


# ---------------------------------------------------------------- figures
def line_chart(series, ymax, ticks, fmt, label, chance=None, H=210):
    """series: (name, colour, values over stages 0-7, None = no point). One axis, 2px lines, 8px markers,
    series named at the line ends, first and last values labelled."""
    W, L, RT, T, B = 690, 44, 150, 14, 34
    xs = [L + 10 + i * (W - L - RT - 20) / 7 for i in range(8)]
    y = lambda v: T + (H - T - B) * (1 - v / ymax)
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(label)}">']
    for g in ticks:
        s.append(f'<line x1="{L}" x2="{W - RT + 10}" y1="{y(g)}" y2="{y(g)}" stroke="{R.GRID}"/>'
                 f'<text x="{L - 6}" y="{y(g) + 3}" font-size="10" fill="{R.INK2}" text-anchor="end">{fmt(g)}</text>')
    for i, x in enumerate(xs):
        s.append(f'<text x="{x}" y="{H - 18}" font-size="10" fill="{R.INK}" text-anchor="middle">{i}</text>')
    s.append(f'<text x="{(xs[0] + xs[-1]) / 2}" y="{H - 4}" font-size="10" fill="{R.INK2}" text-anchor="middle">stage</text>')
    if chance is not None:
        s.append(f'<line x1="{L}" x2="{W - RT + 10}" y1="{y(chance)}" y2="{y(chance)}" stroke="{R.INK2}" stroke-width="1.5" stroke-dasharray="4 3"/>'
                 f'<text x="{W - RT + 16}" y="{y(chance) + 3}" font-size="10" fill="{R.INK2}">chance ({fmt(chance)})</text>')
    for name, col, vals in series:
        pts = [(xs[i], y(v), v) for i, v in enumerate(vals) if v is not None]
        s.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, _ in pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
        for a, b, _ in pts:
            s.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{col}" stroke="#fff" stroke-width="2"/>')
        for a, b, v in (pts[0], pts[-1]):
            s.append(f'<text x="{a:.1f}" y="{b - 9:.1f}" font-size="10" fill="{R.INK}" font-weight="600" text-anchor="middle">{fmt(v)}</text>')
        if len(series) > 1:
            s.append(f'<text x="{pts[-1][0] + 12:.1f}" y="{pts[-1][1] + 4:.1f}" font-size="10" fill="{R.INK}">{html.escape(name)}</text>')
    return "".join(s) + "</svg>"


def jpeg(im, w):
    im = im.convert("RGB").resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + R.base64.b64encode(buf.getvalue()).decode()


def tile(n):
    """Body camera | camera behind the robot, from the stage's preview sheet (row = the scene picked)."""
    if n == 0:                                        # the plane: a closed-loop frame (body camera | follow camera)
        return jpeg(Image.open(os.path.join(RUNS, "kaggle_v4", "frame_deadband_14s.png")).crop((0, 0, 960, 360)), 640)
    row = {1: 1, 2: 0, 3: 1, 4: 3, 5: 2, 6: 0, 7: 0}[n]
    sheet = {1: "kaggle_v9", 2: "kaggle_v10", 3: "kaggle_v13", 4: "kaggle_v13", 5: "kaggle_v13", 6: "kaggle_v14", 7: "kaggle_v13"}[n]
    im = Image.open(os.path.join(RUNS, sheet, f"camera_view_stage{n}.png"))
    return jpeg(im.crop((0, row * 360, 960, row * 360 + 360)), 640)


gallery = "".join(f'<div class="tile"><img src="{tile(n)}"><div><b>{n}</b>&nbsp; {NAMES[n]}</div></div>' for n in range(8))
servo_series = [("trained on that stage", R.S1, [servo(n) for n in range(8)]),
                ("trained on the plane only", R.S2, [servo(0)] + [S[n]["transfer"]["which_servo_acc_single_fault"] for n in range(1, 8)])]
fa_series = [("false alarms", R.S1, [S[n]["cl"]["summary"]["false_alarms_per_min_healthy"] for n in range(8)])]

# ---------------------------------------------------------------- tables
stage_rows = [
    ["0", "plane", "the textured plane of the October report (floor texture and light randomised)", "&ndash;"],
    ["1", "lit room", "walls with skirting, a door, a lit window; wood, tile, carpet or concrete floor; 1&ndash;4 ceiling spotlights with shadows", "no"],
    ["2", "room + objects", "about 10 static objects: boxes, cabinets, bins, balls, tables, chairs, rugs", "no"],
    ["3", "dense clutter", "about twice the objects, plus shelves, sofas, plants, lamps, shoes, papers, pictures; dim to bright, tinted light", "no"],
    ["4", "+ moving objects", "a robot vacuum, rolling balls, a sliding box, a person's legs walking past (visual only)", "no"],
    ["5", "outdoor hills", "grass, dirt or gravel on 5&ndash;9&deg; hills; sky, sun with shadows, rocks, bushes, trees", "yes"],
    ["6", "rough ground", "stage 5 plus bumps of 2&ndash;5&nbsp;mm and stones of 4&ndash;12&nbsp;mm (about half the foot lift of 17&ndash;23&nbsp;mm)", "yes"],
    ["7", "rougher", "7&ndash;11&deg; hills, bumps of 5&ndash;10&nbsp;mm and stones of 8&ndash;25&nbsp;mm (as tall as the foot lift)", "yes"],
]


def off_row(n):
    c, i, t = S[n]["cam"], S[n]["imu"], S[n]["transfer"]
    tr = f"{f3(t['detect_auroc'])} / {pct(t['which_servo_acc_single_fault'])}" if t else "(trained here)"
    return [f"<b>{n}</b> {NAMES[n]}", f3(c["detect_auroc"]), pct(c["which_servo_acc_single_fault"]), pct(c["type_acc_on_faulty_servos"]),
            pct(i["which_servo_acc_single_fault"]), tr]


def cl_row(n):
    s = S[n]
    fa = f"{s['cl']['summary']['false_alarms_per_min_healthy']:.1f}"
    first = pct(s["cl"]["summary"]["first_alarm_correct_servo"])
    if n == 7:
        return [f"<b>7</b> {NAMES[7]}", "not scored: the healthy robot stalls", "", fa, first]
    return [f"<b>{n}</b> {NAMES[n]}", " / ".join(f"{x:.2f}" for x in s["none"]), " / ".join(f"{x:.2f}" for x in s["model"]), fa, first]


walk_rows = [[f"<b>{n}</b> {NAMES[n]}", f"{S[n]['healthy']:.0f}", f"{S[n]['stalled']} of {S[n]['n_cl']}"]
             + [f"{S[n]['flow'][k]:.2f}" for k in (0, 5, 2, 4)] for n in range(8)]

room = [servo(n) for n in range(1, 5)]
room_tr = [S[n]["transfer"]["which_servo_acc_single_fault"] for n in range(1, 5)]
n_val = int(re.search(r"/ (\d+) val", open(os.path.join(RUNS, "kaggle_v13", "runs", "s3_cam", "log.txt")).read()).group(1))
fa = lambda n: S[n]["cl"]["summary"]["false_alarms_per_min_healthy"]

CSS = R.CSS + """
.gallery { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; width: 86%; margin: 6px auto 4px; }
.tile img { width: 100%; border-radius: 3px; display: block; }
.tile div { font-size: 8.8pt; margin-top: 1px; }
"""

body = f"""
<h1>Seeing servo faults without seeing the legs</h1>
<p class="sub">Part 2: more realistic environments, stage by stage (simulation)</p>
<div class="meta">Trishanth Mellimi &middot; 14 November 2026 &middot; <span class="status">Simulation only &middot; no real-robot results yet</span></div>

<div class="key"><b>Key results</b>
<ul>
<li><b>Rooms, clutter and moving objects do not make diagnosis harder.</b> In stages 1&ndash;4 the camera still detects faults
(AUROC {min(S[n]['cam']['detect_auroc'] for n in range(1, 5)):.2f} in all four)
and names the faulty servo in {pct(min(room))}&ndash;{pct(max(room))} of cases, as on the plane ({pct(servo(0))}). The model trained on the
plane works there unchanged ({pct(min(room_tr))}&ndash;{pct(max(room_tr))}).</li>
<li><b>Moving objects raise false alarms</b> ({fa(3):.1f} to {fa(4):.1f} per minute, stage 3 to stage 4).</li>
<li><b>Uneven ground is the hard part.</b> On hills the right servo drops to {pct(servo(5))}, and with stones to {pct(servo(6))}. The
plane-trained model falls to {pct(S[5]['transfer']['which_servo_acc_single_fault'])} and {pct(S[6]['transfer']['which_servo_acc_single_fault'])}.
The camera still measures the body's motion as well as on the plane, so the likely cause is the walking itself: slopes and stones move
the body in ways that resemble faults (not yet tested).</li>
<li><b>On stage 7 the gait is the limit, not the diagnoser.</b> With stones as tall as its step, even the healthy robot stalls (median
{S[7]['healthy']:.0f}&nbsp;mm/s, against {S[5]['healthy']:.0f} on hills).</li>
<li><b>Closed loop: unchanged from the October report.</b> Recovery with the trained model is still slower than doing nothing, in every
environment. False alarms are the cause, and they rise from {fa(0):.1f} per minute on the plane to {fa(6):.1f} on rough ground.</li>
</ul></div>

<figure><div class="gallery">{gallery}</div>
<figcaption><b>Figure 1.</b> One random scene per stage. In each pair, left: the robot's body camera (the input; the legs are out of
view); right: a camera behind the robot, for viewing only. Every episode re-draws the scene: room size, materials, lights and objects,
or terrain, ground, sun and vegetation. Stage 0 shows a frame from the October report.</figcaption></figure>

<div class="keep"><h2>1&nbsp; The stages</h2>
{R.table(["Stage", "", "What it adds", "Walking changes?"], stage_rows)}</div>
<p>Up to stage 4 the floor is flat and nothing solid comes within reach of the robot. The robot walks exactly as on the plane: on the
40 test seeds, stages 0&ndash;4 give identical walking (checked). Only the camera image changes, so any difference in diagnosis comes
from vision. From stage 5 the ground changes the walking too. Textures are procedural (seamless noise), not photographs, and MuJoCo has
no global illumination.</p>
<p><b>Protocol</b>, the same for every stage and as in the October report. Features are optical flow from rendered images.
Each stage has 600 episodes on the same seeds, with {n_val} held out. Per stage we train a camera-only model and a camera + IMU model,
and also test the plane-trained model without retraining. The closed loop runs 40 unseen seeds for 20&nbsp;s each. Speed is relative to
the same seed walking healthy on the same ground.</p>

<div class="keep"><h2>2&nbsp; Results</h2>
<h3>2.1&nbsp; Finding the faulty servo</h3>
<figure>{R.legend(servo_series)}{line_chart(servo_series, 1.0, (0, .25, .5, .75, 1.0), pct, "Right servo identified, by stage", chance=1 / 12)}
<figcaption><b>Figure 2.</b> Right servo identified (out of 12), as a share of single-fault windows in held-out episodes, camera only.
Blue: a model trained on that stage. Orange: the model trained on the plane, tested on each stage without retraining. Dashed: chance.</figcaption></figure></div>
{R.table(["Stage", "Detection AUROC", "Right servo", "Right fault type", "Right servo, with IMU", "Plane-trained model: AUROC / servo"],
         [off_row(n) for n in range(8)], "num")}
<p>Rooms, clutter and moving objects leave detection and localisation where they were on the plane. Differences of about
&plusmn;5 points between these stages are training noise: stages 3 and 4 have identical walking, and their camera-only results differ
by {pct(abs(servo(4) - servo(3)))}. The IMU still adds nothing in simulation. Fault type stays weak everywhere
({pct(min(S[n]['cam']['type_acc_on_faulty_servos'] for n in range(8)))}&ndash;{pct(max(S[n]['cam']['type_acc_on_faulty_servos'] for n in range(8)))}).</p>

<div class="keep"><h3>2.2&nbsp; Why uneven ground is harder</h3>
{R.table(["Stage", "Healthy speed (mm/s, median)", "Healthy runs stalled (&lt; 5 mm/s)", "Flow vs truth: forward", "turn", "vertical", "pitch"],
         walk_rows, "num")}
<p>The optical flow is not the problem: forward and turning speed track the truth as well on hills and stones as on the plane.
Vertical and pitch correlations rise because the body moves more on rough ground. What changes is the walking. Slopes and stones tilt,
slow and jolt the body in the same channels a faulty servo disturbs. A model trained on rough ground partly learns this, and the
plane-trained model does not. On stage 7 the robot lifts its feet 17&ndash;23&nbsp;mm and the stones reach 25&nbsp;mm. The feet catch, and
{S[7]['stalled']} of {S[7]['n_cl']} healthy runs barely move. The robot does not fall. Stage 7 therefore measures the gait, not
fault diagnosis, and stage 6 was added as the walkable step between hills and stage 7.</p></div>

<div class="keep"><h3>2.3&nbsp; Closed loop and false alarms</h3>
<figure>{line_chart(fa_series, 8, (0, 2, 4, 6, 8), lambda v: f"{v:.1f}" if v % 1 else f"{v:.0f}", "False alarms per minute on healthy runs, by stage", H=150)}
<figcaption><b>Figure 3.</b> False alarms per minute on healthy runs (camera-only model, closed loop, 40 seeds).</figcaption></figure></div>
{R.table(["Stage", "No recovery: loss / sag / dead band / all", "Model + recovery: loss / sag / dead band / all", "False alarms / min", "1st alarm right servo"],
         [cl_row(n) for n in range(8)], "num")}
<p>Median walking speed relative to healthy. In no environment does recovery with the trained model beat doing nothing. As on the plane,
each false alarm triggers a pause or a probe that costs walking time. False alarms grow with every step towards realism. They rise
with moving objects and again on uneven ground, where a bump looks like a fault. At 40 seeds the closed-loop speeds swing by about &plusmn;0.1
with no real change: stages 3 and 4 walk identically, yet their model rows differ.</p>

<h2>3&nbsp; Limitations</h2>
<ul>
<li><b>Simulation only.</b> The same fault models and assumed recovery rates as the October report.</li>
<li><b>Rendering.</b> No global illumination, procedural textures, no camera noise, motion blur or rolling shutter. The moving objects are
simple shapes (the person is two swinging legs), and terrain is not combined with moving objects yet.</li>
<li><b>Statistics.</b> One training seed per stage, {n_val} validation episodes, 40 closed-loop seeds, no confidence intervals.</li>
</ul>

<h2>4&nbsp; Next steps</h2>
<ol>
<li><b>Fix the closed loop first.</b> It is now the bottleneck in every environment. Act on the identified servo, escalate the recovery
by cost (resend, then pause, then probe), and calibrate alarms for few false alarms per minute, including on rough ground.</li>
<li><b>Tell a slope from a fault.</b> Give the diagnoser the gravity direction, from the flow or from the IMU. Terrain may be where the IMU
finally matters.</li>
<li><b>Harder environments:</b> terrain with moving objects, camera noise and blur. Stage 7 needs a higher-stepping gait first.</li>
<li><b>Real robot.</b> Print the camera cap and inject the same faults in software, indoors and outdoors.</li>
</ol>
<h3>Questions for discussion</h3>
<ul>
<li>Is stage 6 (rough but walkable ground) far enough for the paper, or should we also make the robot cross stage 7, which needs a new gait?</li>
<li>Fix the closed loop in simulation first, or go to the real robot now that the environments cover rooms, clutter, moving objects and terrain?</li>
</ul>
<p class="note" style="margin-top:14px">Reproducibility: code in the private repository biconcavelens/arachne-hexapod, folder
<code>fault-sim/</code> (environments in <code>environment.py</code>, set with <code>FAULTSIM_STAGE</code>). Runs on Kaggle (T4 GPU): kernel
versions 5 (plane), 9 (stage 1), 10 (stage 2), 13 (stages 3, 4, 5, 7) and 14 (stage 6), about 30 minutes per stage. Every number in this
report is read from those runs' output files by <code>fault-sim/make_stages_report.py</code>.</p>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(f'<!doctype html><html><head><meta charset="utf-8"><title>Environment stages</title>'
                                       f'<style>{CSS}</style></head><body>{body}</body></html>')
print("written", OUT)

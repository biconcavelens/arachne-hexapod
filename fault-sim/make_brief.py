"""
Short brief for the professor: our simulation results plus the closest related papers.
Numbers come from make_report.py (which reads the Kaggle outputs). Writes runs/report/brief.html;
print to PDF as in make_report.py.
"""
import os

import make_report as R

OUT = os.path.join(R.HERE, "runs/report/brief.html")
fn = R.fn


def a(url, text):
    return f'<a href="{url}">{text}</a>'


papers = [
    (a("https://arxiv.org/abs/2207.03386", "Hu, Chen, Lipson") + "<br><span>npj Robotics 2025</span>",
     "Front camera + motor commands predict body motion; a mismatch flags damage. One cut leg link, qualitative only, "
     "about 30 min of retraining to recover, about 5&nbsp;Hz.",
     "<b>Closest work.</b> They notice <i>that</i> something changed; we name <i>which</i> of 12 servos and how, with numbers, "
     "and recover in seconds. Their method is our baseline (code public)."),
    (a("https://arxiv.org/abs/2107.14206", "Thoduka, Gall, Pl&ouml;ger") + "<br><span>IROS 2021</span>",
     "Predicted vs. observed optical flow as an anomaly score for task failures (known kinematics). AUROC 0.80.",
     "Same residual idea as ours, so the mechanism is not new; we apply it to per-servo actuator faults with a camera that cannot see the body."),
    (a("https://arxiv.org/abs/2510.23121", "Santhanam et al.") + "<br><span>RA-L 2025</span>",
     "Anomaly detector plus three escalating recovery levels (pause, perturb, reset) on robot arms.",
     "Precedent for the escalating recovery our results call for (resend &rarr; pause &rarr; probe)."),
    (a("https://arxiv.org/abs/1407.3501", "Cully et al.") + "<br><span>Nature 2015</span>",
     "Hexapod recovers from damage by trial and error, &ldquo;without requiring self-diagnosis&rdquo;.",
     "The comparison to beat: diagnosis must do better than blind adaptation."),
    (a("https://arxiv.org/abs/2604.03397", "Stewart-Height et al.") + " 2026, " + a("https://arxiv.org/abs/2502.05817", "DreamFLEX")
     + "<br><span>arXiv 2026, ICRA 2025</span>",
     "Find the faulty limb or joint from the robot's own joint sensors, then change the gait.",
     "Need joint sensing that MG90S servos lack; an accuracy ceiling, not a competitor."),
]
also = [
    (a("https://arxiv.org/abs/1812.00725", "CRAVES") + " (CVPR 2019), " + a("https://arxiv.org/abs/2104.09359", "Labb&eacute; et al.") + " (CVPR 2021)",
     "Joint angles of a robot from a camera that sees it. Ours never sees the legs."),
    (a("https://arxiv.org/abs/2212.04636", "EgoEgo") + " (CVPR 2023), Yuan &amp; Kitani (ECCV 2018)",
     "Inferring an unseen human body from a head camera: the same structure, shown at CVPR/ECCV, for people."),
    (a("https://arxiv.org/abs/2412.10628", "Qu et al.") + " (arXiv 2024)",
     "SpiderPi hexapod with servos that give no joint feedback; same hardware class, no fault handling."),
    (a("https://arxiv.org/abs/1903.00912", "Zhou et al.") + " (IEEE T-ITS)",
     "Metric speed from the ground plane and known camera height; what our camera stage uses."),
    (a("https://arxiv.org/abs/2606.19067", "Corlito et al.") + " (arXiv 2026)",
     "Global-shutter cameras cut tracking failures on walking robots; guides the real camera choice."),
]

CSS = f"""
@page {{ size: A4; margin: 13mm 15mm; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 9.6pt; line-height: 1.36; color: {R.INK}; margin: 0; }}
h1 {{ font-size: 16pt; margin: 0 0 2px; }}
.meta {{ font-size: 9pt; color: {R.MUTED}; margin-bottom: 10px; }}
.status {{ font-weight: 600; color: #7a4a00; background: #fff3d6; border: 1px solid #f0d49a; border-radius: 4px; padding: 0 5px; }}
h2 {{ font-size: 11.5pt; margin: 12px 0 5px; border-bottom: 1.4px solid {R.INK}; padding-bottom: 2px; }}
.stats {{ display: flex; gap: 8px; margin: 6px 0 6px; }}
.stat {{ flex: 1; border: 1px solid {R.GRID}; border-radius: 6px; padding: 6px 8px; background: #f7f7f5; }}
.stat b {{ display: block; font-size: 16pt; line-height: 1.1; }}
.stat span {{ font-size: 8.5pt; color: {R.INK2}; }}
.stat.bad b {{ color: #a33a12; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.9pt; }}
th {{ text-align: left; border-bottom: 1.2px solid {R.INK}; padding: 3px 5px; }}
td {{ border-bottom: 1px solid {R.GRID}; padding: 4px 5px; vertical-align: top; }}
td span {{ color: {R.MUTED}; font-size: 8.3pt; }}
td:first-child {{ width: 21%; }} td:nth-child(2) {{ width: 38%; }}
a {{ color: {R.S1}; text-decoration: none; }}
.box {{ border-left: 3px solid {R.S1}; background: #f3f7fc; padding: 6px 10px; margin: 10px 0 0; }}
p {{ margin: 4px 0; }}
.note {{ font-size: 8.5pt; color: {R.INK2}; }}
"""

body = f"""
<h1>Seeing servo faults without seeing the legs</h1>
<div class="meta">Trishanth Mellimi &middot; 3 November 2026 &middot; <span class="status">simulation only, no real-robot results yet</span></div>

<h2>Our results (MuJoCo twin of the hexapod, body camera that cannot see the legs)</h2>
<div class="stats">
<div class="stat"><b>{fn['detect_auroc']:.2f}</b><span>fault detection AUROC, camera only</span></div>
<div class="stat"><b>{R.pct(fn['which_servo_acc_single_fault'])}</b><span>right servo out of 12 (chance {R.pct(1 / 12)}, simple baseline {R.pct(fn['baseline_which_servo_acc_single_fault'])})</span></div>
<div class="stat"><b>&asymp; same</b><span>with or without the IMU</span></div>
</div>
<p>Faults: command loss, power sag (2&ndash;5&deg; of movement left) and dead band (15&ndash;20&deg;), on 1&ndash;2 of 12 servos. The camera finds
<i>which</i> servo reliably; it is weak at <i>which kind</i> of fault ({R.pct(min(R.types))}&ndash;{R.pct(max(R.types))}), and running online it still
raises about 2 false alarms per minute on healthy runs. Next: fewer false alarms, then the real robot.</p>

<h2>Closest related work</h2>
<table><thead><tr><th>Paper</th><th>What they do</th><th>Relevance to us</th></tr></thead><tbody>
{"".join(f"<tr><td>{p}</td><td>{w}</td><td>{r}</td></tr>" for p, w, r in papers)}
</tbody></table>

<h2>Also related</h2>
<table><tbody>
{"".join(f"<tr><td style='width:34%'>{p}</td><td>{r}</td></tr>" for p, r in also)}
</tbody></table>

<div class="box"><b>Where we fit.</b> Others detect faults with joint sensors, recover blindly without a diagnosis, or use a camera that sees
the robot. The closest work only notices that something changed. We name which cheap, sensorless servo failed, using a camera that cannot see the
legs, and aim to fix it in seconds. I found no peer-reviewed study of hobby-servo dead band or dropped commands, so measuring real MG90S servos is a
small contribution in itself.</div>
<p class="note" style="margin-top:8px">Full report: fault-sim-results-2026-10-28.pdf. Literature review: docs/literature-review-camera-fault-detection.md.</p>
"""

open(OUT, "w", encoding="utf-8").write(f'<!doctype html><html><head><meta charset="utf-8"><title>Brief</title>'
                                       f'<style>{CSS}</style></head><body>{body}</body></html>')
print("written", OUT)

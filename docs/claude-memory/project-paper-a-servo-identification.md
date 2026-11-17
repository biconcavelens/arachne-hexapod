---
name: project-paper-a-servo-identification
description: "Arachne's research direction (decided 2026-10-17): Paper A, identify sealed hobby servos from IMU data; VLA work is NOT on Arachne"
metadata:
  node_type: memory
  type: project
  originSessionId: cca4871e-0aac-46b0-8408-35683725756e
  modified: 2026-10-17T08:52:15.757Z
---

The research project on Arachne is **Paper A** ("Walking on $3 servos" / "A $150 robot that models itself"): shake the robot ~60 s with the IMU logged, fit ~6 servo params per joint so the MuJoCo twin reproduces the IMU trace, retrain PPO (MJX) on the fitted twin, then walk with the servo model as "virtual encoders". Full plan, reviewer demands, baselines and status checklist live in `research-questions.md` at the repo root ("The paper to write: Paper A").

**Why:** the VLA idea (idea.md T1, RQ3/RQ4) moves to a different robot with the user's professor, because 2-DOF hobby-servo legs don't suit a VLA. Master's applications are irrelevant now; the goal is a top venue (primary CoRL 2027, stretch RSS 2027, backup ICRA 2028; 2027 dates are estimates, unverified).

**How to apply:**
- Do not propose VLA work on Arachne; do not re-pitch applications.
- Kill test comes first: train a "perfect motor" policy in sim, run it on the real robot, measure the gap. Big gap -> proceed; small -> try faster gaits, then pivot.
- Real robot walking numbers do not exist yet; don't claim any. See [[feedback-respect-hardware-choices]] and [[feedback-robot-safety-defaults]].

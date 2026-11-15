---
name: project-camera-fault-detection-idea
description: "Professor-meeting idea (2026-10-22): body-camera detection and online correction of servo faults on the no-feedback hexapod; literature review doc and pilot done"
metadata:
  node_type: memory
  type: project
  originSessionId: cca4871e-0aac-46b0-8408-35683725756e
  modified: 2026-11-15T14:06:09.200Z
---

The user brought a second research idea from their professor: camera-based (body-mounted, cannot see the legs) detection and correction of servo/locomotion errors on the open-loop hexapod, aimed at CVPR 2027, with CMU MSR / NUS robotics applications as the end goal. They present a literature review to the professor on 2026-10-23.

**Why:** closest prior art is Hu, Chen, Lipson (arXiv 2207.03386, npj Robotics 2025); the open gap is per-servo diagnosis + online correction + numbers on hobby servos. CVPR 2027 paper deadline is 16 Nov 2026 (enrol 10 Nov), too tight for real-robot results; IROS 2027 (1 Mar) is the realistic full-paper target. Simulation pilot: knee faults localise ~90%, hips weak, offsets <= 4 deg invisible. Details in the Docs artifact https://claude.ai/artifact/Ax16AW58LEepcwmj5xCWkj and vorpal-mujoco/HANDOVER.md.

**Status 2026-10-28:** fault-sim (loss / sag / 15-20 deg dead band) trained on Kaggle; numbers in fault-sim/RESULTS.md. Sim-only: camera detects (AUROC 0.98) and localises the servo (~75%), IMU adds nothing, fault type weak (~30-40%), model-driven recovery still no better than none (false alarms). Oracle recovery ~1.0 is partly by construction (recovery odds are config guesses).

**Professor feedback ~2026-11-13:** the plane with no proper lighting is too simple; stay purely in sim for now and build environments in stages: 1 lit room, 2 static objects, 3 more clutter, 4 dynamic objects (camera motion not caused by the robot), then gradually rougher terrain. Implemented as FAULTSIM_STAGE in fault-sim/environment.py.

**Status 2026-11-15:** stages 1-7 done and reported (docs/fault-sim-environments-2026-11-14.pdf, fault-sim/RESULTS.md). Rooms, clutter, movers: diagnosis unchanged (~75% servo). Hills/rough ground: drops to 62-67%, plane model 44-51%. Stage 7 (stones as tall as the 17-23 mm foot lift) stalls the healthy gait. Model-driven recovery still below no recovery everywhere; false alarms 1.8 -> 6/min. Waiting on the professor's answer to two questions: is stage 6 far enough, and fix the closed loop in sim first or go to the real robot? Full state in HANDOVER.md.

**How to apply:** treat as a candidate alongside, or merged with, [[project-paper-a-servo-identification]] until the professor decides. Don't claim real-robot results (none exist). First cheap experiment: log command / HTTP reply / PCA9685 readback to test whether the "slider doesn't move" symptom is our own software dropping commands.

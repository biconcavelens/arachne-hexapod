---
name: project-fault-tolerant-path-following
description: "THE research direction since 2026-11-17: one RL policy that follows a given path while servos fail, diagnoses them and walks around dead ones; merges fault-sim and Paper A"
metadata:
  type: project
---

On 2026-11-17 the user merged the two tracks (camera fault detection, Paper A servo identification) into one paper. The full plan is in `PLAN.md` at the repo root.

Main objective: fault detection and self-correction, not path planning. One RL policy, no retraining, follows a GIVEN path to a goal; mid-run it notices a servo fault, says which servo / what kind / correctable or not, compensates if it can, and if the servo is dead walks with the remaining servos along the same path.

Decisions: MuJoCo/MJX only, no Blender (the user trusts Claude to build terrain and scenes in code). Teacher-student: privileged teacher, student with only IMU + past commands + path, a history encoder and an explicit diagnosis head. The fault-sim results are kept on the side, maybe shown in the paper. The current web walk is the hand tripod gait, not RL; the old PPO policy (flat ground, perfect servos) is only a starting point.

**Why:** the user wants error detection and self-correction to be the headline; separately neither track was strong enough for an A* venue.

**How to apply:** treat [[project-camera-fault-detection-idea]] and [[project-paper-a-servo-identification]] as superseded background. Work through PLAN.md's milestones in order. Still open: the professor has not agreed the merge yet; venue decided after the student policy works. Don't claim real-robot results (none exist).

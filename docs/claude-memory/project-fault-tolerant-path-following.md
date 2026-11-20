---
name: project-fault-tolerant-path-following
description: "THE research direction since 2026-11-17: one RL policy that follows a given path while servos fail, diagnoses them and walks around dead ones; merges fault-sim and Paper A"
metadata:
  node_type: memory
  type: project
  originSessionId: 409adc76-3808-4306-80db-c311cdc51662
  modified: 2026-11-18T15:38:36.706Z
---

On 2026-11-17 the user merged the two tracks (camera fault detection, Paper A servo identification) into one paper. The full plan is in `PLAN.md` at the repo root.

Main objective: fault detection and self-correction, not path planning. One RL policy, no retraining, follows a GIVEN path to a goal; mid-run it notices a servo fault, says which servo / what kind / correctable or not, compensates if it can, and if the servo is dead walks with the remaining servos along the same path.

Decisions: MuJoCo/MJX only, no Blender (the user trusts Claude to build terrain and scenes in code). Teacher-student: privileged teacher, student with only IMU + past commands + path, a history encoder and an explicit diagnosis head. The fault-sim results are kept on the side, maybe shown in the paper. The current web walk is the hand tripod gait, not RL; the old PPO policy (flat ground, perfect servos) is only a starting point.

Changes on 2026-11-18 (user's direction):
- Diagnosis names WHICH servo(s), not the fault kind.
- Fault kinds trained: dead, stuck, signal loss and worn gears; sag only goes in the paper. Faults can be present from the start or begin anytime, 0-3 per episode.
- Path tracking matters most: a slowdown to 50% of the target speed is fine, slower is penalised.
- Body camera is a policy input. Compare IMU only, camera only and both. No sensor noise for now.
- The overhead camera is for path drawing and tracking only, never for inference.
- Environment order: furnished room → room with moving objects → rough terrain → rough terrain with movers (terrain_dyn, added 2026-11-19).
- 2026-11-19: body camera stays 24×16 for the local runs. Results: [[project-stage-results-v2]].
- 2026-11-20: the v2 policies crab-walked with the camera pointing sideways (no heading reward), so everything is rerun as v3 on the lab RTX 6000 Ada: heading reward, 96×64 camera with a CNN, IMU+command diagnosis. See [[project-lab-run-v3]] and `LAB_RUN.md`. Code layout: [[reference-code-map]].
- Run locally or on the lab machine; Kaggle isn't set up on this OS.
- An interactive sim demo exists: `sim_app.py`.

**Why:** the user wants error detection and self-correction to be the headline; separately neither track was strong enough for an A* venue.

**How to apply:** treat [[project-camera-fault-detection-idea]] and [[project-paper-a-servo-identification]] as superseded background. Work through PLAN.md's milestones in order. Still open: the professor has not agreed the merge yet; venue decided after the student policy works. Don't claim real-robot results (none exist).

# Plan: fault-tolerant path following (decided 2026-11-17)

This merges the two earlier tracks (camera fault detection in `fault-sim/`, and Paper A's servo
modelling in `research-questions.md`) into one paper. The fault-sim results stay on the side; they
may appear in the paper as a comparison or a supporting result.

## The question

Can one reinforcement-learned policy on a $150 hexapod with sealed hobby servos, and no joint
sensing, follow a given path to a goal while its servos fail mid-run? Without retraining, it has to:

1. **notice** that something is wrong, while walking;
2. **say what**: which servo, which kind of fault, how bad;
3. **decide** whether the fault can be worked around on that servo (dead band, sag, dropped
   commands) or the servo is lost (dead);
4. **keep going**: compensate on the faulty servo if it can, or walk with the remaining servos
   if it is dead, and still follow the same path to the same goal.

The path is given. This is not a path-planning paper; it is about following the path through faults.

## What is ours (against prior work)

- Damage adaptation in legged RL (RMA 2021, LocoFormer 2025) and damage recovery on a hexapod
  (Cully et al., Nature 2015) rely on joint encoders, measured joint positions, or trial-and-error
  search. Arachne has none: MG90S servos report nothing.
- The diagnosis is explicit and part of the policy (which servo, what kind, can it be fixed), not
  only a hidden latent. Reviewers can check it, and it is what an operator needs.
- Correctable versus lost faults are handled in one policy: compensate where possible, re-gait
  where not.
- Real robot, faults injected in software so they are repeatable, plus natural faults if they occur.

To check before writing: fault-tolerant RL locomotion with leg failures (several papers exist),
classical fault-tolerant hexapod gaits, and Hu, Chen, Lipson (camera, damage on a legged robot).

## Faults (applied between the policy's command and the actuator)

| Fault | What happens | Correctable? |
|---|---|---|
| dropped commands | each command lost with probability p | yes: commands repeat every tick |
| dead band | changes below 15-20 deg ignored | partly: overshoot and come back |
| sag | moves only within a few degrees of where it was | partly |
| dead, frozen | holds its last angle (PCA9685 still sends the old pulse) | no: walk around it |
| dead, limp | no torque (signal or power lost; an MG90S goes limp) | no: walk around it |

All of these can be done in the control layer in JAX (limp = command equals the current angle,
which zeroes the servo's force), so no per-environment model changes are needed. Onset is random
mid-episode; zero, one or two faulty servos per episode. A dead knee leaves its leg either dragging or
lifted, depending on the angle it died at; both have to be trained.

## Method

- **Simulator:** MuJoCo / MJX only (no Blender). Terrain is procedural heightfields and meshes
  generated in code: flat, slopes, bumps, stones (the fault-sim stages are the reference).
- **Servo model:** a realistic MG90S (delay, rate limit, small dead band), randomised; replaced by
  the identified values once the real servos are measured (the old Paper A system identification).
- **Teacher policy** (PPO in MJX): sees privileged state: fault labels, true joint angles, terrain
  around it, true velocity.
- **Student policy**: sees only what the real robot has: IMU (gravity, gyro), its own recent
  commands, and the path relative to its pose. It has a history encoder (adapts without retraining)
  and a **diagnosis head** (per-servo: healthy / degraded / dead, fault kind), trained on the sim
  labels alongside imitating the teacher.
- **Path input:** the next few path points in the body frame, plus cross-track and heading error.
  On the real robot the pose comes from the overhead AprilTag tracker (`tools/track_apriltag.py`);
  on-board odometry or the body camera can replace it later.
- **Camera:** not in the first version. The body-camera flow from fault-sim can be added as a
  student input later.

## Evaluation

- Path: goal reached (%), cross-track error, time to goal relative to the healthy robot.
- Diagnosis: time to detect, right servo (%), right kind (%), correctable-vs-dead (%), false
  alarms per minute.
- Conditions: each fault kind on hips and knees, one and two faults, flat and rough ground.
- Baselines: hand tripod gait with a simple path follower; an RL policy trained without faults;
  an RL policy trained with faults but no diagnosis head (RMA-style); an oracle that is told the fault.
- Ablations: no IMU; no history; no correctable/dead split.

## Milestones

1. **Set-up:** RL environment on this machine (Ubuntu, RTX 4060); reproduce the old flat-ground policy.
2. **Healthy path following** on flat ground, walking in any direction.
3. **Terrain:** slopes, bumps, stones.
4. **Faults:** fault layer, teacher with privileged fault labels.
5. **Student:** history encoder and diagnosis head, no privileged inputs.
6. **Evaluation harness** and baselines.
7. **Real robot:** servo measurements, software fault injection, AprilTag-tracked runs.

Code goes in `vorpal-mujoco/rl/` (extending `hexapod_env.py`), reusing the fault definitions from
`fault-sim/faults.py`.

## Demo (later, for the conference talk; not needed now)

- A top-down camera over the arena, not used by the robot, streams to the laptop over HTTP.
- On the laptop you draw a path over that video; it becomes the path the policy follows (the
  same top-down camera tracks the robot's pose via the AprilTag).
- A second panel lets you kill or degrade any servo on the spot (dead, limp, dead band); the
  robot's diagnosis shows live next to it, and it keeps following the drawn path.

## Open

- Agree the merged direction with the professor (the fault thread was their idea).
- Venue: decided later, once milestone 5 works; real-robot results are needed for an A* venue.

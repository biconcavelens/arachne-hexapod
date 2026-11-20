# Plan: fault-tolerant path following (decided 2026-11-17)

This merges the two earlier tracks (camera fault detection in `fault-sim/`, and Paper A's servo
modelling in `research-questions.md`) into one paper. The fault-sim results stay on the side; they
may appear in the paper as a comparison or a supporting result.

## The question

Can one reinforcement-learned policy on a $150 hexapod with sealed hobby servos, and no joint
sensing, follow a given path to a goal while its servos fail mid-run? Without retraining, it has to:

1. **notice** that something is wrong, while walking (or from the start: a servo may already be bad);
2. **say where**: which servo or servos (since 2026-11-18 not the kind; see Faults);
3. **keep going**: walk with what still works and still follow the same path to the same goal,
   slower if it must, but on the path.

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

Updated 2026-11-18 (the user: drop naming the fault kind; dead and stuck matter most; keep it impressive for RSS):

| Fault | What happens | Trained on |
|---|---|---|
| dead | no torque (signal or power lost; an MG90S goes limp) | yes (35% of faults) |
| stuck | holds the angle it had when it failed (stripped gear, jam) | yes (35%) |
| signal loss | each command dropped with probability p (loose wire, brown-out) | yes (15%) |
| worn gears | command changes below 5-20 deg ignored (dead band) | yes (15%) |
| sag | moves only near where it was | no: described in the paper only |

Dead and stuck are the main case: the robot has to walk with what is left. Signal loss and worn
gears are the partial end of the spectrum, kept because a paper that covers the range from
"degraded" to "gone" is stronger than one about dead servos alone, and because a per-kind
breakdown is a natural validation table. 0-3 faulty servos per episode; each one is present from
the start (30%) or starts at a random time in the walk, so faults are not assumed to appear only
after the robot sets off.

The diagnosis names **which servos** are faulty, not the kind: the policy copes with the effect
whatever the kind, so naming the kind adds nothing to the control (the earlier kind classifier
was also the weakest part, about 40%).

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
- **Camera (from 2026-11-18):** the body camera on the camera cap is a policy input. Three variants
  are trained and compared: IMU only, camera only, both. Real IMUs are noisy, so a camera-only robot
  is worth having. No sensor noise yet; it comes back once the real robot is measured. In MJX the
  camera is ray-cast (24 x 16 grey, textured surfaces) because MJX's renderer cannot run inside the
  vmapped training loop. The policy sees frames from now, 0.05 s and 0.2 s ago, so it can in
  principle use motion. At 24 x 16 most of that motion is under a pixel, though.
  Next phase, on a rented cloud GPU after the four local stages: higher resolution, and an
  optical-flow field as an input (exact flow from the ray hits in sim; dense flow, such as DIS,
  pooled to the same grid on the robot).
- **Overhead camera:** used only to draw the path and to track the robot's pose (on the real robot).
  Never an input for fault detection or control.
- **Reward:** following the drawn path first (episode ends beyond 15 cm off it). Speed is measured
  but a slowdown down to half the target speed is not penalised; slower than that is.

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
3. **Environments, in order (2026-11-18):** (a) a furnished room with household objects, path drawn
   around them; (b) the same room with things that move by themselves (robot vacuum, balls, a person)
   crossing the path; (c) rough ground: an 8 x 8 m heightfield with hills, bumps and stones, plus
   rocks, bushes, logs, trees and crates; (d) the same ground with movers (a dog, a rolling
   football, a hiker, a robot mower) crossing the path (added 2026-11-19).
   Mover rules (2026-11-19): people, dogs, the vacuum and the mower stop while the robot is in their
   way; balls don't. An episode ends only when the robot walks into a mover: within 60 deg of its
   heading and closing the gap at least as fast as the mover. Other touches are counted ("struck")
   and don't end the episode. The first dynamic-room policies (dyn_*) were trained before these
   rules; dyn2_* are retrained under them.
   **v3 (2026-11-20):** the v2 policies walked sideways (no reward for facing the path), so the camera
   saw little of where they went. All 12 are rerun on the lab RTX 6000 Ada with a heading reward, a
   96x64 camera read by a small CNN, and a diagnosis from IMU + commands only (`LAB_RUN.md`).
4. **Faults:** fault layer, teacher with privileged fault labels.
5. **Student:** history encoder and diagnosis head, no privileged inputs.
6. **Evaluation harness** and baselines.
7. **Real robot:** servo measurements, software fault injection, AprilTag-tracked runs.

Code goes in `vorpal-mujoco/rl/` (extending `hexapod_env.py`), reusing the fault definitions from
`fault-sim/faults.py`.

## Demo

Built 2026-11-18 as a simulator: `vorpal-mujoco/rl/sim_app.py` (draw on the overhead view, Go, kill
servos live, record GIFs). The real-robot version below follows the same design.

- A top-down camera over the arena, not used by the robot, streams to the laptop over HTTP.
- On the laptop you draw a path over that video; it becomes the path the policy follows (the
  same top-down camera tracks the robot's pose via the AprilTag).
- A second panel lets you kill or degrade any servo on the spot (dead, limp, dead band); the
  robot's diagnosis shows live next to it, and it keeps following the drawn path.

## Open

- Agree the merged direction with the professor (the fault thread was their idea).
- Venue: decided later, once milestone 5 works; real-robot results are needed for an A* venue.

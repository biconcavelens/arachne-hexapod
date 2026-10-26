# Arachne: research questions (frontier scan, 2026-10-05)

Goal: a paper at a top robotics venue (CoRL / RSS / ICRA) built on Arachne
(Vorpal-derived hexapod: 6 legs x 2 DOF, 12 MG90S hobby servos, Arduino Uno Q,
exact MuJoCo twin).

## Where the frontier is (Oct 2026)

- **Legged VLAs are hierarchical.**
  - NaVILA: the VLA emits mid-level language ("turn right 30 deg") and an RL
    policy executes it. Same VLA on Go2 and a humanoid. [2412.04453]
  - QUAR-VLA / QUART-Online: quadruped VLA that emits 12-D action tokens.
    Trained on 259K sim + 3K real episodes; action-chunk discretization for
    latency. [2312.14457, 2412.15576]
  - **No hexapod VLA found.**
- **Generalist (cross-embodiment) locomotion.**
  - URMA: 16 robots including hexapods in sim; real tests on A1 and two
    Badger quadrupeds. MuJoCo, code open. [2409.06366]
  - URMAv2: 50 robots. [2509.02815]
  - Embodiment Scaling Laws: ~1000 generated robots including hexapods;
    real tests on Go2 and H1. CoRL 2025. [2505.05753]
  - LocoFormer: unseen robots and motor failures. CoRL 2025. [2509.23745]
  - **Hexapods only ever appear in sim, and every real test is on a
    torque-controlled quadruped or humanoid.**
- **Real hexapod RL.**
  - Google DeepMind: $600 SpiderPi, 18 servos, no joint feedback,
    teacher-student, depth + VIO. arXiv only. [2412.10628]
  - Motion-prior RL on a real hexapod over complex terrain. [2511.03167]
  - HeLoM: loco-manipulation on a large hexapod. [2509.23651]
- **Cheap hardware is publishable at the top.**
  - ToddlerBot (CoRL 2025) and Berkeley Humanoid Lite (RSS 2025).
  - In both, the contribution is the platform, a validated twin and
    zero-shot transfer.
- **Cheap-actuator sim-to-real is active.**
  - Mini Pupper 2: >50 ms delay; a delay-aware network learns a CPG.
    [2607.26434]
  - Actuator Reality Shaping: drives the real actuator to match the ideal sim
    model. Needs an actuator you can measure and control. [2607.02205]
  - LiteVLA-Edge: on-device VLA at 6.6 Hz on a Jetson Orin, timing only, no
    task evaluation. [2603.03380]
- **Symmetry and language.**
  - Symmetry: MS-PPO [2512.00727] and symmetry-RL [2403.17320] both use
    mirror symmetry.
  - Language to gait: SayTap uses contact patterns as the interface;
    quadruped, CoRL 2023. [2306.07580]

## Research questions (ranked)

### RQ1 (top pick): Do generalist locomotion policies survive a $150 robot?
- **Question:** can cross-embodiment policies (URMA / embodiment-scaling style)
  control a real 2-DOF-per-leg hexapod with position-only hobby servos? Which
  gap breaks them:
  - morphology (2-DOF hinge legs, chiral layout),
  - actuators (deadband, latency, no joint feedback, unit-to-unit variance),
  - or sensing?
- **Gap:** hexapods are in every generalist training set but have never been
  tested for real. There are no real tests on hobby servos.
- **Why us:**
  - Exact MuJoCo twin; URMA itself runs on MuJoCo.
  - The physics model already walks.
  - Measured servo quirks and the coupled hip-collision limit.
- **Experiment:**
  1. Add Arachne to URMA as a held-out robot and test it zero-shot in sim.
  2. Add a "hobby servo" actuator class to the randomization: 20-80 ms
     latency, deadband, 8 deg spline quantization, position-only
     observations. Retrain and re-test.
  3. On the real robot, measure speed, heading drift, falls and leg contacts
     against our hand-tuned tripod gait.
- **Contribution:** the first real hexapod data point for embodiment scaling,
  plus a cheap-actuator axis. A negative result is still a result.
- **Needs:**
  - An IMU (~$5 MPU6050/BNO085 on the same I2C bus).
  - A phone camera + AprilTag for ground truth.
  - A GPU (training cost unverified).
- **Venue:** IROS 2027, RA-L, ICRA 2028 or CoRL 2027.
- **Risk:** compute. Fallback: per-robot RL plus RQ2.

### RQ2: How much servo model does a $3 servo need? (RQ1 component / fallback)
- **Question:** for closed-box position servos (no current, velocity or
  encoder readout), what is the minimum actuator model for zero-shot
  sim-to-real locomotion? Can it be identified in minutes with only a camera
  and an IMU?
- **Gap:** actuator-gap methods assume actuators you can measure or control
  (Reality Shaping; Dynamixel system ID in ToddlerBot). The Mini Pupper work
  models delay only.
- **Experiment:**
  1. Identify the MG90S: step response from a 240 fps camera, deadband, rate
     limit, sag under load.
  2. Build a ladder of sim actuators: ideal PD, then + delay, then
     + deadband and rate limit, then randomized.
  3. Train the same policy on each rung and measure the real gap per rung.
- **Venue:** RA-L or IROS. Low risk, cheap, feeds RQ1.

### RQ3: Language to contact schedules for statically stable walkers
> Moved off Arachne (2026-10-17): the VLA work goes on a different robot with the user's professor.
- **Question:** hexapods have a far larger statically stable gait space than
  quadrupeds (tripod, tetrapod, wave, 5-leg failure gaits). Can a small
  on-device model map instructions such as "leg 2 is broken", "walk quietly"
  or "step high" to 6-leg contact schedules, run by one contact-conditioned
  policy?
- **Gap:** SayTap is quadruped-only and uses a cloud LLM. No hexapod or
  on-device version exists, and no fault-aware gait selection from language.
- **Why us:**
  - Static stability makes schedules safe to run on cheap hardware.
  - The gait generator and clearance checker already exist.
  - A real fault (the cracked hinge) motivates the fault-aware angle.
- **Venue:** ICRA or IROS.
- **Risk:** reviewers may see it as incremental unless the on-device and
  fault-aware parts are strong.

### RQ4: A tiny VLA on a $60 board
> Moved off Arachne (2026-10-17), same as RQ3.
- **Question:** what is the smallest VLA that closes vision-language-locomotion
  on 4x Cortex-A53 (Uno Q; no NPU listed)? Which action space wins at matched
  compute: joint angles, contact schedules, or NaVILA-style mid-level
  language?
- **Gap:** on-device VLA work stops at Jetson Orin, and legged VLAs run
  off-board on large GPUs.
- **Risk:** highest of the five.
  - Needs data and training compute.
  - It may only reach ~1 Hz on the A53.
  - It is best done after RQ1/RQ3 provide the low-level controller.

### RQ5 (short paper): Chirality and symmetry for pinwheel robots
- **Observation (our physics sim, 2026-10-04):** the legs are identical copies
  rotated around the body, each knee offset 21 mm sideways. That makes the
  robot chiral, and it walks with a ~4 deg/s right-turn bias.
- **Gap:** symmetry-RL uses mirror symmetry, but a chiral hexapod only has
  rotational symmetry.
- **Question:** does rotational-symmetry equivariance improve sample
  efficiency and cancel the chirality-induced bias?
- **Venue:** workshop or IROS.

### Demoted
- **End-to-end VLA emitting 12 joint angles:** reviewers will ask why not a
  hierarchy (NaVILA). It is data-hungry and cannot run on the Uno Q.
- **idea.md T1 (primitive-space VLA):** close to NaVILA / QUART-Online. It is
  only new with the on-device, cheap-hardware angle.

## The paper to write: Paper A (decided 2026-10-06, updated 2026-10-17)

Working title: "Walking on $3 servos" / "A $150 robot that models itself".

### Scope decisions (from the 2026-10-17 brainstorm)
- **Arachne = Paper A only.** The VLA ideas (RQ3, RQ4) are NOT on Arachne.
  The user will pursue them on a different robot with their professor
  (2-DOF legs on hobby servos don't suit a VLA).
- **Goal: a top venue. Master's applications no longer drive the plan.**
  - Realistic: ICRA / IROS / RA-L.
  - CoRL / RSS need an extra: the self-repair loop (damage -> shake ->
    re-identify -> retrain in minutes -> walk again, with a timed recovery)
    or a servo-scaling result.

### The idea
Sealed PWM hobby servos (MG90S) have a hidden internal loop: delay, deadband,
rate limit, sag under load, unit-to-unit variance, voltage dependence. You
cannot read their position, current or gains. Plan:
1. **Shake:** a ~60 s scripted motion with the IMU logged.
2. **Identify:** fit ~6 servo params per joint (+ body params) so the MuJoCo
   twin replaying the same commands reproduces the IMU trace.
3. **Retrain:** PPO in MJX on the fitted twin, randomized around the fit.
4. **Walk:** policy on the Uno Q, with the servo model used as "virtual
   encoders" (it predicts true joint angles from the command history).

**Draft contribution:** sealed servos identified from 60 s of on-body IMU data
with no joint sensing. The model both trains the policy and serves as a
virtual encoder. Result: zero-shot learned walking on 3 body configurations
(6/5/4 legs) x 2 robots, closing X% of the sim-to-real gap and matching an
encoder-based actuator network. (X is not measured yet.)

**How it differs from prior work.** All of these had joint feedback or
controllable actuators; ours has neither:
- Tan et al. RSS 2018 (Minitaur actuator + latency model).
- Hwangbo et al. 2019 actuator net.
- ToddlerBot system ID.
- Mini Pupper latency paper (arXiv 2607.26434).
- Actuator Reality Shaping (2607.02205).

**Literature check still TODO:** hobby-servo legged sim-to-real (Petoi Bittle,
Mini Pupper, OpenCat Gym, workshop papers). Make sure nobody does IMU-only
identification of sealed servos.

### What reviewers will demand
- A second robot NOT built by us (the biggest factor).
- At least 10 real trials per condition, on 2 surfaces.
- The ablation ladder: ideal PD -> + delay -> + deadband/rate limit -> + sag
  -> randomized -> full model.
- Baselines:
  - the hand tripod gait,
  - domain randomization only,
  - an encoder-based actuator net as the upper bound (trained on pot-tapped
    bench servos),
  - a delay-aware policy.
- A servo-swap test.
- Open code + twin + video.
- The learned policy must win on command tracking, heading hold, damaged
  configurations and rough foam, not on top speed.

### Kill test FIRST (verdict in ~2-3 weeks)
1. Train a "perfect motor" policy in sim.
2. Run it on the real robot and measure the gap.
- **Big gap:** proceed.
- **Small gap:** try faster gaits / quick turns. If still small, pivot before
  investing more.

### Hardware: the user's choices (respect them, don't re-argue)
- **IMU:** MPU6050 x2 bought (GY-87 10DOF board from robu.in, or GY-521).
  - Use only the MPU6050. The magnetometer is useless near servos and may be
    a QMC5883L clone; the BMP180 is irrelevant.
  - I2C 0x68, on the same bus as the PCA9685 (0x40). Check WHO_AM_I = 0x68.
  - 2 s still gyro-bias calibration at startup; enable the DLPF.
  - Mount rigidly at the body centre, x toward L0.
  - Yaw drifts (no magnetometer). Fine for 10-30 s runs; the camera gives
    true heading.
- **Power:** 2 x 18650 in series (2S: 8.4 V full -> ~6.4 V empty). The user
  declined a regulator; it works fine.
  - Probably above the MG90S 4.8-6 V rating if fed directly to V+, so servo
    behaviour shifts with charge.
  - Mitigation: log pack voltage every run (resistor divider into a Uno Q
    analog pin, or multimeter at session start/end) and make voltage an input
    to the servo model.
- **Declined:** INA power monitor; ADS1115 (pot taps go to Uno Q analog pins
  through a divider; the STM32 takes max 3.3 V); BNO085 (too expensive);
  spare servos for now (buy later for the swap test / burnouts).
- **Feet:** hot-glue blobs. Keep them consistent; measure friction with a tilt
  test (mu = tan of the slide angle).
- **Second robot:** buy ONLY after the kill test.
  - Criteria: plain PWM servos (no feedback), different legs, commercial,
    accessible IMU, CAD/URDF exists.
  - Candidates to verify: Freenove robot dog/hexapod (PCA9685 + PWM), Petoi
    Bittle X (check servo feedback), SpotMicro (we'd print it, so partly
    "ours").
- **Compute:** RTX 4060 Laptop, 8 GB. MuJoCo Playground / MJX runs in WSL2
  (Ubuntu is installed and sees the GPU as of 2026-10-17).
- **Tracking:** AprilTag (tag36h11, ~10 cm) on top of the robot, phone
  overhead at ~1.5 m.

### Status and next steps (2026-10-17)
- [x] Web update (steer slider: index.html, gait.js, gaits/forward.json) is on
      the board. Pushed 2026-10-05 and seen served; re-verify when the board is
      next connected. "Send to robot" stays OFF by default.
- [ ] User: print the AprilTag + mount, rig the phone, wait for the IMU.
- [x] Tracking script: phone video -> AprilTag -> x, y, heading vs time
      (`tools/track_apriltag.py`, passes a synthetic-video self-test; not yet
      run on real footage).
- [x] Servo bench script: 240 fps video -> horn angle vs time
      (`tools/servo_horn_angle.py`, same: synthetic self-test only; needs a
      coloured dot on the horn). Delay/deadband also need command timestamps
      from the sender side.
- [~] Port hexapod_physics.xml to MJX / Playground and train the
      perfect-motor PPO policy (velocity + heading command tracking).
      Ported (`vorpal-mujoco/rl/`, runs in WSL2 on the GPU). The policy sees
      only what the real robot has (IMU gravity+gyro, command, own last action,
      1 Hz clock), 20 Hz control, hips +-25 deg / knees -55..+10 deg (any
      neighbour combination keeps >= 17 mm). 40M-step run `runs/imu` started
      2026-10-17. Result (59 min): sim-to-sim in the full mesh model it tracks
      forward 60 -> 51 mm/s, 100 -> 74 mm/s (hand tripod gait: 33), sideways
      40 -> 37, still -> 4 mm/s drift, tilt <= 2 deg; weak at turning (a v2 retrain
      with a tighter yaw reward was worse everywhere and was dropped). `run_policy.py` (dry run by default) +
      mock-board test ready. Real-robot numbers: none yet (waiting on the IMU).
- [~] IMU: MPU6050 arrived 2026-10-17. Firmware + endpoints written and
      compile-tested (200 Hz, gravity filter, ring buffer; `tools/imu_log.py`),
      NOT flashed yet (needs the board connected and the IMU wired).
- [ ] Together: real baseline of the hand tripod gait (5-10 filmed runs vs the
      physics sim: ~32 mm/s, untrimmed turns right ~4 deg/s, steer -0.15
      straightened it in sim). First real sim-to-real number.
- [ ] Together: servo bench test -> delay, top speed, deadband.
- [ ] IMU arrives: wire 3.3 V / GND / SDA / SCL to the PCA9685's I2C; stream
      gyro/accel at ~200 Hz over the Bridge with calibration; measure the
      Bridge round-trip latency (part of the delay model).
- [ ] Then the kill test.

### Venues and dates (all 2027 dates estimated from 2026, unverified)
| Venue | Rank | Deadline |
|---|---|---|
| CoRL 2027 (primary) | top-tier, CORE unranked | ~late May 2027 (2026 was 28 May) |
| RSS 2027 (stretch) | CORE A* | ~late Jan 2027 (2026 was 30 Jan); Athens 6-11 Jul |
| IROS 2027 | CORE A | 1 Mar 2027 |
| ICRA 2028 (backup) | CORE A* | ~mid-Sep 2027 |
| RA-L | journal + conf option | rolling |

### Later milestones (from the 2026-10-06 plan; gated on the kill test)
- **Servo identification:** bench ID (240 fps camera + pot tap) and variance
  across units; the actuator ladder; first zero-shot real walk.
- **Embodiments:** 5-leg / 4-leg configurations, then the second robot.
  Generalist (URMA) zero-shot comparison if checkpoints exist (unverified).
- **RSS go / no-go (Jan):** needs the full ladder on 2 embodiments.
- **Feb-Apr:** full real evaluation, ablations, baselines.
- **May:** write, video, submit to CoRL 2027.

## Sources
- NaVILA https://arxiv.org/abs/2412.04453
- QUAR-VLA https://arxiv.org/pdf/2312.14457 · QUART-Online https://arxiv.org/pdf/2412.15576
- URMA https://arxiv.org/html/2409.06366 · code https://github.com/nico-bohlinger/one_policy_to_run_them_all
- URMAv2 https://arxiv.org/abs/2509.02815
- Embodiment Scaling Laws https://proceedings.mlr.press/v305/ai25a.html
- LocoFormer https://proceedings.mlr.press/v305/liu25a.html
- Versatile Locomotion Skills for Hexapod Robots https://arxiv.org/abs/2412.10628
- Natural and Robust Hexapod Locomotion https://arxiv.org/abs/2511.03167
- HeLoM https://arxiv.org/html/2509.23651
- ToddlerBot https://proceedings.mlr.press/v305/shi25a.html · Berkeley Humanoid Lite https://engineering.berkeley.edu/news/2025/11/building-bots-on-a-budget/
- RL on cost-constrained quadruped hardware https://arxiv.org/abs/2607.26434
- Actuator Reality Shaping https://arxiv.org/abs/2607.02205
- LiteVLA-Edge https://arxiv.org/abs/2603.03380
- MS-PPO https://arxiv.org/pdf/2512.00727 · Symmetry RL https://arxiv.org/pdf/2403.17320
- SayTap https://arxiv.org/pdf/2306.07580
- CORE ranks https://portal.core.edu.au/conf-ranks/
- ICRA 2027 CfP https://2027.ieee-icra.org/announcements/call-for-technical-papers/ · IROS 2027 https://mldeadlines.com/conference/iros/
- RSS 2026 dates https://mldeadlines.com/conference/rss-2026/ · CoRL 2026 dates https://mldeadlines.com/conference/corl-2026/
- MuJoCo Playground https://playground.mujoco.org/
- Petoi research use https://www.petoi.com/pages/research-spotlight
- CoRL 2026 workshops https://aiworkshoptracker.com/conference/corl/2026/
- Uno Q specs https://arduino.cc/product-uno-q

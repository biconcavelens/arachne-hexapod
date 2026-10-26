# Vorpal Hexapod — Handover (2026-09-08)

## Project
Vorpal Combat Hexapod (open-source, [thingiverse.com/thing:2513566](https://www.thingiverse.com/thing:2513566)),
12x MG90S servos on a single PCA9685, driven by an Arduino Uno Q. 2 DOF per
leg: hip (coxa, swing) + knee (femur, lift). 6 legs.

## Hardware status
All 12 servos wired, legs/hinges physically assembled per the official
[assembly guide](https://vorpalrobotics.com/wiki/index.php/Vorpal_The_Hexapod_Assembly_Instructions),
fitted into the electronics caddy. Board is reachable over USB via `adb`
(see "Known quirks" below — the connection needs re-establishing often).

## Software built so far
- **`servo-web-control`** — an Uno Q App Lab app (sketch + Python `WebUI`
  brick) serving a web page at `http://localhost:7000` (via
  `adb forward tcp:7000 tcp:7000`) with per-channel sliders, a manual jog
  panel, and a live 2D top-down "digital twin" canvas.
  Local copy: `C:\Users\trish\AppData\Local\Temp\claude\...\scratchpad\servo-web-control\`
  (path is session-scoped scratchpad — ask to re-locate if a new session
  can't find it. On-device: `~/ArduinoApps/servo-web-control/`.)
- **`vorpal-sim/`** (in this project dir) — offline Python interference
  simulator (2D rectangles, shapely) + the calibration script that fit it
  to one real hardware touch point. Superseded in spirit by the MuJoCo
  effort below, but still the source of the validated hip pivot geometry.
- **`vorpal-mujoco/`** (in this project dir) — in-progress MuJoCo model
  using the actual STL meshes. **This is the current task.**

## Key measured facts (validated, reusable)
- Hip pivot layout: radius **68.5mm** from base center, 6-fold symmetric,
  at angles **30°/90°/150°/210°/270°/330°** — measured directly off
  `Hexapod_V1r8d_Base_minibrims.stl`'s radial symmetry (see
  `vorpal-sim/hip_interference_sim.py` docstring for method).
- Leg local pivot point: **(106.75, -0.58, -43.33) mm** in the leg STL's
  own local frame (`Hexapod_V1r8a_Single_Leg_6.stl`) — the flat end-cap
  face at the leg's proximal (hip-side) tip. Found via `find_leg_pivot.py`.
- Channel map: hip = channel N, knee = channel N+6, for leg N (0-5). See
  `C:\projects\arachne\vorpal-servo-wiring.md`.
- **One real hardware touch point**: legs 0/1 collided at servo0=112°
  (delta +22 from neutral), servo1=46° (delta -44) — used to calibrate the
  old 2D model, and to sanity-check the MuJoCo model.

## Status: exact model built and working (see 2026-09-12 section below)
The path this section originally described (manually drag hinges into place
in the MuJoCo viewer, no CAD data available) is **abandoned** — a full pull
of the real OnShape assembly (see 2026-09-12 section) made manual placement
unnecessary. Skip straight to that section; this paragraph is kept only so
old context isn't silently deleted.

**Still genuinely open, carried forward**: re-run a MuJoCo collision sweep
(real mesh-vs-mesh, now with the corrected model) to get trustworthy
hip-to-hip safe ranges, then feed that into a live safety clamp on the real
robot — still not implemented anywhere. This was always the next step and
keeps getting deferred by modeling work.

## Known quirks (unrelated to MuJoCo, but will bite again)
- `arm_autostart.sh` (a `@reboot` cron job on the Uno Q) restarts the old
  "Arm Voice Control" app (`pca9685-i2c-test`) on every power cycle,
  stealing the MCU from `servo-web-control`. Fix each time:
  ```
  arduino-app-cli app stop user:pca9685-i2c-test
  arduino-app-cli app start //home/arduino/ArduinoApps/servo-web-control
  ```
  (`pkill -f arm_autostart.sh` too if it keeps re-stealing it mid-session.)
- `adb forward tcp:7000 tcp:7000` drops often — re-run if `localhost:7000`
  stops responding (check `adb forward --list`).
- `adb` device can show `offline` after physical handling of the board —
  `adb kill-server && adb start-server` fixes it without needing a replug.

## File map (current, post 2026-09-12)
```
C:\projects\arachne\
├── vorpal-servo-wiring.md          channel map, status log, quirks
├── vorpal-sim\                     old 2D rectangle model (superseded)
│   ├── hip_interference_sim.py
│   ├── calibrate_from_real_touch.py
│   └── hip_interference_results.json
├── vorpal-mujoco\                  CURRENT — exact CAD-derived model
│   ├── onshape_assembly.json       ground truth: exact mate transforms pulled
│   │                               from the real OnShape assembly's REST API
│   ├── build_from_onshape.py       reads the json above -> hexapod.xml
│   │                               (modes: "static" verifies frames, "tree"
│   │                               builds the real kinematic tree)
│   ├── hexapod.xml                 generated, do not hand-edit
│   ├── render_views.py             headless multi-angle render + collision check
│   ├── analyze_assembly.py         one-off exploration script (kept for reference)
│   ├── find_leg_pivot.py           one-off exploration script (kept for reference)
│   ├── sanity_extreme.py           one-off exploration script (kept for reference)
│   ├── current_sim.png             latest verified render
│   ├── HANDOVER.md                 this file
│   ├── deprecated\                 superseded scripts, see its README - do not use
│   └── meshes\
│       ├── base.stl                decimated to 100k tri (mujoco's face cap)
│       ├── leg6.stl..leg11.stl     one mesh per leg (each its own local frame)
│       └── hinge.stl               one hinge half, CANONICAL frame (not recentered)
└── Vorpal needed\files\            original print-ready STLs
```

## 2026-09-12 update: real bug found and fixed, ChatGPT detour discarded

**What happened**: handed this off to try in ChatGPT. It discarded the exact
OnShape ground truth entirely and rebuilt the model from numbers guessed off
assembly photos (`HIP_TO_KNEE_MM = 44.0`, `KNEE_STAND_DEG = 15.0`, a
hand-picked bracket quaternion). The result looked more like a generic
hexapod at a glance (nicer camera angle, floor, shadows) but every dimension
in it was invented, not measured. Archived under `deprecated/`, not used.

**The real bug**, found by going back to the exact OnShape data: the
hinge→leg/hip classification (`build_from_onshape.py`) compared hinge
**occurrence origins** against hip/knee joint positions. But `hinge.stl`'s
local origin sits ~190mm from its own mesh centroid, so after rotation the
origin can land 150-300mm from anything physically real, while the actual
mesh (centroid-transformed) sits a sane 22-66mm from its true joint. That's
what caused one leg's hinge to be missing/floating - not a wrong CAD pull,
a wrong choice of "which point represents this part's position."

**Fix**: classify using the transformed mesh **centroid**, and use a proper
capacitated bipartite assignment (scipy `linear_sum_assignment`, each hip
gets exactly 2 hinges) instead of naive per-item nearest-neighbor, which
could let two hinges fight over one leg and orphan another (same class of
bug as the earlier hip/knee-servo assignment bug from the same day).

**Result**: every leg now has a clean, consistent 2-hinge chain (matching
"2 hinge joints then the leg" from the real assembly guide) - see
`current_sim.png`. Contacts at rest dropped from 27 (multiple legs missing/
misassigned hinges) to 2 (a 6.4mm overlap between two adjacent legs'
hinges - plausible as a convex-hull collision margin artifact, not a
structural error, and consistent with what real hardware testing already
showed: adjacent hips do get close).

**One remaining minor item**: Legs-11's distal hinge (H1) is 65.7mm from its
knee joint vs ~25.6mm for every other leg - a real outlier worth a second
look eventually, but not blocking.

Authoritative pipeline going forward: `onshape_assembly.json` ->
`build_from_onshape.py` (mode `tree` or `static`) -> `hexapod.xml`, rendered
via `render_views.py`. Do not resurrect anything in `deprecated/`.

## 2026-10-01 CORRECTION - the "fixed" claim above is WRONG about the pose
The 2026-09-12 fix only made hinge assignment consistent. It did NOT make
the model look like the real robot, because the OnShape assembly is a
**posed snapshot**: each leg was left in a different pose by the author.
Measured per leg (knee radius / foot heading off radial / foot elevation):
  Leg 6: 42mm / 190deg (points INWARD, sits inside hub) / -59
  Leg 7: 111mm / 40 / -51     Leg 8: 113mm / 33 / -31
  Leg 9: 108mm / 27 / -53     Leg 10: 112mm / 27 / -32
  Legs-11: 81mm / -32 / -40
So `hexapod.xml` at qpos=0 is six different poses, NOT a neutral stance, and
knee/hip joints rotate about axes that don't match a neutral leg (see
move_knee.png: flexing knee0 swings the leg through the base).
What IS still valid: part meshes, base/hip geometry, per-part local frames
(joint mate origins verified against STL bounds). What is NOT valid: the
assumption that OnShape transforms = neutral pose, and the hinge-pair
kinematic structure (knee radius varies 42-113mm across legs, so the hinge
pair is NOT a rigid link - it articulates).
Next step: define ONE leg's neutral chain by hand, replicate 6x by 60deg.

## 2026-10-04: neutral model built from the real robot  <- CURRENT
`build_neutral.py` -> `hexapod_neutral.xml` is now THE model (supersedes
`hexapod.xml`, which is the posed CAD tree and only useful as a reference).
Render with `render_neutral.py out.png [hip0=30,knee0=35,...]`.
- CAD used only where verified: hip axes (6 base revolutes, exact), the
  hip->knee link (two hinges, rigid: orientation identical across Legs
  7/8/9/10 to 0.00deg; positions = per-part median, which drops Leg 10's
  hinge A (5mm off) and Leg 7's knee/hinge B (2.8mm off)), knee axis.
- Neutral (servos 90/90) from photos/ (2026-10-04): leg points radially out,
  leg servo box level (box axis = leg-local +Z, foot bend = local +X -> down).
  Leg-local Y IS the knee axis: each CAD leg differs from this neutral by a
  pure knee rotation (off-axis residual 0.00-0.03deg).
- One template leg copied onto all 6 hips -> symmetric by construction.
  Legs 6/Legs-11 (mis-mated in CAD) are not used at all.
- Sim leg k sits on CAD hip at 3.3 + 60k deg (base frame). Joints: hip{k}
  (+ = CCW from above), knee{k} (+ = foot lifts). 0 contacts at neutral.
- See compare_photo_vs_sim.png.
NOT YET KNOWN: which servo channel drives which sim leg, and each joint's
sign/offset vs servo degrees -> next step is a jog test on the real robot.
Uncertainty: neutral hip yaw and knee angle judged from handheld photos,
probably +-5-8deg; the jog test + the known legs-0/1 touch point
(servo0=112, servo1=46) will tighten that.

## 2026-10-04: live 3D twin in the web control page
`C:\projects\arachne\servo-web-control\` is now the local source of the Uno Q
app (pulled from the board; push changes with adb to
~/ArduinoApps/servo-web-control/). `assets/sim3d.js` renders the
hexapod_neutral model with three.js (CDN) from `assets/twin_model.json`
(written by build_neutral.py - regenerate + push both if the model changes)
and mirrors every commanded servo angle. Verified against MuJoCo at the same
joint angles: knee frames agree to <0.001mm / <0.05deg.
Channel mapping (sim angle = sign*(servo-90)+offset) is edited in the page and
saved on the board at ~/ArduinoApps/servo-web-control/twin_mapping.json
(GET/POST /mapping). Default is a guess: ch N<6 -> hipN, ch N>=6 -> knee(N-6),
sign +, offset 0. Offsets double as per-servo trims.
NEXT: user confirms the mapping channel by channel; then read
twin_mapping.json and compute collision thresholds in MuJoCo in servo degrees.

## 2026-10-04: channel mapping confirmed + first gait
- Channel map confirmed on the robot (see vorpal-servo-wiring.md): L0 front;
  L0..L5 = hip ch 0,3,4,5,2,1 / knee ch 6,9,10,11,8,7. All hips sign -1.
- Collision accuracy: at the real legs-touch point (ch0=112, ch1=46 ->
  L0 hip -22, L5 hip +44) the sim still reports a 4.8mm gap, i.e. the sim is
  ~5mm optimistic (convex hulls + calibration). Clearance margin for gaits is
  10mm (gaits.py CLEARANCE_MM).
- gaits.py: generates gaits in SIM joint angles, verifies min gap between
  different legs over every frame + the ramp from neutral, writes
  servo-web-control/assets/gaits/<name>.json. forward.json = tripod gait,
  hip amp 18deg, knee lift 25deg, 1.2s cycle, 20fps; min gap 28mm (L1-L2).
  L0/L3 can't push along x with a hip-yaw leg, so they only lift/plant.
- Web page: "Animations" panel (gait.js). Off by default ("Send to robot"
  unchecked) = preview in the twin only; on = one /set_all per frame via the
  new sketch setAll(String "a0,...,a11").
- Standing pose: STANCE_KNEE = -30 (knee servos 60). At 90 the knee hinges
  cleared the ground by only ~13mm and scraped; at -30 ~40mm, body ~96mm up.
  Gaits run from and end in this pose ("rest" in the gait json).
- Web player is wall-clock timed (pose from elapsed time, frames interpolated)
  and the twin renders on demand: the old nonstop render loop ran ~6fps in
  the desktop app's browser and slowed the player down. Buttons: Walk
  forward / Stop / Stand / Sit (all 90). Uno Q is USB-powered (servos on
  batteries), so servo current spikes can't brown out the board.

## 2026-10-04: physics walk test + steer (hinge being reprinted, sim only)
- build_physics.py -> hexapod_physics.xml: neutral twin + gravity, freejoint,
  floor, masses (522 g total, estimates), MG90S as position servos (kp 2,
  +-0.2 N.m). Stands stably at the -30 knee pose (body 96 mm up).
- sim_walk.py [gait=forward] [seconds=8] [speed=1] [steer=0] [video=1]: plays
  the exact web-page gait json open loop, reports distance/drift/tilt/body
  contacts, writes sim_<gait>.gif.
- Result: ~32 mm/s forward, body level (<1 deg tilt), no body/hinge contact.
  Untrimmed it turns right ~4 deg/s: the legs are a pinwheel (knee offset
  -21 mm tangential on every leg), so the robot is chiral and walking forward
  isn't mirror-symmetric.
- Steer = left/right stride difference, hip swing x (1 + steer*side)
  (gaits.steered, hip_side in the json). + turns right. steer -0.15 holds
  heading in physics (+4 deg in 6 s), but heading is open-loop neutral: over
  16 s it still wanders +-20-35 deg, and 0.005 of trim flips the direction.
  Needs feedback (IMU/camera) to walk truly straight.
- Clearance holds to steer +-0.6 (11.5 mm, checked in gaits.py ->
  steer_max); that turns ~12 deg/s. Web page: live "steer" slider in the
  Animations panel (double-click resets), default 0; tune it on the real
  robot.
- Pushed to the board 2026-10-05 and seen served (index.html, gait.js,
  gaits/forward.json with the steer slider). Board was offline again by
  2026-10-17; re-verify with md5sum next time it is connected.
- 2026-10-17: direction decided, see research-questions.md ("Paper A"):
  sealed-servo identification from IMU data, kill test first (perfect-motor
  policy on the real robot).

## 2026-10-17: RL pipeline (Paper A kill test) and IMU firmware
- `rl/` (code in the repo, environment in WSL2 Ubuntu `~/rl`, GPU JAX 0.11 +
  MuJoCo 3.16 + Playground; run via `wsl -d Ubuntu -- bash <repo>/rl/wslrun.sh python <script>`):
  - `build_mjx.py` -> `hexapod_mjx.xml`: physics model with meshes replaced by
    foot spheres (r 5 mm, a guess for the hot-glue blobs) + base box; masses and
    inertias identical (522 g); stands at 95 mm.
  - `check_limits.py`: widest hip range keeping >= 10 mm between neighbouring
    legs for ANY combination = +-25 deg (worst 17.4 mm); +-30 gives 9.7 mm.
  - `common.py` (shared obs/action definitions), `hexapod_env.py` (Playground
    env, JAX backend, ideal "perfect motor" servos), `train.py` (Brax PPO,
    shim for the removed `jax.device_put_replicated`), `play_policy.py` (NumPy
    inference in the full mesh model, same as the Uno Q will run).
  - Training speed ~6k env-steps/s (22k for pure stepping). A 4.4M-step test
    run learns slowly (reward 2.6 -> 11.7); the policy hardly walks yet.
- IMU (MPU6050): `servo-web-control/sketch/sketch.ino` now samples it at 200 Hz
  in loop() and ALL Bridge handlers are `provide_safe` (so I2C is never used by
  two threads); compiles for `arduino:zephyr:unoq` (13% flash). Backup of the
  working version in `backup-pre-imu/`. New endpoints in `python/main.py`:
  `/imu_status`, `/imu`, `/imu_batch`, `/imu_calibrate`, `/imu_axes`.
  Laptop helper `tools/imu_log.py`. Not flashed yet. Unknowns: 400 kHz I2C on
  this wiring, and whether Bridge returns long Strings (batch of 40 samples).
- Tools: `tools/track_apriltag.py`, `tools/servo_horn_angle.py` (synthetic
  self-tests pass; no real footage yet).

## 2026-10-17 (later): perfect-motor policy v1 trained, runner ready
- `rl/runs/imu` (40M steps, 59 min, 8192 envs, ~11.6k env-steps/s): tracking
  error 133 -> 38 mm/s over training. Sim-to-sim in the FULL mesh model
  (`play_policy.py`, 20 s each, ideal servos, no noise): fwd cmd 60 -> 51 mm/s;
  back -60 -> -48; fwd 100 -> 74 (the open-loop tripod gait does 33); left 40 ->
  37; stand still -> 4 mm/s drift; max tilt ~2 deg; nothing but feet touches
  the ground. Weak at turning: yaw cmd 0.5 rad/s (29 deg/s) gives 8.5 deg/s, and
  fwd+yaw gives ~0.5 deg/s. Cause: yaw range +-0.8 rad/s exceeds what a hip-yaw
  hexapod can do, and the yaw reward (sigma 0.1) was weak.
- `rl/runs/imu_v2` (yaw +-0.3 rad/s, sigma_ang 0.02, 40M steps): WORSE and
  discarded. Full-model eval: fwd 60 -> 25 mm/s, 100 -> 35, left 40 -> 18, and
  yaw cmd +-0.25 gives about +8 and +10 deg/s (same sign: it does not steer).
  The training reward looked the same as v1 (lin 0.45), so training reward
  alone does not predict transfer; always run play_policy.py. Turning needs
  a different approach (e.g. command-conditioned curriculum or a heading
  reward); not needed for the kill test. v1 stays the kill-test policy.
- `rl/run_policy.py`: laptop-in-the-loop on the real robot, 20 Hz, one HTTP
  round trip per tick via the new `/policy_step` endpoint. DRY RUN by default;
  `send=1` moves the robot (eases to stand, ramps the command, eases back on
  exit/Ctrl-C). `rl/test_run_policy.py` checks it against a mock board (passes).
  Lesson: use 127.0.0.1, not localhost, from Windows Python (IPv6-first = ~2 s
  per request); tools/imu_log.py had the same bug, fixed.
- Kill-test procedure once the IMU is flashed: (1) `tools/imu_log.py status`,
  `live` (check axes), `calibrate`; (2) `run_policy.py` dry run; (3) `send=1` in
  the air; (4) on the floor under the overhead phone with the AprilTag, 5-10
  runs of cmd=0.06,0,0 (and 0.10,0,0, 0,0.04,0, 0,0,0); (5) track_apriltag.py
  per video; compare with the sim numbers above. Big gap -> proceed.

## 2026-10-22: IMU wired and flashed; first checks
- Verified from the Uno Q core files: `Wire` = I2C2 = D20 (SDA) / D21 (SCL);
  `Wire1` = Qwiic (I2C4); `Wire2` = A4/A5. The PCA9685 must be on D20/D21 (the
  first-session A4/A5 advice was wrong for this board). See vorpal-servo-wiring.md.
- IMU firmware flashed; `imu_log.py status` -> WHO_AM_I 104 (MPU6050 OK). At rest
  (robot on the table) gyro ~0.001 rad/s after calibration; |accel| ~1.05 g (sensor
  scale ~5% high, harmless: only the direction is used). The earlier noisy readings
  were the user handling the robot, not a calibration error.
- Bridge limits (measured): GET /imu 18.7 ms, POST /set_all 17 ms, POST /policy_step
  29.8 ms (p95 31) -> a 20 Hz policy loop fits its 50 ms budget with ~40% slack.
- 200 Hz streaming does NOT work through the Bridge: each request stalls the MCU main
  loop (~20 ms + ~2.4 ms per sample returned, slow UART), so polling gives ~100 Hz with
  gaps up to 210 ms. Moving the getters to the Bridge thread (`provide`) changed
  nothing. For the shake / system-ID log, record on the MCU into RAM (200 Hz, ~12 B per
  sample, 60 s = ~150 KB of 786 KB SRAM) and download afterwards. Not built yet.
- The board rebooted once while the robot was being handled (cable tug or a brown-out
  from loose wiring on the 3V3 rail are the candidates); recovered with the usual
  reclaim sequence.
- Still to do: axes check by tilting (imu_log.py live), then run_policy.py dry run.

## 2026-10-22: camera-based servo-fault detection idea (for the professor meeting 2026-10-23)
- Literature review + plan written as a Claude Docs document ("Camera-based error detection
  and correction for low-cost legged robots", artifact https://claude.ai/artifact/Ax16AW58LEepcwmj5xCWkj).
  Closest prior art: Hu, Chen, Lipson (arXiv 2207.03386; npj Robotics 2025): front camera +
  12 commands predict ego-motion, mismatch flags damage on a 12-DoF legged robot; one anomaly
  (cut end-link), no numbers, no per-joint localisation, relearns in ~30 min.
- fault-pilot/pilot_observability.py (simulation only, ~20 s): single-servo faults (stuck, +-8 deg
  offset) in the lean twin walking the tripod gait; camera-style ego-motion alone names the faulty
  servo 67% of the time (chance 8%); knees 89-100%, hips 17-58%; offsets <= 4 deg undetectable;
  camera-style beat IMU-style at low noise. Idealised signals, no sim-to-real. Results in
  fault-pilot/pilot_results.json / .png.
- Slider finding: servo-web-control/assets/app.js sends one fetch per slider change with no error
  check, ack or retry, so a failed request leaves the UI and twin out of sync with the robot (likely
  the "doesn't move until I wiggle it" symptom). Untested: log command / HTTP reply / PCA9685
  register (/raw) for 200 small moves to separate software drops from servo dead band.
- CVPR 2027: enrolment 10 Nov, paper 16 Nov 2026 (AoE), decisions 25 Feb 2027; authors must review;
  no substantially similar submission elsewhere until 22 Feb. Recommended: do not rely on it.

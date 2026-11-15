# Arachne

A Vorpal-derived hexapod: 6 legs x 2 DOF (12 MG90S servos on one PCA9685), an Arduino
Uno Q as the brain, a MuJoCo digital twin, a web control page, an IMU, an RL pipeline,
and the research notes around it. Status as of 2026-10-26. Private repository.

**Nothing here has been measured on a walking real robot yet.** The hinge that cracked
was reprinted and the IMU was only just wired; every walking number below is simulation.

## What is where

| Path | What it is |
| --- | --- |
| `HANDOVER.md` | **Start here:** current state, setup on a new machine, next steps (2026-11-15) |
| `fault-sim/` | Camera-based servo-fault detection in simulation, environment stages 0-7, Kaggle runs; results in `fault-sim/RESULTS.md` |
| `vorpal-servo-wiring.md` | Channel map (confirmed on the robot), I2C pins of the Uno Q, status log |
| `vorpal-mujoco/HANDOVER.md` | The engineering log: every decision, bug and number, newest last |
| `vorpal-mujoco/build_neutral.py` | Builds the CAD-derived twin from the OnShape data and photos |
| `vorpal-mujoco/gaits.py`, `sim_walk.py`, `build_physics.py` | Tripod gait with leg-clearance check; physics walk test |
| `vorpal-mujoco/rl/` | MJX training environment, PPO, policy player, real-robot runner (WSL2 + GPU) |
| `servo-web-control/` | The app on the board: sketch (servos + 200 Hz IMU), FastAPI endpoints, 3D twin web page |
| `tools/` | AprilTag path tracker, 240 fps servo-horn tracker, IMU logger |
| `fault-pilot/` | Simulation pilot: can a faulty servo be found from body motion? |
| `research-questions.md` | Literature scan, paper plans, venues and dates |
| `docs/literature-review-camera-fault-detection.md` | Literature review for the camera-based fault-detection idea |
| `idea.md` | Early research-agent seed (partly superseded; see its caveats) |
| `portfolio-handover.md` | Handover for building a portfolio site |
| `backup-pre-imu/` | The last servo-only sketch and app, before the IMU firmware |
| `vorpal-sim/` | First hip-interference model, calibrated on one real touch point |

## Findings so far

- **Twin.** CAD-derived MuJoCo model, validated against photos at servos=90. Collision
  distances are about 5 mm optimistic at the one real touch point we measured, so gaits
  keep a 10 mm margin.
- **Open-loop tripod gait** (hip +-18 deg, knee lift 25 deg, 1.2 s cycle): minimum leg gap
  28 mm. In the physics model it walks at about 33 mm/s but turns right about 4 deg/s
  (the legs are a pinwheel); a stride trim of -0.15 holds the heading for a few seconds.
- **Learned "perfect motor" walking policy** (`vorpal-mujoco/rl`, MJX + PPO, 40M steps):
  in the full mesh model, 60 mm/s forward command gives 51 mm/s, 100 gives 74 (the
  tripod gait does 33), sideways 40 gives 37; weak at turning. A retrain with a tighter
  turn reward (v2) was worse everywhere and was dropped. Training reward alone did not
  predict transfer, so test every policy in the full model.
- **IMU on the board** (MPU6050, 200 Hz): works, but polling through the Bridge gives
  about 100 Hz with gaps (every request stalls the board for about 20 ms). The 20 Hz
  policy loop fits (one request is about 30 ms). For system identification, record on the
  board and download afterwards.
- **Slider symptom.** `servo-web-control/assets/app.js` sends one `fetch` per slider move
  with no error check or retry, so a failed request leaves the page and twin out of sync
  with the robot. Probably the "it only moves when I wiggle it" behaviour; untested.
- **Camera-based servo-fault detection** (professor idea, 2026-10-22): the closest prior
  work is Hu, Chen, Lipson (arXiv 2207.03386; npj Robotics 2025). Our simulation pilot
  (`fault-pilot/`): camera-style ego-motion names the faulty servo 67% of the time
  (chance 8%); knees are readable (89-100%), hips are not (17-58%); offsets of 4 deg or
  less are invisible. Idealised signals, no sim-to-real. Full review in `docs/`.
- **CVPR 2027** needs the paper by 16 Nov 2026 (enrolment 10 Nov); too tight for real-robot
  results. IROS 2027 (1 Mar) is the realistic full-paper target.

## Running things

- Simulation scripts run on Windows Python 3.13 with `mujoco`, `numpy`, `scipy`, `PIL`
  (and `scikit-learn`, `joblib`, `matplotlib` for the pilot): for example
  `python vorpal-mujoco/gaits.py`, `python vorpal-mujoco/sim_walk.py steer=-0.15`,
  `python fault-pilot/pilot_observability.py`.
- Training runs inside WSL2 Ubuntu with a GPU: see `vorpal-mujoco/rl/setup_wsl.sh` and
  `wslrun.sh`, then `python train.py name=imu steps=40000000 envs=8192`.
- The app on the board is deployed with `adb push` to `~/ArduinoApps/servo-web-control`
  and started with `arduino-app-cli`; the exact steps, and the reboot/autostart quirks,
  are in `vorpal-mujoco/HANDOVER.md` and `vorpal-servo-wiring.md`.
- `vorpal-mujoco/rl/run_policy.py` is a dry run unless you pass `send=1`. Do not send to
  the robot without reading its docstring.

## Not in the repository

Photos of the robot, the original Vorpal download, scratch downloads, the superseded
`sim/` model from August, videos (`*.gif`, regenerate with `sim_walk.py` and
`play_policy.py`), pickled training state, a third-party PDF and the Python environment.
See `.gitignore` and `NOTICE.md`.

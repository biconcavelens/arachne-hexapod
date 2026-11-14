# Results

Kaggle, T4, kernel `trishanthmellimi/arachne-fault-sim` version 4 (2026-10-28), `kaggle_run.py` mode `gt`.
Fault-free and faulty runs share each seed's scene, gait and noise. Outputs are in `runs/kaggle_v4/`
(gitignored). Version 3 has the camera-pipeline numbers.

## Stage B diagnoser, offline (2000 episodes, 400 held out, gt ego-motion)

| | Transformer | logistic regression (pilot) |
|---|---|---|
| detection AUROC | **0.97** | 0.65 |
| right servo, single fault | **74%** | 34% |
| right type on the faulty servo | 43% | |
| size error (when type is right) | loss p 0.12, sag 1.0°, deadband 1.7° | |

Validation loss is lowest at epoch 7 of 30 (that checkpoint is kept), so more data or regularisation
should help. Sag is often called healthy, and loss is confused with both other types: a frozen servo
and one that moves only 2–5° look alike from the body.

## Closed loop (60 test seeds, 20 s, speed over the last 4 s, median vs the same seed without faults)

| first fault | no recovery | oracle diagnosis + recovery | model + recovery |
|---|---|---|---|
| loss | 0.67 | **0.99** | 0.56 |
| sag | 0.90 | **1.00** | 0.70 |
| deadband | 0.86 | **0.99** | 0.59 |

With the true labels, the recoveries work: resend, pause and resend, and the dead-band probe plus
jump-between-extremes gait. With the model, the robot walks slower than with no recovery. It raises
7.2 false alarms/min on healthy runs and picks the wrong type, and wrong probes and pauses cost
standing time. The first alarm names the right servo 60% of the time and the right type 33%.

Dead-band estimate from the probe (oracle runs): within 1° of the truth in 9 of 15 probes; the rest
saw no jump and fell back to 24°. Summing the two signals' z-scores (now in `recovery.py`) gives 11
of 15 on the same traces. That rule was chosen on those traces, so it needs a fresh run to confirm.
The estimate only decides whether the workaround is usable, so it matters less than it looks.

## Camera pipeline (version 3: 24 episodes, DIS optical flow, plane ego-motion)

| | vx | vy | vz | wx | wy | wz |
|---|---|---|---|---|---|---|
| correlation with true twist | 0.91 | 0.72 | 0.26 | 0.16 | 0.16 | 0.86 |

Forward speed and turning are tracked well, body roll and pitch rates are not (600 episodes in
version 5: 0.91 / 0.76 / 0.26 / 0.23 / 0.21 / 0.88). The diagnoser does not need them; see the IMU
comparison below. Camera frames are dark in the renders, which may limit flow.

## Without the IMU (version 5, 2026-10-28)

There are two ways to lose the IMU. "IMU dropped": the IMU-trained model runs with its IMU inputs
held at the training mean. "Camera only": a model trained and run with no IMU. Offline numbers are
from held-out episodes: 400 for gt, 120 for flow. Flow models saw 600 episodes, gt models 2000.

| | detection AUROC | right servo, single fault | right type |
|---|---|---|---|
| gt, with IMU | 0.967 | 74% | 43% |
| gt, IMU dropped | 0.955 | 63% | 34% |
| gt, camera only | 0.966 | 74% | 43% |
| flow, with IMU | 0.984 | 75% | 28% |
| flow, IMU dropped | 0.984 | 75% | 29% |
| flow, camera only | 0.984 | **77%** | 32% |

Closed loop: median speed vs healthy. gt runs use 60 seeds, where no recovery scores 0.86. Flow runs
use the first 40 of those seeds, where no recovery scores 0.82.

| | loss | sag | deadband | all | false alarms/min | first alarm right servo |
|---|---|---|---|---|---|---|
| gt, with IMU | 0.56 | 0.70 | 0.57 | 0.65 | 7.3 | 60% |
| gt, IMU dropped | 0.57 | 0.96 | 0.92 | 0.74 | 2.0 | 66% |
| gt, camera only | 0.56 | 0.66 | 0.54 | 0.60 | 6.2 | 67% |
| flow, with IMU | 0.82 | 0.83 | 0.83 | 0.82 | 2.2 | 58% |
| flow, IMU dropped | 0.52 | 0.83 | 0.75 | 0.75 | 1.9 | 45% |
| flow, camera only | 0.55 | 0.91 | 0.75 | 0.75 | 1.8 | 53% |

In simulation the IMU adds nothing to the diagnosis: camera-only models match IMU models on both
feature sets. The IMU-trained gt model leans on the IMU somewhat (servo 74% to 63% when it is
removed), but a model trained without it does not miss it. Closed-loop speed mostly tracks the
false-alarm rate (each false alarm costs standing time). Every model is still at or below no
recovery, so the alarm policy, not the sensor, is the bottleneck. Caveats: the simulated IMU has
only white noise (no bias, vibration or mount flex), and the simulated flow has no blur, rolling
shutter or real floor; flow results rest on 120 validation and 40 closed-loop episodes.

## Environment stages (versions 9 and 10, 2026-11-14)

The professor's plan: build realistic environments step by step, still purely in sim. They are set with
`FAULTSIM_STAGE` (`environment.py`).

- **Stage 0:** the textured plane (everything above).
- **Stage 1:** a lit room. Painted or wallpapered walls with skirting, a door and a lit window.
  The floor is wood planks, tiles, carpet or concrete, with gloss on wood and tiles. One to four
  ceiling spotlights (warm to cool white) and dim ambient light; two lights cast shadows.
- **Stage 2:** stage 1 plus about 10 static objects per room: boxes, cardboard boxes, cabinets
  against the walls, bins, balls, tables, chairs, and rugs.

Room size, room angle, materials, lights and objects are re-drawn every episode. Textures are
procedural. No solid object is placed within 1.5 m of the start or on the strip ahead, so the robot
never touches anything: on the same 40 test seeds, stages 0, 1 and 2 walk identically (checked).
The first stage 2 run (version 9) let the steering robot bump into furniture on 5 of 40 seeds, so
stage 2 was rerun (version 10). Previews of four random rooms per stage are in
`runs/kaggle_v9/camera_view_stage1.png` and `runs/kaggle_v10/camera_view_stage2.png`, and videos in
`runs/kaggle_v9/runs/s1_videos/` and `runs/kaggle_v10/runs/s2_videos/`.

All stages use optical flow, 600 episodes and the same seeds, with 120 held-out episodes:

| | camera only: AUROC / right servo / right type | with IMU: AUROC / servo / type | IMU dropped: servo | plane-trained model, tested here: AUROC / servo |
|---|---|---|---|---|
| stage 0, plane | 0.984 / 77% / 32% | 0.984 / 75% / 28% | 75% | (trained here) |
| stage 1, room | 0.984 / 77% / 29% | 0.982 / 79% / 26% | 76% | 0.982 / 76% |
| stage 2, room + objects | 0.984 / 75% / 30% | 0.983 / 79% / 27% | 79% | 0.984 / 73% |

Closed loop, camera-only model: 40 seeds, median speed relative to healthy.

| | no recovery: loss / sag / dead band / all | model: loss / sag / dead band / all | false alarms/min | first alarm right servo |
|---|---|---|---|---|
| stage 0 | 0.54 / 0.81 / 0.91 / 0.82 | 0.55 / 0.91 / 0.75 / 0.75 | 1.8 | 53% |
| stage 1 | 0.54 / 0.81 / 0.91 / 0.82 | 0.58 / 0.78 / 0.68 / 0.69 | 3.0 | 53% |
| stage 2 | 0.54 / 0.81 / 0.91 / 0.82 | 0.60 / 0.82 / 0.67 / 0.73 | 3.9 | 63% |

Optical-flow ego-motion, correlation with the true body velocity (vx / vy / vz / wx / wy / wz):

- plane: 0.91 / 0.76 / 0.26 / 0.23 / 0.21 / 0.88
- stage 1: 0.88 / 0.71 / 0.17 / 0.23 / 0.21 / 0.87
- stage 2: 0.86 / 0.72 / 0.15 / 0.23 / 0.21 / 0.87

What changed:

- **The diagnosis does not get harder in a lit room with furniture.** Detection stays at 0.98 and the
  right servo at 74–79%.
- **The model trained on the plane works in the rooms without retraining** (76% / 73% right servo).
  The diagnoser's inputs (body motion plus a coarse 4×4 flow grid) do not depend on the scene, as
  long as the floor has texture.
- **The plane-motion fit gets slightly worse.** Walls and objects break the flat-floor assumption;
  forward speed falls from 0.91 to 0.86, but the robust fit absorbs most of it.
- **False alarms rise** (1.8, 3.0, 3.9 per minute). The closed loop is still below no recovery in
  every stage, so the alarm policy is still the bottleneck.

Caveats:

- MuJoCo's renderer has no global illumination, and the textures are procedural, not photographs.
- The objects are static, the floor is flat, and there is no camera noise or motion blur yet.
- One training seed per stage; closed-loop differences of a few hundredths are within noise at 40 seeds.

## Environment stages 3-7 (versions 13 and 14, 2026-11-14)

- **Stage 3, dense clutter:** about twice the objects of stage 2. It adds bookshelves, sofas, plants, floor
  lamps with their own light, shoes, papers on the floor and pictures on the walls. Lighting runs
  from dim to bright, sometimes tinted.
- **Stage 4, things that move by themselves:** stage 3 plus a robot vacuum, rolling balls, a sliding box and a
  person's legs walking past, crossing the view or passing alongside, starting at 1-10 s. They are
  visual only, so the camera sees motion the robot did not cause.
- **Stage 5, outdoor hills:** grass, dirt or gravel ground on hills of 5-9°, with sky, a sun at 20-75° casting
  shadows, rocks, bushes and trees. The ground is a detailed patch where the robot walks plus a
  coarse far field, because one fine heightfield made rendering 40× slower.
- **Stage 6, rough ground:** stage 5 plus bumps of 2-5 mm and stones of 4-12 mm, about half of what the
  gait lifts a foot (17-23 mm).
- **Stage 7, rougher still:** hills of 7-11°, bumps of 5-10 mm and stones of 8-25 mm, as tall as the foot
  lift. Version 13 ran this as "stage 6". Same code path and random draws, so the results in
  `runs/kaggle_v13/runs/s7_*` were renamed. Stage 6 was then made milder (version 14), because the
  healthy robot stalls on stage 7.

Stages 3 and 4 walk identically to stages 0-2 on all 40 test seeds (checked), so only the image
changes. From stage 5 the ground changes the walking, so those stages have their own no-recovery
reference. Previews are in `runs/kaggle_v13/camera_view_stage{3,4,5,7}.png`, videos in
`runs/kaggle_v13/runs/s*_videos/` and `runs/kaggle_v14/runs/s6_videos/`.

Same protocol as above: optical flow, 600 episodes, 120 held out.

| | camera only: AUROC / right servo / right type | with IMU: AUROC / servo / type | IMU dropped: servo | plane-trained model, tested here: AUROC / servo |
|---|---|---|---|---|
| stage 0, plane | 0.984 / 77% / 32% | 0.984 / 75% / 28% | 75% | (trained here) |
| stage 3, clutter | 0.984 / 71% / 26% | 0.980 / 81% / 30% | 79% | 0.972 / 79% |
| stage 4, + movers | 0.982 / 76% / 31% | 0.980 / 78% / 27% | 78% | 0.971 / 78% |
| stage 5, hills | 0.957 / 62% / 24% | 0.959 / 66% / 23% | 62% | 0.825 / 51% |
| stage 6, stones up to half a step | 0.957 / 67% / 24% | 0.952 / 62% / 21% | 62% | 0.787 / 44% |
| stage 7, stones as tall as a step | 0.901 / 50% / 16% | 0.913 / 49% / 17% | 50% | 0.741 / 33% |

Closed loop, camera-only model: 40 seeds, median speed relative to the same seed walking healthy.

| | no recovery: loss / sag / dead band / all | model: loss / sag / dead band / all | false alarms/min | first alarm right servo |
|---|---|---|---|---|
| stage 0 | 0.54 / 0.81 / 0.91 / 0.82 | 0.55 / 0.91 / 0.75 / 0.75 | 1.8 | 53% |
| stage 3 | 0.54 / 0.81 / 0.91 / 0.82 | 0.53 / 0.83 / 0.51 / 0.67 | 2.1 | 63% |
| stage 4 | 0.54 / 0.81 / 0.91 / 0.82 | 0.73 / 0.65 / 0.65 / 0.65 | 3.6 | 51% |
| stage 5 | 0.51 / 0.74 / 0.84 / 0.71 | 0.61 / 0.64 / 0.39 / 0.61 | 4.5 | 49% |
| stage 6 | 0.52 / 0.68 / 0.83 / 0.67 | 0.45 / 0.64 / 0.56 / 0.58 | 6.0 | 57% |
| stage 7 | (not meaningful, see below) | | 6.8 | 45% |

Optical-flow ego-motion, correlation with the true body velocity (vx / vy / vz / wx / wy / wz):

- plane: 0.91 / 0.76 / 0.26 / 0.23 / 0.21 / 0.88
- stage 3: 0.86 / 0.75 / 0.15 / 0.23 / 0.21 / 0.88
- stage 4: 0.85 / 0.70 / 0.14 / 0.23 / 0.21 / 0.87
- stage 5: 0.90 / 0.74 / 0.29 / 0.37 / 0.34 / 0.88
- stage 6: 0.90 / 0.81 / 0.45 / 0.45 / 0.40 / 0.87
- stage 7: 0.89 / 0.83 / 0.61 / 0.55 / 0.52 / 0.87

What changed:

- **Clutter and moving objects do not hurt the offline diagnosis.** Detection stays at 0.98, and the
  plane-trained model still finds the right servo 78-79% of the time. The camera-only 71% on stage 3
  is within training noise: the stage 3 model with IMU, which in sim does not use the IMU (79% with
  it dropped), gets 81%.
- **Moving objects raise false alarms** (2.1 to 3.6 per minute, stage 3 to stage 4). The windows where
  something crosses the view look like a fault.
- **Hills are the first real drop:** 62% right servo when trained on hills, and 51% for the
  plane-trained model. The flow measures the body motion as well as on the plane or better, so the
  likely cause is the walking itself: slopes tilt and slow the body in ways that look like faults.
  This is untested.
- **Stones up to half a step (stage 6) add little beyond the hills when trained there** (67% right
  servo, against 62% on stage 5, within noise). The healthy robot still walks (median 37 mm/s, against
  43 on the hills; none of 40 stall). But the plane-trained model falls further (44%) and false
  alarms rise to 6 per minute.
- **On stage 7 the gait, not the diagnoser, is the limit.** The healthy robot's median speed falls to
  12 mm/s (43 mm/s on stage 5), and 16 of 40 healthy runs move less than 5 mm/s. It does not fall;
  its feet catch on stones as tall as its step. Speed ratios against a stalled healthy run mean
  nothing, so stage 7's closed loop is not scored. Its offline numbers (50% right servo) mix fault
  signatures with legs blocked by stones. The higher vz/wx/wy flow correlations on stages 5-7 come
  from the body bouncing more (larger true motion), not from better flow.

Caveats as above, plus: the movers are simple shapes (the person is two swinging legs), and the
closed loop at 40 seeds swings by about ±0.1 between stages without any real change (stages 3 and 4
walk identically, yet their model rows differ).

## Next

1. Choose the recovery from the servo alone, escalating by cost: resend, then pause, then probe.
   This avoids the weak type head.
2. Cut false alarms with a higher threshold, more consecutive windows, or more training data.
   Moving objects now account for part of them.
3. Terrain: give the diagnoser what it needs to tell slope from fault, for example the gravity
   direction from the flow or the IMU (the IMU may finally matter here). Then stage 7 needs a
   higher-stepping gait before it can say anything about faults.
4. Combine terrain with moving objects, and add camera noise and motion blur.

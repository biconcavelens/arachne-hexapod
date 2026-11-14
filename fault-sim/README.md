# fault-sim: servo faults seen through the body camera

The MuJoCo twin walks the tripod gait with faults on one or two servos. The robot works
out from its forward camera (plus IMU and its own commands) which servo is faulty and
how, then recovers. Nothing here has been run yet: start with `selftest.py`.

## Faults (`faults.py`, sizes in `config.py`)

| fault | what the servo does | size | recovery (`recovery.py`) |
|---|---|---|---|
| loss | each command dropped with probability p; 30% of losses are fully frozen (p = 1) | p 0.5–1.0 | resend: 3 copies per command; a frozen channel clears on reset with chance 0.7 |
| sag | still moves, but only ±span around where it was when the sag began | 2–5° | stand still 1 s, then resend; clears with chance 0.5 |
| deadband | ignores a commanded change smaller than db | 15–20° | probe (below), then a modified gait for the rest of the run |

Stall/blocked leg, horn slip and dead servo are left out for now. The odds in the recovery
column are guesses about the real robot; change them in `config.py`.

**Dead-band probe and the new gait.** The robot stands. The suspect servo is forced to one
limit and then the other (a jump bigger than any dead band, so its angle is known again),
returned to stand, then ramped slowly. The command offset at which the camera first
sees the body move is the dead-band estimate. That servo then jumps between its two
cycle extremes instead of moving smoothly (`gaits_lib.DeadbandWorkaround`), switching
where the smooth trajectory crosses its midpoint, so the gait keeps its timing. If the
servo's travel is not more than db + 3°, the gait is left as it is and the event is logged.

## Pipeline

```
config.py        every number
environment.py   environment stages (FAULTSIM_STAGE; 0 = plane): 1 lit room, 2 room + objects, 3 dense clutter,
                 4 clutter + things moving by themselves, 5 outdoor hills, 6 rough ground (stones up to half
                 a step), 7 stones as tall as a step
build_scene.py   the scene XML: lean physics model + visual meshes + the new cap + body camera (+ room)
                 (--check renders a sheet of random scenes and reports how much of the view is robot)
generate.py      episodes -> data/<name>/ep_*.npz   (resumable)
train.py         Stage B: healthy-motion model, then a Transformer/GRU diagnoser -> runs/<name>/
closed_loop.py   walk, diagnose online, recover; compares healthy / no recovery / recovery
selftest.py      checks of faults, probe, workaround and ego-motion; no MuJoCo or torch needed
```

`--egomotion gt` skips rendering and uses the true body motion with noise and scale error
in place of the camera's estimate (runs anywhere). `--egomotion flow` renders the camera and
runs optical flow (needs OpenGL; on a headless machine set `MUJOCO_GL=egl`).

## Order

```bash
pip install -r requirements.txt
python selftest.py
python build_scene.py --check
python generate.py --out data/gt --episodes 2000 --egomotion gt
python train.py --data data/gt --out runs/gt_transformer --baseline
python closed_loop.py --mode oracle --episodes 60
python closed_loop.py --mode none   --episodes 60
python closed_loop.py --mode model  --ckpt runs/gt_transformer/model.pt --episodes 60
```

Then the same with `--egomotion flow` (fewer episodes: about 400) to see what the real camera
pipeline costs. For Kaggle, see [KAGGLE.md](KAGGLE.md).

## Assumptions to check against the real robot

- The camera board is 30 × 25 × 1.6 mm, in the cap mount, tilted 20° down, 45° vertical FOV
  (see `hardware/camera_cap/`).
- 20 Hz commands. A dead band is measured against the angle the servo is holding, not the
  last command.
- In closed-loop runs, the gait jumps to the current phase after a pause or probe (no
  blend). Add a ramp if it trips the robot.

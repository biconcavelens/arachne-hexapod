# Running the Arachne experiments on the lab GPU

This is the hand-off for the lab machine (Ubuntu, RTX 6000 Ada, 48 GB). The whole experiment runs
from one script, and the results are pushed back to this repository on the branch `lab-results`.
Everything is simulation. Nothing here talks to the real robot.

Background on the project: `PLAN.md` (the research plan) and `docs/claude-memory/` (notes on decisions,
earlier results and where the code lives). Start with `docs/claude-memory/project-lab-run-v3.md` and
`reference-code-map.md`.

## What it runs

The robot is a hexapod that must walk along a path drawn on an overhead view, while 0–3 of its 12
servos fail (dead, stuck, losing signal, worn gears). There are four environment stages and three
sensor sets, so 12 policies are trained (PPO, MuJoCo MJX, 40M steps each):

| Stage | Env name | Run names |
|---|---|---|
| Furnished room | `room` | `v3_room_imu`, `v3_room_cam`, `v3_room_both` |
| Room with moving objects | `room_dyn` | `v3_dyn_*` |
| Rough outdoor terrain | `terrain` | `v3_terr_*` |
| Terrain with moving objects | `terrain_dyn` | `v3_tdyn_*` |

`imu` = IMU only, `cam` = body camera only, `both` = both. Then come the evaluations (256 episodes per
fault scenario), a servo-fault detector per stage, and videos.

The `v3` runs differ from the earlier laptop runs in three ways (20 Nov 2026):

1. The robot is rewarded for facing along the path, so the body camera looks where it walks and the
   robot turns at bends. Before, it walked sideways or backwards, and the camera was nearly useless.
2. The camera image is larger: 96×64 by default instead of 24×16. A small convolutional network
   reads it (`policy_nets.py`), seeing the newest frame plus its differences to two older frames.
3. The servo-fault detector reads only the IMU and the last commands, not the camera.

## Steps

All commands run from the repository root unless a step says otherwise.

**1. Get the code.**

```bash
git clone https://github.com/biconcavelens/arachne-hexapod.git
cd arachne-hexapod
```

Pushing needs write access to this private repository from this machine. Check with
`git push --dry-run origin HEAD:refs/heads/lab-results`.

**2. Set up Python** (about 5 minutes, creates `~/rl`):

```bash
bash vorpal-mujoco/rl/lab/setup.sh
```

It must end with `setup ok` and list a CUDA device.

**3. Start the run.** It keeps going after the terminal or Claude session closes:

```bash
cd vorpal-mujoco/rl
setsid nohup bash lab/run_all.sh > lab/run_all.log 2>&1 &
```

The script first runs a preflight of about 10 minutes at 96×64. It checks that the camera fits in GPU
memory and prints a time estimate to `lab/progress.txt`.

- If the preflight fails with out-of-memory, or the estimate is too long, start again at a smaller size:
  ```bash
  rm -f lab/logs/preflight.ok
  ARACHNE_CAM=72x48 setsid nohup bash lab/run_all.sh > lab/run_all.log 2>&1 &
  ```
  Keep the same size for the whole run. Every script refuses to load a policy trained at another size.
- The camera size must keep a 3:2 width-to-height ratio, for example 48x32, 72x48, 96x64 or 120x80.

**4. Watch it.**

```bash
cat vorpal-mujoco/rl/lab/progress.txt
tail -n 3 vorpal-mujoco/rl/runs/v3_*/log.txt
nvidia-smi
```

Each training logs a line about every 1.6M steps. `SUCCESS` is the share of episodes that reach
the goal.

**5. If it stops** (crash, reboot), run the same start command again. Finished steps are skipped,
and a training that was cut off starts over.

**6. When `ALL DONE` appears** in `progress.txt`, everything is on the branch `lab-results`. Results
are also pushed after every step, so partial results show up there along the way.

## What gets pushed

Pushed to the branch `lab-results`:

- `runs/v3_*/`: `config.json`, `log.txt` and `params.pkl` (trained weights) for each policy.
- Evaluation results: `results.json` and `table.md`.
- Fault-detector results.
- `media/lab/*.mp4` videos.
- `lab/progress.txt` and the step logs in `lab/logs/`.

GIFs and evaluation trajectories stay on the lab machine, in `vorpal-mujoco/rl/media/lab/` and
`vorpal-mujoco/rl/runs/`.

## Rules for whoever runs this

- One GPU job at a time. The script already runs everything in sequence.
- Don't change the code or settings beyond the camera size, unless a step fails. If one does, note the
  error and the fix in `lab/progress.txt`, so it is pushed with the results.
- Don't push to `main`. Results go only to `lab-results`.
- Nothing here connects to the robot hardware.

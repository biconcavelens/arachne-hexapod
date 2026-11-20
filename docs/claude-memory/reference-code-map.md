---
name: reference-code-map
description: Where the RL code lives in vorpal-mujoco/rl and how to run each piece (envs, training, eval, diagnosis, videos, sim app, job queue)
metadata:
  type: reference
---

Everything runs from `vorpal-mujoco/rl/` through `bash wslrun.sh python ...`. That script activates `~/rl` and sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MUJOCO_GL=egl`.

**Environments**, picked by name through `scenes.env_class(name)`:

| Name | Code | Contents |
|---|---|---|
| (flat) | `fault_path_env.py` | Flat-ground task; also defines `DS` = 1 cm path points, `make_path`, the fault kinds |
| `room` | `room_env.FaultRoom` | Task, rewards, faults, body camera (mjx.ray cast), object placement. Scene from `room_scene.py` → `room_scene.xml` |
| `room_dyn` | `room_dyn_env.FaultRoomDyn` | Movers: vacuum, 2 balls, person. Yielding rules and contact logic |
| `terrain` | `terrain_env.FaultTerrain` | Heightfield from `terrain.npz` (`terrain_scene.py`). Camera ray-marches the ground |
| `terrain_dyn` | `terrain_dyn_env.FaultTerrainDyn` | Terrain + dog, football, hiker, mower |

**Scripts:**
- **Training:** `train_fault.py name= env= sensors=imu|cam|both steps= envs= [encoder=cnn|mlp] [faults=0]` → `runs/<name>/` (`params.pkl`, `config.json`, `log.txt`).
- **Networks:** `policy_nets.py` (CNN actor).
- **Rollouts:** `fault_rollout.py`: `load_policy`, `make_runner` (records xy, yaw, cross-track, heading_err, struck, obs, diag_x, mocap).
- **Evaluation:** `eval_room.py policies=a,b env= n=256 out=` → `runs/<out>/results.json`, `table.md`. 12 scenarios: healthy; 1 dead / stuck / signal loss / worn, each from the start or mid-walk; 2 dead; 3 dead; 3 mixed.
- **Diagnosis:** `diagnose_room.py policy= out= [episodes=1536]` → per-servo fault detector, `results.json`.
- **Videos:** `render_room.py policy= env= kind=dead,stuck servo=knee1,hip4 onset=40 seed= [diag=] out=x.gif`. Servo names: hip/knee 0–5. Legs are front, front-left, rear-left, rear, rear-right, front-right.
- **Interactive demo:** `sim_app.py` + `sim_app.html`: draw a path, kill servos live.
- **Speed test:** `speed_terrain.py [envs] [sensors] [scene]`.
- **Laptop job queue:**
  - `queue_runner.sh` reads `runs/queue.txt` and runs each line in a memory-capped systemd unit (`arachne-job`).
  - Start the runner as unit `arachne-queue`.
  - Prefix a line with `@mem=9G` to change its cap.
- **Lab machine:** `lab/` (see [[project-lab-run-v3]]).

**Settings:**
- **Camera:** `ARACHNE_CAM=WxH`; frames `room_env.CAM_FRAMES = (0, 1, 4)`, i.e. now, −0.05 s and −0.2 s.
- **Control:** 20 Hz (`common.CTRL_DT`); episode 600 steps (30 s); target speed 0.12 m/s.

**Not in git:** GIFs, `params.pkl`, `diag_params.pkl`, eval `traj_*.npz`, `runs_*.out` logs. On the `lab-results` branch, the lab's `params.pkl` and `diag_params.pkl` are force-added.

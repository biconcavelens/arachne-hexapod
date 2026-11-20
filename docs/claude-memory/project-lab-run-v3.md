---
name: project-lab-run-v3
description: v3 rerun of all 12 policies on the lab RTX 6000 Ada (2026-11-20): heading reward, 96x64 camera with CNN, IMU+command diagnosis; instructions in LAB_RUN.md, results on branch lab-results
metadata:
  type: project
---

On 2026-11-20 the user decided to rerun everything on a lab computer: an RTX 6000 Ada (48 GB), Ubuntu. This laptop has no SSH to it. A Claude on a different account runs it there and pushes the results back.

**What changed from v2** (see [[project-stage-results-v2]]):
1. **Heading:**
   - New reward `heading` (scale 1.0, sigma 0.35 rad). It compares body yaw, which is where the body camera looks, with the path direction 10 cm ahead, so the robot turns before a bend.
   - The start yaw is now the path's first direction ± 90° (`start_yaw_spread`), no longer random over the full circle.
   - New metric `heading_err`. The evaluation reports `heading_mean_deg` and `heading_within30_pct`.
   - Code: `room_env.py`, `_heading_err`, `HEAD_LOOK`.
2. **Resolution:**
   - Set with the env var `ARACHNE_CAM=WxH` (`room_scene.py`). Lab default 96x64; fallback 72x48. Keep the 3:2 ratio.
   - Each run stores `cam` in `config.json`, and `fault_rollout.load_policy` refuses a mismatch.
   - The 4060 test at 4096 envs: 24x16 gave 4,260 env-steps/s, 48x32 gave 2,380, 72x48 and 96x64 ran out of memory.
3. **CNN actor** (`policy_nets.py`, `encoder=cnn`, the default in `train_fault.py`):
   - 4 conv layers on the newest frame plus two frame differences, then a 128-number feature, joined to IMU/commands/path, then the MLP.
   - The critic is unchanged (privileged state). Old runs load as `encoder=mlp`.
4. **Diagnosis input** (`diagnose_room.py`):
   - IMU (gravity, gyro) + last 12 commands, every 0.2 s over 2 s.
   - Recorded as `diag_x` in `fault_rollout`. It no longer reads the policy's observation.

**Hand-off:**
- **Instructions:** `LAB_RUN.md` at the repo root.
- **Scripts:** `vorpal-mujoco/rl/lab/`:
  - `setup.sh` (venv `~/rl`)
  - `run_all.sh`: preflight, then 12 trainings, 4 evaluations, 4 detectors and videos. It is resumable, uses run prefix `v3_`, and pushes after every step.
  - `push_results.sh` → branch `lab-results`
  - `to_mp4.py`
- **What comes back** on `lab-results`: run configs, logs, `params.pkl`, eval `results.json` / `table.md`, `diag_params.pkl`, `media/lab/*.mp4`, `lab/progress.txt`.
- **ETA:** rough guess 5–8 h per camera training at 96x64, so about 2–3 days in total. The preflight prints a measured estimate.
- **Checks:** the CNN training compiled and ran, a CNN policy loaded back, and `diag_x` recorded, all on the laptop. Whether the heading reward actually makes the robot face the path was NOT verified. The user chose to check that on the lab machine.

**When results arrive:**
- Fetch `lab-results`.
- Check `heading_mean_deg` first: is the camera facing the path?
- Then compare v3 against v2: camera vs IMU, movers, diagnosis.
- Rebuild the four-stage report.
- Mover waiting (more collision cost or a waiting reward) is still unsolved and not part of v3.

---
name: project-stage-results-v2
description: Laptop results of the four environment stages (room, room+movers, terrain, terrain+movers), 2026-11-18..20, 24x16 camera, before the heading fix; numbers and lessons
metadata:
  type: project
---

Trained on the RTX 4060 laptop, 2026-11-18 to 2026-11-20. PPO (Brax) + MuJoCo MJX, 40M steps, 4096 envs, about 2.5 h per training. The body camera was 24×16 grey, three frames, fed flat into an MLP. Report (private artifact): https://claude.ai/artifact/9kvTCyod1AsXrLreeAycyp. Local copy: `vorpal-mujoco/rl/media/stages/report/`. GIFs: `media/stages/*.gif`.

**Big flaw, found 2026-11-20:** the robot started at a random yaw, and nothing rewarded facing along the path. The yaw rate was even penalised. So the policies crab-walked sideways or backwards, with the camera looking elsewhere. All camera results below are therefore weak. This led to the v3 rerun, see [[project-lab-run-v3]].

Results: success % over 256 episodes per scenario; xt = mean distance from the path in mm.
- **Room** (room_both / room_cam / room_imu):
  - healthy 98 / 98 / 100%
  - 1 dead 97–100%
  - 3 dead 64 / 64 / 70%
  - xt about 3–5 mm
- **room_both_healthy** (never trained with faults) did as well as the fault-trained policy in the flat room.
- **Room + movers** (dyn2 = retrained under the fair mover rules):
  - dyn2_imu: healthy 84%, 1 dead 79%, 3 dead 48%
  - dyn2_both: healthy 80%, 1 dead 69%, 3 dead 37%
  - These are no better than room_both, which never saw movers. Struck 25–56%.
- **Terrain:**
  - terr_imu: 100% on single faults, 2 dead 91%, 3 mixed faults 94%
  - terr_both: 100%, 86%, 91%
  - room_both tested on terrain: 1 dead 82%, 3 mixed 66%. Terrain training clearly helps.
- **Terrain + movers:**
  - tdyn_both: healthy 90%, 1 dead 84%, 3 dead 45%
  - tdyn_imu: 90%, 84%, 48%
  - terr_both (no movers in training) scores about the same.
- **IMU only ≥ IMU + camera** everywhere at 24×16.
- **Diagnosis** (right servo named, single fault; chance about 8%):
  - room_imu 56%, AUROC 0.85, delay 3.4 s
  - room_both 19%, room_cam 10%, terr_both 16%, tdyn_both 13%
  - The detector read the policy's observation including camera pixels. v3 changes it to IMU + commands only.

**Lessons:**
- **Movers:** policies trained with movers have not learned to wait. A larger collision cost or an explicit waiting reward is needed. Still open.
- **Fault training** pays off on rough ground, not in the flat room.

**Run names:** `room_*`, `dyn2_*` (`dyn_*` were trained under the old, unfair mover rules), `terr_*`, `tdyn_*` in `vorpal-mujoco/rl/runs/`.

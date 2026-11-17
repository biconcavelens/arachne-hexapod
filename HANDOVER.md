# Handover (2026-11-15)

Written before reinstalling the operating system. It covers the state of the whole project, how
to set up a fresh machine, and what to do next. The repository map is in [README.md](README.md),
and the full engineering log of the robot and twin is in
[vorpal-mujoco/HANDOVER.md](vorpal-mujoco/HANDOVER.md).

## 1. Where things stand

Two research threads share this robot (a Vorpal-style hexapod with 12 MG90S servos on a PCA9685,
an Arduino Uno Q and an MPU6050).

**A. Camera-based servo-fault detection** (the professor's idea, 22 Oct 2026). This is the active
thread. The code is in `fault-sim/`, everything runs in simulation, and there are no real-robot
results.

- A forward body camera that cannot see the legs detects servo faults (command loss, power sag,
  15–20° dead band) at AUROC 0.98 and names the faulty servo about 75% of the time (chance 8%).
  The IMU adds nothing in simulation. Fault type is weak (16–32%).
- The professor asked for realistic environments, built up in stages. Stages 1–7 are done:

  | Stage | Environment |
  |---|---|
  | 1 | lit room |
  | 2 | room with objects |
  | 3 | dense clutter |
  | 4 | moving objects |
  | 5 | outdoor hills |
  | 6 | stones up to half a step |
  | 7 | stones as tall as a step |

  Rooms, clutter and moving objects leave the diagnosis unchanged. Uneven ground costs 10–15
  points, more for a model trained only on the plane. On stage 7 the healthy gait stalls: the feet
  lift 17–23 mm and the stones reach 25 mm.
- Recovery with the trained model in the loop is still slower than doing nothing, in every
  environment. False alarms are the cause; they rise from 1.8 per minute on the plane to 6 on
  rough ground. This is the main open problem.
- Reports sent to the professor:
  - [docs/fault-sim-results-2026-10-28.pdf](docs/fault-sim-results-2026-10-28.pdf), the plane results;
  - [docs/fault-sim-brief-2026-11-03.pdf](docs/fault-sim-brief-2026-11-03.pdf), with related work;
  - [docs/fault-sim-environments-2026-11-14.pdf](docs/fault-sim-environments-2026-11-14.pdf), stages 1–7.
- Every number, with caveats, is in [fault-sim/RESULTS.md](fault-sim/RESULTS.md).
- **Waiting on the professor:**
  - Is stage 6 (rough but walkable ground) far enough for the paper, or must the robot cross
    stage 7, which needs a new gait?
  - Should we fix the closed loop in simulation first, or go to the real robot now?
- **Venues:** CVPR 2027 (16 Nov 2026) is too tight. IROS 2027 (about 1 Mar) is the realistic
  target. Dates for 2027 are estimates.
- **Literature review:** the live version is the Claude Docs artifact
  https://claude.ai/artifact/Ax16AW58LEepcwmj5xCWkj (online, not affected by the reinstall). A
  snapshot is in `docs/`.

**B. Paper A: identify the sealed servos from IMU data, then walk on the fitted twin.** This is
paused behind thread A. Plan, kill test and status are in [research-questions.md](research-questions.md)
("The paper to write: Paper A"). A "perfect motor" PPO policy exists in simulation; the kill test
on the real robot has not been run.

## 2. Next steps for the fault-sim thread

1. **Fix the closed loop.** Act on the identified servo and escalate the recovery by cost (resend,
   then pause, then probe) instead of trusting the weak fault-type head. Calibrate alarms for few
   false alarms per minute, including on rough ground. Target: clearly beat no recovery.
2. **Tell a slope from a fault.** Give the diagnoser the gravity direction (from the flow or the
   IMU). Terrain may be where the IMU finally helps.
3. **Harder environments.** Combine terrain with moving objects, and add camera noise and blur.
   Stage 7 needs a higher-stepping gait first.
4. **Real robot.** Print the camera cap (`hardware/camera_cap/`), inject the same faults in
   software on the controller, and record naturally faulty servos too. A cheap first test: log
   command, HTTP reply and PCA9685 readback to see whether the "slider doesn't move" symptom is
   our own software dropping commands.

## 3. Setting up the new machine

1. **Git and GitHub.** Clone the private repo
   `git clone https://github.com/biconcavelens/arachne-hexapod.git arachne`, then `gh auth login`.
2. **Results that are not in git.** The Kaggle outputs (`fault-sim/runs/`, about 135 MB) and the RL
   runs (`vorpal-mujoco/rl/runs/`, policy `params.pkl` and GIFs) are assets of the private GitHub
   release `results-2026-11-15`:
   ```bash
   gh release download results-2026-11-15 -R biconcavelens/arachne-hexapod -D /tmp/arachne-results
   ```
   Unzip `fault-sim-runs.zip` inside `fault-sim/` (it holds `runs/`) and `rl-runs.zip` inside
   `vorpal-mujoco/rl/`. `kaggle_push.sh` needs `fault-sim/runs/kaggle_v5` (the plane model the
   stage runs test).
3. **Python.** Any 3.11–3.13. For the local checks and reports:
   ```bash
   pip install -r fault-sim/requirements.txt pymupdf
   ```
   This needs `mujoco==3.12.0`. Reports are printed to PDF with headless Chrome:
   ```bash
   chrome --headless --no-pdf-header-footer --print-to-pdf=OUT.pdf fault-sim/runs/report/stages.html
   ```
   Build the HTML first with `python fault-sim/make_stages_report.py`. RL training uses its own
   environment: `vorpal-mujoco/rl/setup_wsl.sh` (runs natively on Ubuntu with a GPU, uses `uv`;
   the name is from the old WSL2 setup).
4. **Kaggle.**
   - Install the CLI with `pip install kaggle`.
   - Back up and restore your own API token, `~/.kaggle/kaggle.json`. It is not in the repo, and
     no assistant should read or type it.
   - The account is `trishanthmellimi`, with the private dataset `arachne-hexapod` and the private
     kernel `arachne-fault-sim`.
5. **Robot tools.** You need `adb` for the Uno Q. The board quirks are in `vorpal-servo-wiring.md`
   and `vorpal-mujoco/HANDOVER.md`.
6. **Claude Code memory.** The assistant's saved notes (standing rules and project state) are
   copied in [docs/claude-memory/](docs/claude-memory/). On the new machine, start Claude Code in
   the repo once, then copy those files into its memory folder,
   `~/.claude/projects/<repo path with / replaced by ->/memory/`. Or ask it to read
   `docs/claude-memory/` and save them.

## 4. Running fault-sim

Until 2026-11-15 the laptop's fans were broken, so everything heavy (simulation, rendering, data
generation, training, videos) ran on Kaggle. The fans were fixed on 2026-11-17, so heavy runs may
now be done locally too (RTX 4060 Laptop GPU, 8 GB; 14 GB RAM). Kaggle stays useful for long
batches. The Kaggle workflow:

1. Set `MODE` and `STAGES` at the top of `fault-sim/kaggle_run.py`:

   | MODE | What it runs |
   |---|---|
   | `stages` | full run per stage, about 30 min each |
   | `stages_smoke` | a quick test of every step |
   | `gt`, `full`, `noimu` | the older plane runs |

2. Run `fault-sim/kaggle_push.sh`, or `KAGGLE=/path/to/kaggle fault-sim/kaggle_push.sh` if the CLI
   is not on PATH. Add `--dry` to only stage the files.
3. Wait: `kaggle kernels status trishanthmellimi/arachne-fault-sim`.
4. Download in pieces. Whole-output downloads are slow, and some files come back as HTML error pages.
   ```bash
   kaggle kernels output trishanthmellimi/arachne-fault-sim -p out --file-pattern 'results/status.txt'
   kaggle kernels output trishanthmellimi/arachne-fault-sim -p out --file-pattern 'results/runs/s6_'
   ```
5. Copy `out/results/` to `fault-sim/runs/kaggle_vN` and add the numbers to `RESULTS.md`.
   Environments are set with `FAULTSIM_STAGE`; see the docstring of `fault-sim/environment.py`.

**Kaggle run log** (kernel versions):

| Version | What | Kept in |
|---|---|---|
| 3 | first camera pipeline, 24 flow episodes | `runs/kaggle_v3` |
| 4 | full gt run, videos, dead-band probe traces | `runs/kaggle_v4` |
| 5 | IMU ablation, gt and flow; **the plane baseline** | `runs/kaggle_v5` |
| 6–8 | smoke tests for stage 1 | not kept |
| 9 | stage 1, plus a first stage 2 that bumped furniture (superseded) | `runs/kaggle_v9` |
| 10 | stage 2, rerun with a wider keep-out zone | `runs/kaggle_v10` |
| 11–12 | smoke tests for stages 3–6 | not kept |
| 13 | stages 3, 4, 5, and today's stage 7 (it ran as "stage 6" before the split) | `runs/kaggle_v13` |
| 14 | stage 6 | `runs/kaggle_v14` |

## 5. Standing rules

- **The real robot.**
  - "Send to robot" stays off by default, and the twin previews first.
  - First real moves happen in the air at 0.5× speed.
  - Don't kill processes you didn't start.
  - Taking over the Uno Q from its autostart app is described in `docs/claude-memory/feedback-robot-safety-defaults.md`.
- **Hardware choices are final.** These are the 2× MPU6050, 2× 18650 in series with no regulator,
  and hot-glue feet. Work around their side effects in software.
- **Not in the repository on purpose** (privacy or licence, see `.gitignore` and `NOTICE.md`):
  - `photos/`, which show you;
  - `Vorpal needed/`, the original CC BY-SA download;
  - `scratch/` and the superseded `sim/`;
  - the third-party Hu et al. PDF;
  - `.venv/` and `.claude/`.

  **Back these up yourself before reinstalling** if you want to keep them.
- No real-robot walking or fault numbers exist yet; don't claim any.

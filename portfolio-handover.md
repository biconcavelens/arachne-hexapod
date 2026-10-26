# Portfolio website: handover (written 2026-10-08)

For the agent building the user's portfolio site. Read sections 0-3 before
writing any code or copy. Section 5 is the project inventory; section 6 is
the flagship project (Arachne) in detail.

Provenance tags used below: **[file]** = I read it in the repo, **[README]** =
quoted from a project README I only skimmed (first ~1800 chars), **[user]** =
the user told me in chat, **[unverified]** = I have not checked it.

---

## 0. Rules for this job

1. **Claims must come from the files.** If a number, ranking, acceptance or
   outcome is not in the repo or confirmed by the user, leave it out or ask.
   This user's projects have a history of "fixed" claims that were wrong
   (see Arachne, section 6); the portfolio is where an overclaim does the most
   damage.
2. **Never publish secrets.** Project folders may hold `.env` files, API keys,
   tokens or board addresses. Do not copy a file you have not read. Scan
   anything you copy for keys before it goes in the site folder.
3. **Do not publish contact details** (email, phone, socials) the user has not
   chosen. Section 3 asks which to use.
4. **Do not touch the robot.** The Arduino Uno Q board for Arachne is
   currently stopped and not needed. You do not need `adb` or
   `arduino-app-cli` for this job. If you ever do, the board's autostart
   ("Arm Voice Control") relaunches on every reboot and the user's other app
   lives there.
5. **Don't kill processes you didn't start.** The user's experiment run in
   `C:\projects\robotics research thing online` may be running.
6. **Pronouns:** none stated. Write "the user", or use their name, and avoid
   "he/she".

## 1. Who the site is for

- **Name:** Trishanth Mellimi (from git config **[file]**). Confirm the display
  form with the user.
- **GitHub:** `biconcavelens` (from repo remotes **[file]**; the account page
  itself is **[unverified]**).
- **Domain:** two project sites are hosted under `triixd.tech`
  (`har.triixd.tech`, `bas.triixd.tech` **[README]**). Whether the user owns
  the domain and could host the portfolio there is **[unverified]**.
- **Purpose, as I understand it:** support master's applications (ETH Zurich
  MSc Robotics, Systems and Control; NUS MSc Robotics) and look credible to
  robotics researchers. Earlier in this project the user said exactly that.
  But `research-questions.md` (it contains a note dated 2026-10-06) says "applications don't"
  drive the paper plan, and on 2026-10-17 the user decided master's applications
  are irrelevant now. **Ask what the site is for.** Also: the VLA work is moving
  to a different robot with the user's professor and is NOT part of Arachne;
  Arachne's research is "Paper A" (servo identification), still a plan, no results.
- **Dates I looked up (2026-10-05):** ETH MSc RSC applications open 1 Nov and
  close 30 Nov 2026 (programme starts Sep 2027). NUS MSc Robotics August 2027
  intake: window probably May-Jun 2027, not confirmed (the Aug 2026 window was
  4 May-1 Jun). If the site is meant to be seen by ETH, it should be live
  well before 30 Nov.

## 2. Suggested shape (change it if the user disagrees)

- **Lead with robotics and research**, since that is who reads it: Arachne,
  Argus, the solar-filament paper, the BAS human-activity project, and the
  earlier arm projects. Put the rest in a short "other work" list.
- **Static site, no framework, no build step** unless the user wants one.
  One landing page plus one page per featured project. Hosted free (GitHub
  Pages or similar) unless the user has a domain.
- **A narrative thread that is true to the repos:** hobby-grade robots that
  cannot report their own state, and what that forces you to build. The
  MG996R arm used cameras and coloured dots as external encoders
  (`robot_arm`), the SO-101 replaced that with servos that report position
  (`so101_arm`, `so101_act`), and Arachne goes back to $3 position-only
  servos on purpose (`arachne`, `research-questions.md` RQ2). This is my
  suggestion; check it with the user.
- Show real artifacts (photos, videos, the in-browser twin, the paper PDF),
  not just text.

## 3. Ask the user before building (one short message)

1. Is the site still for the NUS/ETH applications? Target launch date?
2. Display name, and which contact email / links to publish (git config has an
   address; the user has not said to publish it).
3. Hosting: GitHub Pages, the `triixd.tech` domain, or something else?
4. Which projects to feature (my suggested set is in section 5), and any to
   leave out.
5. Status of items I could not verify: is the solar-filament paper accepted,
   what happened at the Arduino Physical AI Challenge and SIH, and were the
   open-source PRs (griptape, OpenCV, jaeger-ui) merged?
6. Has the repaired Arachne hinge walked on the ground yet, and are there
   real-robot numbers? None were reported to me.
7. Tone and look: any sites they like? Dark/light?

## 4. Environment notes

- Windows 11. Shells: Git Bash and PowerShell. Python 3.13 at
  `C:\Users\trish\AppData\Local\Programs\Python\Python313\python.exe` (has
  MuJoCo, numpy, PIL). `python -m http.server` is enough to preview a static
  site.
- The built-in browser pane can open `http://127.0.0.1:<port>` to check a
  page, including at phone width (resize_window mobile preset).
- `C:\projects\arachne` is **not** a git repo. Copy assets you need into the
  site folder; do not point the site at that folder.

## 5. Project inventory

Everything lives under `C:\projects\`. Grouped by how well they suit the
purpose above. One-liners are from READMEs I skimmed, not audited.

### Feature (robotics / research)

| Project | Path | What it is | Notes |
|---|---|---|---|
| **Arachne** (hexapod, MuJoCo twin) | `arachne\` | See section 6. | Flagship. Active. |
| **Argus** | `so101_act\` (repo `argus-desk-robot-assistant-arm`) | Voice-controlled SO-101 arm; two ACT pick-and-place policies; the whole voice-to-motion pipeline runs on an Arduino Uno Q, no laptop at demo time. Built for the Arduino Physical AI Challenge India 2026 **[README]**. | Has `docs\ARGUS_technical_report.pdf`, `ARGUS_challenge_submission.pdf`, `media\`, `demo_slides.html`, and a local `ARGUS_demo_deck.pptx` (README says 271 MB, too big for the repo). Competition outcome **[unverified]**. |
| **Solar filament segmentation** | `kaggle sun thiing\` (repo `filament-segmentation-2026`) | Instance segmentation of solar filaments in GONG H-alpha images; calibrated ensemble of detectors. Kaggle challenge plus a paper: T. Mellimi, R. Bhargava, R. Gupta, "Instance Segmentation of Solar Filaments in GONG H-alpha Images with a Calibrated Ensemble of Detectors", **submitted** to SABiD 2026 (IEEE BigData 2026 workshop). Public leaderboard 0.40, out-of-fold PQ 0.461 **[README]**. | Paper source and PDF in `paper\` (`main.pdf`, `main.tex`, figures as PNG+PDF). Say "submitted" unless the user confirms acceptance. Folder is large (578 items). |
| **BAS human activity recognition** | `lunar thing\` (repo `bas-har`) | Entry for an ISRO BAS problem statement: on-board human activity recognition from two payload cameras. Hosted sites `har.triixd.tech` and `bas.triixd.tech` **[README]**. | README results (e.g. 34/34 injected mistakes alerted, 687 B/s vs 108 KB/s video) are on **synthetic Blender runs** and held-out NASA ISS footage. Keep that caveat next to any number. |
| **Varuna: oil-spill detection and attribution** | `sih oil spill\` | Smart India Hackathon problem statement 26143 (NTRO). Sentinel-1 SAR slick detection (DeepLabV3+), OpenDrift hindcast, AIS vessel attribution, forecast, hosted dashboard **[README]**. | No git remote seen. Hackathon outcome **[unverified]**. |
| **6-DOF arm (paused)** | `robot_arm\` (repo `robot-arm-physical-ai`) | MG996R open-loop arm tracked with two cameras and coloured dots; chat-driven control with a local Gemma VLM. README says paused 2026-08-12 after switching to the SO-101 **[README]**. | The honest "what carries over" section in its README is good portfolio material. |
| **SO-101 arm** | `so101_arm\` | Successor project: Feetech servos with absolute position readout, learned manipulation **[README]**. | |

### Feature (ML / software), shorter write-ups

| Project | Path | What it is |
|---|---|---|
| Poker collusion detection | `kaggle poker thing\` (repo `poker-collusion-detection`) | Kaggle: find colluding player pairs in ~2M synthetic hands. LightGBM; best public leaderboard score 0.78370 (up from 0.131 first submission) **[README]**. |
| quietype | `quietype\` (repo `quietype`) | Local-first open-source dictation + voice assistant (Whisper or Gemma 4, hold-to-talk hotkeys) **[README]**. |
| fly brain vs ARC-AGI-2 | `fly brain experiments\` | Asks whether a real fly connectome's motion neurons can identify rigid transforms in ARC puzzles. README is explicit that it does **not** solve ARC and that only 11/1000 tasks qualify. Good example of an honest negative-framed study **[README]**. |
| Zen Agent | `agentic_browser\` | Local agentic browsing assistant as a Firefox/Zen extension with a Python sidecar and cost-aware model routing **[README]**. |
| Harness | `harness\` | Open-source CLI coding agent, provider-agnostic, approval gates **[README]**. |
| ARC Prize | `arc agi prize\` | Kaggle ARC-AGI entry. Contents (`solver`, `notebook`, `research`) **not reviewed**. |
| Voice call agent | `call_voice_agent\` (repo `call-voice-agent`) | **Not reviewed** (has `website\`, `worker\`, setup docs). |

### Not reviewed (do not describe without opening them)

`Chest Xray Few shot`, `eggplant_fewshot`, `cctv_surviallance`, `codeforces`,
`eyantra robotics competition` (one file, `task1a.py`), `hackerhouse`,
`worldquant*`, `shorts automation`, `stt_product_idea`, `visual-perception`,
`pi-zero-2w-server`, `qwen-image-2.1`, `chi bench`, `ai sports thing`,
`ai_agent_security_kaggle`, `finapp-*`, `browser_agent`,
`agentic-assistance-project`.

`light weight vla research\` is empty. `robotics research thing online\` is
the user's own experiment run (SmolVLA baseline, eval, videos); I have not
read it and it may be unfinished or running. Leave it out unless the user
points at it.

### Open-source contributions

Folders `open source contributions\` (griptape, opencv) and
`github-opensource-contributions\` (jaeger-ui) exist **[file]**. Whether any
PR was opened or merged is **[unverified]**. Check the GitHub account (or ask)
and link only merged PRs.

## 6. Arachne (flagship): what is true, what is not

**What it is.** A Vorpal hexapod (open-source, Thingiverse thing 2513566, by
user "vorpal") built by the user: 6 legs x 2 DOF = 12 MG90S servos on one
PCA9685, controlled by an Arduino Uno Q (STM32U585 MCU plus a Qualcomm
Linux SoC). Wiring and channel map: `arachne\vorpal-servo-wiring.md`. Full
engineering log: `arachne\vorpal-mujoco\HANDOVER.md`. Research plan:
`arachne\research-questions.md`.

**Verified, safe to state (with the caveat in brackets):**
- A MuJoCo digital twin built from the OnShape CAD assembly plus photos of the
  real robot at servos=90/90 (`vorpal-mujoco\build_neutral.py`). The CAD file
  is a posed display model; only 4 of 6 legs are mechanically consistent, so
  the neutral pose uses those four (per-part medians) and the photos.
- A web control page with the twin running in the browser (three.js), a
  per-channel mapping the user confirmed on the real robot (2026-10-04), and
  an animations panel (Stand, Walk forward, Sit, steer slider). **[file]**
- Collision clearance: the twin's convex-hull geometry is about **5 mm
  optimistic** at the one real touch point the user measured (servo 0=112,
  servo 1=46). Gaits are therefore verified to a 10 mm margin. Forward gait
  minimum leg gap: 28 mm in sim, checked over every frame (`gaits.py`).
- Physics simulation (`build_physics.py`, `sim_walk.py`): forward tripod gait
  walks at about 32-33 mm/s, tilt under 1 degree, no body or hinge ground
  contact. Untrimmed it veers right (-139 mm sideways, -24 degrees in 6.3 s);
  with steer = -0.15 it is +17 mm and +4 degrees. Open loop, heading still
  wanders +-20-35 degrees over 16 s.
  **All of this is simulation.** Body mass (522 g) and servo torque are
  estimates, not measurements.
- Each wrong-model episode is a real story (guessed rectangles, STL-only
  pivots, a "fixed" model that was still mis-posed). It is described in
  `vorpal-mujoco\HANDOVER.md` and makes honest "lessons learned" material.

**Not true or not yet known. Do not claim:**
- **No real-robot walking results exist** as far as I know. A hinge cracked
  during assembly and was reprinted. After that I brought the web control back
  up on the board (between 2026-10-05 and 10-08) and the user said it was
  "working well", but no walk measurements were reported to me. Ask the user
  for any real test results.
- **Do not repeat "0.0000 mm reproduction error vs CAD" or "exact" twin.**
  That phrase is in `idea.md` from an early build that turned out to be posed
  wrong. Say "CAD-derived, validated against photos and a measured collision
  point".
- **No paper exists.** `research-questions.md` is a plan ("Paper A: Walking
  on $3 servos", target CoRL 2027, RSS 2027 stretch). Describe it as ongoing
  research, never as a result, and don't name venues as targets unless the
  user agrees to that on a public page.
- Don't claim novelty ("first", "no one has") beyond what
  `research-questions.md` says, and only with its sources.
- Don't use `vorpal-mujoco\deprecated\` (ChatGPT's guessed model) or
  `current_sim.png` (2026-09-12, from the old wrong model).

**Assets (all under `C:\projects\arachne`):**

| File | Size | Use |
|---|---|---|
| `photos\WIN_20261004_*.jpg` (9) | ~250-300 KB each | Real robot at 90/90 from top, side and bottom. Open them to see which is which. |
| `vorpal-mujoco\compare_photo_vs_sim.png` | 0.9 MB | Photo vs twin side by side. Strong hero or twin-validation image. |
| `vorpal-mujoco\neutral_v1.png` | 0.3 MB | Twin render. |
| `vorpal-mujoco\sim_forward_frames.png` | 0.5 MB | Contact sheet of the physics walk. |
| `vorpal-mujoco\sim_forward.gif` | **9.9 MB** | Physics walk, 160 frames. Too heavy for a page: convert to MP4/WebM (ffmpeg availability **[unverified]**) or re-render smaller with `sim_walk.py` (`video=1`). |
| `servo-web-control\assets\` | meshes: `base.stl` 4.8 MB, `leg10.stl` 0.6 MB, `hinge.stl` 0.2 MB | The interactive twin: `index.html`, `app.js`, `sim3d.js`, `gait.js`, `style.css`, `twin_model.json`, `gaits\forward.json`. |

**Embedding the live twin.** The page ran from a plain static server on
2026-10-04 (`python -m http.server`): the twin rendered, the steer slider
worked, and "Walk forward" played in preview mode. three.js loads from
jsdelivr, so it needs internet. For the portfolio, copy the folder into the
site (for example `site\arachne\twin\`) and:
- hide or remove the "Send to robot" checkbox and the channel-mapping save
  (they call board endpoints `/set_all`, `/set`, `/mapping`, which don't exist
  on a static host). Whether the page logs errors there I did not check;
  open the console.
- keep the default mapping and the preview-only mode.
- add a short note that it is a simulation of the robot, not live video.

**License and attribution (important).** The Vorpal parts are
**Creative Commons Attribution-ShareAlike** (`Vorpal needed\LICENSE.txt`
**[file]**). The twin's STL meshes are derived from them, so publishing the
twin means crediting "Vorpal The Hexapod Walking Robot by vorpal on
Thingiverse (thing 2513566)" with a link, and licensing the derived meshes
under the same CC BY-SA terms. Check the license text before publishing and
tell the user. Vorpal sells kits; link to the original.

**Hardware as described by Arduino** (https://arduino.cc/product-uno-q):
Qualcomm Dragonwing QRB2210, quad-core Cortex-A53 up to 2.0 GHz, Adreno 702
GPU; STM32U585 (Cortex-M33, 160 MHz) as the MCU.

## 7. Suggested order of work

1. Send the section 3 questions to the user, one message.
2. Pick the stack (default: static HTML/CSS), set up the site folder and
   preview with `python -m http.server`.
3. Build the landing page and the Arachne page first (most assets, the
   strongest story); then the solar paper page; then the rest.
4. For every number you put on a page, put its source file next to it in a
   comment or in a `CLAIMS.md` in the site folder, so the user can audit it.
5. Check at phone width, with images compressed and no file over ~1 MB except
   the twin meshes and the video.
6. Before publishing: secret scan, license/attribution check, and the user
   reads the claims list.

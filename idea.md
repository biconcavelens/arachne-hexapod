# Arachne Research Agent — idea.md

Seed document for a Hermes agent (orchestrator + subagents) whose job is to
**find and develop research-paper-worthy topics while I build Arachne**, a
Vorpal-derived hexapod, with a bias toward **generalist, lightweight VLAs
(vision-language-action models) for hexapods**.

The agent does two things continuously: (1) **logs** the build as a lab
notebook so nothing we learn is lost, and (2) **researches** the literature
against what we're actually doing, so paper ideas come from real friction,
not armchair speculation.

---

## 1. The platform (what the agent is researching *on*)

- **Robot**: Vorpal Combat Hexapod (open-source, 3D-printed). 6 legs × 2 DOF
  (hip = horizontal swing, knee = lift) = **12 MG90S servos** on one PCA9685.
  Hinge-linkage legs, not serial 3-DOF chains. ~$150 BOM.
- **Brain**: Arduino **Uno Q** — STM32 MCU (Zephyr, real-time servo loop) +
  Qualcomm Dragonwing Linux SoC (runs Python, Docker, small NN models via
  App Lab "bricks"). Two cores talk over an RPC Bridge. **This is an edge-AI
  legged robot for under $200. That is the whole pitch.**
- **Digital twin**: MuJoCo model built from the *actual OnShape assembly*
  (exact mate axes, 0.0000 mm reproduction error vs CAD). Real hip/knee
  joint locations, real hinge geometry. Sim-to-real is on the table.
- **Control surface already built**: web UI (sliders, jog, live 2D twin),
  REST endpoints (`/set`, `/state`, `/raw`), `arduino-app-cli` deployment.
- **Known real-world messiness (research fodder, not bugs)**: per-servo
  range variance (ch0 tops at 168°, ch1 at 170°, "180°" nominal), spline-tooth
  8° horn quantization needing per-joint trim, adjacent-hip self-collision
  (real touch at Δ+22°/−44°, far tighter than naive geometry predicted),
  I2C/bridge latency, USB/adb flakiness.

Ground-truth files live in the repo: `vorpal-servo-wiring.md`,
`vorpal-mujoco/HANDOVER.md`, `vorpal-mujoco/onshape_assembly.json`.

---

## 2. Research thesis (the bet)

> Big VLAs (OpenVLA, π0, RT-2-class) target 6–7-DOF arms with rich action
> spaces. A hexapod with 12 low-precision servos and a **small, structured
> gait manifold** is a different regime: the action space is low-dimensional
> and highly compressible, the hardware is cheap and reproducible, and the
> compute is edge-class. This regime is under-served and is where a
> **tiny generalist VLA** can plausibly work end-to-end on-device.

Everything below is a way to attack that bet, or a spin-off that pays for
itself even if the bet fails.

---

## 3. Candidate paper topics (ranked by feasibility × novelty)

Each has a one-line "why us" — the unfair advantage from building Arachne.

### T1 — Primitive-space VLA for legged locomotion  ★ top pick
Instead of emitting 12 joint angles per step, the VLA emits **gait
primitives + parameters** (walk/turn/strafe/lift-leg-N, speed, stride,
height). Vorpal already ships ~60 hand-authored motions — a ready-made
primitive vocabulary. A sub-100M model choosing among primitives is a
tractable on-device target; a 450M SmolVLA-style model regressing joint
angles is not.
*Why us*: we have the primitive library, the twin to render every primitive,
and the hardware to test whether language→primitive generalizes.
*Experiment*: language-conditioned primitive selection, sim-trained, real-
evaluated. Compare vs. direct joint-angle regression at matched param count.

### T2 — How exact does the twin need to be? (CAD-exact vs. simplified sim-to-real)
We built the model three ways: hand-guessed rectangles → STL-only meshes →
OnShape-exact. Each gave a *different* self-collision boundary (56° → 32° →
real). Quantify how twin fidelity affects transfer of collision-aware
policies on cheap servos.
*Why us*: we have all three fidelity levels of the *same* robot, already.
*Experiment*: train the same policy in each twin, measure real-world
collision rate / trim needed.

### T3 — Self-collision-aware action manifolds for hinge-linkage legs
Adjacent-hip interference is coupled (leg i's safe range depends on leg
i±1), not a per-joint limit. Learn or enforce a collision-free action
manifold and show it as a drop-in constraint for any policy head.
*Why us*: we measured the real coupled boundary and have the exact hinge
geometry in the twin. Small, clean, publishable.

### T4 — Servo-variance-robust control on hobby hardware
Per-unit MG90S variance and 8° horn quantization are real and systematic.
Treat them as a domain-randomization axis; show policies that survive
un-calibrated servo swaps. Nobody publishes on $3 servos; everyone uses them.
*Why us*: we have the measurements and 12 real units.

### T5 — On-device VLA on Uno Q: latency-aware action chunking
Quantized tiny VLA on the Dragonwing SoC driving a 50 Hz MCU servo loop over
an RPC bridge. Async action chunking to hide inference latency; measure the
accuracy-vs-latency frontier on real hardware.
*Why us*: the dual-core MCU+SoC split is exactly the deployment shape this
needs, and it's a $60 board.

### T6 — Arachne-Bench: a <$200 reproducible hexapod benchmark for VLA research
Open twin + open hardware + standard tasks + language instructions. Benchmark
papers get cited and force us to do everything above properly.
*Why us*: it's all open-source already; we just have to package it.

### T7 — Language-conditioned gait modulation ("walk carefully", "hurry", "step high")
Continuous modulation of gait params from natural language. Cheap to train,
easy to demo, natural extension of T1.

### T8 — Digital twin as a teleop/data engine
Use the web UI + twin to collect demonstrations *without* the robot (twin-
only teleop), then measure how much real data is needed on top. Data-
efficiency angle for small VLAs.

**Kill criteria** (the critic subagent applies these): needs >1 GPU-week;
needs >12 DOF; needs sensors we don't have (no depth, no IMU yet — camera
is the only rich sensor); already done with a hexapod in the last 18 months
without a cheap-hardware or on-device angle.

---

## 4. Agent architecture

One orchestrator, four subagents. Each subagent owns a directory under
`research/` and writes only there. The orchestrator merges into `digest/`.

```
research/
  journal/      <- Logger        (append-only lab notebook)
  lit/          <- Scout         (papers, notes, bib)
  topics/       <- Critic        (one file per topic, living scorecard)
  experiments/  <- Planner       (minimal runnable experiment specs)
digest/         <- Orchestrator  (weekly summary, decisions, next actions)
```

### Orchestrator
- Runs on a schedule (nightly cheap pass, weekly deep pass) and on demand.
- Reads new journal entries → asks Scout for related lit → asks Critic to
  rescore affected topics → asks Planner to (re)spec the top 2.
- Produces `digest/YYYY-WW.md`: what changed, what we learned, what's now
  the most promising paper, and the single next experiment to run.
- Never invents build facts. If it needs a number it doesn't have, it asks
  me or reads the repo.

### Logger subagent (the lab notebook)
Turns build activity into a dated, searchable record. Sources:
- Git commits / diffs in `C:\projects\arachne`
- Handover docs (`vorpal-servo-wiring.md`, `vorpal-mujoco/HANDOVER.md`)
- Anything I paste in chat tagged `#log`
- Calibration/measurement artifacts (JSON, PNG renders)

Each entry: `journal/YYYY-MM-DD.md` with sections
`Did / Measured / Broke / Decided / Open`. Measurements get a machine-
readable block (yaml fenced) so the Planner can pull numbers later.
**Rule: log the number, the method, and the uncertainty. "~30°" is not a
measurement; "32° via 2-point binary search, ±2°, symmetric case" is.**

### Scout subagent (literature)
- Weekly sweep: arXiv (cs.RO, cs.LG, cs.CV), Semantic Scholar, HF papers,
  for {VLA, legged, hexapod, quadruped, edge, on-device, tiny, distillation,
  action chunking, sim-to-real, self-collision, cheap hardware}.
- Maintains `lit/bib.bib` + one note per paper in `lit/notes/` with:
  claim, method, hardware, param count, what they *didn't* do.
- Flags **gaps** explicitly: "no paper does X on <1B params on a legged
  platform" is the output we want, not summaries.
- Seed reading list to verify/expand (do not trust from memory — fetch):
  OpenVLA, π0 / π0.5, SmolVLA, TinyVLA, Octo, RT-2, plus whatever the
  current SOTA small-VLA and legged-locomotion-foundation-model papers are.

### Critic subagent (topic scoring)
- One scorecard per topic in `topics/T*.md`: novelty (1–5), feasibility on
  Arachne (1–5), effort (person-weeks), venue fit (RA-L / CoRL / ICRA /
  workshop), *strongest objection*, *what evidence would kill it*.
- Rescore whenever Scout finds prior art or Logger records a blocker.
- Must be adversarial. Its job is to save us from a paper that's already
  been written.

### Planner subagent (minimal experiments)
- For the top 2 topics, write `experiments/E*.md`: hypothesis, the
  *smallest* experiment that could falsify it, exact assets needed (twin
  config, servo channels, data), success metric, time estimate.
- Prefers experiments runnable in the MuJoCo twin first, real robot second.
- Outputs runnable scaffolding when cheap (a config, a script stub), never
  full pipelines unprompted.

---

## 5. Logging schema (so subagents can read each other)

```yaml
# inside any journal entry
measurements:
  - name: hip_symmetric_collision_limit
    value: 32
    unit: deg
    method: 2-point binary search, real hardware, legs 0/1
    uncertainty: ±2
    date: 2026-09-07
    supersedes: hip_symmetric_collision_limit@2026-09-06   # was 56, naive model
decisions:
  - what: use OnShape mate axes instead of STL-derived pivots
    why: STL pivot radius was 68.5mm, true is 57.89mm
    date: 2026-09-08
open:
  - Uno Q SoC: what is the real inference budget (ms) for a 50M-param model?
```

Every claim in a digest must trace to a journal entry or a lit note.

---

## 6. Cadence & deliverables

- **Nightly** (cheap): Logger ingests the day, Orchestrator updates open
  questions. <5 min compute.
- **Weekly** (deep): full Scout sweep, Critic rescore, Planner respec,
  `digest/` entry. Ends with **one sentence**: "Most promising paper this
  week is T_, next experiment is E_."
- **Milestone**: when a topic hits novelty≥4 & feasibility≥4, Planner
  produces a 1-page extended abstract draft in `topics/T*_abstract.md`.

Deliverables over ~3 months: a ranked, evidence-backed shortlist of 2–3
paper topics with a spec'd first experiment each, a complete lab notebook
of the build, and a bib of the actual current SOTA.

---

## 7. Constraints & guardrails

- **Hardware truth wins.** If the twin and the robot disagree, the robot is
  right and the discrepancy is itself a logged finding (see T2).
- **No invented numbers, no invented citations.** Scout fetches; Critic
  checks; anything unverifiable is tagged `unverified` and can't reach a
  digest.
- **Small first.** Every topic must have a version runnable on one laptop
  GPU and the twin before touching the robot.
- **Log failures louder than successes.** The self-collision saga (three
  wrong models before ground truth) is worth more to a paper's "lessons"
  section than any success.
- The agent proposes; I decide. Nothing is "the plan" until it's in a
  digest I've acknowledged.

---

## 8. Open questions for the first Scout sweep

1. What is the smallest published VLA (params) that controls a legged robot
   end-to-end, and what hardware did it run on?
2. Has anyone used *hand-authored motion primitives* as a VLA action space?
3. What's the state of on-device VLA inference on Qualcomm-class edge SoCs
   (QNN / ONNX / ExecuTorch paths)?
4. Any prior work on self-collision constraints for hinge-linkage (not
   serial-chain) legs?
5. Is there an existing sub-$500 open legged benchmark? If yes, how does
   Arachne differ (cheaper? more DOF? edge compute?).

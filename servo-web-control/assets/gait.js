// Animation player. Gaits live in gaits/*.json, generated and clearance-checked
// in MuJoCo by vorpal-mujoco/gaits.py, in SIM joint angles. They're converted
// to servo angles here with the channel mapping (Twin.simPoseToServos), so a
// mapping change never needs a regenerate. Each gait carries a "rest" pose =
// standing (knees pushed down so the knee hinges clear the ground).
//
// Timing is wall-clock based: every tick computes the pose from elapsed time,
// so a slow page skips frames instead of slowing the robot down.
//
// "Send to robot" is off by default: the twin previews the motion and then
// returns to wherever the sliders were, so the page never shows a pose the
// robot isn't in. With it on, poses go out as /set_all calls (~20/s max).
const RAMP_S = 0.6;     // ease between poses
const TICK_MS = 30;     // UI/twin update interval
const GAIT = 'gaits/forward.json';

let running = false;
let stopRequested = false;
let inFlight = false;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const $ = (id) => document.getElementById(id);
const lerp = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
const ease = (t) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, t));

// Steer, read live: side legs' hip swing x (1 + steer * side), same as
// gaits.py steered(); clamped to the range gaits.py clearance-checked.
const steerInput = $('anim-steer');
const showSteer = () => { $('anim-steer-val').textContent = Number(steerInput.value).toFixed(2); };
steerInput.addEventListener('input', showSteer);
steerInput.addEventListener('dblclick', () => { steerInput.value = 0; showSteer(); });
function steered(gait, pose) {
  const s = Math.max(-gait.steer_max, Math.min(gait.steer_max, Number(steerInput.value)));
  return pose.map((v, k) => (k < gait.hip_side.length ? v * (1 + s * gait.hip_side[k]) : v));
}

function sendFrame(servos) {
  if (inFlight) return;  // drop a frame rather than queue up behind the board
  inFlight = true;
  fetch(`/set_all?a=${servos.map((v) => Math.round(v)).join(',')}`, { method: 'POST' })
    .catch(() => {})
    .finally(() => { inFlight = false; });
}

function show(servos, toRobot) {
  servos.forEach((v, ch) => window.ServoUI.setServo(ch, Math.round(v)));
  if (toRobot) sendFrame(servos);
}

async function ramp(to, toRobot, seconds = RAMP_S) {
  const from = window.ServoUI.angles();
  const t0 = performance.now();
  for (;;) {
    const t = (performance.now() - t0) / (seconds * 1000);
    show(lerp(from, to, ease(t)), toRobot);
    if (t >= 1) break;
    await sleep(TICK_MS);
  }
  if (toRobot) { await sleep(80); inFlight = false; sendFrame(to); }  // make sure the final pose lands
}

// Runs `body` with a loaded gait; in preview mode, puts the sliders back after.
async function run(label, body) {
  if (running) return;
  running = true;
  stopRequested = false;
  const toRobot = $('anim-send').checked;
  const speed = Number($('anim-speed').value);
  const start = window.ServoUI.angles();
  try {
    const gait = await (await fetch(GAIT)).json();
    const servos = (pose) => window.Twin.simPoseToServos(gait.joints, pose);
    $('anim-status').textContent = `${label}${toRobot ? '' : ' (preview)'} — min leg gap ${gait.verified.min_gap_mm} mm in sim`;
    await body({ gait, toRobot, speed, servos });
    if (!toRobot) await ramp(start, false);
    $('anim-status').textContent = toRobot ? `${label}: done` : 'preview done';
  } catch (e) {
    $('anim-status').textContent = `${label} failed: ${e}`;
    console.error(e);
  } finally {
    running = false;
  }
}

$('anim-walk').addEventListener('click', () => run('Walk forward', async ({ gait, toRobot, speed, servos }) => {
  await ramp(servos(gait.rest), toRobot);        // stand up first
  await ramp(servos(steered(gait, gait.frames[0])), toRobot);   // then into the gait
  const n = gait.frames.length;
  const t0 = performance.now();
  while (!stopRequested) {
    const f = (((performance.now() - t0) / 1000) * gait.fps * speed) % n;   // fractional frame
    const i = Math.floor(f);
    show(servos(steered(gait, lerp(gait.frames[i], gait.frames[(i + 1) % n], f - i))), toRobot);
    await sleep(TICK_MS);
  }
  await ramp(servos(gait.rest), toRobot);        // end standing
}));

$('anim-stand').addEventListener('click', () => run('Stand', async ({ gait, toRobot, servos }) => {
  await ramp(servos(gait.rest), toRobot);
  if (!toRobot) await sleep(1200);                // let the preview be seen
}));

$('anim-sit').addEventListener('click', () => run('Sit (all 90)', async ({ gait, toRobot, servos }) => {
  await ramp(servos(new Array(gait.joints.length).fill(0)), toRobot);
  if (!toRobot) await sleep(1200);
}));

$('anim-stop').addEventListener('click', () => { stopRequested = true; });

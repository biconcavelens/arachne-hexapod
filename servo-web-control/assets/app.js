const NUM_SERVOS = 12;
// The 3D twin (sim3d.js) loads three.js from a CDN. If that fails, the sliders
// must still work, so fall back to a no-op twin instead of breaking the page.
const Twin = window.Twin || { init: async () => {}, setChannel: () => {} };
const rows = document.getElementById('rows');
const servoInputs = []; // index -> {input, out}

function setServo(ch, angle) {
  const { input, out } = servoInputs[ch];
  input.value = angle;
  out.textContent = angle;
  Twin.setChannel(ch, Number(angle));
}

const setTimers = {}; // per-channel debounce, so the "all" slider doesn't drop channels
function sendSet(ch, angle) {
  clearTimeout(setTimers[ch]);
  setTimers[ch] = setTimeout(() => {
    fetch(`/set?channel=${ch}&angle=${angle}`, { method: 'POST' });
  }, 80); // debounce while dragging
}

function makeRow(ch, initial) {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML =
    `<label for="s${ch}">Servo ${ch}</label>` +
    `<input type="range" id="s${ch}" min="0" max="180" value="${initial}">` +
    `<output for="s${ch}">${initial}</output>`;
  rows.appendChild(row);

  const input = row.querySelector('input');
  const out = row.querySelector('output');
  servoInputs[ch] = { input, out };
  Twin.setChannel(ch, Number(initial)); // sync twin to the real starting angle, not just the default 90
  input.addEventListener('input', () => {
    out.textContent = input.value;
    Twin.setChannel(ch, Number(input.value));
    sendSet(ch, input.value);
  });
}

function jogSet(ch, angle) {
  fetch(`/set?channel=${ch}&angle=${angle}`, { method: 'POST' });
}

function initJogPanel() {
  const select = document.getElementById('jog-channel');
  const angleOut = document.getElementById('jog-angle');

  for (let ch = 0; ch < NUM_SERVOS; ch++) {
    const opt = document.createElement('option');
    opt.value = ch;
    opt.textContent = `Servo ${ch}`;
    select.appendChild(opt);
  }

  const currentAngle = () => Number(servoInputs[select.value].input.value);
  const applyAngle = (angle) => {
    angle = Math.max(0, Math.min(180, angle));
    const ch = Number(select.value);
    setServo(ch, angle); // updates that channel's row slider too
    angleOut.textContent = angle;
    jogSet(ch, angle); // sends immediately, no debounce - jog clicks are discrete
  };

  select.addEventListener('change', () => {
    angleOut.textContent = currentAngle();
  });

  document.querySelectorAll('.jog-buttons button[data-delta]').forEach((btn) => {
    btn.addEventListener('click', () => {
      applyAngle(currentAngle() + Number(btn.dataset.delta));
    });
  });

  document.getElementById('jog-center').addEventListener('click', () => applyAngle(90));

  angleOut.textContent = currentAngle();
}

function initAllSlider() {
  const allInput = document.getElementById('all');
  const allOut = document.querySelector('output[for="all"]');
  allInput.addEventListener('input', () => {
    allOut.textContent = allInput.value;
    for (let ch = 0; ch < NUM_SERVOS; ch++) {
      setServo(ch, allInput.value);
      sendSet(ch, allInput.value);
    }
  });
}

// For the animation player (gait.js)
window.ServoUI = {
  setServo,
  angles: () => servoInputs.map(({ input }) => Number(input.value)),
};

async function init() {
  try {
    await Twin.init();
  } catch (e) {
    console.error('3D twin failed to load; controls still work', e);
  }
  let angles = new Array(NUM_SERVOS).fill(90);
  try {
    const res = await fetch('/state');
    const data = await res.json();
    if (Array.isArray(data.angles)) angles = data.angles;
  } catch (e) {
    // fall back to defaults if the board isn't ready yet
  }
  for (let ch = 0; ch < NUM_SERVOS; ch++) {
    makeRow(ch, angles[ch] ?? 90);
  }
  initJogPanel();
  initAllSlider();
}

init();

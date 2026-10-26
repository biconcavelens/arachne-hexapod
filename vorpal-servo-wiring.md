# Vorpal Hexapod — Servo Wiring

12 MG90S servos on a single PCA9685 (I2C address 0x40), driven from the Uno Q via
the `servo-web-control` app (`sketch/sketch.ino` + `python/main.py`).

## I2C wiring (Uno Q)

The sketch uses `Wire`. In the Uno Q Arduino core (`arduino:zephyr` 0.90.0,
`arduino_uno_q_stm32u585xx.overlay`, `i2cs = <&i2c2>, <&i2c4>, <&i2c3>`):

| Object | Bus | Pins |
|---|---|---|
| `Wire`  | I2C2 | **D20 = SDA, D21 = SCL** (the SDA/SCL header pins next to AREF) |
| `Wire1` | I2C4 | Qwiic connector (GND, +3V3, SDA, SCL) |
| `Wire2` | I2C3 | A4 = SDA, A5 = SCL |

So the PCA9685 (which works with `Wire`) must be on **D20/D21**, not A4/A5 (the
A4/A5 advice in the first-session wiring guide is the UNO R3 convention and is
wrong for the Uno Q). Inferred from the board files; not seen in a photo (the
electronics are under the dome). The IMU (MPU6050, 0x68) goes on the same two
wires. Logic is 3.3 V (pins are 5 V tolerant except A0/A1), so the IMU is a
3.3 V part on this bus: check that the PCA9685's VCC pin is on 3.3 V, not 5 V
(a 5 V VCC pulls the bus to 5 V through the breakout's pull-ups).

## Channel map (confirmed on the robot 2026-10-04)

Leg numbering follows the digital twin: L0 is the front leg, then L1..L5 go
counter-clockwise seen from above (sim hip axes at 3.3 + 60k deg).

| Leg | Position (from above) | Hip channel | Knee channel |
|-----|-----------------------|-------------|--------------|
| L0  | front                 | 0           | 6            |
| L1  | front-left (60 deg CCW)  | 3        | 9            |
| L2  | rear-left (120 deg)   | 4           | 10           |
| L3  | rear (180 deg)        | 5           | 11           |
| L4  | rear-right (240 deg)  | 2           | 8            |
| L5  | front-right (300 deg) | 1           | 7            |

Hip and knee of a leg are still channel N and N+6, but the legs are NOT in
channel order around the body. All hip servos turn opposite to the sim's +
(CCW from above): sim hip angle = -(servo - 90). Knees: sim knee angle =
+(servo - 90), + lifts the foot. This is saved on the board as
twin_mapping.json and is the web page's default.

## Status log

- **2026-09-01**: Hips (channels 0–5) wired, tested via manual jog and 90/60/120
  sweep — confirmed moving correctly after a wiring fix (root cause: suspected
  PCA9685 V+ power rail, not the I2C/software path — software commands and
  register readback were confirmed correct throughout, see `/raw` diagnostic
  endpoint in `python/main.py`).
- Knee servos (6–11) not yet connected.
- Servo-specific notes: channel 0 tops out ~168°, channel 1 ~170° (normal
  per-unit MG90S variance, no buzzing/straining — not a software clipping
  issue, confirmed via the `map()` math and PCA9685 register readback).
- **2026-09-07**: Legs and hinges assembled per the official assembly guide.
  Hips confirmed moving after re-fitting into the electronics caddy.

## Hip-to-hip interference (digital twin)

Adjacent hips can physically collide depending on each other's angle — this
isn't a fixed per-channel min/max, it's coupled between neighbors. Simulated
in `vorpal-sim/hip_interference_sim.py` using real geometry measured from the
actual STL files (hip pivot layout off the base, leg envelope off a single
leg part) — see script docstring for full measurement provenance. Results in
`vorpal-sim/hip_interference_results.json`.

**2026-09-07 CORRECTION**: the first version of this model (bare-leg width,
15mm half-width, no hinge) was proven wrong by a real hardware touch test —
legs 0/1 actually made contact at servo0=112° (delta +22), servo1=46°
(delta −44), far inside what that model predicted safe. The hinge that
connects the leg to the hip horn adds real bulk right at the pivot that the
leg-only STL never captured. Half-width was refit to 32.0mm (rounded up
from an exact 31.58mm fit) to reproduce that touch point — see
`calibrate_from_real_touch.py`. Re-verified live: with the corrected model,
the twin now flags legs 0/1 as colliding at exactly that same servo0=112/
servo1=46 configuration.

**This is a one-point calibration and should be treated as provisional**
until 2-3 more real touch points (a roughly-symmetric one, and a different
leg pair) confirm it. One data point with one free parameter (half-width)
can't distinguish "the leg is wider than measured" from "the true shape
isn't a uniform rectangle" (plausible — hinge bulk concentrated at the
pivot, tapering toward the foot, would fit better than a uniform width).

**Findings with the corrected model** (same for all 6 adjacent pairs, by
symmetry):
- One hip alone, neighbor neutral: safe up to **+50°** toward the neighbor
  (down from the original model's 90° — the hinge bulk matters even for a
  single leg against a stationary neighbor, not just the symmetric case).
- Both adjacent hips swinging toward each other symmetrically: safe up to
  **|delta| ≤ 32°, on each leg individually** — not a sum of the two.

**Known limitation**: still a full-width rectangle for the leg's entire
reach, now further complicated by the hinge not being modeled as its own
shape at all (only absorbed into a uniform width fudge factor). Upgrade
path: either gather more real touch points to fit a better (non-uniform)
shape, or pull the public OnShape assembly for the real leg+hinge geometry.

This constraint still needs to be applied somewhere before the trim tool is
"done" — either as a live clamp in `sketch.ino` (reject/clip a hip command
if it would violate the neighbor pair's budget) or as a check in the gait
planner later. Not yet implemented as a hard clamp — currently only
visualized live (flashes red) in the digital twin, see below.

## Digital twin (real-time, browser)

`assets/twin.js` + the `#twin-panel` in `assets/index.html`: a live 2D
top-down canvas showing all 6 legs, driven by the exact same angle state as
the sliders/jog panel (every path routes through `setServo()` in `app.js`,
which forwards to `Twin.setChannel()`). Uses the same measured geometry as
the offline sim (`hip_interference_results.json`, copied into `assets/` so
both stay in sync) and does a live SAT polygon-overlap check between
adjacent legs each frame — a colliding pair fills red and a warning banner
names the legs. Knee angle is shown as a text readout per leg (no height/3D
— this is intentionally 2D only, matching what the underlying model actually
represents; a 3D twin would need the leg's true assembled orientation, which
isn't recoverable from the print-oriented STL with confidence, see the
"Known limitation" above).

## Known quirks

- The board's `arm_autostart.sh` (a `@reboot` cron job) restarts the old
  "Arm Voice Control" app (`pca9685-i2c-test`) on every power cycle, which
  steals the MCU from `servo-web-control`. After a power cycle, run:
  ```
  arduino-app-cli app stop user:pca9685-i2c-test
  arduino-app-cli app start //home/arduino/ArduinoApps/servo-web-control
  ```
  (and `pkill -f arm_autostart.sh` if it keeps re-stealing it).
- `adb forward tcp:7000 tcp:7000` needs re-running most times the USB
  connection is touched — check `adb forward --list` if `localhost:7000`
  stops responding.

## Next steps

1. Wire knee servos 6–11 as legs go on.
2. Fill in physical position column above.
3. Find and record safe min/max angle per channel (Phase 5 of the calibration
   plan) once all legs are assembled.
4. Build the trim/calibration tool (per-channel offset, persisted) — Phase 3/4.

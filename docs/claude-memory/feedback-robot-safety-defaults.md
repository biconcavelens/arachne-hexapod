---
name: feedback-robot-safety-defaults
description: "Standing rules for touching the real robot, the Uno Q board and the user's processes"
metadata:
  node_type: memory
  type: feedback
  originSessionId: cca4871e-0aac-46b0-8408-35683725756e
  modified: 2026-10-17T08:52:22.308Z
---

- "Send to robot" on the web page stays OFF by default; the twin previews first. First real moves are in the air at 0.5x before the ground.
- Don't kill processes I didn't start (the user's experiment in `C:\projects\robotics research thing online`, on the old Windows install, was left alone pending their word).
- On the Uno Q, the autostart cron relaunches "Arm Voice Control" at every boot. To take the board over: `pkill -f '[a]rm_autostart.sh'`, `arduino-app-cli app stop user:pca9685-i2c-test`, then start `servo-web-control`. When asked to take things down: stop the app, remove the `adb forward`, close the browser pane.
- A board reset drops adb ("offline"); `adb reconnect offline` brings it back, then redo `adb forward tcp:7000 tcp:7000`.

**Why:** unexpected robot motion or killing the user's work are the costly mistakes here; the rest are time-wasters I already hit once.

**How to apply:** follow these without asking again; ask only when a step falls outside them. Related: [[project-paper-a-servo-identification]].
